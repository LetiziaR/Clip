import ast
import os
import re

import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


class PTBXL(Dataset):
    """Minimal PTB-XL dataset for CoCa.

    Output format matches trainer/evaluation expectations:
    - always: ecg, input_ids, attention_mask
    - if dual tokenizer: decoder_input_ids, decoder_attention_mask
    - if return_labels: labels (multi-hot from label_col)
    """

    def __init__(
        self,
        root,
        tokenizer=None,
        encoder_tokenizer=None,
        decoder_tokenizer=None,
        use_dual_tokenizer=False,
        sampling_rate=500,
        folds=None,
        target_length=None,
        return_text=True,
        normalize=True,
        text_max_length=128,
        return_labels=False,
        label_col="scp_codes",
        label_threshold=0.0,
        label_map=None,
        text_source="report",
        normalize_mode="global",
        return_demographics=False,
    ):
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if text_source not in ("report", "pseudo_report"):
            raise ValueError("text_source must be 'report' or 'pseudo_report'")

        self.root = root
        self.return_text = return_text
        self.normalize = normalize
        self.text_max_length = int(text_max_length)
        self.return_labels = return_labels
        self.label_col = label_col
        self.label_threshold = float(label_threshold)
        self.label_map = label_map
        self.text_source = text_source
        self.normalize_mode = normalize_mode
        self.use_dual_tokenizer = use_dual_tokenizer
        self.return_demographics = return_demographics

        self.encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else tokenizer
        self.decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else tokenizer

        if self.return_text and self.encoder_tokenizer is None:
            raise ValueError("A tokenizer is required when return_text=True")

        db_path = os.path.join(root, "ptbxl_database.csv")
        df = pd.read_csv(db_path, index_col="ecg_id")

        if folds is not None:
            df = df[df["strat_fold"].isin(folds)]

        if df.empty:
            raise ValueError("No records found after applying fold filter")

        filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.records = df[filename_col].astype(str).tolist()
        self.reports = [self._clean_report_text(x) for x in df["report"].tolist()]
        self.meta_df = df[["age", "sex", "height", "weight", "pacemaker", "heart_axis"]].copy()

        self.labels = None
        if self.return_labels:
            if self.label_col == "diagnostic_superclass":
                # Map individual SCP codes → 5 diagnostic superclasses
                # (NORM, MI, STTC, CD, HYP) using scp_statements.csv.
                scp_map = self._load_superclass_map(root)
                parsed = [
                    self._scp_to_superclass(v, scp_map)
                    for v in df["scp_codes"].tolist()
                ]
            else:
                if self.label_col not in df.columns:
                    raise ValueError(f"{self.label_col} not found in ptbxl_database.csv")
                parsed = [self._parse_label_dict(v) for v in df[self.label_col].tolist()]

            if self.label_map is None:
                all_labels = sorted({k for d in parsed for k in d})
                self.label_map = {k: i for i, k in enumerate(all_labels)}
                # NOTE: pass label_map=train_dataset.label_map to val/test datasets
                # so all splits share the same label ordering.

            self.labels = [self._labels_to_vector(d) for d in parsed]

        if target_length is None:
            header = wfdb.rdheader(os.path.join(self.root, self.records[0]))
            self.target_length = int(header.sig_len)
        else:
            self.target_length = int(target_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal, _ = wfdb.rdsamp(os.path.join(self.root, self.records[idx]))
        x = torch.tensor(signal, dtype=torch.float32).T  # (12, T)

        if self.normalize:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # Global normalization: single mean/std across all leads — preserves inter-lead amplitude ratios.
            mean = x.mean()
            std = x.std().clamp(min=1e-6)
            x = (x - mean) / std

        # Truncate or zero-pad AFTER normalization so padding zeros represent the mean
        if x.shape[1] > self.target_length:
            x = x[:, : self.target_length]
        elif x.shape[1] < self.target_length:
            pad_len = self.target_length - x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], pad_len), dtype=x.dtype)], dim=1)

        if not self.return_text:
            if self.return_labels:
                return x, self.labels[idx]
            return x

        if self.text_source == "pseudo_report":
            encoder_text = self._build_pseudo_report(idx)
        else:
            encoder_text = self.reports[idx]

        decoder_text = self.reports[idx]

        enc = self.encoder_tokenizer(
            encoder_text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )

        output = {
            "ecg": x,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.use_dual_tokenizer:
            if self.decoder_tokenizer is None:
                raise ValueError("decoder_tokenizer is required when use_dual_tokenizer=True")
            dec = self.decoder_tokenizer(
                decoder_text,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt",
            )
            output["decoder_input_ids"] = dec["input_ids"].squeeze(0)
            output["decoder_attention_mask"] = dec["attention_mask"].squeeze(0)

        if self.return_demographics:
            demo_text = self._build_pseudo_report(idx)
            demo_enc = self.encoder_tokenizer(
                demo_text,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt",
            )
            output["demo_input_ids"] = demo_enc["input_ids"].squeeze(0)
            output["demo_attention_mask"] = demo_enc["attention_mask"].squeeze(0)

        if self.return_labels:
            output["labels"] = self.labels[idx]

        return output

    def _normalize_text(self, value):
        if pd.isna(value):
            return "no report available"
        text = str(value).strip()
        return text if text else "no report available"

    def _clean_report_text(self, value):
        text = self._normalize_text(value)
        if text == "no report available":
            return text

        # Remove PTB-XL machine-edit suffixes like "Edit: NORM 100, ...".
        text = re.sub(r"\bedit\s*:\s*.*$", "", text, flags=re.IGNORECASE)

        # Trim leftover whitespace/punctuation after suffix removal.
        text = re.sub(r"\s+", " ", text).strip(" ,.;")
        return text if text else "no report available"

    def _parse_label_dict(self, value):
        if pd.isna(value):
            return {}
        if isinstance(value, dict):
            d = value
        else:
            try:
                d = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                return {}

        out = {}
        for k, v in d.items():
            try:
                if float(v) > self.label_threshold:
                    out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _load_superclass_map(self, root):
        """Load scp_statements.csv and return {scp_code: diagnostic_class}."""
        path = os.path.join(root, "scp_statements.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"scp_statements.csv not found in {root}. "
                "It is required when label_col='diagnostic_superclass'."
            )
        scp_df = pd.read_csv(path, index_col=0)
        # Keep only rows that are diagnostic (diagnostic == 1.0)
        diag = scp_df[scp_df["diagnostic"] == 1.0]
        return diag["diagnostic_class"].dropna().to_dict()

    def _scp_to_superclass(self, value, scp_map):
        """Convert an scp_codes dict string to superclass labels."""
        raw = self._parse_label_dict(value)
        superclasses = {}
        for code in raw:
            sc = scp_map.get(code)
            if sc is not None:
                superclasses[sc] = 1.0
        return superclasses

    def _labels_to_vector(self, label_dict):
        vec = torch.zeros(len(self.label_map), dtype=torch.float32)
        for label in label_dict:
            idx = self.label_map.get(label)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def _build_pseudo_report(self, idx):
        row = self.meta_df.iloc[idx]
        age = self._fmt_int(row.get("age"), "unknown")
        sex = self._fmt_sex(row.get("sex"))
        height = self._fmt_int(row.get("height"), None)
        weight = self._fmt_int(row.get("weight"), None)
        pacemaker = self._to_bool(row.get("pacemaker"))

        parts = [f"{age}-year-old {sex}"]
        if weight is not None:
            parts.append(f"weight {weight} kg")
        if height is not None:
            parts.append(f"height {height} cm")
        parts.append("pacemaker present" if pacemaker else "no pacemaker")
        return ". ".join(parts) + "."

    def _fmt_int(self, value, default):
        if pd.isna(value):
            return default
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return default

    def _fmt_text(self, value, default):
        if pd.isna(value):
            return default
        text = str(value).strip()
        return text if text else default

    def _fmt_sex(self, value):
        if pd.isna(value):
            return "unknown"
        try:
            code = int(float(value))
            if code == 0:
                return "male"
            if code == 1:
                return "female"
            return "unknown"
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in {"female", "male"}:
                return text
            return "unknown"

    def _to_bool(self, value):
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "present", "pacemaker"}
