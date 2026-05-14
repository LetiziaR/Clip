import os
from glob import glob

import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


class MIMIC(Dataset):
	"""MIMIC-IV-ECG dataset for CoCa.

	Output format matches trainer/evaluation expectations (same as PTBXL):
	- always: ecg, input_ids, attention_mask
	- if dual tokenizer: decoder_input_ids, decoder_attention_mask
	"""

	def __init__(
		self,
		root,
		tokenizer=None,
		encoder_tokenizer=None,
		decoder_tokenizer=None,
		use_dual_tokenizer=False,
		target_length=5000,
		normalize=True,
		text_max_length=128,
		record_list_file="record_list.csv",
		machine_measurements_file="machine_measurements.csv",
		waveform_note_links_file="waveform_note_links.csv",
		folds_file="mimic_folds.csv",
		files_dir="files",
		text_source="report",
		notes_root=None,
		demographics_dir=None,
		folds=None,
		max_samples=None,
		normalize_mode="global",
		return_demographics=False,
		return_labels=False,
		labels_file=None,
		label_map=None,
	):
		if text_source not in {"report", "note", "pseudo_report"}:
			raise ValueError("text_source must be 'report', 'note', or 'pseudo_report'")
		if folds is None:
			raise ValueError("folds must be provided (e.g. list(range(1,9)) for train)")


		self.root = root
		self.target_length = int(target_length)
		self.normalize = bool(normalize)
		self.text_max_length = int(text_max_length)
		self.text_source = text_source
		self.notes_root = notes_root
		self.demographics_dir = demographics_dir
		self.folds = folds
		self.normalize_mode = normalize_mode
		self.use_dual_tokenizer = use_dual_tokenizer
		self.return_demographics = return_demographics
		self.return_labels = return_labels

		self.encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else tokenizer
		self.decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else tokenizer

		if self.encoder_tokenizer is None:
			raise ValueError("A tokenizer is required (pass tokenizer or encoder_tokenizer)")

		self.waveform_root = os.path.join(root, files_dir)

		# ── Load metadata ──
		record_list_path = os.path.join(root, record_list_file)
		machine_measurements_path = os.path.join(root, machine_measurements_file)
		waveform_note_links_path = os.path.join(root, waveform_note_links_file)

		record_usecols = ["subject_id", "study_id", "path"]
		if self.text_source == "pseudo_report":
			record_usecols.append("ecg_time")
		records_df = pd.read_csv(
			record_list_path,
			usecols=record_usecols,
		)

		report_cols = [f"report_{i}" for i in range(18)]
		machine_usecols = ["subject_id", "study_id", *report_cols]
		machine_df = pd.read_csv(
			machine_measurements_path,
			usecols=lambda c: c in machine_usecols,
			low_memory=False,
		)

		df = records_df.merge(machine_df, on=["subject_id", "study_id"], how="left")
		df["machine_text"] = self._build_machine_text(df, report_cols)
		df["path"] = df["path"].fillna("").astype(str).str.strip()
		df = df[df["path"] != ""]

		# ── Subject-level split via pre-computed folds ──
		folds_path = os.path.join(root, folds_file)
		if not os.path.isfile(folds_path):
			raise FileNotFoundError(
				f"Fold file not found: {folds_path}\n"
				"Run  python generate_mimic_folds.py --data_root <root>  first."
			)
		folds_df = pd.read_csv(folds_path)
		folds_df = folds_df[folds_df["strat_fold"].isin(self.folds)]
		df = df[df["subject_id"].isin(folds_df["subject_id"])].copy()
		if df.empty:
			raise ValueError(
				f"No samples available for folds={self.folds}"
			)

		# ── Load classification labels if requested ──
		self.labels = None
		self.label_map = label_map
		if self.return_labels:
			if labels_file is None:
				raise ValueError(
					"labels_file is required when return_labels=True. "
					"Run prepare_ecg_labels.py first to generate it."
				)
			labels_path = (
				labels_file if os.path.isabs(labels_file)
				else os.path.join(root, labels_file)
			)
			labels_df = pd.read_csv(labels_path)
			# Label columns = everything except study_id and subject_id
			label_cols = [c for c in labels_df.columns if c not in ("study_id", "subject_id")]

			if self.label_map is None:
				self.label_map = {code: i for i, code in enumerate(label_cols)}

			# Merge labels onto main df by study_id
			df = df.merge(
				labels_df[["study_id"] + label_cols],
				on="study_id",
				how="inner",
			)
			if df.empty:
				raise ValueError(
					"No samples matched between record_list and labels_file. "
					"Check that study_id values align."
				)

			# Label columns kept in df; vectors built after all merges below.

		# ── Load cardiologist notes if needed ──
		if self.text_source in {"note", "pseudo_report"}:
			if not self.notes_root:
				raise ValueError("notes_root is required when text_source is 'note' or 'pseudo_report'")
			notes_df = pd.read_csv(
				waveform_note_links_path,
				usecols=["subject_id", "study_id", "note_id"],
			)
			df = df.merge(notes_df, on=["subject_id", "study_id"], how="left")

			note_ids = self._clean_string_series(
				df["note_id"] if "note_id" in df.columns else pd.Series(dtype=str)
			)
			note_text_map = self._load_note_text_map(note_ids)

			df["note_text"] = ""
			if "note_id" in df.columns:
				clean_note_id = self._clean_string_series(df["note_id"])
				df["note_text"] = clean_note_id.map(note_text_map).fillna("")

		# ── Merge demographics if needed ──
		need_demographics = (
			self.text_source == "pseudo_report" or self.return_demographics
		)
		if need_demographics:
			if not self.demographics_dir:
				raise ValueError(
					"demographics_dir is required when text_source='pseudo_report' "
					"or return_demographics=True"
				)
			df = self._merge_demographics(df)

		# ── Build text columns ──
		if self.text_source == "note":
			df["text"] = df["note_text"].where(
				df["note_text"].str.len() > 0, df["machine_text"]
			)
		elif self.text_source == "pseudo_report":
			# decoder_text = cardiologist note (fallback to machine report)
			df["decoder_text"] = df["note_text"].where(
				df["note_text"].str.len() > 0, df["machine_text"]
			)
			df["decoder_text"] = self._clean_string_series(df["decoder_text"]).replace(
				"", "no report available"
			)
			# encoder_text = pseudo-report from demographics
			df["text"] = df.apply(self._build_pseudo_report_row, axis=1)
		else:
			df["text"] = df["machine_text"]

		df["text"] = self._clean_string_series(df["text"]).replace("", "no report available")

		# ── Store demographics texts for return_demographics ──
		if self.return_demographics and need_demographics:
			df["demo_text"] = df.apply(self._build_pseudo_report_row, axis=1)

		if max_samples is not None:
			df = df.iloc[: int(max_samples)].copy()

		if df.empty:
			raise ValueError(
				f"No samples available after preprocessing (folds={self.folds})"
			)

		# ── Build label vectors (after all merges / max_samples) ──
		if self.return_labels and self.label_map is not None:
			ordered_cols = sorted(self.label_map.keys(), key=lambda c: self.label_map[c])
			label_matrix = df[ordered_cols].values.astype(float)
			self.labels = [
				torch.tensor(label_matrix[i], dtype=torch.float32)
				for i in range(len(df))
			]

		self.records = df["path"].tolist()
		self.texts = df["text"].astype(str).tolist()
		self.decoder_texts = (
			df["decoder_text"].astype(str).tolist()
			if "decoder_text" in df.columns
			else self.texts
		)
		self.demo_texts = (
			df["demo_text"].astype(str).tolist()
			if "demo_text" in df.columns
			else None
		)

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx, _retries=10):
		record_rel_path = self.records[idx]
		record_path = self._resolve_record_path(record_rel_path)

		try:
			signal, _ = wfdb.rdsamp(record_path)
		except FileNotFoundError:
			if _retries <= 0:
				raise
			import random
			return self.__getitem__(random.randint(0, len(self) - 1), _retries - 1)

		ecg = torch.tensor(signal, dtype=torch.float32)

		if ecg.ndim != 2:
			raise ValueError(f"Expected 2D ECG signal, got shape {tuple(ecg.shape)}")

		# Ensure shape is (12, T)
		if ecg.shape[1] == 12:
			ecg = ecg.T
		elif ecg.shape[0] != 12:
			raise ValueError(
				f"Expected one ECG dimension to be 12 leads, got shape {tuple(ecg.shape)}"
			)

		# Normalize
		if self.normalize:
			ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
			# Global normalization: single mean/std across all leads — preserves inter-lead amplitude ratios.
			mean = ecg.mean()
			std = ecg.std().clamp(min=1e-6)
			ecg = (ecg - mean) / std

		# Pad or truncate
		ecg = self._pad_or_truncate(ecg)

		# Text — encoder gets pseudo-report (or report), decoder gets original report
		encoder_text = self.texts[idx].strip() or "no report available"
		decoder_text = self.decoder_texts[idx].strip() or "no report available"

		enc = self.encoder_tokenizer(
			encoder_text,
			padding="max_length",
			truncation=True,
			max_length=self.text_max_length,
			return_tensors="pt",
		)

		output = {
			"ecg": ecg,
			"input_ids": enc["input_ids"].squeeze(0),
			"attention_mask": enc["attention_mask"].squeeze(0),
		}

		if self.use_dual_tokenizer:
			if self.decoder_tokenizer is None:
				raise ValueError(
					"decoder_tokenizer is required when use_dual_tokenizer=True"
				)
			dec = self.decoder_tokenizer(
				decoder_text,
				padding="max_length",
				truncation=True,
				max_length=self.text_max_length,
				return_tensors="pt",
			)
			output["decoder_input_ids"] = dec["input_ids"].squeeze(0)
			output["decoder_attention_mask"] = dec["attention_mask"].squeeze(0)

		if self.return_demographics and self.demo_texts is not None:
			demo_text = self.demo_texts[idx].strip() or "unknown patient"
			demo_enc = self.encoder_tokenizer(
				demo_text,
				padding="max_length",
				truncation=True,
				max_length=self.text_max_length,
				return_tensors="pt",
			)
			output["demo_input_ids"] = demo_enc["input_ids"].squeeze(0)
			output["demo_attention_mask"] = demo_enc["attention_mask"].squeeze(0)

		if self.return_labels and self.labels is not None:
			output["labels"] = self.labels[idx]

		return output

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _resolve_record_path(self, record_path):
		if os.path.isabs(record_path):
			return record_path
		rel = record_path.lstrip("/")
		# Strip the CSV prefix (e.g. "files/") so paths resolve against waveform_root
		prefix = "files/"
		if rel.startswith(prefix):
			rel = rel[len(prefix):]
		return os.path.join(self.waveform_root, rel)

	def _pad_or_truncate(self, ecg):
		length = ecg.shape[1]
		if length > self.target_length:
			return ecg[:, : self.target_length]
		if length < self.target_length:
			pad_len = self.target_length - length
			pad = torch.zeros((ecg.shape[0], pad_len), dtype=ecg.dtype)
			return torch.cat([ecg, pad], dim=1)
		return ecg

	@staticmethod
	def _clean_string_series(series):
		clean = series.fillna("").astype(str).str.strip()
		clean = clean.replace({"nan": "", "None": "", "<NA>": ""})
		return clean

	def _build_machine_text(self, df, report_cols):
		present_report_cols = [col for col in report_cols if col in df.columns]
		if not present_report_cols:
			return pd.Series(
				["" for _ in range(len(df))], index=df.index, dtype=str
			)
		text_df = df[present_report_cols].fillna("")
		text_df = text_df.astype(str).apply(lambda col: col.str.strip())
		text_df = text_df.replace({"nan": "", "None": "", "<NA>": ""})
		return text_df.apply(
			lambda row: ". ".join([part for part in row.tolist() if part]),
			axis=1,
		)

	def _merge_demographics(self, df):
		"""Merge patients.csv and omr.csv to add age, sex, height, weight."""
		patients_path = os.path.join(self.demographics_dir, "patients.csv")
		omr_path = os.path.join(self.demographics_dir, "omr.csv")

		patients = pd.read_csv(
			patients_path,
			usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
		)
		df = df.merge(patients, on="subject_id", how="left")

		# Compute age at ECG time
		df["ecg_year"] = pd.to_datetime(df["ecg_time"], errors="coerce").dt.year
		df["age"] = df["anchor_age"] + (df["ecg_year"] - df["anchor_year"])
		df["age"] = df["age"].clip(lower=0)

		# Sex mapping
		df["sex"] = df["gender"].map({"M": "male", "F": "female"}).fillna("unknown")

		# Height and weight from OMR (latest per subject)
		omr = pd.read_csv(omr_path, usecols=["subject_id", "result_name", "result_value", "chartdate"])
		omr["result_value"] = pd.to_numeric(omr["result_value"], errors="coerce")
		omr = omr.dropna(subset=["result_value"])

		for measure, col_name in [("Height", "height"), ("Weight", "weight")]:
			subset = omr[omr["result_name"].str.contains(measure, case=False, na=False)]
			subset = (
				subset.sort_values("chartdate")
				.drop_duplicates("subject_id", keep="last")[["subject_id", "result_value"]]
				.rename(columns={"result_value": col_name})
			)
			df = df.merge(subset, on="subject_id", how="left")

		return df

	@staticmethod
	def _build_pseudo_report_row(row):
		"""Build a PTB-XL-style pseudo-report from demographics.

		MIMIC OMR stores height in inches and weight in pounds —
		convert to metric (cm / kg) for consistency with PTB-XL.
		"""
		age = int(row["age"]) if pd.notna(row.get("age")) else "unknown"
		sex = row.get("sex", "unknown")
		parts = [f"{age}-year-old {sex}"]

		weight_lbs = row.get("weight")
		if pd.notna(weight_lbs) and float(weight_lbs) > 0:
			weight_kg = float(weight_lbs) * 0.453592
			parts.append(f"weight {int(round(weight_kg))} kg")

		height_in = row.get("height")
		if pd.notna(height_in) and float(height_in) > 0:
			height_cm = float(height_in) * 2.54
			parts.append(f"height {int(round(height_cm))} cm")

		return ". ".join(parts) + "."

	def _load_note_text_map(self, note_ids, chunksize=200000):
		note_ids = pd.Series(note_ids).dropna().astype(str).str.strip()
		note_ids = set(note_ids[note_ids != ""].tolist())
		if not note_ids:
			return {}

		notes_pattern = os.path.join(self.notes_root, "note", "*.csv.gz")
		note_files = sorted(glob(notes_pattern))
		if not note_files:
			return {}

		remaining = set(note_ids)
		text_map = {}
		for note_file in note_files:
			if not remaining:
				break
			for chunk in pd.read_csv(
				note_file,
				compression="gzip",
				usecols=lambda c: c in {"note_id", "text"},
				chunksize=chunksize,
			):
				if "note_id" not in chunk.columns or "text" not in chunk.columns:
					continue
				chunk["note_id"] = self._clean_string_series(chunk["note_id"])
				chunk["text"] = self._clean_string_series(chunk["text"])
				chunk = chunk[
					(chunk["note_id"].isin(remaining)) & (chunk["text"] != "")
				]
				if chunk.empty:
					continue
				for row in chunk.itertuples(index=False):
					note_id = row.note_id
					if note_id not in text_map:
						text_map[note_id] = row.text
				remaining -= set(chunk["note_id"].unique().tolist())
				if not remaining:
					break

		return text_map
