import os
from glob import glob

import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


class MIMIC(Dataset):
	def __init__(
		self,
		root,
		tokenizer,
		target_length,
		normalize=True,
		text_max_length=128,
		record_list_file="record_list.csv",
		machine_measurements_file="machine_measurements.csv",
		waveform_note_links_file="waveform_note_links.csv",
		files_dir="files",
		text_source="machine",
		notes_root=None,
		split="train",
		split_ratio=(0.8, 0.1, 0.1),
		seed=42,
		max_samples=None,
	):
		if tokenizer is None:
			raise ValueError("tokenizer is required")
		if text_source not in {"machine", "cardiologist"}:
			raise ValueError("text_source must be either 'machine' or 'cardiologist'")
		if split not in {"train", "val", "test"}:
			raise ValueError("split must be one of: 'train', 'val', 'test'")
		if len(split_ratio) != 3:
			raise ValueError("split_ratio must contain exactly 3 values for train/val/test")
		if any(r < 0 for r in split_ratio):
			raise ValueError("split_ratio values must be non-negative")
		ratio_sum = float(sum(split_ratio))
		if ratio_sum <= 0:
			raise ValueError("split_ratio must sum to a positive value")

		self.root = root
		self.tokenizer = tokenizer
		self.target_length = int(target_length)
		self.normalize = bool(normalize)
		self.text_max_length = int(text_max_length)
		self.text_source = text_source
		self.notes_root = notes_root
		self.split = split
		self.split_ratio = tuple(float(r) / ratio_sum for r in split_ratio)
		self.seed = int(seed)

		record_list_path = os.path.join(root, record_list_file)
		machine_measurements_path = os.path.join(root, machine_measurements_file)
		waveform_note_links_path = os.path.join(root, waveform_note_links_file)
		self.waveform_root = os.path.join(root, files_dir)

		records_df = pd.read_csv(
			record_list_path,
			usecols=["subject_id", "study_id", "path"],
		)

		report_cols = [f"report_{i}" for i in range(18)]
		machine_usecols = ["subject_id", "study_id", *report_cols]
		machine_df = pd.read_csv(
			machine_measurements_path,
			usecols=lambda c: c in machine_usecols,
		)

		df = records_df.merge(machine_df, on=["subject_id", "study_id"], how="left")
		df["machine_text"] = self._build_machine_text(df, report_cols)
		df["path"] = df["path"].fillna("").astype(str).str.strip()
		df = df[df["path"] != ""]

		df = self._apply_subject_split(df)

		if self.text_source == "cardiologist":
			if not self.notes_root:
				raise ValueError("notes_root is required when text_source='cardiologist'")
			notes_df = pd.read_csv(
				waveform_note_links_path,
				usecols=["subject_id", "study_id", "note_id"],
			)
			df = df.merge(notes_df, on=["subject_id", "study_id"], how="left")

			note_ids = self._clean_string_series(df["note_id"] if "note_id" in df.columns else pd.Series(dtype=str))
			note_text_map = self._load_note_text_map(note_ids)

			df["note_text"] = ""
			if "note_id" in df.columns:
				clean_note_id = self._clean_string_series(df["note_id"])
				df["note_text"] = clean_note_id.map(note_text_map).fillna("")

			df["text"] = df["note_text"].where(df["note_text"].str.len() > 0, df["machine_text"])
		else:
			df["text"] = df["machine_text"]

		df["text"] = self._clean_string_series(df["text"]).replace("", "no report available")

		if max_samples is not None:
			df = df.iloc[: int(max_samples)].copy()

		if df.empty:
			raise ValueError("No samples available after preprocessing")

		self.records = df["path"].tolist()
		self.texts = df["text"].astype(str).tolist()

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx):
		record_rel_path = self.records[idx]
		record_path = self._resolve_record_path(record_rel_path)

		try:
			signal, _ = wfdb.rdsamp(record_path)
		except FileNotFoundError as exc:
			raise FileNotFoundError(f"WFDB record not found for path: {record_path}") from exc

		ecg = torch.tensor(signal, dtype=torch.float32)

		if ecg.ndim != 2:
			raise ValueError(f"Expected 2D ECG signal, got shape {tuple(ecg.shape)}")

		if ecg.shape[1] == 12:
			ecg = ecg.T
		elif ecg.shape[0] == 12:
			pass
		else:
			raise ValueError(f"Expected one ECG dimension to be 12 leads, got shape {tuple(ecg.shape)}")

		ecg = self._pad_or_truncate(ecg)

		if self.normalize:
			ecg = self._normalize_per_lead(ecg)

		text = self.texts[idx].strip()
		if not text:
			text = "no report available"

		encoded = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=self.text_max_length,
			return_tensors="pt",
		)

		return {
			"ecg": ecg,
			"input_ids": encoded["input_ids"].squeeze(0),
			"attention_mask": encoded["attention_mask"].squeeze(0),
		}

	def _resolve_record_path(self, record_path):
		if os.path.isabs(record_path):
			return record_path

		rel = record_path.lstrip("/")

		if rel.startswith("files/"):
			rel = rel[len("files/") :]

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
	def _normalize_per_lead(ecg):
		ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
		mean = ecg.mean(dim=1, keepdim=True)
		std = ecg.std(dim=1, keepdim=True).clamp(min=1e-6)
		return (ecg - mean) / std

	@staticmethod
	def _clean_string_series(series):
		clean = series.fillna("").astype(str).str.strip()
		clean = clean.replace({"nan": "", "None": "", "<NA>": ""})
		return clean

	def _build_machine_text(self, df, report_cols):
		present_report_cols = [col for col in report_cols if col in df.columns]
		if not present_report_cols:
			return pd.Series(["" for _ in range(len(df))], index=df.index, dtype=str)

		text_df = df[present_report_cols].fillna("")
		text_df = text_df.astype(str).apply(lambda col: col.str.strip())
		text_df = text_df.replace({"nan": "", "None": "", "<NA>": ""})
		return text_df.apply(
			lambda row: " ".join([part for part in row.tolist() if part]),
			axis=1,
		)

	def _apply_subject_split(self, df):
		df = df.copy()
		subjects = df["subject_id"].dropna().unique().tolist()
		if not subjects:
			raise ValueError("No subjects available for splitting")

		rng = torch.Generator()
		rng.manual_seed(self.seed)
		perm = torch.randperm(len(subjects), generator=rng).tolist()
		shuffled_subjects = [subjects[i] for i in perm]

		n_subjects = len(shuffled_subjects)
		n_train = int(n_subjects * self.split_ratio[0])
		n_val = int(n_subjects * self.split_ratio[1])

		train_subjects = set(shuffled_subjects[:n_train])
		val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
		test_subjects = set(shuffled_subjects[n_train + n_val :])

		if self.split == "train":
			selected_subjects = train_subjects
		elif self.split == "val":
			selected_subjects = val_subjects
		else:
			selected_subjects = test_subjects

		df = df[df["subject_id"].isin(selected_subjects)].copy()
		if df.empty:
			raise ValueError(f"No samples available after applying split='{self.split}'")
		return df

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

				chunk = chunk[(chunk["note_id"].isin(remaining)) & (chunk["text"] != "")]
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



from transformers import AutoTokenizer

def main():
    root = "/dss/mcmlscratch/0F/ra59ver2/mimic_ecg_project/mimic_dataset"
    notes_root = "/dss/mcmlscratch/0F/ra59ver2/mimic_ecg_project/mimic_notes"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = MIMIC(
        root=root,
        tokenizer=tokenizer,
        target_length=5000,
        text_source="machine",  # 🔁 change to "cardiologist" if needed
        notes_root=notes_root,
        split="train",
    )

    print(f"Dataset size: {len(dataset)}\n")

    # Check first 3 samples
    for i in range(3):
        print("=" * 50)
        print(f"SAMPLE {i}")

        # 🔹 RAW TEXT (before tokenizer)
        raw_text = dataset.texts[i]

        print("\nRAW TEXT:")
        print(raw_text)

        # 🔹 TOKENIZED VERSION (decoded back)
        sample = dataset[i]
        decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

        print("\nTOKENIZED → DECODED:")
        print(decoded)

        # 🔹 ECG INFO
        print("\nECG SHAPE:")
        print(sample["ecg"].shape)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()