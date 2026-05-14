import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from ts2vec.ts2vec import TS2Vec

from data.ptbxl_dataset import PTBXL


# ------------------------------------------------------------------
# Lightweight MIMIC ECG-only dataset (no tokenizer needed)
# ------------------------------------------------------------------

class MIMICEcgOnly(Dataset):
    """Loads raw MIMIC-IV ECG waveforms for self-supervised pretraining."""

    def __init__(self, root, files_dir="mimic_ecg", target_length=5000,
                 normalize=True, normalize_mode="global",
                 folds=None, folds_file="mimic_folds.csv",
                 max_samples=None):
        if folds is None:
            raise ValueError("folds must be provided (e.g. list(range(1,9)) for train)")
        self.target_length = target_length
        self.normalize = normalize
        self.normalize_mode = normalize_mode

        waveform_root = os.path.join(root, files_dir)
        self.waveform_root = waveform_root

        records_df = pd.read_csv(
            os.path.join(root, "record_list.csv"),
            usecols=["subject_id", "path"],
        )
        records_df["path"] = records_df["path"].fillna("").astype(str).str.strip()
        records_df = records_df[records_df["path"] != ""]

        # Subject-level split via pre-computed folds
        folds_path = os.path.join(root, folds_file)
        if not os.path.isfile(folds_path):
            raise FileNotFoundError(
                f"Fold file not found: {folds_path}\n"
                "Run  python generate_mimic_folds.py --data_root <root>  first."
            )
        folds_df = pd.read_csv(folds_path)
        selected = set(folds_df[folds_df["strat_fold"].isin(folds)]["subject_id"])
        records_df = records_df[records_df["subject_id"].isin(selected)]

        if max_samples is not None:
            records_df = records_df.iloc[:int(max_samples)]

        self.records = records_df["path"].tolist()
        print(f"MIMICEcgOnly: {len(self.records)} ECGs (folds={folds})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx, _retries=10):
        rel = self.records[idx].lstrip("/")
        if rel.startswith("files/"):
            rel = rel[len("files/"):]
        record_path = os.path.join(self.waveform_root, rel)

        try:
            signal, _ = wfdb.rdsamp(record_path)
        except FileNotFoundError:
            if _retries <= 0:
                raise
            return self.__getitem__(random.randint(0, len(self) - 1), _retries - 1)

        ecg = torch.tensor(signal, dtype=torch.float32)
        if ecg.shape[1] == 12:
            ecg = ecg.T
        elif ecg.shape[0] != 12:
            if _retries <= 0:
                raise ValueError(f"Bad ECG shape {tuple(ecg.shape)} at {record_path}")
            return self.__getitem__(random.randint(0, len(self) - 1), _retries - 1)

        if self.normalize:
            ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            if self.normalize_mode == "global":
                mean = ecg.mean()
                std = ecg.std().clamp(min=1e-6)
            else:
                mean = ecg.mean(dim=1, keepdim=True)
                std = ecg.std(dim=1, keepdim=True).clamp(min=1e-6)
            ecg = (ecg - mean) / std

        # Pad or truncate to target_length
        if ecg.shape[1] > self.target_length:
            ecg = ecg[:, :self.target_length]
        elif ecg.shape[1] < self.target_length:
            pad_len = self.target_length - ecg.shape[1]
            ecg = torch.cat([ecg, torch.zeros(12, pad_len)], dim=1)

        return ecg


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain TS2Vec on ECG data")
    parser.add_argument("--dataset", type=str, default="ptbxl",
                        choices=["ptbxl", "mimic", "combined"],
                        help="Dataset to pretrain on (default: ptbxl)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root (PTB-XL for ptbxl, MIMIC for mimic, PTB-XL for combined)")
    parser.add_argument("--mimic_root", type=str, default=None,
                        help="Path to MIMIC root (only used with --dataset combined)")
    parser.add_argument("--save_path", type=str, default="ts2vec_pretrained.pt")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500],
                        help="Must match the sampling rate used in CoCa training (default: 500)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dims", type=int, default=320)
    parser.add_argument("--hidden_dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for TS2Vec training (reduce if OOM)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of ECGs (useful for large MIMIC dataset)")
    parser.add_argument("--normalize_mode", type=str, default="global",
                        choices=["global"],
                        help="ECG normalization mode")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = 0 if torch.cuda.is_available() else "cpu"

    if args.dataset == "ptbxl":
        train_folds = list(range(1, 9))
        dataset = PTBXL(
            root=args.data_root,
            sampling_rate=args.sampling_rate,
            folds=train_folds,
            return_text=False,
        )
    elif args.dataset == "mimic":
        dataset = MIMICEcgOnly(
            root=args.data_root,
            folds=list(range(1, 9)),
            normalize_mode=args.normalize_mode,
            max_samples=args.max_samples,
        )
    elif args.dataset == "combined":
        if args.mimic_root is None:
            raise ValueError("--mimic_root is required when --dataset=combined")
        ptbxl_ds = PTBXL(
            root=args.data_root,
            sampling_rate=args.sampling_rate,
            folds=list(range(1, 9)),
            return_text=False,
        )
        mimic_ds = MIMICEcgOnly(
            root=args.mimic_root,
            folds=list(range(1, 9)),
            normalize_mode=args.normalize_mode,
            max_samples=args.max_samples,
        )
        dataset = torch.utils.data.ConcatDataset([ptbxl_ds, mimic_ds])
        print(f"Combined: {len(ptbxl_ds)} PTB-XL + {len(mimic_ds)} MIMIC = {len(dataset)} total")

    print(f"Loaded {len(dataset)} ECG signals ({args.dataset})")

    first_x = dataset[0]
    train_data = np.zeros(
        (len(dataset), first_x.shape[0], first_x.shape[1]),
        dtype=np.float32,
    )
    for i in range(len(dataset)):
        train_data[i] = dataset[i].numpy()

    # Transpose from (N, C, T) to (N, T, C) for TS2Vec
    train_data = np.transpose(train_data, (0, 2, 1))
    print("Train data shape:", train_data.shape)

    model = TS2Vec(
        input_dims=12,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        batch_size=args.batch_size,
    )

    print("Starting TS2Vec pretraining...")
    model.fit(train_data, n_epochs=args.epochs, verbose=True)

    model.save(args.save_path)
    print(f"Saved pretrained TS2Vec model to {args.save_path}")


if __name__ == "__main__":
    main()
