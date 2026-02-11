import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


class PTBXL(Dataset):
    
    def __init__(self, root, tokenizer=None, sampling_rate=100, folds=None, 
    target_length=None, return_text=True, normalize=True, text_max_length=128):

        self.root = root
        self.tokenizer = tokenizer
        self.return_text = return_text
        self.normalize = normalize
        self.text_max_length = text_max_length
        
        if return_text and tokenizer is None:
            raise ValueError("tokenizer is required when return_text=True")


        df = pd.read_csv(
            os.path.join(root, "ptbxl_database.csv"),
            index_col="ecg_id"
        )

        if folds is not None:
            df = df[df["strat_fold"].isin(folds)]

        if sampling_rate == 100:
            filename_col = "filename_lr"
        else:
            filename_col = "filename_hr"

        self.records = df[filename_col].astype(str).tolist()
        self.reports = df["report"].fillna("No report available").astype(str).tolist()

        if target_length is None:
            first_record = self.records[0]
            first_signal, _ = wfdb.rdsamp(os.path.join(self.root, first_record))
            self.target_length = first_signal.shape[0]
        else:
            self.target_length = int(target_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        signal, _ = wfdb.rdsamp(os.path.join(self.root, record))
        x = torch.tensor(signal, dtype=torch.float32)

        if x.shape[0] > self.target_length:
            x = x[: self.target_length]
        elif x.shape[0] < self.target_length:
            pad_len = self.target_length - x.shape[0]
            pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        if self.normalize:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std
            
        if not self.return_text:
            return x
        text = self.reports[idx]

        # Tokenize the text report 
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return x, input_ids, attention_mask
