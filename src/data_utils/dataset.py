# src/data_utils/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SSTDataset(Dataset):
    
    def __init__(self, csv_path: str, tokenizer_name: str, max_length: int):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["sentence"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )

        return {
            "inputs_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }