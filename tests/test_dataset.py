# tests/test_dataset.py
import sys
sys.path.append(".")
from src.data_utils.dataset import SSTDataset

ds = SSTDataset(
    csv_path="data/splits/train.csv",
    tokenizer_name="distilbert-base-uncased",
    max_length=128,
)   

print(f"Dataset length : {len(ds)}")
sample = ds[0]
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"attention_mask : {sample['attention_mask'].shape}")
print(f"label          : {sample['label']}")