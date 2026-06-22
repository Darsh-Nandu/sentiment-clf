# scripts/preprocess.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import yaml
import argparse

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str:
    """Minimal cleaning - lowercase and strip whitespaces"""
    return text.strip().lower()

def run(config_path: str):
    config = load_config(config_path)
    seed = config["data"]["random_seed"]
    test_size = config["data"]["test_size"]

    # 1. Load raw CSVs saved during EDA
    raw_train = pd.read_csv("data/raw/train.csv")
    raw_val = pd.read_csv("data/raw/val.csv")     # This becomes our test size

    # 2. Clean text
    raw_train["sentence"] = raw_train["sentence"].apply(clean_text)
    raw_val["sentence"]   = raw_val["sentence"].apply(clean_text)

    # 3. Split train -> train + val
    # We will use 10% of train as val and the original val set becomes our test set.
    train_df, val_df = train_test_split(
        raw_train,
        test_size=test_size,
        random_state=seed,
        stratify=raw_train["label"]              # Keeps class ratio the same in both splits
    )
    test_df = raw_val

    # 4. Save splits 
    os.makedirs("data/splits", exist_ok=True)

    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv",     index=False)
    test_df.to_csv("data/splits/test.csv",   index=False)

    print("✅ Splits saved to data/splits/")
    print(f"   train : {len(train_df):>6} rows  ({train_df['label'].mean():.2%} positive)")
    print(f"   val   : {len(val_df):>6} rows  ({val_df['label'].mean():.2%} positive)")
    print(f"   test  : {len(test_df):>6} rows  ({test_df['label'].mean():.2%} positive)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run(args.config)
