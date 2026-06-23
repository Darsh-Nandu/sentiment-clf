# scripts/train.py

import sys
import os
sys.path.append(os.path.join(os.path.dirnam(__file__), ".."))

import yaml
import argparse
import torch
import mlflow
from torch.utils.data import DataLoader

from src.models.classifier import SentimentClassifier
from src.data_utils.dataset import SSTDataset
from src.training.trainer import Trainer

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main(config_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = SSTDataset("data/splits/train.csv", config["model"]["name"], config["data"]["max_length"])
    val_dataset = SSTDataset("data/splits/val.csv", config["model"]["name"], config["data"]["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model
    model = SentimentClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )

    # Mlflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_params({
            "model_name":    config["model"]["name"],
            "batch_size":    config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs":    config["training"]["num_epochs"],
            "max_length":    config["data"]["max_length"],
        })

        trainer = Trainer(model, config, device)
        trainer.fit(train_loader, val_loader)

        # Save model artifact
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), "outputs/model.pt")
        mlflow.log_artifact("outputs/model.pt")
        print("✅ Model saved and logged to MLflow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)