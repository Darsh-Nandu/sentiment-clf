# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import mlflow
from tqdm import tqdm


class Trainer:

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.criterion = nn.CrossEntropyLoss()

    def _setup_scheduler(self, num_traning_steps: int):
        warmup_steps = int(num_traning_steps * self.config["training"]["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_traning_steps
        )


    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss/len(dataloader)


    def _eval_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss   = self.criterion(logits, labels)

                preds   = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy


    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        num_epochs = self.config["training"]["num_epochs"]
        num_training_steps = num_epochs * len(train_loader)

        self._setup_scheduler(num_training_steps)

        for epoch in range(1, num_epochs + 1):
            train_loss            = self._train_epoch(train_loader)
            val_loss, val_accuracy = self._eval_epoch(val_loader)

            print(f"Epoch {epoch}/{num_epochs} — "
                  f"train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | "
                  f"val_acc: {val_accuracy:.4f}")

            # Log everything to MLflow
            mlflow.log_metric("train_loss",    train_loss,    step=epoch)
            mlflow.log_metric("val_loss",      val_loss,      step=epoch)
            mlflow.log_metric("val_accuracy",  val_accuracy,  step=epoch)