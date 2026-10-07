import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from src.utils.logger import logger


class EarlyStopping:
    """Early stopping to prevent overfitting when validation loss stops improving."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


class ModelCheckpoint:
    """Saves the best model weights when validation loss or F1 improves."""

    def __init__(self, filepath: str, mode: str = "min"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else -float("inf")

    def __call__(self, current_metric: float, model: nn.Module, metadata: Dict[str, Any] = None) -> bool:
        improved = (current_metric < self.best_metric) if self.mode == "min" else (current_metric > self.best_metric)
        if improved:
            self.best_metric = current_metric
            save_payload = {
                "state_dict": model.state_dict(),
                "model_name": getattr(model, "model_architecture_name", "model"),
                "best_metric": current_metric,
                "metadata": metadata or {},
                "idx_to_class": metadata.get("idx_to_class", {}) if metadata else {}
            }
            torch.save(save_payload, self.filepath)
            logger.info(f"ModelCheckpoint: Saved best weights to {self.filepath} ({self.mode}={current_metric:.4f})")
            return True
        return False
