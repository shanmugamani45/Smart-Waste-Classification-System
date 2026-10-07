import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from src.utils.logger import logger


class ModelEvaluator:
    """Calculates comprehensive classification, latency, and resource metrics."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
    ) -> Dict[str, Any]:
        """Calculates standard classification metrics without fabricated values."""
        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        report_text = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )

        per_class_metrics = {}
        for idx, cls_name in enumerate(class_names):
            per_class_metrics[cls_name] = {
                "precision": round(float(prec_per_class[idx]), 4),
                "recall": round(float(rec_per_class[idx]), 4),
                "f1_score": round(float(f1_per_class[idx]), 4),
                "support": int(support[idx])
            }

        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_text
        }

    @staticmethod
    def measure_inference_latency(
        model: nn.Module,
        input_tensor_shape: tuple = (1, 3, 224, 224),
        device: torch.device = torch.device("cpu"),
        num_warmup: int = 5,
        num_runs: int = 30
    ) -> float:
        """Measures average inference latency per sample in milliseconds."""
        model.eval()
        model.to(device)
        dummy_input = torch.randn(input_tensor_shape).to(device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            total_elapsed = time.perf_counter() - start_time

        avg_latency_ms = (total_elapsed / num_runs) * 1000.0
        return round(avg_latency_ms, 2)

    @staticmethod
    def get_model_size_mb(filepath: str) -> float:
        """Returns file size on disk in megabytes."""
        p = Path(filepath)
        if p.exists():
            return round(p.stat().st_size / (1024 * 1024), 2)
        return 0.0
