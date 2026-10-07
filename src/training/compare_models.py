import os
import csv
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import torch

from src.training.train import TrainingPipeline
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import ModelEvaluator
from src.utils.config_loader import get_config
from src.utils.logger import logger


class ModelComparator:
    """Trains and compares multiple candidate architectures to generate model_comparison.csv."""

    CANDIDATE_MODELS = ["MobileNetV3", "EfficientNet-B0"]

    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.comparison_csv_path = Path(self.config["paths"]["comparison_csv"])
        self.comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def run_comparison(self) -> pd.DataFrame:
        """Runs the training and evaluation protocol for all candidate models."""
        results = []

        for model_name in self.CANDIDATE_MODELS:
            logger.info(f"=== Benchmarking Candidate: {model_name} ===")
            pipeline = TrainingPipeline(model_name=model_name, config=self.config)

            try:
                metrics = pipeline.run()
                row = {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["macro_precision"],
                    "Recall": metrics["macro_recall"],
                    "F1": metrics["macro_f1"],
                    "Parameters": metrics["total_parameters"],
                    "Size_MB": metrics["model_size_mb"],
                    "Inference_Time_MS": metrics["inference_latency_ms"]
                }
            except Exception as e:
                logger.warning(f"Model {model_name} evaluation skipped/failed: {e}")
                row = {
                    "Model": model_name,
                    "Accuracy": "Not evaluated yet",
                    "Precision": "Not evaluated yet",
                    "Recall": "Not evaluated yet",
                    "F1": "Not evaluated yet",
                    "Parameters": "Not evaluated yet",
                    "Size_MB": "Not evaluated yet",
                    "Inference_Time_MS": "Not evaluated yet"
                }

            results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(self.comparison_csv_path, index=False)
        logger.info(f"Model comparison saved to: {self.comparison_csv_path}")
        return df

    @classmethod
    def generate_initial_comparison_placeholder(cls, output_path: str = None) -> None:
        """Generates the initial CSV with 'Not evaluated yet' per academic integrity rules."""
        config = get_config()
        path = Path(output_path or config["paths"]["comparison_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "Model": m,
                "Accuracy": "Not evaluated yet",
                "Precision": "Not evaluated yet",
                "Recall": "Not evaluated yet",
                "F1": "Not evaluated yet",
                "Parameters": "Not evaluated yet",
                "Size_MB": "Not evaluated yet",
                "Inference_Time_MS": "Not evaluated yet"
            }
            for m in cls.CANDIDATE_MODELS
        ]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info(f"Initialized benchmark comparison table at: {path}")


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.generate_initial_comparison_placeholder()
