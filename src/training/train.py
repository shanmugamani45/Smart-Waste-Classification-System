import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_factory import ModelFactory
from src.data.dataset_loader import WasteDatasetLoader
from src.data.dataset_validator import DatasetValidator
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import TrainingVisualizer
from src.utils.config_loader import get_config
from src.utils.logger import logger


def set_seed(seed: int = 42):
    """Enforces deterministic behavior for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingPipeline:
    """Production training engine orchestrating two-stage transfer learning."""

    def __init__(self, model_name: str = None, config: Dict = None):
        self.config = config or get_config()
        self.model_name = model_name or self.config["model"]["name"]
        self.classes = self.config["classes"]
        self.num_classes = len(self.classes)

        set_seed(self.config["training"].get("seed", 42))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for training: {self.device}")

        # Ensure directories exist
        self.model_dir = Path(self.config["paths"]["model_dir"])
        self.eval_dir = Path(self.config["paths"]["evaluation_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Executes the complete two-stage training, fine-tuning, and evaluation workflow."""
        # 1. Dataset validation
        validator = DatasetValidator()
        validation_report = validator.validate_full_dataset()
        if not validation_report["is_valid"]:
            logger.warning(
                f"Dataset validation produced issues: {validation_report['errors']}. "
                "Ensure valid images are present before full training."
            )

        # 2. Load DataLoaders
        loader_factory = WasteDatasetLoader(self.config)
        dataloaders, idx_to_class = loader_factory.get_dataloaders()

        # 3. Instantiate model
        model = ModelFactory.build_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=True,
            dropout_rate=self.config["model"].get("dropout_rate", 0.3)
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_model_path = self.model_dir / f"{self.model_name.lower()}_best.pt"
        checkpoint_callback = ModelCheckpoint(str(best_model_path), mode="min")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # ================= STAGE 1: Train Head Only =================
        logger.info(f"--- Starting Stage 1: Frozen Backbone Head Training ({self.model_name}) ---")
        ModelFactory.freeze_backbone(model)
        optimizer_s1 = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        early_stopping_s1 = EarlyStopping(
            patience=self.config["training"]["early_stopping_patience"],
            mode="min"
        )

        for epoch in range(1, self.config["training"]["epochs_stage1"] + 1):
            train_loss, train_acc = self._train_one_epoch(model, dataloaders["train"], criterion, optimizer_s1)
            val_loss, val_acc, _, _ = self._evaluate_split(model, dataloaders["validation"], criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                f"Stage 1 [Epoch {epoch}/{self.config['training']['epochs_stage1']}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            checkpoint_callback(val_loss, model, {"epoch": epoch, "stage": 1, "idx_to_class": idx_to_class})
            if early_stopping_s1(val_loss):
                logger.info(f"Stage 1 early stopping triggered at epoch {epoch}.")
                break

        # ================= STAGE 2: Fine-Tuning =================
        logger.info(f"--- Starting Stage 2: Fine-Tuning Unfrozen Top Layers ({self.model_name}) ---")
        ModelFactory.unfreeze_top_layers(model, num_blocks=3)
        optimizer_s2 = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config["training"]["fine_tune_learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        scheduler_s2 = ReduceLROnPlateau(
            optimizer_s2,
            mode="min",
            factor=0.2,
            patience=self.config["training"]["lr_reduce_patience"]
        )
        early_stopping_s2 = EarlyStopping(
            patience=self.config["training"]["early_stopping_patience"],
            mode="min"
        )

        for epoch in range(1, self.config["training"]["epochs_stage2"] + 1):
            train_loss, train_acc = self._train_one_epoch(model, dataloaders["train"], criterion, optimizer_s2)
            val_loss, val_acc, _, _ = self._evaluate_split(model, dataloaders["validation"], criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler_s2.step(val_loss)

            logger.info(
                f"Stage 2 [Epoch {epoch}/{self.config['training']['epochs_stage2']}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            checkpoint_callback(val_loss, model, {"epoch": epoch, "stage": 2, "idx_to_class": idx_to_class})
            if early_stopping_s2(val_loss):
                logger.info(f"Stage 2 early stopping triggered at epoch {epoch}.")
                break

        # Load best model weights for final evaluation
        final_best_path = Path(self.config["paths"]["best_model_path"])
        if best_model_path.exists():
            ckpt = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(ckpt["state_dict"])
            logger.info("Loaded best model checkpoint for final test evaluation.")
            # Also sync to central best_model_path for predictor
            import shutil
            shutil.copy2(best_model_path, final_best_path)
            logger.info(f"Synchronized best checkpoint to: {final_best_path}")

        # ================= FINAL EVALUATION ON UNSEEN TEST SET =================
        logger.info("--- Performing Final Evaluation on Unseen Test Set ---")
        test_loss, test_acc, y_true, y_pred = self._evaluate_split(model, dataloaders["test"], criterion)
        metrics = ModelEvaluator.calculate_classification_metrics(y_true, y_pred, self.classes)
        metrics["test_loss"] = round(test_loss, 4)
        metrics["test_accuracy"] = round(test_acc, 4)

        # Performance profiling
        latency_ms = ModelEvaluator.measure_inference_latency(model, device=self.device)
        model_size_mb = ModelEvaluator.get_model_size_mb(str(best_model_path))
        param_summary = ModelFactory.get_parameter_summary(model)

        metrics["inference_latency_ms"] = latency_ms
        metrics["model_size_mb"] = model_size_mb
        metrics.update(param_summary)

        # Visualizations
        TrainingVisualizer.plot_learning_curves(history, str(self.eval_dir))
        TrainingVisualizer.plot_confusion_matrix(
            metrics["confusion_matrix"],
            self.classes,
            str(self.eval_dir / "confusion_matrix.png")
        )

        # Save classification report text
        with open(self.eval_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(metrics["classification_report"])

        # Save metrics JSON
        with open(self.eval_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Training pipeline finished for {self.model_name}. Test Accuracy: {test_acc:.4f}, F1: {metrics['macro_f1']:.4f}")
        return metrics

    def _train_one_epoch(self, model, dataloader, criterion, optimizer) -> Tuple[float, float]:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return (total_loss / total) if total > 0 else 0.0, (correct / total) if total > 0 else 0.0

    def _evaluate_split(self, model, dataloader, criterion) -> Tuple[float, float, list, list]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        avg_loss = (total_loss / total) if total > 0 else 0.0
        acc = (correct / total) if total > 0 else 0.0
        return avg_loss, acc, y_true, y_pred


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
