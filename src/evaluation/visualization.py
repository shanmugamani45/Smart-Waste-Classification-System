from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import logger


class TrainingVisualizer:
    """Visualizes training dynamics, convergence, and evaluation confusion matrices."""

    @staticmethod
    def plot_learning_curves(
        history: Dict[str, List[float]],
        output_dir: str = "./evaluation"
    ) -> Dict[str, str]:
        """Plots and saves loss and accuracy progression curves."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        epochs = range(1, len(history["train_loss"]) + 1)

        # 1. Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#3498db")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        loss_file = str(out_path / "loss_curve.png")
        plt.savefig(loss_file, dpi=200)
        plt.close()
        saved_files["loss_curve"] = loss_file

        # 2. Accuracy Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ecc71")
        if "val_acc" in history and history["val_acc"]:
            plt.plot(epochs, history["val_acc"], "s-", label="Validation Accuracy", color="#9b59b6")
        plt.title("Training & Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        acc_file = str(out_path / "accuracy_curve.png")
        plt.savefig(acc_file, dpi=200)
        plt.close()
        saved_files["accuracy_curve"] = acc_file

        logger.info(f"Saved learning curves to: {saved_files}")
        return saved_files

    @staticmethod
    def plot_confusion_matrix(
        cm_matrix: List[List[int]],
        class_names: List[str],
        output_path: str = "./evaluation/confusion_matrix.png"
    ) -> str:
        """Plots confusion matrix as a formatted heatmap."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cm = np.array(cm_matrix)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Confusion Matrix - Waste Material Classification",
            ylabel="True Class",
            xlabel="Predicted Class"
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        fig.tight_layout()
        plt.savefig(str(out_p), dpi=200)
        plt.close()
        logger.info(f"Saved confusion matrix plot to: {out_p}")
        return str(out_p)
