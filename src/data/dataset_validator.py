import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from PIL import Image
from src.utils.config_loader import get_config
from src.utils.logger import logger


class DatasetValidator:
    """
    Validates filesystem dataset integrity, file formats, image decodability,
    split hygiene, duplicates, and class balance.
    """

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, dataset_root: str = None, classes: List[str] = None):
        config = get_config()
        self.dataset_root = Path(dataset_root or config["paths"]["dataset_root"])
        self.expected_classes = classes or config["classes"]
        self.splits = ["train", "validation", "test"]

    def validate_full_dataset(self) -> Dict[str, Any]:
        """Runs the complete suite of dataset hygiene and validation tests."""
        report = {
            "dataset_root": str(self.dataset_root),
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "split_statistics": {},
            "total_images": 0,
            "duplicates_detected": 0,
            "corrupted_images": 0
        }

        if not self.dataset_root.exists():
            report["is_valid"] = False
            report["errors"].append(f"Dataset root does not exist: {self.dataset_root}")
            return report

        all_hashes: Dict[str, str] = {}  # hash -> file_path for cross-split duplicate checks

        for split in self.splits:
            split_dir = self.dataset_root / split
            split_stats = self._validate_split(split_dir, split, report, all_hashes)
            report["split_statistics"][split] = split_stats
            report["total_images"] += split_stats["total_images"]

        # Check for class imbalance across training split
        train_stats = report["split_statistics"].get("train", {}).get("class_counts", {})
        if train_stats:
            counts = [c for c in train_stats.values() if c > 0]
            if len(counts) > 1:
                max_c, min_c = max(counts), min(counts)
                ratio = max_c / (min_c if min_c > 0 else 1)
                if ratio > 3.0:
                    report["warnings"].append(
                        f"Class imbalance detected in train set: max class has {max_c} samples vs min {min_c} (ratio {ratio:.1f}:1)"
                    )

        if report["errors"]:
            report["is_valid"] = False

        return report

    def _validate_split(
        self,
        split_dir: Path,
        split_name: str,
        report: Dict[str, Any],
        all_hashes: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validates an individual split directory (train/validation/test)."""
        stats = {
            "total_images": 0,
            "class_counts": {cls_name: 0 for cls_name in self.expected_classes},
            "corrupted_files": [],
            "duplicate_files": [],
            "unexpected_files": []
        }

        if not split_dir.exists():
            report["errors"].append(f"Split directory missing: {split_dir}")
            return stats

        # Verify each expected class exists
        for cls_name in self.expected_classes:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                report["errors"].append(f"Missing category folder: '{cls_name}' in {split_name}")
                continue

            # Iterate files inside class folder
            for file_path in cls_dir.iterdir():
                if file_path.name == ".gitkeep":
                    continue

                if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                    stats["unexpected_files"].append(str(file_path))
                    report["warnings"].append(f"Unsupported file format ignored: {file_path}")
                    continue

                # Verify image can be decoded
                is_valid_img, img_hash = self._inspect_image(file_path)
                if not is_valid_img:
                    stats["corrupted_files"].append(str(file_path))
                    report["corrupted_images"] += 1
                    report["errors"].append(f"Corrupted image cannot be decoded: {file_path}")
                    continue

                # Check duplicate hash
                if img_hash in all_hashes:
                    stats["duplicate_files"].append({
                        "file": str(file_path),
                        "original": all_hashes[img_hash]
                    })
                    report["duplicates_detected"] += 1
                    report["warnings"].append(f"Duplicate image hash detected: {file_path} matches {all_hashes[img_hash]}")
                else:
                    all_hashes[img_hash] = str(file_path)

                stats["class_counts"][cls_name] += 1
                stats["total_images"] += 1

        return stats

    @staticmethod
    def _inspect_image(path: Path) -> Tuple[bool, str]:
        """Attempts to open, verify, and hash image bytes."""
        try:
            with Image.open(path) as img:
                img.verify()  # Verifies file integrity
            # Reopen to read bytes for hashing
            with open(path, "rb") as f:
                content = f.read()
                file_hash = hashlib.md5(content).hexdigest()
            return True, file_hash
        except Exception:
            return False, ""


def run_validation():
    validator = DatasetValidator()
    report = validator.validate_full_dataset()
    logger.info(f"Dataset validation completed. Total images found: {report['total_images']}")
    for split, s in report["split_statistics"].items():
        logger.info(f"Split '{split}': {s['class_counts']}")
    return report

if __name__ == "__main__":
    run_validation()
