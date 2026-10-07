import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from src.utils.config_loader import get_config
from src.utils.logger import logger


class DatasetImporter:
    """
    Cleans synthetic placeholder verification images and automates
    importing and splitting real-world waste images into train/validation/test.
    """

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self):
        self.config = get_config()
        self.dataset_root = Path(self.config["paths"]["dataset_root"])
        self.classes = self.config["classes"]

    def clean_synthetic_placeholders(self) -> int:
        """Removes all synthetic verification_sample_*.jpg placeholder files."""
        removed_count = 0
        for split in ["train", "validation", "test"]:
            for cls_name in self.classes:
                folder = self.dataset_root / split / cls_name
                if folder.exists():
                    for file in folder.glob("verification_sample_*.jpg"):
                        file.unlink()
                        removed_count += 1
        logger.info(f"Removed {removed_count} synthetic placeholder images.")
        return removed_count

    def import_and_split_from_raw_folder(
        self,
        raw_data_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> None:
        """
        Takes a raw folder organized by category:
            raw_folder/
                plastic/ (any image filenames)
                paper/
                glass/
                ...
        Validates every image, removes synthetic placeholders, and splits them
        automatically into dataset/train, dataset/validation, dataset/test.
        """
        random.seed(seed)
        raw_path = Path(raw_data_dir)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {raw_path}")

        # Clean placeholders first
        self.clean_synthetic_placeholders()

        for cls_name in self.classes:
            cls_source = raw_path / cls_name
            if not cls_source.exists() or not cls_source.is_dir():
                logger.warning(f"Category folder '{cls_name}' not found in {raw_path}. Skipping.")
                continue

            # Gather all valid image files
            valid_files = []
            for f in cls_source.iterdir():
                if f.suffix.lower() in self.ALLOWED_EXTENSIONS:
                    try:
                        with Image.open(f) as img:
                            img.verify()
                        valid_files.append(f)
                    except Exception:
                        logger.warning(f"Skipping corrupted image: {f}")

            random.shuffle(valid_files)
            total = len(valid_files)
            if total == 0:
                logger.warning(f"No valid images found for '{cls_name}'.")
                continue

            n_train = int(total * train_ratio)
            n_val = int(total * val_ratio)
            train_files = valid_files[:n_train]
            val_files = valid_files[n_train:n_train + n_val]
            test_files = valid_files[n_train + n_val:]

            # Copy to respective split folders
            for split_name, files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
                target_folder = self.dataset_root / split_name / cls_name
                target_folder.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, target_folder / f.name)

            logger.info(
                f"Imported '{cls_name}': {len(train_files)} train, "
                f"{len(val_files)} val, {len(test_files)} test."
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean placeholders and import real waste images.")
    parser.add_argument("--clean-placeholders", action="store_true", help="Delete synthetic placeholder squares")
    parser.add_argument("--import-raw", type=str, help="Path to raw folder containing class subdirectories to split")

    args = parser.parse_args()
    importer = DatasetImporter()

    if args.clean_placeholders:
        importer.clean_synthetic_placeholders()
    elif args.import_raw:
        importer.import_and_split_from_raw_folder(args.import_raw)
    else:
        print("""
Usage:
  1. To delete the synthetic test squares:
     python -m src.data.dataset_importer --clean-placeholders

  2. To import and auto-split a folder of real images:
     python -m src.data.dataset_importer --import-raw <path-to-folder>
""")
