from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.data.preprocessing import get_data_transforms
from src.utils.config_loader import get_config
from src.utils.logger import logger


class WasteDatasetLoader:
    """Manages torchvision ImageFolder datasets and PyTorch DataLoaders."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.paths = self.config["paths"]
        self.training_cfg = self.config["training"]
        self.classes = self.config["classes"]
        self.image_size = self.config["model"]["image_size"]

        self.transforms = get_data_transforms(self.image_size)

    def get_datasets(self) -> Dict[str, datasets.ImageFolder]:
        """Initializes and returns ImageFolder datasets for train, val, and test splits."""
        dataset_splits = {}
        split_map = {
            "train": self.paths["train_dir"],
            "validation": self.paths["val_dir"],
            "test": self.paths["test_dir"]
        }

        for split_name, split_path in split_map.items():
            path_obj = Path(split_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Dataset split directory not found: {path_obj}")

            # Instantiate ImageFolder with corresponding transform
            dataset = datasets.ImageFolder(
                root=str(path_obj),
                transform=self.transforms[split_name]
            )

            # Validate classes match configured schema
            found_classes = sorted(dataset.classes)
            expected_classes = sorted(self.classes)
            if found_classes != expected_classes:
                logger.warning(
                    f"Class mismatch in {split_name} split! Found: {found_classes}, Expected: {expected_classes}"
                )

            dataset_splits[split_name] = dataset

        return dataset_splits

    def get_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[Dict[str, DataLoader], Dict[int, str]]:
        """
        Creates optimized DataLoaders and returns an index-to-class dictionary.
        """
        batch_sz = batch_size or self.training_cfg.get("batch_size", 32)
        workers = num_workers if num_workers is not None else self.training_cfg.get("num_workers", 0)

        datasets_dict = self.get_datasets()

        # Check if train dataset contains images
        if len(datasets_dict["train"]) == 0:
            raise ValueError(
                f"Training dataset at {self.paths['train_dir']} contains 0 samples. "
                "Please place labeled waste images into the respective class directories."
            )

        dataloaders = {
            "train": DataLoader(
                datasets_dict["train"],
                batch_size=batch_sz,
                shuffle=True,
                num_workers=workers,
                pin_memory=torch.cuda.is_available()
            ),
            "validation": DataLoader(
                datasets_dict["validation"],
                batch_size=batch_sz,
                shuffle=False,
                num_workers=workers,
                pin_memory=torch.cuda.is_available()
            ),
            "test": DataLoader(
                datasets_dict["test"],
                batch_size=batch_sz,
                shuffle=False,
                num_workers=workers,
                pin_memory=torch.cuda.is_available()
            )
        }

        idx_to_class = {v: k for k, v in datasets_dict["train"].class_to_idx.items()}

        logger.info(
            f"DataLoaders prepared: train={len(datasets_dict['train'])}, "
            f"val={len(datasets_dict['validation'])}, test={len(datasets_dict['test'])} samples."
        )

        return dataloaders, idx_to_class
