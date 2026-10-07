import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from src.utils.config_loader import get_config
from src.utils.logger import logger

# Synthetic visual distinctiveness mapping for pipeline verification only
SYNTHETIC_COLOR_MAP = {
    "plastic": (50, 150, 255),
    "paper": (240, 240, 230),
    "glass": (100, 220, 200),
    "metal": (160, 160, 170),
    "organic": (80, 180, 70),
    "textile": (220, 100, 150),
    "e_waste": (70, 70, 80)
}


def populate_synthetic_verification_dataset(samples_per_class: int = 5):
    """
    Generates minimal synthetic patterned images strictly for verifying that
    the training loop, loss computation, backpropagation, and checkpointing
    execute without errors. 
    NOTE: Real academic evaluations require real-world datasets (e.g. TrashNet).
    """
    config = get_config()
    dataset_root = Path(config["paths"]["dataset_root"])
    classes = config["classes"]

    logger.warning("Generating SYNTHETIC verification dataset. DO NOT cite as academic ground truth!")

    splits_dist = {"train": samples_per_class, "validation": max(2, samples_per_class // 2), "test": max(2, samples_per_class // 2)}

    for split, count in splits_dist.items():
        for cls_name in classes:
            folder = dataset_root / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            base_color = SYNTHETIC_COLOR_MAP.get(cls_name, (128, 128, 128))

            for i in range(count):
                img = Image.new("RGB", (224, 224), color=base_color)
                draw = ImageDraw.Draw(img)
                # Draw class-specific geometric patterns
                draw.rectangle([20 + i*10, 20 + i*10, 200 - i*10, 200 - i*10], outline=(255, 255, 255), width=3)
                draw.text((30, 30), f"{cls_name.upper()} #{i+1}", fill=(255, 255, 255))

                img_path = folder / f"verification_sample_{i+1}.jpg"
                img.save(img_path, "JPEG", quality=90)

    logger.info("Synthetic verification dataset generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Ingestion & Synthetic Setup")
    parser.add_argument("--synthetic-for-test-only", action="store_true", help="Generate synthetic samples to test train.py")
    args = parser.parse_args()

    if args.synthetic_for_test_only:
        populate_synthetic_verification_dataset()
    else:
        print("""
======================================================================
REAL-WORLD DATASET INGESTION INSTRUCTIONS
======================================================================
To train the circular economy waste classification model on real data:

1. Obtain a standardized waste classification benchmark dataset:
   - TrashNet (2,527 images across 6-7 waste classes)
   - TACO (Trash Annotations in Context)
   - RealWaste Dataset
   
2. Place the images into the following directory layout:
   dataset/
   ├── train/
   │   ├── plastic/
   │   ├── paper/
   │   ├── glass/
   │   ├── metal/
   │   ├── organic/
   │   ├── textile/
   │   └── e_waste/
   ├── validation/
   │   └── [same 7 categories]
   └── test/
       └── [same 7 categories]

3. Run dataset validation before training:
   python -m src.data.dataset_validator

4. Start training:
   python -m src.training.train
======================================================================
""")
