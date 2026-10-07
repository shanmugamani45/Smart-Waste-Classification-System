import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from src.data.dataset_validator import DatasetValidator

@pytest.fixture
def dummy_dataset_dir():
    temp_dir = tempfile.mkdtemp()
    splits = ["train", "validation", "test"]
    classes = ["plastic", "paper", "glass", "metal", "organic", "textile", "e_waste"]

    for s in splits:
        for c in classes:
            p = Path(temp_dir) / s / c
            p.mkdir(parents=True, exist_ok=True)
            # Add one valid image
            img = Image.new("RGB", (100, 100), color=(50, 100, 150))
            img.save(p / "sample_1.jpg", format="JPEG")

    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_dataset_validator_valid_dataset(dummy_dataset_dir):
    validator = DatasetValidator(dataset_root=dummy_dataset_dir)
    report = validator.validate_full_dataset()
    assert report["is_valid"] is True
    assert report["total_images"] == 21  # 3 splits * 7 classes * 1 image
    assert report["corrupted_images"] == 0

def test_dataset_validator_corrupted_file(dummy_dataset_dir):
    corrupt_file = Path(dummy_dataset_dir) / "train" / "plastic" / "bad.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"not_an_image_header_bytes")

    validator = DatasetValidator(dataset_root=dummy_dataset_dir)
    report = validator.validate_full_dataset()
    assert report["corrupted_images"] >= 1
    assert report["is_valid"] is False

def test_dataset_validator_duplicate_detection(dummy_dataset_dir):
    orig = Path(dummy_dataset_dir) / "train" / "glass" / "sample_1.jpg"
    dup = Path(dummy_dataset_dir) / "train" / "glass" / "sample_2.jpg"
    # Copy exact bytes
    with open(orig, "rb") as f_in, open(dup, "wb") as f_out:
        f_out.write(f_in.read())

    validator = DatasetValidator(dataset_root=dummy_dataset_dir)
    report = validator.validate_full_dataset()
    assert report["duplicates_detected"] >= 1
