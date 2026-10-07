import io
import pytest
import torch
from PIL import Image
from src.data.preprocessing import get_data_transforms, preprocess_image_bytes, denormalize_tensor

def test_data_transforms_output_shapes():
    transforms_dict = get_data_transforms(image_size=224)
    assert "train" in transforms_dict
    assert "validation" in transforms_dict
    assert "inference" in transforms_dict

    # Create dummy PIL image
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = transforms_dict["train"](img)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

def test_preprocess_image_bytes_valid():
    # Create byte buffer of a JPEG image
    img = Image.new("RGB", (250, 250), color=(200, 50, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    tensor, pil_out = preprocess_image_bytes(img_bytes, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert pil_out.size == (250, 250)

def test_preprocess_image_bytes_corrupted():
    corrupted_bytes = b"not_a_valid_image_header_bytes"
    with pytest.raises(ValueError) as excinfo:
        preprocess_image_bytes(corrupted_bytes)
    assert "Corrupted or invalid image format" in str(excinfo.value)

def test_preprocess_image_empty():
    with pytest.raises(ValueError):
        preprocess_image_bytes(b"")

def test_denormalize_tensor():
    dummy_tensor = torch.zeros(3, 224, 224)
    denorm = denormalize_tensor(dummy_tensor)
    assert denorm.shape == (3, 224, 224)
