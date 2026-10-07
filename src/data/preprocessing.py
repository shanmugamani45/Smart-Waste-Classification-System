import io
from typing import Tuple, Dict, Any, Union
from PIL import Image
import torchvision.transforms as transforms
import torch
from src.utils.config_loader import get_config

# Standard ImageNet normalization coefficients
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Constructs PyTorch transform pipelines for training, validation, and inference.
    Training uses conservative, realistic augmentations to prevent domain shift.
    """
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        "train": train_transform,
        "validation": val_test_transform,
        "test": val_test_transform,
        "inference": val_test_transform
    }


def preprocess_image_bytes(
    image_bytes: bytes,
    image_size: int = 224
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Safely validates, sanitizes, and transforms raw image bytes into a PyTorch model tensor.
    
    Args:
        image_bytes: Raw binary payload from HTTP upload.
        image_size: Expected spatial resolution.

    Returns:
        (tensor [1, 3, H, W], original_pil_rgb_image)
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Received empty image payload.")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image format: {str(e)}")

    # Ensure 3-channel RGB representation
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    transform = get_data_transforms(image_size)["inference"]
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

    return tensor, pil_image


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization for visualization and explainability overlays."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean
