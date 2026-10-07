import pytest
import torch
from PIL import Image
import numpy as np
from src.models.model_factory import ModelFactory
from src.explainability.gradcam import GradCAM

def test_build_mobilenetv3_head():
    model = ModelFactory.build_model("MobileNetV3", num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)

def test_build_efficientnet_head():
    model = ModelFactory.build_model("EfficientNet-B0", num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)

def test_freeze_and_unfreeze():
    model = ModelFactory.build_model("MobileNetV3", num_classes=7, pretrained=False)
    ModelFactory.freeze_backbone(model)
    summary_frozen = ModelFactory.get_parameter_summary(model)
    assert summary_frozen["frozen_parameters"] > 0
    assert summary_frozen["trainable_parameters"] > 0

    ModelFactory.unfreeze_top_layers(model, num_blocks=2)
    summary_unfrozen = ModelFactory.get_parameter_summary(model)
    assert summary_unfrozen["trainable_parameters"] > summary_frozen["trainable_parameters"]

def test_gradcam_execution():
    model = ModelFactory.build_model("MobileNetV3", num_classes=7, pretrained=False)
    model.eval()
    gradcam = GradCAM(model)

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = gradcam.generate_heatmap(dummy_tensor, target_class_idx=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert 0.0 <= heatmap.min() <= 1.0
    assert 0.0 <= heatmap.max() <= 1.0

    # Test overlay
    pil_dummy = Image.new("RGB", (224, 224), color=(100, 150, 200))
    blended, b64_str = gradcam.overlay_on_image(pil_dummy, heatmap)
    assert blended.size == (224, 224)
    assert b64_str.startswith("data:image/png;base64,")
