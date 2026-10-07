import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models
from src.models.classifier import CustomWasteClassifierHead
from src.utils.config_loader import get_config
from src.utils.logger import logger


class ModelFactory:
    """
    Factory for instantiating, configuring, and modifying transfer learning models:
    - MobileNetV3-Large (Primary)
    - EfficientNet-B0 (Alternative)
    - ResNet50 (Optional benchmark)
    """

    SUPPORTED_MODELS = ["MobileNetV3", "EfficientNet-B0", "ResNet50"]

    @classmethod
    def build_model(
        cls,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ) -> nn.Module:
        config = get_config()
        name = model_name or config["model"]["name"]
        n_classes = num_classes or config["model"]["num_classes"]

        logger.info(f"Building model: {name} (num_classes={n_classes}, pretrained={pretrained})")

        if name == "MobileNetV3":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)
            in_features = base_model.classifier[0].out_features if hasattr(base_model.classifier[0], 'out_features') else 960
            # MobileNetV3 classifier is sequential: [0: Linear(960, 1280), 1: Hardswish, 2: Dropout, 3: Linear(1280, 1000)]
            # We replace with our custom head from the 960 pool output or from 1280
            in_features = base_model.classifier[0].in_features
            base_model.classifier = nn.Sequential(
                base_model.classifier[0],
                base_model.classifier[1],
                CustomWasteClassifierHead(
                    in_features=base_model.classifier[0].out_features,
                    num_classes=n_classes,
                    hidden_dim=256,
                    dropout_rate=dropout_rate
                )
            )
            base_model.gradcam_target_layer = base_model.features[-1]

        elif name == "EfficientNet-B0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = CustomWasteClassifierHead(
                in_features=in_features,
                num_classes=n_classes,
                hidden_dim=256,
                dropout_rate=dropout_rate
            )
            base_model.gradcam_target_layer = base_model.features[-1]

        elif name == "ResNet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = CustomWasteClassifierHead(
                in_features=in_features,
                num_classes=n_classes,
                hidden_dim=256,
                dropout_rate=dropout_rate
            )
            base_model.gradcam_target_layer = base_model.layer4[-1]

        else:
            raise ValueError(
                f"Unsupported model: '{name}'. Choose from: {cls.SUPPORTED_MODELS}"
            )

        base_model.model_architecture_name = name
        return base_model

    @staticmethod
    def freeze_backbone(model: nn.Module) -> None:
        """Freezes all parameters except the custom classification head."""
        name = getattr(model, "model_architecture_name", "unknown")
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze head
        head = model.fc if hasattr(model, "fc") else model.classifier
        for param in head.parameters():
            param.requires_grad = True

        logger.info(f"Backbone frozen for {name}. Head is trainable.")

    @staticmethod
    def unfreeze_top_layers(model: nn.Module, num_blocks: int = 3) -> None:
        """
        Unfreezes the top feature extraction blocks for Stage 2 fine-tuning.
        """
        name = getattr(model, "model_architecture_name", "")
        if name in ["MobileNetV3", "EfficientNet-B0"]:
            features = model.features
            total_blocks = len(features)
            unfreeze_from = max(0, total_blocks - num_blocks)
            for idx, block in enumerate(features):
                if idx >= unfreeze_from:
                    for param in block.parameters():
                        param.requires_grad = True
            logger.info(f"Unfroze last {num_blocks} blocks of {name} features for fine-tuning.")

        elif name == "ResNet50":
            # Unfreeze layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            logger.info("Unfroze layer4 of ResNet50 for fine-tuning.")

    @staticmethod
    def get_parameter_summary(model: nn.Module) -> Dict[str, Any]:
        """Calculates total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }
