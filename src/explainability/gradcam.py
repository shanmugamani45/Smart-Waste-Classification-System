import base64
import io
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from src.utils.logger import logger


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) generator
    for visualizing influential convolutional regions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Identify target layer if not explicitly supplied
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "gradcam_target_layer"):
            self.target_layer = model.gradcam_target_layer
        elif hasattr(model, "features"):
            self.target_layer = model.features[-1]
        elif hasattr(model, "layer4"):
            self.target_layer = model.layer4[-1]
        else:
            raise ValueError("Unable to automatically identify target convolutional layer for Grad-CAM.")

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap array in range [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        target_score = output[0, target_class_idx]
        target_score.backward()

        # Channel-wise mean pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        # Weight activations by pooled gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def overlay_on_image(
        self,
        original_pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[Image.Image, str]:
        """
        Overlays the Grad-CAM heatmap onto the original PIL image.
        Returns (blended_pil_image, base64_png_string).
        """
        width, height = original_pil_image.size
        resized_heatmap = cv2.resize(heatmap, (width, height))
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        orig_np = np.array(original_pil_image)
        blended = cv2.addWeighted(orig_np, 1 - alpha, colored_heatmap, alpha, 0)

        blended_pil = Image.fromarray(blended)

        # Convert to base64 string for API response
        buffered = io.BytesIO()
        blended_pil.save(buffered, format="PNG")
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return blended_pil, f"data:image/png;base64,{b64_str}"
