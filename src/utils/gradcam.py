"""
Module: src/utils/gradcam.py
Mục đích: Trích xuất bản đồ nhiệt Grad-CAM (XAI - Explainable AI) cho mô hình phân loại ResNet-34.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling của gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def overlay_cam_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Ép bản đồ nhiệt màu JET lên ảnh RGB gốc.
    """
    h, w = image_rgb.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    return np.uint8(np.clip(overlayed, 0, 255))

