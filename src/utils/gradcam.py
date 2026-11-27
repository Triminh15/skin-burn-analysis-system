"""
Module: src/utils/gradcam.py
Mục đích: Cung cấp giải thuật Grad-CAM (Gradient-weighted Class Activation Mapping)
thuần PyTorch để giải thích vùng quyết định lâm sàng (Explainable AI - XAI)
của mạng phân loại cấp độ bỏng (ResNet-34 / ConvNeXt).
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM thuần PyTorch tương thích mọi mô hình CNN (ResNet, ConvNeXt, VGG...)
    hoạt động bằng cơ chế Forward/Backward Hook, không cần thư viện ngoài.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Khởi tạo GradCAM:
        - model: Mô hình mạng nơ-ron phân loại (đã nạp trọng số)
        - target_layer: Lớp tích chập cuối cùng (ví dụ: model.layer4[-1] của ResNet-34)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Đăng ký hook bắt activations và gradients"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] chứa tensor gradient wrt output của target_layer
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h_fwd, h_bwd])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Tính toán bản đồ nhiệt Grad-CAM chuẩn hóa [0, 1] cho ảnh đầu vào:
        - input_tensor: Tensor kích thước (1, 3, H, W)
        - class_idx: Chỉ số lớp cần giải thích (0, 1, 2). Nếu None, tự lấy lớp có xác suất cao nhất.
        Trả về: Ma trận numpy 2D kích thước (H, W) với giá trị [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()

        # Đảm bảo tensor có batch size = 1 và đòi hỏi tính gradient
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass (không dùng torch.no_grad() vì cần đồ thị tính gradient)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        # Target score cho lớp được chọn
        score = output[0, class_idx]

        # Backward pass để tính đạo hàm ngược tới target_layer
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # 1. Tính trọng số alpha bằng Global Average Pooling trên không gian (H, W) của gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 2. Nhân trọng số alpha với các bản đồ đặc trưng (activations) và cộng dồn lại theo kênh C
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 3. Áp dụng ReLU: Chỉ giữ lại các đặc trưng làm TĂNG điểm số chẩn đoán của lớp mục tiêu
        cam = F.relu(cam)

        # Chuyển về numpy 2D
        cam = cam.squeeze().cpu().numpy()

        # 4. Chuẩn hóa dải giá trị về [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phóng to heatmap về kích thước ảnh RGB, phủ màu JET và trộn đè (Alpha Blending):
        - image_rgb: Ảnh gốc định dạng RGB (H, W, 3), giá trị uint8 0-255
        - heatmap: Ma trận heatmap 2D [0.0, 1.0]
        - alpha: Tỷ lệ ảnh gốc (0.55 ảnh gốc + 0.45 bản đồ nhiệt)
        - colormap: cv2.COLORMAP_JET hoặc cv2.COLORMAP_TURBO
        Trả về: (overlay_rgb, colored_heatmap_rgb)
        """
        h, w = image_rgb.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image_rgb, alpha, colored_rgb, 1.0 - alpha, 0)
        return overlay, colored_rgb

    def remove_hooks(self):
        """Hủy đăng ký hooks để giải phóng bộ nhớ khi không dùng nữa"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def overlay_cam_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Ép bản đồ nhiệt màu JET lên ảnh RGB gốc (tương thích cho burn_analyzer.py).
    """
    h, w = image_rgb.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    return np.uint8(np.clip(overlayed, 0, 255))


def get_target_layer_for_model(model: nn.Module, model_name: str = "resnet34") -> nn.Module:
    """
    Tự động tìm lớp tích chập cuối cùng phù hợp với từng kiến trúc mạng.
    """
    name_clean = model_name.lower().replace("-", "_")

    if "resnet" in name_clean:
        if hasattr(model, "layer4"):
            return model.layer4[-1]
    elif "convnext" in name_clean:
        if hasattr(model, "stages"):
            return model.stages[-1].blocks[-1]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.Conv2d, nn.Sequential)):
            return module

    raise ValueError(f"Không thể tự động xác định target layer cho kiến trúc: {model_name}")
