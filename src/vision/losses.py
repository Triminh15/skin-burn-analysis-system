"""
Module: src/vision/losses.py
Mục đích: Định nghĩa các hàm mất mát (Loss Functions) chuyên dụng cho:
1. Giai đoạn Phân đoạn (Segmentation): ComboLoss (BCE + Dice), Focal Tversky Loss.
2. Giai đoạn Phân loại (Classification): Weighted Cross-Entropy, Class-Balanced Focal Loss kết hợp Label Smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ==============================================================================
# PHẦN 1: HÀM MẤT MÁT CHO PHÂN ĐOẠN (SEGMENTATION LOSSES)
# ==============================================================================

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss cho bài toán phân đoạn nhị phân (Binary Segmentation).
    Đầu vào: Raw Logits (chưa qua Sigmoid) từ mô hình.
    Đầu ra: Giá trị sai số diện tích trùng khớp (1 - Dice Coefficient).
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Chuyển logits về xác suất [0, 1]
        probs = torch.sigmoid(logits)

        # Làm phẳng tensor để tính toán trên toàn bộ batch
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """
    ComboLoss (BCEWithLogitsLoss + SoftDiceLoss) - Mốc Baseline chuẩn mực.
    Kết hợp:
    - BCE: Phạt sai số độc lập ở từng điểm ảnh (Pixel-level).
    - Dice: Tối ưu sự trùng khớp hình dạng tổng thể vùng tổn thương (Region-level).
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Tiêu chuẩn vàng trong phân đoạn ảnh Y tế.
    Giải quyết 2 bài toán lớn:
    1. Tversky Index: Phạt nặng False Negatives (FN - Bỏ sót vết bỏng) hơn False Positives (FP).
       alpha (FP weight) = 0.3, beta (FN weight) = 0.7.
    2. Focal parameter (gamma = 1.33): Dồn gradient tập trung vào các vùng viền mờ khó đoán (Hard Examples).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Chuyển logits về xác suất
        probs = torch.sigmoid(logits)

        # Làm phẳng
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Tính True Positives, False Positives, False Negatives
        true_pos = (probs * targets).sum()
        false_pos = ((1.0 - targets) * probs).sum()
        false_neg = (targets * (1.0 - probs)).sum()

        # Tversky Index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        # Focal Tversky Loss: (1 - TI)^gamma
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss


def get_segmentation_loss(name: str = "combo", **kwargs) -> nn.Module:
    """
    Factory function khởi tạo hàm loss phân đoạn.
    Hỗ trợ: 'combo', 'focal_tversky', 'bce', 'dice'
    """
    name_clean = name.lower().replace("-", "_")
    if name_clean == "combo":
        return ComboLoss(**kwargs)
    elif name_clean in ["focal_tversky", "ftl"]:
        return FocalTverskyLoss(**kwargs)
    elif name_clean == "dice":
        return SoftDiceLoss(**kwargs)
    elif name_clean == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Hàm loss phân đoạn không hợp lệ: {name}. Chọn: combo, focal_tversky, dice, bce.")


# ==============================================================================
# PHẦN 2: HÀM MẤT MÁT CHO PHÂN LOẠI (CLASSIFICATION LOSSES)
# ==============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss kết hợp Label Smoothing (Cui et al., CVPR 2019).
    Đặc trị bài toán mất cân bằng mẫu nghiêm trọng:
    - Sử dụng 'Effective Number of Samples' E_n = (1 - beta) / (1 - beta^n)
    - Focal parameter (gamma = 2.0) giúp dồn trọng số vào ca khó phân biệt.
    - Label Smoothing (0.05 - 0.1) chống hiện tượng overconfident / overfitting trên dữ liệu y tế.
    """
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Tính toán Class-Balanced Weights
        effective_num = [1.0 - (beta ** n) if n > 0 else 1.0 for n in samples_per_class]
        weights = [(1.0 - beta) / en for en in effective_num]
        total_weight = sum(weights)
        # Chuẩn hóa trọng số sao cho tổng trọng số = số lượng lớp (C)
        num_classes = len(samples_per_class)
        norm_weights = [w / total_weight * num_classes for w in weights]
        
        # Đăng ký buffer để tự động chuyển theo thiết bị (CPU/GPU) của mô hình
        self.register_buffer("class_weights", torch.tensor(norm_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-Entropy với Label Smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )

        # Tính xác suất pt của lớp thực tế để áp dụng Focal Modulation: (1 - pt)^gamma
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Focal Loss hoàn chỉnh
        loss = focal_weight * ce_loss
        return loss.mean()


def get_classification_loss(
    name: str = "weighted_ce",
    samples_per_class: Optional[List[int]] = None,
    label_smoothing: float = 0.05,
    **kwargs
) -> nn.Module:
    """
    Factory function khởi tạo hàm loss phân loại.
    Hỗ trợ:
    - 'weighted_ce': Nghịch đảo tần số mẫu cơ bản (Baseline).
    - 'cb_focal': Class-Balanced Focal Loss + Label Smoothing (SOTA).
    """
    name_clean = name.lower().replace("-", "_")
    
    if samples_per_class is None:
        # Mặc định theo phân bố tập Train: Độ 1 (359), Độ 2 (346), Độ 3 (153)
        samples_per_class = [359, 346, 153]

    if name_clean in ["weighted_ce", "ce"]:
        # Trọng số nghịch đảo tần số
        total_samples = sum(samples_per_class)
        num_classes = len(samples_per_class)
        weights = [total_samples / (num_classes * n) for n in samples_per_class]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    elif name_clean in ["cb_focal", "class_balanced_focal"]:
        return ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            label_smoothing=label_smoothing,
            **kwargs
        )
    else:
        raise ValueError(f"Hàm loss phân loại không hợp lệ: {name}. Chọn: weighted_ce, cb_focal.")


# ==============================================================================
# KIỂM THỬ ĐƠN VỊ NHANH (SANITY CHECK)
# ==============================================================================
if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("--- Testing Unit Tests for src/vision/losses.py ---")
    
    # 1. Test Segmentation Losses
    dummy_logits = torch.randn(4, 1, 256, 256, requires_grad=True)
    dummy_masks = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    loss_combo_fn = get_segmentation_loss("combo")
    loss_combo = loss_combo_fn(dummy_logits, dummy_masks)
    loss_combo.backward()
    print(f"[OK] ComboLoss test: {loss_combo.item():.4f} (Grad OK)")

    loss_ftl_fn = get_segmentation_loss("focal_tversky")
    loss_ftl = loss_ftl_fn(dummy_logits, dummy_masks)
    print(f"[OK] FocalTverskyLoss test: {loss_ftl.item():.4f} (Success)")

    # 2. Test Classification Losses
    dummy_cls_logits = torch.randn(8, 3, requires_grad=True)
    dummy_labels = torch.randint(0, 3, (8,)).long()

    loss_ce_fn = get_classification_loss("weighted_ce", samples_per_class=[359, 346, 153])
    loss_ce = loss_ce_fn(dummy_cls_logits, dummy_labels)
    print(f"[OK] Weighted CrossEntropy test: {loss_ce.item():.4f} (Success)")

    loss_cbf_fn = get_classification_loss("cb_focal", samples_per_class=[359, 346, 153])
    loss_cbf = loss_cbf_fn(dummy_cls_logits, dummy_labels)
    loss_cbf.backward()
    print(f"[OK] Class-Balanced Focal Loss test: {loss_cbf.item():.4f} (Grad OK)")
    print("[SUCCESS] All loss functions passed sanity checks perfectly!")
