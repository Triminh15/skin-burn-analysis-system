"""
Module: src/vision/metrics.py
Mục đích: Cung cấp đầy đủ các độ đo đánh giá chuyên dụng theo chuẩn Y khoa:
1. Độ đo Phân đoạn (Segmentation): Dice Score, IoU (Jaccard Index), Pixel Accuracy.
2. Độ đo Phân loại (Classification): Overall Accuracy, Macro F1-Score, Per-class Recall/Precision, Confusion Matrix.
3. Lớp Tracker đo lường xuyên suốt toàn bộ Epoch hoặc toàn bộ tập Test.
"""

import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# ==============================================================================
# PHẦN 1: ĐỘ ĐO PHÂN ĐOẠN (SEGMENTATION METRICS)
# ==============================================================================

def calculate_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Tính chỉ số Dice Score (F1 cấp độ Pixel) cho nhãn nhị phân.
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)
        return float(dice.item())


def calculate_iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Tính chỉ số IoU (Intersection over Union / Jaccard Index).
    IoU = |P ∩ G| / |P ∪ G|
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        return float(iou.item())


def calculate_pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Tính tỷ lệ số pixel đoán đúng trên tổng số pixel (Pixel Accuracy).
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        correct = (preds == targets).float().sum()
        total = torch.numel(targets)
        return float((correct / total).item())


class SegmentationMetricTracker:
    """
    Bộ theo dõi và tích lũy các độ đo phân đoạn trên toàn bộ tập dữ liệu (Epoch/Test set).
    Tránh lỗi sai lệch khi lấy trung bình của trung bình.
    """
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_intersection = 0.0
        self.total_preds = 0.0
        self.total_targets = 0.0
        self.total_union = 0.0
        self.total_correct_pixels = 0
        self.total_pixels = 0
        self.batch_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()

            preds_flat = preds.view(-1)
            targets_flat = targets.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            p_sum = preds_flat.sum().item()
            t_sum = targets_flat.sum().item()
            union = p_sum + t_sum - intersection

            self.total_intersection += intersection
            self.total_preds += p_sum
            self.total_targets += t_sum
            self.total_union += union

            self.total_correct_pixels += (preds_flat == targets_flat).sum().item()
            self.total_pixels += targets_flat.numel()
            self.batch_count += 1

    def compute(self) -> Dict[str, float]:
        dice = (2.0 * self.total_intersection + self.eps) / (
            self.total_preds + self.total_targets + self.eps
        )
        iou = (self.total_intersection + self.eps) / (self.total_union + self.eps)
        pixel_acc = (
            self.total_correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
        )
        return {
            "dice": round(float(dice), 4),
            "iou": round(float(iou), 4),
            "pixel_accuracy": round(float(pixel_acc), 4)
        }


# ==============================================================================
# PHẦN 2: ĐỘ ĐO PHÂN LOẠI (CLASSIFICATION METRICS)
# ==============================================================================

class ClassificationMetricTracker:
    """
    Bộ tích lũy và tính toán độ đo phân loại chuẩn y tế:
    - Overall Accuracy
    - Macro F1-score (coi trọng cả 3 lớp như nhau)
    - Per-class Recall: đặc biệt là Recall của ca nguy kịch (Bỏng Độ 3)
    - Confusion Matrix (Ma trận nhầm lẫn)
    """
    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            self.class_names = ["Độ 1 (Epidermal)", "Độ 2 (Dermal)", "Độ 3 (Full-thickness)"]
        else:
            self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            self.all_preds.extend(preds.cpu().numpy().tolist())
            self.all_targets.extend(targets.cpu().numpy().tolist())

    def compute(self) -> Dict[str, Any]:
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # Recall và Precision chi tiết cho từng lớp
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Ma trận nhầm lẫn (Confusion Matrix)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        metrics = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_precision": round(float(macro_precision), 4),
            "macro_recall": round(float(macro_recall), 4),
            # Chỉ số y đức: Recall của lớp bỏng Độ 3 (Class 2)
            "recall_degree_3": round(float(recalls[2]), 4) if len(recalls) > 2 else 0.0,
            "per_class": {
                name: {
                    "precision": round(float(precisions[i]), 4),
                    "recall": round(float(recalls[i]), 4),
                    "f1": round(float(f1s[i]), 4)
                }
                for i, name in enumerate(self.class_names) if i < len(recalls)
            },
            "confusion_matrix": cm.tolist()
        }
        return metrics


# ==============================================================================
# KIỂM THỬ ĐƠN VỊ NHANH (SANITY CHECK)
# ==============================================================================
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("--- Testing Unit Tests for src/vision/metrics.py ---")

    # 1. Test Segmentation Metrics
    dummy_logits = torch.randn(4, 1, 256, 256)
    dummy_targets = torch.randint(0, 2, (4, 1, 256, 256)).float()

    dice = calculate_dice_score(dummy_logits, dummy_targets)
    iou = calculate_iou_score(dummy_logits, dummy_targets)
    acc = calculate_pixel_accuracy(dummy_logits, dummy_targets)
    print(f"[OK] Single batch: Dice={dice:.4f}, IoU={iou:.4f}, PixelAcc={acc:.4f}")

    seg_tracker = SegmentationMetricTracker()
    seg_tracker.update(dummy_logits, dummy_targets)
    seg_tracker.update(dummy_logits, dummy_targets)
    seg_results = seg_tracker.compute()
    print(f"[OK] Segmentation Tracker: {seg_results}")

    # 2. Test Classification Metrics
    cls_tracker = ClassificationMetricTracker()
    dummy_cls_logits = torch.randn(10, 3)
    dummy_cls_targets = torch.randint(0, 3, (10,)).long()

    cls_tracker.update(dummy_cls_logits, dummy_cls_targets)
    cls_results = cls_tracker.compute()
    print(f"[OK] Classification Accuracy: {cls_results['accuracy']:.4f}")
    print(f"[OK] Classification Macro F1: {cls_results['macro_f1']:.4f}")
    print(f"[OK] Critical Recall Degree 3: {cls_results['recall_degree_3']:.4f}")
    print(f"[OK] Confusion Matrix shape: {len(cls_results['confusion_matrix'])}x{len(cls_results['confusion_matrix'][0])}")
    print("[SUCCESS] All metrics computed correctly!")
