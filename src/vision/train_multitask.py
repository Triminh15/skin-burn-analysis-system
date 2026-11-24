"""
Module: src/vision/train_multitask.py
Mục đích: Pipeline huấn luyện đầu-cuối (End-to-End) mạng đa nhiệm MT-EANet-EDL
(Multi-Task Erythema-Aware Network with Evidential Deep Learning) theo đặc tả
kỹ thuật tại Section 7 & Section 9.3 của docs/RESEARCH_ARCHITECTURE_AND_ROADMAP.md.

Tính năng chuẩn y sinh:
- Shared ResNet-34 Backbone cập nhật trọng số đồng thời từ cả 3 nhánh nhiệm vụ.
- Hàm mất mát kết hợp đa nhiệm (Comprehensive Multi-Task Joint Loss):
      L_total = w_seg * L_seg + w_bnd * L_bnd + w_cls * L_cls
  Trong đó:
      + L_seg: Focal Tversky Loss (hoặc ComboLoss) chống bỏ sót viền tổn thương.
      + L_bnd: Binary Cross-Entropy cho nhánh phát hiện viền ban đỏ mờ ngoại vi.
      + L_cls: Evidential Loss (EDL) dựa trên phân phối Dirichlet kèm Annealing Schedule.
- Theo dõi đầy đủ các độ đo chuẩn y tế:
      + Phân đoạn: Dice Score, IoU (Jaccard Index), Pixel Accuracy.
      + Phân loại: Overall Accuracy, Macro F1, Recall ca nguy kịch Độ 3, Confusion Matrix.
      + Độ bất định y sinh: Epistemic Uncertainty (u), so sánh độ bất định trên ca đúng vs sai.
- Quản lý Checkpoint (best_model.pth, last_model.pth) dựa trên điểm tổng hợp (Dice + Macro F1).
- Cơ chế Early Stopping chống Overfitting.
- Vẽ và xuất biểu đồ Loss & Metric curves tự động.
- Đánh giá khách quan trên tập Test độc lập (187 ảnh) và lưu test_results.json.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Tuple, Optional, List

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # Chế độ headless an toàn cho máy chủ / Docker / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.vision.dataset_multitask import SkinBurnMultiTaskDataset, get_multitask_dataloaders
from src.vision.models_multitask import MTEANet, build_multitask_model, count_parameters
from src.vision.edl_loss import EDLLoss, get_edl_metrics
from src.vision.losses import get_segmentation_loss, FocalTverskyLoss
from src.vision.metrics import SegmentationMetricTracker, ClassificationMetricTracker


# ==============================================================================
# 1. HÀM CỐ ĐỊNH SEED TÁI LẬP (REPRODUCIBILITY)
# ==============================================================================

def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên đảm bảo tính tái lập 100% trong nghiên cứu y sinh."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2. BỘ THEO DÕI ĐỘ ĐO ĐA NHIỆM (MULTITASK METRIC TRACKER)
# ==============================================================================

class MultiTaskMetricTracker:
    """
    Bộ theo dõi và tích lũy số liệu cho cả 3 nhánh nhiệm vụ trong một Epoch:
    - Nhánh Phân đoạn: Dice, IoU, Pixel Accuracy
    - Nhánh Viền: Boundary BCE Loss
    - Nhánh Phân loại: Accuracy, Macro F1, Recall Độ 3, Per-class metrics
    - Nhánh Độ bất định (EDL): Epistemic Uncertainty u, so sánh u đúng vs u sai
    """

    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.seg_tracker = SegmentationMetricTracker()
        self.cls_tracker = ClassificationMetricTracker(class_names=class_names)
        self.reset()

    def reset(self):
        self.seg_tracker.reset()
        self.cls_tracker.reset()
        self.all_uncertainties = []
        self.correct_uncertainties = []
        self.incorrect_uncertainties = []
        self.total_samples = 0
        self.loss_seg_accum = 0.0
        self.loss_bnd_accum = 0.0
        self.loss_cls_accum = 0.0
        self.loss_total_accum = 0.0
        self.batch_count = 0

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        boundaries: torch.Tensor,
        labels: torch.Tensor,
        losses: Dict[str, float],
        use_edl: bool = True,
    ):
        batch_size = labels.size(0)
        self.total_samples += batch_size
        self.batch_count += 1

        self.loss_seg_accum += losses.get("loss_seg", 0.0) * batch_size
        self.loss_bnd_accum += losses.get("loss_bnd", 0.0) * batch_size
        self.loss_cls_accum += losses.get("loss_cls", 0.0) * batch_size
        self.loss_total_accum += losses.get("loss_total", 0.0) * batch_size

        # Cập nhật Segmentation
        self.seg_tracker.update(outputs["mask_logits"], masks)

        # Cập nhật Classification & Uncertainty
        with torch.no_grad():
            if use_edl:
                evidence = outputs["evidence"]
                probs, uncertainty = get_edl_metrics(evidence, num_classes=self.num_classes)
            else:
                cls_logits = outputs["cls_logits"]
                probs = torch.softmax(cls_logits, dim=-1)
                # Softmax uncertainty: 1 - max probability
                uncertainty = 1.0 - torch.max(probs, dim=-1).values

            preds = torch.argmax(probs, dim=-1)

            # Cập nhật phân loại
            self.cls_tracker.update(probs, labels)

            # Phân tách độ bất định trên ca đúng và ca sai
            u_np = uncertainty.cpu().numpy()
            correct_mask = (preds == labels).cpu().numpy()

            self.all_uncertainties.extend(u_np.tolist())
            self.correct_uncertainties.extend(u_np[correct_mask].tolist())
            self.incorrect_uncertainties.extend(u_np[~correct_mask].tolist())

    def compute(self) -> Dict[str, Any]:
        seg_res = self.seg_tracker.compute()
        cls_res = self.cls_tracker.compute()

        n = max(self.total_samples, 1)
        mean_loss_total = self.loss_total_accum / n
        mean_loss_seg = self.loss_seg_accum / n
        mean_loss_bnd = self.loss_bnd_accum / n
        mean_loss_cls = self.loss_cls_accum / n

        u_all = np.array(self.all_uncertainties) if self.all_uncertainties else np.array([0.0])
        u_correct = np.array(self.correct_uncertainties) if self.correct_uncertainties else np.array([0.0])
        u_incorrect = np.array(self.incorrect_uncertainties) if self.incorrect_uncertainties else np.array([0.0])

        # Tỷ lệ các ca cần kích hoạt hội chẩn y tế (Triage Abstention Rates)
        abstain_02 = float(np.mean(u_all >= 0.2)) if len(u_all) > 0 else 0.0
        abstain_04 = float(np.mean(u_all >= 0.4)) if len(u_all) > 0 else 0.0

        # Điểm tổng hợp cho việc chọn checkpoint tối ưu: 50% Dice + 50% Macro F1
        composite_score = round(0.5 * seg_res["dice"] + 0.5 * cls_res["macro_f1"], 4)

        return {
            "loss_total": round(mean_loss_total, 4),
            "loss_seg": round(mean_loss_seg, 4),
            "loss_bnd": round(mean_loss_bnd, 4),
            "loss_cls": round(mean_loss_cls, 4),
            # Phân đoạn
            "dice": seg_res["dice"],
            "iou": seg_res["iou"],
            "pixel_accuracy": seg_res["pixel_accuracy"],
            # Phân loại
            "accuracy": cls_res["accuracy"],
            "macro_f1": cls_res["macro_f1"],
            "macro_precision": cls_res["macro_precision"],
            "macro_recall": cls_res["macro_recall"],
            "recall_degree_3": cls_res["recall_degree_3"],
            "per_class": cls_res.get("per_class", {}),
            "confusion_matrix": cls_res.get("confusion_matrix", []),
            # Độ bất định y tế
            "mean_uncertainty": round(float(np.mean(u_all)), 4),
            "uncertainty_correct": round(float(np.mean(u_correct)), 4),
            "uncertainty_incorrect": round(float(np.mean(u_incorrect)), 4),
            "abstention_rate_02": round(abstain_02, 4),
            "abstention_rate_04": round(abstain_04, 4),
            # Điểm lựa chọn mô hình
            "composite_score": composite_score,
        }


# ==============================================================================
# 3. VÒNG LẶP HUẤN LUYỆN (EPOCH LOOPS)
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    seg_loss_fn: nn.Module,
    bce_loss_fn: nn.Module,
    edl_loss_fn: nn.Module,
    epoch: int,
    w_seg: float,
    w_bnd: float,
    w_cls: float,
    device: torch.device,
    use_edl: bool = True,
) -> Dict[str, Any]:
    """Huấn luyện mô hình MT-EANet qua 1 Epoch với Joint Loss duy nhất."""
    model.train()
    tracker = MultiTaskMetricTracker()

    pbar = tqdm(dataloader, desc=f"  Train Ep {epoch:02d}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boundaries = batch["boundary"].to(device)
        labels = batch["severity"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # 1. Tính toán từng thành phần Loss
        loss_seg = seg_loss_fn(outputs["mask_logits"], masks)
        if outputs["boundary_logits"] is not None and w_bnd > 0.0:
            loss_bnd = bce_loss_fn(outputs["boundary_logits"], boundaries)
        else:
            loss_bnd = torch.tensor(0.0, device=device)

        if use_edl:
            loss_cls = edl_loss_fn(outputs["evidence"], labels, epoch=epoch)
        else:
            # Fallback nếu dùng CrossEntropy
            loss_cls = edl_loss_fn(outputs["cls_logits"], labels)

        # 2. Tổng hợp Hàm Mất Mát Đa Nhiệm
        loss_total = (w_seg * loss_seg) + (w_bnd * loss_bnd) + (w_cls * loss_cls)

        # 3. Lan truyền ngược 1 lần duy nhất cập nhật Shared Encoder & các Heads
        loss_total.backward()
        # Cắt ngưỡng gradient tránh xung nhọn từ Digamma EDL + Class Weights làm hỏng Encoder
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Cập nhật số liệu
        losses_dict = {
            "loss_total": loss_total.item(),
            "loss_seg": loss_seg.item(),
            "loss_bnd": loss_bnd.item(),
            "loss_cls": loss_cls.item(),
        }
        tracker.update(outputs, masks, boundaries, labels, losses_dict, use_edl=use_edl)
        pbar.set_postfix({
            "Loss": f"{loss_total.item():.3f}",
            "Seg": f"{loss_seg.item():.3f}",
            "Bnd": f"{loss_bnd.item():.3f}",
            "Cls": f"{loss_cls.item():.3f}",
        })

    return tracker.compute()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    bce_loss_fn: nn.Module,
    edl_loss_fn: nn.Module,
    epoch: int,
    w_seg: float,
    w_bnd: float,
    w_cls: float,
    device: torch.device,
    desc: str = "Val",
    use_edl: bool = True,
) -> Dict[str, Any]:
    """Đánh giá mô hình trên tập Validation hoặc Test."""
    model.eval()
    tracker = MultiTaskMetricTracker()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"  {desc} Ep {epoch:02d}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            boundaries = batch["boundary"].to(device)
            labels = batch["severity"].to(device)

            outputs = model(images)

            loss_seg = seg_loss_fn(outputs["mask_logits"], masks)
            if outputs["boundary_logits"] is not None and w_bnd > 0.0:
                loss_bnd = bce_loss_fn(outputs["boundary_logits"], boundaries)
            else:
                loss_bnd = torch.tensor(0.0, device=device)

            if use_edl:
                loss_cls = edl_loss_fn(outputs["evidence"], labels, epoch=epoch)
            else:
                loss_cls = edl_loss_fn(outputs["cls_logits"], labels)

            loss_total = (w_seg * loss_seg) + (w_bnd * loss_bnd) + (w_cls * loss_cls)

            losses_dict = {
                "loss_total": loss_total.item(),
                "loss_seg": loss_seg.item(),
                "loss_bnd": loss_bnd.item(),
                "loss_cls": loss_cls.item(),
            }
            tracker.update(outputs, masks, boundaries, labels, losses_dict, use_edl=use_edl)
            pbar.set_postfix({"Loss": f"{loss_total.item():.3f}"})

    return tracker.compute()


# ==============================================================================
# 4. HÀM TRỰC QUAN HÓA VÀ XUẤT BIỂU ĐỒ (VISUALIZATION)
# ==============================================================================

def plot_training_curves(history: Dict[str, list], save_path: str, run_name: str):
    """
    Vẽ và lưu hệ thống biểu đồ huấn luyện đa nhiệm 2x2:
    - Ô 1: Tổng Loss và Loss chi tiết từng nhánh (Train vs Val)
    - Ô 2: Chỉ số Phân đoạn (Dice & IoU)
    - Ô 3: Chỉ số Phân loại (Accuracy, Macro F1, Recall Độ 3)
    - Ô 4: Chỉ số Độ bất định (Uncertainty u đúng vs sai, Tỷ lệ cảnh báo)
    """
    epochs = history.get("epoch", list(range(1, len(history["train_loss_total"]) + 1)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle(f"MT-EANet-EDL Multi-Task Training Curves: {run_name}", fontsize=15, fontweight="bold")

    # 1. Total & Task Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss_total"], label="Train Total Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss_total"], label="Val Total Loss", color="#ff7f0e", lw=2, linestyle="--")
    if "val_loss_seg" in history:
        ax1.plot(epochs, history["val_loss_seg"], label="Val Seg Loss", color="#2ca02c", alpha=0.7)
    if "val_loss_cls" in history:
        ax1.plot(epochs, history["val_loss_cls"], label="Val Cls Loss", color="#d62728", alpha=0.7)
    ax1.set_title("Loss Curves (Total & Sub-tasks)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Segmentation Metrics (Dice & IoU)
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["train_dice"], label="Train Dice", color="#1f77b4", lw=2)
    ax2.plot(epochs, history["val_dice"], label="Val Dice", color="#2ca02c", lw=2.5)
    ax2.plot(epochs, history["val_iou"], label="Val IoU", color="#9467bd", lw=2, linestyle="--")
    ax2.set_title("Segmentation Performance (Dice / IoU)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    # 3. Classification Metrics (Accuracy, Macro F1, Recall Deg 3)
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["train_macro_f1"], label="Train Macro F1", color="#1f77b4", lw=1.5, linestyle=":")
    ax3.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#e377c2", lw=2.5)
    ax3.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#ff7f0e", lw=2)
    if "val_recall_degree_3" in history:
        ax3.plot(epochs, history["val_recall_degree_3"], label="Val Recall Độ 3 (Critical)", color="#d62728", lw=2, linestyle="--")
    ax3.set_title("Classification Performance (F1 / Accuracy / Recall Độ 3)", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.set_ylim([0.0, 1.0])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right")

    # 4. Uncertainty & Medical Ethics
    ax4 = axes[1, 1]
    if "val_mean_uncertainty" in history:
        ax4.plot(epochs, history["val_mean_uncertainty"], label="Mean Uncertainty (u)", color="#8c564b", lw=2)
    if "val_uncertainty_correct" in history and "val_uncertainty_incorrect" in history:
        ax4.plot(epochs, history["val_uncertainty_correct"], label="u (Correct Cases)", color="#2ca02c", linestyle="--")
        ax4.plot(epochs, history["val_uncertainty_incorrect"], label="u (Incorrect Cases)", color="#d62728", lw=2)
    if "val_abstention_rate_04" in history:
        ax4.plot(epochs, history["val_abstention_rate_04"], label="Abstention Rate (u >= 0.4)", color="#bcbd22", linestyle=":")
    ax4.set_title("Evidential Epistemic Uncertainty & Triage", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Uncertainty / Rate")
    ax4.set_ylim([0.0, 1.0])
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: str, title: str):
    """Vẽ ma trận nhầm lẫn chuẩn trực quan."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Severity Label",
        xlabel="Predicted Severity Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 5. HÀM CHÍNH ĐIỀU PHỐI HUẤN LUYỆN (MAIN WORKFLOW)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Task MT-EANet-EDL Training Pipeline")

    # Dữ liệu & Đường dẫn
    parser.add_argument("--train_csv", "--train-csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val_csv", "--val-csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--test_csv", "--test-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--output_dir", "--output-dir", type=str, default="outputs/checkpoints/mteanet_multitask")
    parser.add_argument("--run_name", "--run-name", type=str, default="run_mteanet_edl")

    # Mô hình
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--num_classes", "--num-classes", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", "--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument(
        "--model_type", "--model-type",
        type=str,
        default="mteanet_edl",
        choices=["mteanet_edl", "multitask_edl"],
        help="mteanet_edl (M3): Multi-Task + Dual-Branch; multitask_edl (M2): Multi-Task thuần (không Dual-Branch)"
    )
    parser.add_argument("--use_boundary", "--use-boundary", action="store_true", default=True, help="Bật nhánh viền ban đỏ Dual-Branch (MT-EANet)")
    parser.add_argument("--no_boundary", "--no-boundary", action="store_false", dest="use_boundary", help="Tắt nhánh viền ban đỏ (Multi-Task thuần)")

    # Trọng số Joint Loss tối ưu chống Gradient Dominance (cân bằng biên độ Seg vs Weighted Digamma EDL)
    parser.add_argument("--w_seg", "--w-seg", type=float, default=2.0, help="Trọng số hàm loss phân đoạn (mặc định 2.0 bảo vệ Dice)")
    parser.add_argument("--w_bnd", "--w-bnd", type=float, default=0.1, help="Trọng số hàm loss viền ban đỏ (mặc định 0.1 tránh nhiễu)")
    parser.add_argument("--w_cls", "--w-cls", type=float, default=0.15, help="Trọng số hàm loss phân loại tại Encoder (mặc định 0.15 cân bằng Digamma)")

    # Cấu hình hàm mất mát
    parser.add_argument("--seg_loss", "--seg-loss", type=str, default="combo", choices=["focal_tversky", "combo", "dice"])
    parser.add_argument("--cls_loss", "--cls-loss", type=str, default="edl", choices=["edl", "ce"])
    parser.add_argument("--no_edl", "--no-edl", action="store_true", help="Tắt EDL, dùng Cross-Entropy thuần (Softmax) cho nhánh phân loại")
    parser.add_argument("--annealing_epochs", "--annealing-epochs", type=int, default=10, help="Số epoch ủ nhiệt cho KL divergence trong EDL")
    parser.add_argument("--edl_type", "--edl-type", type=str, default="digamma", choices=["digamma", "log", "mse"], help="Dạng Bayes Risk cho EDL (digamma tối ưu nhất)")
    parser.add_argument("--kl_weight", "--kl-weight", type=float, default=0.15, help="Trọng số cực đại phạt KL trong EDL (tránh triệt tiêu evidence)")
    parser.add_argument("--no_class_weights", "--no-class-weights", action="store_true", help="Tắt trọng số đền bù mất cân bằng lớp")
    parser.add_argument("--boundary_kernel", "--boundary-kernel", type=int, default=5, help="Kích thước kernel hình thái học trích xuất viền")

    # Siêu tham số Huấn luyện & Differential Learning Rates
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate gốc cho Shared Encoder")
    parser.add_argument("--dec_lr_mult", "--dec-lr-mult", type=float, default=3.0, help="Hệ số nhân LR cho Decoder phân đoạn")
    parser.add_argument("--cls_lr_mult", "--cls-lr-mult", type=float, default=5.0, help="Hệ số nhân LR cho Head phân loại (bù w_cls nhỏ)")
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img_size", "--img-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12, help="Ngưỡng Early Stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", "--eval-only", action="store_true", help="Chỉ chạy đánh giá trên tập Test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn nạp checkpoint có sẵn")

    args = parser.parse_args()

    # Đồng bộ hóa cấu hình kiến trúc cho Ablation Study (M2 vs M3 vs No-EDL)
    if args.no_edl:
        args.cls_loss = "ce"

    use_edl = (args.cls_loss == "edl")
    loss_tag = "EDL" if use_edl else "CE"

    if args.model_type == "multitask_edl" or not args.use_boundary or args.w_bnd <= 0.0:
        args.use_boundary = False
        args.w_bnd = 0.0
        model_name_display = f"Multi-Task Thuần (Shared + {loss_tag}, Không Dual-Branch)"
    else:
        args.use_boundary = True
        model_name_display = f"MT-EANet-{loss_tag} (Multi-Task + Dual-Branch Viền + {loss_tag})"

    set_seed(args.seed)
    device = torch.device(args.device)

    run_dir = os.path.join(PROJECT_ROOT, args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(" " * 14 + f"HỆ THỐNG HUẤN LUYỆN: {model_name_display}")
    print("=" * 80)
    print(f"[Device]         : {device}")
    print(f"[Architecture]   : {model_name_display}")
    print(f"[Backbone]       : {args.encoder} (Pretrained: {args.pretrained})")
    print(f"[Joint Loss]     : {args.w_seg} * L_seg({args.seg_loss}) + {args.w_bnd} * L_bnd(BCE) + {args.w_cls} * L_cls({args.cls_loss})")
    print(f"[EDL Anneal]     : {args.annealing_epochs} epochs")
    print(f"[Epochs / Batch] : {args.epochs} / {args.batch_size}")
    print(f"[Output Dir]     : {run_dir}")
    print("=" * 80)

    # 1. Nạp Dữ liệu Đa Nhiệm
    print("[1/5] Khởi tạo DataLoaders Đa nhiệm...")
    loaders = get_multitask_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=PROJECT_ROOT,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        boundary_kernel_size=args.boundary_kernel,
    )
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = None

    print(f"  + Train samples: {len(train_loader.dataset)}")
    print(f"  + Val samples:   {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"  + Test samples:  {len(test_loader.dataset)}")

    # 2. Xây dựng Kiến trúc Mô hình
    print(f"[2/5] Xây dựng mô hình {model_name_display}...")
    model = build_multitask_model(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        encoder_name=args.encoder,
        use_boundary=args.use_boundary,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  + Tổng tham số: {total_params / 1e6:.2f}M | Cần huấn luyện: {trainable_params / 1e6:.2f}M")

    # 3. Khởi tạo Hàm Mất Mát & Tối ưu hóa
    print("[3/5] Khởi tạo Hàm mất mát và Trình tối ưu AdamW...")
    if args.seg_loss == "focal_tversky":
        seg_loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    else:
        seg_loss_fn = get_segmentation_loss(args.seg_loss)

    bce_loss_fn = nn.BCEWithLogitsLoss()

    use_edl = (args.cls_loss == "edl")
    # Tính trọng số nghịch đảo tần số lớp từ tập Train thực tế (đền bù ca nặng Độ 3)
    class_weights_tensor = None
    if not args.no_class_weights:
        train_df = train_loader.dataset.df
        class_counts = [int((train_df["severity_degree"] == c).sum()) for c in range(args.num_classes)]
        if all(c > 0 for c in class_counts):
            total_n = sum(class_counts)
            cw_list = [total_n / (args.num_classes * c) for c in class_counts]
            class_weights_tensor = torch.tensor(cw_list, dtype=torch.float32, device=device)
            print(f"  + Class Counts: {class_counts} -> Class Weights: {[round(w, 3) for w in cw_list]}")

    if use_edl:
        edl_loss_fn = EDLLoss(
            num_classes=args.num_classes,
            annealing_epochs=args.annealing_epochs,
            loss_type=args.edl_type,
            kl_weight=args.kl_weight,
            class_weights=class_weights_tensor,
        ).to(device)
        print(f"  + EDL Config: type={args.edl_type}, kl_weight={args.kl_weight}, annealing={args.annealing_epochs}")
    else:
        edl_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)

    # Tách nhóm Learning Rate (Differential Learning Rates):
    # - Shared Encoder: lr gốc (1e-4) để bảo toàn đặc trưng ImageNet & chống giằng co
    # - Seg Decoder: lr * 3.0 để hội tụ nhanh viền tổn thương
    # - Cls Head + Attention: lr * 5.0 để học hết tốc lực dù w_cls nhỏ (0.15)
    enc_params, dec_params, cls_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(param)
        elif name.startswith("cls_head.") or name.startswith("mask_guided_attn."):
            cls_params.append(param)
        else:
            dec_params.append(param)

    param_groups = [
        {"params": enc_params, "lr": args.lr},
        {"params": dec_params, "lr": args.lr * args.dec_lr_mult},
        {"params": cls_params, "lr": args.lr * args.cls_lr_mult},
    ]
    print(
        f"  + Differential LR: Encoder={args.lr:.1e} | "
        f"Decoder={args.lr * args.dec_lr_mult:.1e} | "
        f"ClsHead={args.lr * args.cls_lr_mult:.1e}"
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Nếu chế độ eval_only hoặc nạp checkpoint
    best_checkpoint_path = os.path.join(run_dir, "best_model.pth")
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] Nạp trọng số từ checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif args.eval_only and os.path.exists(best_checkpoint_path):
        print(f"[INFO] Chế độ Eval-Only: Nạp {best_checkpoint_path}")
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    if args.eval_only:
        if test_loader is None:
            raise ValueError("Không tìm thấy test_loader để thực hiện eval_only!")
        print("[EVAL] Đang đánh giá trên tập Test...")
        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            seg_loss_fn=seg_loss_fn,
            bce_loss_fn=bce_loss_fn,
            edl_loss_fn=edl_loss_fn,
            epoch=0,
            w_seg=args.w_seg,
            w_bnd=args.w_bnd,
            w_cls=args.w_cls,
            device=device,
            desc="Test",
            use_edl=use_edl,
        )
        print("\n" + "=" * 60)
        print(f"KẾT QUẢ ĐÁNH GIÁ TẬP TEST:")
        print(f"  - Dice Score:        {test_metrics['dice']:.4f}")
        print(f"  - IoU Score:         {test_metrics['iou']:.4f}")
        print(f"  - Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
        print(f"  - Macro F1:          {test_metrics['macro_f1'] * 100:.2f}%")
        print(f"  - Recall Độ 3:       {test_metrics['recall_degree_3'] * 100:.2f}%")
        print(f"  - Epistemic Unc (u): {test_metrics['mean_uncertainty']:.4f}")
        print("=" * 60)
        return

    # Lưu file config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    # 5. Tiến trình Huấn luyện Chính
    print("[4/5] Bắt đầu Vòng lặp Huấn luyện Đa nhiệm...")
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss_total": [], "val_loss_total": [],
        "val_loss_seg": [], "val_loss_bnd": [], "val_loss_cls": [],
        "train_dice": [], "val_dice": [],
        "val_iou": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "val_accuracy": [], "val_recall_degree_3": [],
        "val_mean_uncertainty": [],
        "val_uncertainty_correct": [],
        "val_uncertainty_incorrect": [],
        "val_abstention_rate_04": [],
        "val_composite_score": [],
    }

    best_score = -1.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train 1 Epoch
        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            seg_loss_fn=seg_loss_fn,
            bce_loss_fn=bce_loss_fn,
            edl_loss_fn=edl_loss_fn,
            epoch=epoch,
            w_seg=args.w_seg,
            w_bnd=args.w_bnd,
            w_cls=args.w_cls,
            device=device,
            use_edl=use_edl,
        )

        # Validate
        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            seg_loss_fn=seg_loss_fn,
            bce_loss_fn=bce_loss_fn,
            edl_loss_fn=edl_loss_fn,
            epoch=epoch,
            w_seg=args.w_seg,
            w_bnd=args.w_bnd,
            w_cls=args.w_cls,
            device=device,
            desc="Val",
            use_edl=use_edl,
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Lưu lịch sử
        history["epoch"].append(epoch)
        history["train_loss_total"].append(train_res["loss_total"])
        history["val_loss_total"].append(val_res["loss_total"])
        history["val_loss_seg"].append(val_res["loss_seg"])
        history["val_loss_bnd"].append(val_res["loss_bnd"])
        history["val_loss_cls"].append(val_res["loss_cls"])
        history["train_dice"].append(train_res["dice"])
        history["val_dice"].append(val_res["dice"])
        history["val_iou"].append(val_res["iou"])
        history["train_macro_f1"].append(train_res["macro_f1"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["val_accuracy"].append(val_res["accuracy"])
        history["val_recall_degree_3"].append(val_res["recall_degree_3"])
        history["val_mean_uncertainty"].append(val_res["mean_uncertainty"])
        history["val_uncertainty_correct"].append(val_res["uncertainty_correct"])
        history["val_uncertainty_incorrect"].append(val_res["uncertainty_incorrect"])
        history["val_abstention_rate_04"].append(val_res["abstention_rate_04"])
        history["val_composite_score"].append(val_res["composite_score"])

        # In tóm tắt Epoch theo phong cách bảng điều khiển y tế
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss: T={train_res['loss_total']:.3f}/V={val_res['loss_total']:.3f} | "
            f"Dice: {val_res['dice']:.4f} (IoU: {val_res['iou']:.4f}) | "
            f"F1: {val_res['macro_f1'] * 100:.1f}% | "
            f"Rec Độ 3: {val_res['recall_degree_3'] * 100:.1f}% | "
            f"u: {val_res['mean_uncertainty']:.3f} | "
            f"Score: {val_res['composite_score']:.4f}"
        )

        # Kiểm tra và Lưu Best Checkpoint dựa trên Composite Score
        curr_score = val_res["composite_score"]
        if curr_score > best_score:
            best_score = curr_score
            patience_counter = 0
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_res,
                "config": vars(args),
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  >>> [*] Best Model Updated: Score={best_score:.4f} (Dice={val_res['dice']:.4f}, F1={val_res['macro_f1']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[EARLY STOPPING] Kích hoạt dừng sớm sau {epoch} epochs không cải thiện.")
                break

    # Lưu checkpoint cuối cùng
    last_checkpoint_path = os.path.join(run_dir, "last_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
    }, last_checkpoint_path)

    # Lưu history ra JSON
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Vẽ biểu đồ tổng kết
    curve_save_path = os.path.join(run_dir, "multitask_training_curves.png")
    plot_training_curves(history, curve_save_path, args.run_name)

    total_training_time = time.time() - start_time
    print(f"\n[INFO] Huấn luyện hoàn tất sau {total_training_time / 60:.1f} phút.")

    # 6. Đánh giá khách quan trên tập Test độc lập (187 ảnh)
    if test_loader is not None and os.path.exists(best_checkpoint_path):
        print("\n[5/5] Đang nạp Best Model và chạy Đánh giá Khách quan trên Tập Test (187 ảnh)...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_results = evaluate(
            model=model,
            dataloader=test_loader,
            seg_loss_fn=seg_loss_fn,
            bce_loss_fn=bce_loss_fn,
            edl_loss_fn=edl_loss_fn,
            epoch=best_ckpt["epoch"],
            w_seg=args.w_seg,
            w_bnd=args.w_bnd,
            w_cls=args.w_cls,
            device=device,
            desc="Test",
            use_edl=use_edl,
        )

        test_json_path = os.path.join(run_dir, "test_results.json")
        with open(test_json_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=4)

        # Xuất biểu đồ ma trận nhầm lẫn của tập Test
        if "confusion_matrix" in test_results and test_results["confusion_matrix"]:
            cm_save_path = os.path.join(run_dir, "confusion_matrix_test.png")
            class_names = ["Độ 1", "Độ 2", "Độ 3"]
            plot_confusion_matrix(
                test_results["confusion_matrix"],
                class_names=class_names,
                save_path=cm_save_path,
                title=f"MT-EANet Test Confusion Matrix ({args.run_name})"
            )

        print("\n" + "=" * 80)
        print(" " * 22 + "BÁO CÁO NGHIỆM THU TẬP TEST ĐỘC LẬP")
        print("=" * 80)
        print(f"  + Phân đoạn Dice Score:          {test_results['dice']:.4f}")
        print(f"  + Phân đoạn IoU Score:           {test_results['iou']:.4f}")
        print(f"  + Phân loại Overall Accuracy:    {test_results['accuracy'] * 100:.2f}%")
        print(f"  + Phân loại Macro F1-Score:      {test_results['macro_f1'] * 100:.2f}%")
        print(f"  + Nhạy ca nguy kịch Độ 3:        {test_results['recall_degree_3'] * 100:.2f}% (Chỉ số đạo đức lâm sàng)")
        print(f"  + Độ bất định trung bình (u):    {test_results['mean_uncertainty']:.4f}")
        print(f"  + Độ bất định ca đúng:           {test_results['uncertainty_correct']:.4f}")
        print(f"  + Độ bất định ca sai:            {test_results['uncertainty_incorrect']:.4f}")
        print(f"  + Tỷ lệ ca kích hoạt hội chẩn:   {test_results['abstention_rate_04'] * 100:.2f}% (u >= 0.4)")
        print("=" * 80)
        print(f"[SUCCESS] Checkpoints và Báo cáo khoa học đã lưu đầy đủ tại: {run_dir}")


if __name__ == "__main__":
    main()
