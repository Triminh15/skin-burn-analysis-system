"""
Module: src/vision/train_cls.py
Mục đích: Script chuẩn mực huấn luyện các mô hình Phân loại Cấp độ bỏng (Classification):
Hỗ trợ:
- Huấn luyện ResNet-34 (Baseline) và ConvNeXt-Tiny (SOTA)
- Tùy chọn hàm mất mát: Weighted Cross-Entropy vs Class-Balanced Focal Loss + Label Smoothing
- Cố định Seed đảm bảo tính tái lập (Reproducibility)
- Thanh tiến trình trực quan (tqdm), theo dõi Accuracy, Macro F1 và Recall ca nặng Độ 3
- Checkpoint Manager lưu best_model.pth dựa trên Macro F1 và Early Stopping chống overfit
- Vẽ và lưu biểu đồ Loss, Accuracy curves (.png)
- Đánh giá khách quan trên tập Test (187 ảnh ROI) và xuất Ma trận nhầm lẫn (Confusion Matrix Heatmap)
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

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.vision.losses import get_classification_loss
from src.vision.metrics import ClassificationMetricTracker
from src.vision.models import build_classification_model, count_parameters


# ==============================================================================
# 1. DATASET & TRANSFORMS CHO PHÂN LOẠI ROI
# ==============================================================================

def get_cls_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_cls_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BurnROIDataset(Dataset):
    """
    Dataset nạp ảnh ROI vết bỏng (224x224) từ các file CSV *_roi.csv.
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[A.Compose] = None
    ):
        self.root_dir = root_dir if root_dir is not None else PROJECT_ROOT
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.root_dir, csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file danh sách ROI: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        roi_rel = row.get("roi_path", row.get("image_path"))
        roi_full = os.path.join(self.root_dir, roi_rel)

        img = cv2.imread(roi_full)
        if img is None:
            raise FileNotFoundError(f"Lỗi khi đọc file ảnh ROI: {roi_full}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        severity = torch.tensor(int(row["severity_degree"]), dtype=torch.long)
        return {
            "image": img_tensor,
            "severity": severity,
            "image_id": str(row["image_id"])
        }


# ==============================================================================
# 2. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ (EPOCH LOOPS)
# ==============================================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    tracker = ClassificationMetricTracker()

    pbar = tqdm(dataloader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device)
        targets = batch["severity"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        tracker.update(logits, targets)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return epoch_loss, metrics["accuracy"], metrics["macro_f1"]


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    running_loss = 0.0
    tracker = ClassificationMetricTracker()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  Eval ", leave=False, dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device)
            targets = batch["severity"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            tracker.update(logits, targets)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    eval_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return eval_loss, metrics


def plot_cls_curves(history: Dict[str, list], save_path: str, run_name: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title(f"Loss Curves - {run_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Accuracy & Macro F1
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title(f"Validation Performance - {run_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], save_path: str, run_name: str):
    """Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix) đẹp mắt phục vụ Báo cáo."""
    cm_array = np.array(cm)
    classes = ["Độ 1 (Epidermal)", "Độ 2 (Dermal)", "Độ 3 (Full)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix - {run_name}",
        ylabel="Nhãn Thực tế (Ground Truth)",
        xlabel="Nhãn AI Dự đoán (Prediction)"
    )

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # In giá trị số lên từng ô
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = cm_array[i, j]
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12, fontweight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Script Huấn luyện Phân loại Cấp độ Bỏng (Classification Training)")
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34", "convnext_tiny"], help="Kiến trúc mô hình")
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "cb_focal"], help="Hàm mất mát")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch tối đa")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Tên phiên chạy")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = f"cls_{args.model}_{args.loss}"

    print("=" * 75)
    print(f"🚀 KHỞI ĐỘNG PHIÊN HUẤN LUYỆN PHÂN LOẠI: {args.run_name.upper()}")
    print("=" * 75)
    print(f"• Thiết bị tính toán  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Kiến trúc mô hình   : {args.model} (3 classes: Độ 1, Độ 2, Độ 3)")
    print(f"• Hàm mất mát         : {args.loss} (Label smoothing: {args.label_smoothing})")
    print(f"• Batch size / Epochs : {args.batch_size} / {args.epochs}")

    checkpoint_dir = "outputs/checkpoints"
    figures_dir = "outputs/figures"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(checkpoint_dir, f"{args.run_name}_best.pth")
    history_csv_path = os.path.join(figures_dir, f"{args.run_name}_history.csv")
    curve_png_path = os.path.join(figures_dir, f"{args.run_name}_curves.png")
    cm_png_path = os.path.join(figures_dir, f"{args.run_name}_confusion_matrix.png")
    test_json_path = os.path.join(figures_dir, f"{args.run_name}_test_results.json")

    # 3. Nạp DataLoaders từ các file ROI CSV
    print("\n📦 Đang nạp tập dữ liệu ROI...")
    train_dataset = BurnROIDataset("data/splits/train_roi.csv", transform=get_cls_train_transforms())
    val_dataset = BurnROIDataset("data/splits/val_roi.csv", transform=get_cls_val_transforms())
    test_dataset = BurnROIDataset("data/splits/test_roi.csv", transform=get_cls_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  -> Train ROI: {len(train_dataset)} ảnh")
    print(f"  -> Val ROI  : {len(val_dataset)} ảnh")
    print(f"  -> Test ROI : {len(test_dataset)} ảnh")

    # 4. Khởi tạo Mô hình và Hàm Loss
    print("\n🧠 Đang khởi tạo mô hình...")
    model = build_classification_model(
        model_name=args.model,
        num_classes=3,
        pretrained=True,
        drop_rate=0.2
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  -> Tổng số tham số: {total_params / 1e6:.2f}M")

    # Phân bố mẫu: Độ 1 (359), Độ 2 (346), Độ 3 (153)
    criterion = get_classification_loss(
        name=args.loss,
        samples_per_class=[359, 346, 153],
        label_smoothing=args.label_smoothing
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Vòng lặp Huấn luyện (Training Loop)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_recall_deg3": []
    }

    best_val_macro_f1 = 0.0
    patience_counter = 0
    start_time = time.time()

    print("\n⚡ BẮT ĐẦU HUẤN LUYỆN:")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_rec3 = val_metrics["recall_degree_3"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["train_macro_f1"].append(round(train_f1, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_recall_deg3"].append(round(val_rec3, 4))

        improved_str = ""
        # Tiêu chí chọn Best Model: Dựa vào Macro F1 (chống mất cân bằng)
        if val_f1 > best_val_macro_f1:
            diff = val_f1 - best_val_macro_f1
            best_val_macro_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            improved_str = f" 🔥 [KỶ LỤC MỚI: Macro F1={val_f1:.4f} (+{diff:.4f}) -> Đã lưu Best Checkpoint]"
        else:
            patience_counter += 1
            improved_str = f" (Patience: {patience_counter}/{args.patience})"

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Acc: {val_acc*100:.1f}% (F1: {val_f1*100:.1f}%, Rec Độ 3: {val_rec3*100:.1f}%) | "
            f"Time: {epoch_time:.1f}s{improved_str}"
        )

        if patience_counter >= args.patience:
            print(f"\n🛑 DỪNG SỚM (EARLY STOPPING): Macro F1 không cải thiện trong {args.patience} epoch liên tiếp.")
            break

    total_training_time = time.time() - start_time
    print("-" * 75)
    print(f"⏱️ Tổng thời gian huấn luyện: {total_training_time / 60:.2f} phút.")

    # 6. Lưu file Lịch sử CSV và Biểu đồ Curves
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    plot_cls_curves(history, curve_png_path, args.run_name)
    print(f"📊 Đã lưu lịch sử huấn luyện: {history_csv_path}")
    print(f"📈 Đã lưu đồ thị Curves: {curve_png_path}")

    # 7. ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (187 ẢNH ĐỘC LẬP)
    print("\n" + "=" * 75)
    print("🏆 ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ROI (187 ẢNH)")
    print("=" * 75)

    if os.path.exists(best_checkpoint_path):
        print(f"• Đang nạp mô hình xuất sắc nhất từ: {best_checkpoint_path}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    plot_confusion_matrix(test_metrics["confusion_matrix"], cm_png_path, args.run_name)

    test_results = {
        "run_name": args.run_name,
        "model": args.model,
        "loss_fn": args.loss,
        "parameters_m": round(total_params / 1e6, 2),
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_recall_degree_3": test_metrics["recall_degree_3"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "per_class_results": test_metrics["per_class"],
        "training_time_minutes": round(total_training_time / 60, 2)
    }

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)

    print("\n📋 KẾT QUẢ TỔNG KẾT ĐỂ ĐIỀN VÀO BÁO CÁO (CLASSIFICATION STUDY):")
    print(f"┌───────────────────────────────┬───────────────────────────────┐")
    print(f"│ Chỉ số Đánh giá               │ Giá trị đạt được              │")
    print(f"├───────────────────────────────┼───────────────────────────────┤")
    print(f"│ 1. Overall Accuracy           │ {test_metrics['accuracy'] * 100:6.2f}%                       │")
    print(f"│ 2. Macro F1-Score             │ {test_metrics['macro_f1'] * 100:6.2f}%                       │")
    print(f"│ 3. RECALL BỎNG ĐỘ 3 (Ca nặng) │ {test_metrics['recall_degree_3'] * 100:6.2f}%                       │")
    print(f"│ 4. Test Loss                  │ {test_loss:6.4f}                         │")
    print(f"│ 5. Best Val Macro F1          │ {best_val_macro_f1 * 100:6.2f}%                       │")
    print(f"│ 6. Model Parameters           │ {total_params / 1e6:6.2f}M                        │")
    print(f"└───────────────────────────────┴───────────────────────────────┘")
    print(f"🖼️ Ma trận nhầm lẫn đã lưu tại: {cm_png_path}")
    print(f"💾 File kết quả JSON đã lưu tại: {test_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
