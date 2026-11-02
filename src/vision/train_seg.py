"""
Module: src/vision/train_seg.py
Mục đích: Script chuẩn mực huấn luyện các mô hình Phân đoạn vết bỏng (Segmentation):
Hỗ trợ:
- Chạy 4 thực nghiệm linh hoạt qua tham số dòng lệnh (--arch, --backbone, --loss, --attention)
- Cố định Seed đảm bảo tính tái lập (Reproducibility)
- Thanh tiến trình trực quan (tqdm), theo dõi Train/Val Loss, Dice, IoU
- Cơ chế Checkpoint Manager lưu best_model.pth và Early Stopping chống overfit
- Tự động vẽ và lưu biểu đồ Loss & Dice curves (.png)
- Tự động chạy đánh giá khách quan trên tập Test (187 ảnh) và xuất bảng kết quả tổng kết
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Tuple

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.dataset import SkinBurnDataset
from src.vision.transforms import get_train_transforms, get_val_transforms
from src.vision.losses import get_segmentation_loss
from src.vision.metrics import SegmentationMetricTracker
from src.vision.models import build_segmentation_model, count_parameters


def set_seed(seed: int = 42):
    """Cố định seed ngẫu nhiên đảm bảo kết quả khoa học tái lập 100%."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Huấn luyện mô hình qua 1 Epoch."""
    model.train()
    running_loss = 0.0
    tracker = SegmentationMetricTracker()

    pbar = tqdm(dataloader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        tracker.update(logits, masks)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return epoch_loss, metrics["dice"], metrics["iou"]


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Đánh giá mô hình trên tập Validation hoặc Test."""
    model.eval()
    running_loss = 0.0
    tracker = SegmentationMetricTracker()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  Eval ", leave=False, dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item() * images.size(0)
            tracker.update(logits, masks)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    eval_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return eval_loss, metrics["dice"], metrics["iou"], metrics["pixel_accuracy"]


def plot_training_curves(history: Dict[str, list], save_path: str, run_name: str):
    """Vẽ và lưu biểu đồ Loss curves và Dice/IoU curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị 1: Loss Curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title(f"Loss Curves - {run_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Đồ thị 2: Dice & IoU Score
    axes[1].plot(epochs, history["val_dice"], label="Val Dice (F1)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_iou"], label="Val IoU (Jaccard)", color="#9467bd", linewidth=2, linestyle="--")
    axes[1].set_title(f"Validation Metrics - {run_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Script Huấn luyện Phân đoạn Vết bỏng (Segmentation Training)")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "unetplusplus"], help="Kiến trúc mô hình")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone encoder (resnet34, tu-convnext_tiny)")
    parser.add_argument("--attention", type=str, default="none", choices=["none", "scse"], help="Cơ chế Attention Gate")
    parser.add_argument("--loss", type=str, default="combo", choices=["combo", "focal_tversky"], help="Hàm mất mát")
    parser.add_argument("--epochs", type=int, default=35, help="Số epoch tối đa")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate khởi tạo")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay chống overfit")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--img-size", type=int, default=256, help="Kích thước ảnh vuông")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Tên phiên chạy (tự sinh nếu để trống)")
    args = parser.parse_args()

    # 1. Thiết lập Seed và Device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đặt tên phiên chạy thông minh nếu không truyền
    if args.run_name is None:
        att_str = f"_{args.attention}" if args.attention != "none" else ""
        bb_clean = args.backbone.replace("tu-", "").replace("-", "_")
        args.run_name = f"{args.arch}_{bb_clean}{att_str}_{args.loss}"

    print("=" * 75)
    print(f"🚀 KHỞI ĐỘNG PHIÊN HUẤN LUYỆN: {args.run_name.upper()}")
    print("=" * 75)
    print(f"• Thiết bị tính toán  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Kiến trúc mô hình   : {args.arch} (Backbone: {args.backbone}, Attention: {args.attention})")
    print(f"• Hàm mất mát         : {args.loss}")
    print(f"• Batch size / Epochs : {args.batch_size} / {args.epochs} (Patience: {args.patience})")
    print(f"• Learning rate       : {args.lr} (AdamW + CosineAnnealing)")

    # 2. Chuẩn bị Thư mục lưu kết quả
    checkpoint_dir = "outputs/checkpoints"
    figures_dir = "outputs/figures"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(checkpoint_dir, f"{args.run_name}_best.pth")
    history_csv_path = os.path.join(figures_dir, f"{args.run_name}_history.csv")
    curve_png_path = os.path.join(figures_dir, f"{args.run_name}_curves.png")
    test_json_path = os.path.join(figures_dir, f"{args.run_name}_test_results.json")

    # 3. Nạp DataLoaders
    print("\n📦 Đang nạp tập dữ liệu...")
    train_dataset = SkinBurnDataset(
        csv_path="data/splits/train.csv",
        transform=get_train_transforms(img_size=args.img_size)
    )
    val_dataset = SkinBurnDataset(
        csv_path="data/splits/val.csv",
        transform=get_val_transforms(img_size=args.img_size)
    )
    test_dataset = SkinBurnDataset(
        csv_path="data/splits/test.csv",
        transform=get_val_transforms(img_size=args.img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 là an toàn tuyệt đối trên Windows
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"  -> Train: {len(train_dataset)} ảnh ({len(train_loader)} batches)")
    print(f"  -> Val  : {len(val_dataset)} ảnh ({len(val_loader)} batches)")
    print(f"  -> Test : {len(test_dataset)} ảnh ({len(test_loader)} batches)")

    # 4. Xây dựng Mô hình, Loss, Optimizer & Scheduler
    print("\n🧠 Đang khởi tạo mô hình...")
    model = build_segmentation_model(
        arch=args.arch,
        backbone=args.backbone,
        attention=args.attention,
        in_channels=3,
        classes=1,
        pretrained=True
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  -> Tổng số tham số: {total_params / 1e6:.2f}M ({trainable_params / 1e6:.2f}M cần huấn luyện)")

    criterion = get_segmentation_loss(name=args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Vòng lặp Huấn luyện (Training Loop)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_pixel_acc": [],
        "lr": []
    }

    best_val_dice = 0.0
    patience_counter = 0
    start_time = time.time()

    print("\n⚡ BẮT ĐẦU HUẤN LUYỆN:")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Huấn luyện 1 epoch
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Đánh giá trên tập Validation
        val_loss, val_dice, val_iou, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Cập nhật scheduler
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Lưu lịch sử
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_dice"].append(round(train_dice, 4))
        history["train_iou"].append(round(train_iou, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_dice"].append(round(val_dice, 4))
        history["val_iou"].append(round(val_iou, 4))
        history["val_pixel_acc"].append(round(val_acc, 4))
        history["lr"].append(current_lr)

        # Kiểm tra kỷ lục và lưu Checkpoint
        improved_str = ""
        if val_dice > best_val_dice:
            diff = val_dice - best_val_dice
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            improved_str = f" 🔥 [KỶ LỤC MỚI: {val_dice:.4f} (+{diff:.4f}) -> Đã lưu Best Checkpoint]"
        else:
            patience_counter += 1
            improved_str = f" (Patience: {patience_counter}/{args.patience})"

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Dice: {val_dice:.4f}, IoU: {val_iou:.4f}) | "
            f"Time: {epoch_time:.1f}s{improved_str}"
        )

        # Kiểm tra Early Stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 DỪNG SỚM (EARLY STOPPING): Val Dice không cải thiện trong {args.patience} epoch liên tiếp.")
            break

    total_training_time = time.time() - start_time
    print("-" * 75)
    print(f"⏱️ Tổng thời gian huấn luyện: {total_training_time / 60:.2f} phút.")

    # 6. Lưu file Lịch sử CSV và Biểu đồ Curves
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    plot_training_curves(history, curve_png_path, args.run_name)
    print(f"📊 Đã lưu lịch sử huấn luyện: {history_csv_path}")
    print(f"📈 Đã lưu đồ thị Loss/Dice curves: {curve_png_path}")

    # 7. ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (UNSEEN DATA)
    print("\n" + "=" * 75)
    print("🏆 ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST (187 ẢNH ĐỘC LẬP)")
    print("=" * 75)

    if os.path.exists(best_checkpoint_path):
        print(f"• Đang nạp mô hình xuất sắc nhất từ: {best_checkpoint_path}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    else:
        print("• Cảnh báo: Không tìm thấy checkpoint tốt nhất, dùng trọng số epoch cuối.")

    test_loss, test_dice, test_iou, test_acc = validate(
        model, test_loader, criterion, device
    )

    test_results = {
        "run_name": args.run_name,
        "arch": args.arch,
        "backbone": args.backbone,
        "attention": args.attention,
        "loss_fn": args.loss,
        "parameters_m": round(total_params / 1e6, 2),
        "best_val_dice": round(best_val_dice, 4),
        "test_loss": round(test_loss, 4),
        "test_dice": round(test_dice, 4),
        "test_iou": round(test_iou, 4),
        "test_pixel_accuracy": round(test_acc, 4),
        "training_time_minutes": round(total_training_time / 60, 2),
    }

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)

    print("\n📋 KẾT QUẢ TỔNG KẾT ĐỂ ĐIỀN VÀO BÁO CÁO (ABLATION STUDY):")
    print(f"┌───────────────────────────────┬───────────────────────────────┐")
    print(f"│ Chỉ số Đánh giá               │ Giá trị đạt được              │")
    print(f"├───────────────────────────────┼───────────────────────────────┤")
    print(f"│ 1. Test Dice Score (F1)       │ {test_dice * 100:6.2f}%                       │")
    print(f"│ 2. Test IoU (Jaccard Index)   │ {test_iou * 100:6.2f}%                       │")
    print(f"│ 3. Test Pixel Accuracy        │ {test_acc * 100:6.2f}%                       │")
    print(f"│ 4. Test Loss                  │ {test_loss:6.4f}                         │")
    print(f"│ 5. Best Val Dice              │ {best_val_dice * 100:6.2f}%                       │")
    print(f"│ 6. Model Parameters           │ {total_params / 1e6:6.2f}M                        │")
    print(f"└───────────────────────────────┴───────────────────────────────┘")
    print(f"💾 File kết quả JSON đã lưu tại: {test_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
