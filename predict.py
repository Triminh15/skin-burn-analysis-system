"""
Script: predict.py
Mục đích: Pipeline Chẩn đoán Toàn diện End-to-End cho Hệ thống Phân tích Vết bỏng:
1. Nhận ảnh thô bất kỳ từ bên ngoài (chụp từ điện thoại hoặc máy ảnh).
2. Tiền xử lý chuẩn hóa kích thước và dải điểm ảnh.
3. Giai đoạn 1 (U-Net): Khoanh vùng tự động vết bỏng và tính % diện tích tổn thương (% TBSA).
4. Hậu xử lý & Cắt ROI: Lọc nhiễu hình thái học, nới 15% đệm an toàn và đệm khung vuông 224x224.
5. Giai đoạn 2 (ResNet-34): Phân loại chính xác cấp độ bỏng lâm sàng (Độ 1, Độ 2, Độ 3) kèm xác suất.
6. Trực quan hóa kết quả tổng hợp ra 1 bức ảnh hoàn chỉnh lưu trên đĩa.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple, Optional

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.vision.models import build_segmentation_model, build_classification_model
from src.vision.postprocessing import clean_binary_mask, calculate_burn_area_percentage, get_burn_bounding_box, crop_roi
from src.vision.transforms import IMAGENET_MEAN, IMAGENET_STD


CLASS_NAMES = ["Bỏng Độ 1 (Erythema)", "Bỏng Độ 2 (Blisters)", "Bỏng Độ 3 (Full Thickness)"]
CLASS_COLORS = ["#22c55e", "#eab308", "#ef4444"] # Xanh lá - Vàng - Đỏ


def load_pipeline_models(
    seg_checkpoint: str,
    cls_checkpoint: str,
    device: str = "cpu"
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Nạp 2 mô hình vô địch tốt nhất của Giai đoạn 1 và Giai đoạn 2.
    """
    print("🧠 Đang nạp mô hình Giai đoạn 1: U-Net (ResNet-34)...")
    seg_model = build_segmentation_model(arch="unet", backbone="resnet34", classes=1, pretrained=False)
    if os.path.exists(seg_checkpoint):
        checkpoint = torch.load(seg_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        seg_model.load_state_dict(state_dict)
        print(f"  -> Đã nạp thành công Checkpoint Phân đoạn: {seg_checkpoint}")
    else:
        print(f"  ⚠️ Cảnh báo: Không tìm thấy checkpoint phân đoạn tại: {seg_checkpoint}")

    seg_model.to(device)
    seg_model.eval()

    print("🧠 Đang nạp mô hình Giai đoạn 2: Classifier (ResNet-34)...")
    cls_model = build_classification_model(model_name="resnet34", num_classes=3, pretrained=False)
    if os.path.exists(cls_checkpoint):
        checkpoint = torch.load(cls_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cls_model.load_state_dict(state_dict)
        print(f"  -> Đã nạp thành công Checkpoint Phân loại: {cls_checkpoint}")
    else:
        print(f"  ⚠️ Cảnh báo: Không tìm thấy checkpoint phân loại tại: {cls_checkpoint}")

    cls_model.to(device)
    cls_model.eval()

    return seg_model, cls_model


def predict_pipeline(
    image_path: str,
    seg_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    device: str = "cpu",
    output_path: str = "outputs/figures/prediction_result.png"
) -> Dict[str, Any]:
    """
    Quy trình xử lý hoàn chỉnh từ ảnh thô đến chẩn đoán cuối cùng.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {image_path}")

    # 1. ĐỌC ẢNH GỐC
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_rgb.shape[:2]

    # 2. TIỀN XỬ LÝ CHO GIAI ĐOẠN 1 (U-NET: 256x256)
    seg_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    seg_input = seg_transforms(image=orig_rgb)["image"].unsqueeze(0).to(device)

    # 3. GIAI ĐOẠN 1: SUY LUẬN PHÂN ĐOẠN (U-NET)
    print("\n🔍 [1/3] Đang thực thi phân đoạn tổn thương (U-Net)...")
    with torch.no_grad():
        seg_logits = seg_model(seg_input)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

    # Resize mask xác suất về kích thước ảnh gốc
    mask_prob_resized = cv2.resize(seg_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    raw_binary_mask = (mask_prob_resized > 0.5).astype(np.uint8) * 255

    # 4. HẬU XỬ LÝ HÌNH THÁI HỌC & TÍNH TOÁN % TBSA
    print("✂️ [2/3] Đang lọc nhiễu, tính % TBSA và cắt vùng quan tâm (ROI)...")
    cleaned_mask = clean_binary_mask(raw_binary_mask, min_area=50)
    burn_percentage = calculate_burn_area_percentage(cleaned_mask)
    bbox = get_burn_bounding_box(cleaned_mask, padding_ratio=0.15)

    # Cắt ROI hình vuông 224x224
    roi_bgr = crop_roi(orig_bgr, bbox, target_size=(224, 224), make_square=True)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # 5. TIỀN XỬ LÝ & SUY LUẬN GIAI ĐOẠN 2 (RESNET-34: 224x224)
    print("🏷️ [3/3] Đang phân tích cấp độ tổn thương (ResNet-34)...")
    cls_transforms = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    cls_input = cls_transforms(image=roi_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits = cls_model(cls_input)
        cls_probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()

    pred_class_idx = int(np.argmax(cls_probs))
    pred_class_name = CLASS_NAMES[pred_class_idx]
    pred_confidence = float(cls_probs[pred_class_idx]) * 100.0

    # 6. TRỰC QUAN HÓA KẾT QUẢ TỔNG HỢP (4 Ô LIÊN HOÀN)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)

    # Ô 1: Ảnh gốc
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"1. Ảnh gốc ban đầu\n({orig_w}x{orig_h} px)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Ô 2: Mặt nạ phân đoạn phủ lên vết thương
    overlay = orig_rgb.copy()
    colored_mask = np.zeros_like(orig_rgb)
    colored_mask[cleaned_mask > 0] = [0, 255, 255] # Màu Cyan
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2) # Viền xanh lá
    if bbox is not None:
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) # Hộp bao ROI màu đỏ

    axes[1].imshow(overlay)
    axes[1].set_title(f"2. Khoanh vùng U-Net\nDiện tích: {burn_percentage}% khung hình", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Ô 3: Vùng quan tâm (ROI 224x224)
    axes[2].imshow(roi_rgb)
    axes[2].set_title(f"3. Vùng ROI đệm vuông\n(224x224 px)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Ô 4: Biểu đồ phân bố xác suất cấp độ
    y_pos = np.arange(len(CLASS_NAMES))
    axes[3].barh(y_pos, cls_probs * 100.0, color=CLASS_COLORS, edgecolor="black", height=0.55)
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels(["Độ 1", "Độ 2", "Độ 3"], fontsize=11, fontweight="bold")
    axes[3].set_xlim(0, 100)
    axes[3].set_xlabel("Độ tin cậy (%)", fontsize=10, fontweight="bold")
    axes[3].set_title(f"4. Kết luận: {pred_class_name.split()[0]} {pred_class_name.split()[1]}\n(Độ tin cậy: {pred_confidence:.1f}%)",
                      fontsize=11, fontweight="bold", color=CLASS_COLORS[pred_class_idx])

    for i, v in enumerate(cls_probs * 100.0):
        axes[3].text(v + 2, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.suptitle("HỆ THỐNG PHÂN TÍCH VẾT BỎNG ĐA PHƯƠNG THÁI (SKIN BURN ANALYSIS PIPELINE)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 65)
    print("📋 KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG TỔNG HỢP:")
    print("=" * 65)
    print(f"• Diện tích tổn thương (% TBSA) : {burn_percentage}%")
    print(f"• Cấp độ bỏng dự đoán          : {pred_class_name}")
    print(f"• Độ tin cậy (Confidence)      : {pred_confidence:.2f}%")
    print(f"  - Xác suất Bỏng Độ 1         : {cls_probs[0]*100:.2f}%")
    print(f"  - Xác suất Bỏng Độ 2         : {cls_probs[1]*100:.2f}%")
    print(f"  - Xác suất Bỏng Độ 3 (Nặng)  : {cls_probs[2]*100:.2f}%")
    print(f"🖼️ Đã xuất ảnh chẩn đoán trực quan tại: {output_path}")
    print("=" * 65)

    return {
        "burn_percentage": burn_percentage,
        "predicted_class": pred_class_name,
        "confidence": pred_confidence,
        "probabilities": cls_probs,
        "output_figure": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="Chẩn đoán ảnh vết bỏng bằng chuỗi Pipeline 2 Giai đoạn")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh thô cần chẩn đoán")
    parser.add_argument("--seg_checkpoint", type=str, default="outputs/checkpoints/unet_resnet34_focal_tversky_best.pth",
                        help="Đường dẫn file trọng số tốt nhất của U-Net")
    parser.add_argument("--cls_checkpoint", type=str, default="outputs/checkpoints/cls_resnet34_weighted_ce_best.pth",
                        help="Đường dẫn file trọng số tốt nhất của Classifier")
    parser.add_argument("--output", type=str, default="outputs/figures/prediction_result.png",
                        help="Đường dẫn lưu ảnh kết quả trực quan hóa")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán (cuda hoặc cpu)")

    args = parser.parse_args()

    seg_model, cls_model = load_pipeline_models(
        seg_checkpoint=args.seg_checkpoint,
        cls_checkpoint=args.cls_checkpoint,
        device=args.device
    )

    predict_pipeline(
        image_path=args.image,
        seg_model=seg_model,
        cls_model=cls_model,
        device=args.device,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
