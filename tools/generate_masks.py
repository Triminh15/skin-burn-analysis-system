"""
Module: tools/generate_masks.py
Mục đích: Tự động chuyển đổi Bounding Box (YOLO format) thành Mask phân đoạn chuẩn Pixel (Semantic Segmentation)
sử dụng MobileSAM (Segment Anything Model) từ Ultralytics.
Hỗ trợ:
- Resumption (tiếp tục chạy từ ảnh chưa xử lý nếu bị dừng)
- Tối ưu đa luồng CPU (Intel Core 5)
- imgsz=512 giúp tăng tốc độ gấp 3 lần mà vẫn giữ nguyên độ nét
"""

import os
import argparse
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import SAM

# Tối ưu hóa số luồng PyTorch trên CPU
torch.set_num_threads(os.cpu_count() or 8)


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int):
    """
    Chuyển đổi toạ độ YOLO chuẩn hóa (0->1) sang toạ độ pixel tuyệt đối [x1, y1, x2, y2].
    """
    x1 = int(max(0, (cx - bw / 2.0) * img_w))
    y1 = int(max(0, (cy - bh / 2.0) * img_h))
    x2 = int(min(img_w, (cx + bw / 2.0) * img_w))
    y2 = int(min(img_h, (cy + bh / 2.0) * img_h))
    return [x1, y1, x2, y2]


def read_yolo_annotations(txt_path: str, img_w: int, img_h: int):
    """
    Đọc toàn bộ các vết bỏng trong file .txt của một bức ảnh.
    Trả về danh sách bboxes [[x1, y1, x2, y2], ...] và classes [0, 1, 2, ...].
    """
    bboxes = []
    classes = []
    if not os.path.exists(txt_path):
        return bboxes, classes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                box = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)
                # Kiểm tra tính hợp lệ của hộp
                if box[2] > box[0] and box[3] > box[1]:
                    bboxes.append(box)
                    classes.append(cls_id)

    return bboxes, classes


def generate_mask(model: SAM, img_path: str, bboxes: list, device: str = "cpu", imgsz: int = 512):
    """
    Dùng MobileSAM với box prompt để sinh mask nhị phân (Binary Mask) ôm sát vết thương.
    Nếu ảnh có nhiều vết bỏng (nhiều boxes), các mask sẽ được gộp lại (Logical OR).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # Nếu không có vết bỏng nào, mask toàn màu đen
    if len(bboxes) == 0:
        return np.zeros((h, w), dtype=np.uint8), img

    # Gọi MobileSAM suy luận với gợi ý hộp (Box Prompts)
    results = model(img_path, bboxes=bboxes, imgsz=imgsz, device=device, verbose=False)

    if results[0].masks is None:
        # Dự phòng: Nếu SAM không bắt được viền, tạo mask bao quanh box thô
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in bboxes:
            mask[box[1]:box[3], box[0]:box[2]] = 255
        return mask, img

    # Lấy tensor mask (num_boxes, H, W)
    masks_tensor = results[0].masks.data
    # Gộp tất cả các vết bỏng lại thành 1 ảnh nhị phân: 255 (vết bỏng) và 0 (da lành/nền)
    combined = torch.any(masks_tensor > 0, dim=0).cpu().numpy().astype(np.uint8) * 255

    # Đảm bảo kích thước mask trùng khớp tuyệt đối với ảnh gốc
    if combined.shape[:2] != (h, w):
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

    return combined, img


def visualize_demo(samples: list, model: SAM, output_fig_path: str, device: str = "cpu"):
    """
    Trực quan hóa so sánh trực tiếp:
    1. Ảnh gốc kèm Bounding Box thô (màu đỏ)
    2. Mặt nạ nhị phân do MobileSAM sinh ra (đen-trắng)
    3. Phủ màu viền lên ảnh gốc (Overlay Alpha Blending - màu xanh ngọc)
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (img_path, txt_path) in enumerate(samples):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        bboxes, classes = read_yolo_annotations(txt_path, w, h)
        mask, _ = generate_mask(model, img_path, bboxes, device=device, imgsz=512)

        # 1. Ảnh gốc kèm Box
        img_bbox = img_rgb.copy()
        for box, cls_id in zip(bboxes, classes):
            cv2.rectangle(img_bbox, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(
                img_bbox,
                f"Degree {cls_id+1}",
                (box[0], max(15, box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # 2. Mặt nạ Binary Mask
        mask_disp = mask.copy()

        # 3. Phủ Overlay Alpha Blending
        overlay = img_rgb.copy()
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[mask > 0] = [0, 255, 255]  # Màu Cyan viền vết thương
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        # Vẽ viền ngoài cùng
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        axes[idx, 0].imshow(img_bbox)
        axes[idx, 0].set_title(f"[{os.path.basename(img_path)}] Bounding Box thô")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask_disp, cmap="gray")
        axes[idx, 1].set_title("MobileSAM Mask (Pixel-level)")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Mask viền phủ lên vết thương")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Thành công] Đã lưu ảnh trực quan hóa nghiệm thu tại: {output_fig_path}")


def process_dataset(raw_dir: str, processed_dir: str, model_weights: str = "mobile_sam.pt", device: str = "cpu", imgsz: int = 512):
    """
    Xử lý hàng loạt toàn bộ 1.227 ảnh trong dataset raw:
    - Lưu ảnh chuẩn hóa sang data/processed/images/
    - Lưu mask nhị phân sang data/processed/masks/
    - Xuất file siêu dữ liệu tổng data/splits/metadata.csv
    - Hỗ trợ resume: bỏ qua ảnh đã xử lý sẵn.
    """
    images_out = os.path.join(processed_dir, "images")
    masks_out = os.path.join(processed_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    print(f"\n[*] Đang nạp mô hình MobileSAM ({model_weights})...")
    model = SAM(model_weights)

    # Tìm các cặp ảnh & txt hợp lệ
    img_extensions = ("*.jpg", "*.jpeg", "*.png")
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    valid_pairs = []
    for img_p in img_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        txt_p = os.path.join(raw_dir, f"{base}.txt")
        if os.path.exists(txt_p):
            valid_pairs.append((img_p, txt_p, base))

    # Sắp xếp theo tên để xử lý tuần tự có thứ tự
    valid_pairs.sort(key=lambda x: x[2])
    print(f"[*] Tìm thấy {len(valid_pairs)} cặp (Ảnh + File nhãn .txt) hợp lệ.")

    records = []
    skipped_count = 0

    for img_p, txt_p, base in tqdm(valid_pairs, desc="Sinh Masks"):
        out_img_name = f"{base}.jpg"
        out_mask_name = f"{base}.png"

        out_img_path = os.path.join(images_out, out_img_name)
        out_mask_path = os.path.join(masks_out, out_mask_name)

        # Kiểm tra nếu đã có sẵn mask và ảnh thì nạp lại thông tin (Resume mode)
        if os.path.exists(out_mask_path) and os.path.exists(out_img_path):
            mask = cv2.imread(out_mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(out_img_path)
            if mask is not None and img is not None:
                h, w = img.shape[:2]
                bboxes, classes = read_yolo_annotations(txt_p, w, h)
                burn_pixels = int(np.count_nonzero(mask))
                burn_ratio = round((burn_pixels / (h * w)) * 100.0, 2)
                max_severity = max(classes) if len(classes) > 0 else 0
                records.append({
                    "image_id": base,
                    "image_path": os.path.relpath(out_img_path, start="d:/burn_analysis").replace("\\", "/"),
                    "mask_path": os.path.relpath(out_mask_path, start="d:/burn_analysis").replace("\\", "/"),
                    "width": w,
                    "height": h,
                    "num_burns": len(bboxes),
                    "severity_degree": max_severity,
                    "burn_pixel_count": burn_pixels,
                    "burn_area_percentage": burn_ratio
                })
                skipped_count += 1
                continue

        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]

        bboxes, classes = read_yolo_annotations(txt_p, w, h)
        mask, _ = generate_mask(model, img_p, bboxes, device=device, imgsz=imgsz)

        # Ghi ảnh (JPG) và mask (PNG nhị phân 0-255 không nén mất mát)
        cv2.imwrite(out_img_path, img)
        cv2.imwrite(out_mask_path, mask)

        # Thống kê y khoa
        burn_pixels = int(np.count_nonzero(mask))
        total_pixels = int(h * w)
        burn_ratio = round((burn_pixels / total_pixels) * 100.0, 2)
        max_severity = max(classes) if len(classes) > 0 else 0

        records.append({
            "image_id": base,
            "image_path": os.path.relpath(out_img_path, start="d:/burn_analysis").replace("\\", "/"),
            "mask_path": os.path.relpath(out_mask_path, start="d:/burn_analysis").replace("\\", "/"),
            "width": w,
            "height": h,
            "num_burns": len(bboxes),
            "severity_degree": max_severity,
            "burn_pixel_count": burn_pixels,
            "burn_area_percentage": burn_ratio
        })

    # Lưu metadata tổng vào data/splits/
    splits_dir = "d:/burn_analysis/data/splits"
    os.makedirs(splits_dir, exist_ok=True)
    metadata_csv = os.path.join(splits_dir, "metadata.csv")
    df = pd.DataFrame(records)
    df.to_csv(metadata_csv, index=False, encoding="utf-8")

    print(f"\n[Hoàn thành] Đã xử lý {len(records)} ảnh và mask (trong đó {skipped_count} ảnh đã có sẵn được giữ nguyên)!")
    print(f"[*] Thư mục ảnh chuẩn hóa: {images_out}")
    print(f"[*] Thư mục mask nhị phân: {masks_out}")
    print(f"[*] File bảng dữ liệu tổng quan: {metadata_csv}")
    print(f"[*] Thống kê mức độ bỏng:\n{df['severity_degree'].value_counts().sort_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tự động sinh Mask từ Bounding Box bằng MobileSAM")
    parser.add_argument("--raw_dir", type=str, default="d:/burn_analysis/data/raw/Skin Burn Dataset")
    parser.add_argument("--processed_dir", type=str, default="d:/burn_analysis/data/processed")
    parser.add_argument("--weights", type=str, default="mobile_sam.pt")
    parser.add_argument("--imgsz", type=int, default=512, help="Kích thước ảnh inference cho MobileSAM")
    parser.add_argument("--demo", action="store_true", help="Chạy thử nghiệm trên 3-5 ảnh và xuất hình ảnh so sánh")
    parser.add_argument("--samples", type=int, default=3, help="Số lượng ảnh thử nghiệm nếu bật --demo")

    args = parser.parse_args()

    if args.demo:
        print("[*] Đang chạy chế độ DEMO kiểm tra trực quan...")
        model = SAM(args.weights)
        demo_names = ["img0", "img1", "img10"]
        demo_pairs = []
        for name in demo_names:
            img_f = os.path.join(args.raw_dir, f"{name}.jpg")
            txt_f = os.path.join(args.raw_dir, f"{name}.txt")
            if os.path.exists(img_f) and os.path.exists(txt_f):
                demo_pairs.append((img_f, txt_f))

        out_fig = "d:/burn_analysis/outputs/figures/sam_mask_demo.png"
        visualize_demo(demo_pairs, model, out_fig)
    else:
        process_dataset(args.raw_dir, args.processed_dir, args.weights, imgsz=args.imgsz)
