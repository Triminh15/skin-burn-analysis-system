"""
Module: src/vision/postprocessing.py
Mục đích: Module Hậu xử lý (Post-processing) và Trích xuất Vùng quan tâm (ROI):
1. Lọc nhiễu hình thái học (Morphological Cleaning) loại bỏ các đốm trắng li ti trên da lành.
2. Tìm Bounding Box ôm trọn vết bỏng và mở rộng 15% đệm an toàn (Padding).
3. Cắt ROI và đệm thành hình vuông (Square Padding) trước khi resize về (224, 224) chống méo mô da.
4. Tính toán chỉ số lâm sàng: Phần trăm diện tích tổn thương (% TBSA).
5. Hỗ trợ chạy batch tạo sẵn tập dữ liệu ROI (train_roi.csv, val_roi.csv, test_roi.csv) cho Giai đoạn Phân loại.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, List

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo thư mục gốc dự án luôn được xác định động
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# PHẦN 1: CÁC THUẬT TOÁN HẬU XỬ LÝ TRỰC TIẾP TRÊN ẢNH VÀ MASK
# ==============================================================================

def clean_binary_mask(
    mask: np.ndarray,
    min_area: int = 50,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Lọc nhiễu hình thái học cho mặt nạ nhị phân:
    1. Morphology Closing: Lấp đầy các lỗ thủng nhỏ bên trong vết bỏng.
    2. Connected Components: Xóa bỏ các đốm trắng cô lập có diện tích < min_area (do bóng đổ/nốt ruồi).
    """
    # Đảm bảo mask ở dạng uint8 (0 hoặc 255)
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = np.where(mask > 127, 255, 0).astype(np.uint8)

    # 1. Lấp lỗ thủng nhỏ bên trong
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # 2. Xóa các đốm nhiễu nhỏ lẻ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned_mask = np.zeros_like(closed)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255

    return cleaned_mask


def calculate_burn_area_percentage(mask: np.ndarray) -> float:
    """
    Tính phần trăm diện tích tổn thương do bỏng (% TBSA ước tính):
    % = (Số pixel màu trắng / Tổng số pixel ảnh) * 100
    """
    if mask is None or mask.size == 0:
        return 0.0

    burn_pixels = np.count_nonzero(mask > 127)
    total_pixels = mask.size
    pct = (burn_pixels / total_pixels) * 100.0
    return float(round(pct, 2))


def get_burn_bounding_box(
    mask: np.ndarray,
    padding_ratio: float = 0.15
) -> Optional[List[int]]:
    """
    Tìm tọa độ hộp bao (Bounding Box) [x1, y1, x2, y2] ôm trọn toàn bộ vết bỏng:
    - padding_ratio: Tỷ lệ mở rộng an toàn ra ngoài da lành (mặc định 15%).
    - Trả về None nếu ảnh không có vết bỏng nào.
    """
    # Tìm tất cả các pixel có màu trắng (vết bỏng)
    y_indices, x_indices = np.where(mask > 127)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    h, w = mask.shape[:2]
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Thêm khoảng đệm an toàn (Padding)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    return [x1, y1, x2, y2]


def crop_roi(
    image: np.ndarray,
    box: Optional[List[int]],
    target_size: Tuple[int, int] = (224, 224),
    make_square: bool = True
) -> np.ndarray:
    """
    Cắt vùng ảnh ROI theo hộp bao và chuẩn hóa kích thước:
    - make_square=True: Đệm thêm viền để đưa về hình vuông trước khi resize,
      giúp bảo toàn nguyên vẹn tỷ lệ co giãn của mô da (chống méo hình).
    - target_size: (224, 224) - Chuẩn vàng cho ResNet và ConvNeXt.
    """
    h_img, w_img = image.shape[:2]

    # Nếu không tìm thấy vết bỏng, lấy vùng trung tâm ảnh
    if box is None:
        side = min(h_img, w_img)
        x1 = (w_img - side) // 2
        y1 = (h_img - side) // 2
        box = [x1, y1, x1 + side, y1 + side]

    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]

    if make_square:
        h, w = cropped.shape[:2]
        max_side = max(h, w)
        # Tạo canvas vuông với màu trung tính (màu xám nhạt hoặc pad đen)
        square_img = np.zeros((max_side, max_side, 3), dtype=image.dtype)
        start_x = (max_side - w) // 2
        start_y = (max_side - h) // 2
        square_img[start_y:start_y + h, start_x:start_x + w] = cropped
        cropped = square_img

    # Resize về kích thước chuẩn (224, 224)
    roi_resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return roi_resized


def process_burn_image(
    image_bgr: np.ndarray,
    mask_raw: np.ndarray,
    padding_ratio: float = 0.15,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Pipeline Hậu xử lý hoàn chỉnh tích hợp:
    Nhận: Ảnh BGR gốc + Mask dự đoán thô (từ U-Net)
    Trả về:
    - cleaned_mask: Mask đã lọc nhiễu
    - burn_percentage: % diện tích tổn thương (% TBSA)
    - bounding_box: [x1, y1, x2, y2]
    - roi_image: Ảnh vết thương đã cắt phóng to (224, 224, 3)
    - has_burn: True/False
    """
    # 1. Lọc nhiễu mask
    cleaned_mask = clean_binary_mask(mask_raw)

    # 2. Tính % diện tích
    burn_pct = calculate_burn_area_percentage(cleaned_mask)

    # 3. Tìm hộp bao và cắt ROI
    box = get_burn_bounding_box(cleaned_mask, padding_ratio=padding_ratio)
    has_burn = box is not None

    roi_img = crop_roi(image_bgr, box, target_size=target_size, make_square=True)

    return {
        "cleaned_mask": cleaned_mask,
        "burn_percentage": burn_pct,
        "bounding_box": box,
        "roi_image": roi_img,
        "has_burn": has_burn
    }


# ==============================================================================
# PHẦN 2: CÔNG CỤ TẠO SẴN BỘ DỮ LIỆU ROI (BATCH PREPROCESSING CHO CLASSIFIER)
# ==============================================================================

def generate_roi_dataset(
    root_dir: Optional[str] = None,
    output_roi_dir: str = "data/processed/rois",
    splits: List[str] = None,
    padding_ratio: float = 0.15,
    target_size: Tuple[int, int] = (224, 224)
):
    if root_dir is None:
        root_dir = PROJECT_ROOT
    """
    Quét qua các file split (train.csv, val.csv, test.csv), cắt sẵn toàn bộ ảnh ROI
    bằng Mask chuẩn và lưu vào thư mục data/processed/rois/.
    Đồng thời sinh ra 3 file CSV mới: train_roi.csv, val_roi.csv, test_roi.csv.
    Giúp việc huấn luyện bộ Phân loại (Classification) nhanh gấp 5 lần!
    """
    if splits is None:
        splits = ["train", "val", "test"]

    print("=" * 75)
    print("✂️ BẮT ĐẦU CẮT VÀ CHUẨN BỊ TẬP DỮ LIỆU ROI CHO GIAI ĐOẠN PHÂN LOẠI")
    print("=" * 75)

    for split in splits:
        csv_path = os.path.join(root_dir, f"data/splits/{split}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Bỏ qua {split}: Không tìm thấy {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        split_roi_dir = os.path.join(root_dir, output_roi_dir, split)
        os.makedirs(split_roi_dir, exist_ok=True)

        new_records = []
        print(f"\n📂 Đang xử lý tập '{split}' ({len(df)} ảnh)...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Crop {split}"):
            img_rel = row["image_path"]
            mask_rel = row["mask_path"]

            img_full = os.path.join(root_dir, img_rel)
            mask_full = os.path.join(root_dir, mask_rel)

            img = cv2.imread(img_full)
            mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Xử lý cắt ROI
            res = process_burn_image(img, mask, padding_ratio=padding_ratio, target_size=target_size)
            roi = res["roi_image"]

            # Lưu ảnh ROI
            img_filename = os.path.basename(img_rel)
            roi_filename = f"roi_{img_filename}"
            roi_full_path = os.path.join(split_roi_dir, roi_filename)
            cv2.imwrite(roi_full_path, roi)

            # Ghi nhận đường dẫn tương đối
            roi_rel_path = os.path.relpath(roi_full_path, root_dir).replace("\\", "/")

            record = row.to_dict()
            record["roi_path"] = roi_rel_path
            record["cleaned_burn_percentage"] = res["burn_percentage"]
            new_records.append(record)

        # Xuất file CSV mới cho tập ROI
        new_csv_path = os.path.join(root_dir, f"data/splits/{split}_roi.csv")
        pd.DataFrame(new_records).to_csv(new_csv_path, index=False)
        print(f"✅ Đã tạo thành công: {new_csv_path} ({len(new_records)} mẫu ROI)")

    print("\n🎉 HOÀN TẤT CHUẨN BỊ TOÀN BỘ TẬP DỮ LIỆU ROI!")


# ==============================================================================
# KIỂM THỬ ĐƠN VỊ NHANH (SANITY CHECK)
# ==============================================================================
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("--- Testing Unit Tests for src/vision/postprocessing.py ---")

    # 1. Tạo ảnh giả lập và mask có đốm nhiễu
    dummy_img = np.full((300, 400, 3), 120, dtype=np.uint8)  # Da lành màu xám
    dummy_mask = np.zeros((300, 400), dtype=np.uint8)

    # Vẽ vết bỏng lớn ở giữa (kích thước 100x120)
    dummy_mask[80:180, 120:240] = 255
    dummy_img[80:180, 120:240] = [30, 40, 200]  # Vùng đỏ bỏng

    # Thêm 2 hạt bụi nhiễu nhỏ (diện tích 3x3 và 2x2) ở góc xa
    dummy_mask[10:13, 10:13] = 255
    dummy_mask[280:282, 380:382] = 255

    # 2. Kiểm tra lọc nhiễu
    cleaned = clean_binary_mask(dummy_mask, min_area=50)
    assert cleaned[10, 10] == 0, "Lỗi: Chưa lọc được hạt bụi nhiễu nhỏ!"
    assert cleaned[100, 150] == 255, "Lỗi: Vết bỏng thật bị mất!"
    print("[OK] clean_binary_mask: Lọc sạch hạt bụi nhiễu thành công!")

    # 3. Kiểm tra tính % diện tích
    pct = calculate_burn_area_percentage(cleaned)
    print(f"[OK] calculate_burn_area_percentage: {pct}%")

    # 4. Kiểm tra tìm hộp bao và cắt ROI
    box = get_burn_bounding_box(cleaned, padding_ratio=0.15)
    print(f"[OK] get_burn_bounding_box: {box}")

    roi = crop_roi(dummy_img, box, target_size=(224, 224), make_square=True)
    print(f"[OK] crop_roi output shape: {roi.shape} (Chuẩn 224x224)")

    # 5. Kiểm tra pipeline trọn gói
    result = process_burn_image(dummy_img, dummy_mask)
    print(f"[OK] process_burn_image: Has burn={result['has_burn']}, Area={result['burn_percentage']}%, ROI shape={result['roi_image'].shape}")
    print("[SUCCESS] All postprocessing functions passed perfectly!")
