"""
Module: src/vision/anatomical_tbsa.py
Mục đích: Khắc phục "Lỗ hổng Y khoa" khi quy đổi diện tích ảnh 2D sang % TBSA cho Công thức Parkland.

Giải quyết triệt để 3 sai số cốt lõi khi đánh giá bỏng qua ảnh chụp lâm sàng:
1. Sai số Phông nền (Background Contamination):
   - Tỷ lệ `% pixel vết bỏng / tổng pixel bức ảnh` bị nhiễu nặng bởi nền ngoại cảnh (ga giường, quần áo, tường).
   - Giải pháp: Phân đoạn tự động Vùng da bộ phận chụp được (Multi-space YCrCb + HSV + Peri-lesional Adaptive
     Chrominance + Morphological Silhouette Filling), từ đó tính chính xác:
       Tỷ lệ tổn thương giải phẫu = Diện tích Vết bỏng / Diện tích Vùng da bộ phận chụp được.
2. Sai số Quy mô Giải phẫu (Anatomical Scale - Wallace Rule of Nines, Palmar 1%, Lund-Browder):
   - Chuẩn hóa theo phân vùng giải phẫu chi tiết: Lòng bàn tay / Bàn tay (1%), Cẳng tay (4%), Cánh tay (9%),
     Đầu-Mặt-Cổ (9% người lớn / hiệu chỉnh Lund-Browder cho trẻ em < 15 tuổi), Thân trước (18%), Thân sau (18%),
     Đùi (9%), Cẳng chân (7%), Bàn chân (2%), Toàn chân (18%).
3. Sai số Tiêu cự / Góc chụp cận cảnh (Camera Framing / Field-of-View Coverage):
   - Cho phép hiệu chỉnh tỷ lệ phần bộ phận nằm trong khung hình (`region_coverage` từ 10% cận cảnh sát vết thương
     đến 100% toàn bộ bộ phận) kèm cảnh báo tự động khi phát hiện ảnh chụp dí sát (Macro/Close-up).
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np


# Danh mục Phân vùng Giải phẫu Chuẩn Y khoa (Wallace Rule of Nines + Sub-regions + Palmar 1%)
ANATOMICAL_REGIONS: Dict[str, Dict[str, Any]] = {
    # Chi trên (Upper Extremity - Tổng 9% mỗi bên)
    "palm_hand": {
        "name": "Bàn tay / Lòng bàn tay (Quy tắc 1%)",
        "short_name": "Bàn tay",
        "adult_tbsa": 1.0,
        "group": "upper_limb",
    },
    "forearm": {
        "name": "Cẳng tay (1 bên)",
        "short_name": "Cẳng tay",
        "adult_tbsa": 4.0,
        "group": "upper_limb",
    },
    "upper_arm": {
        "name": "Cánh tay trên (1 bên)",
        "short_name": "Cánh tay trên",
        "adult_tbsa": 4.5,
        "group": "upper_limb",
    },
    "left_arm": {
        "name": "Toàn bộ cánh tay trái",
        "short_name": "Tay trái",
        "adult_tbsa": 9.0,
        "group": "upper_limb",
    },
    "right_arm": {
        "name": "Toàn bộ cánh tay phải",
        "short_name": "Tay phải",
        "adult_tbsa": 9.0,
        "group": "upper_limb",
    },
    # Đầu - Mặt - Cổ (Head & Neck - Tổng 9% người lớn, thay đổi theo tuổi ở trẻ em)
    "face_only": {
        "name": "Vùng Mặt",
        "short_name": "Mặt",
        "adult_tbsa": 4.5,
        "group": "head_neck",
    },
    "head_neck": {
        "name": "Đầu - Mặt - Cổ",
        "short_name": "Đầu - Mặt - Cổ",
        "adult_tbsa": 9.0,
        "group": "head_neck",
    },
    # Thân mình (Trunk - Trước 18%, Sau 18%)
    "chest_only": {
        "name": "Nửa thân trước - Ngực",
        "short_name": "Ngực trước",
        "adult_tbsa": 9.0,
        "group": "trunk",
    },
    "abdomen_only": {
        "name": "Nửa thân trước - Bụng",
        "short_name": "Bụng trước",
        "adult_tbsa": 9.0,
        "group": "trunk",
    },
    "chest_abdomen": {
        "name": "Toàn bộ Ngực & Bụng - Thân trước",
        "short_name": "Thân trước",
        "adult_tbsa": 18.0,
        "group": "trunk",
    },
    "back_only": {
        "name": "Vùng Lưng (không gồm Mông - Lund-Browder)",
        "short_name": "Vùng Lưng",
        "adult_tbsa": 13.0,
        "group": "trunk",
    },
    "upper_back": {
        "name": "Nửa lưng trên",
        "short_name": "Nửa lưng trên",
        "adult_tbsa": 9.0,
        "group": "trunk",
    },
    "back_buttocks": {
        "name": "Toàn bộ Lưng & Thân sau",
        "short_name": "Lưng / Thân sau",
        "adult_tbsa": 18.0,
        "group": "trunk",
    },
    # Chi dưới (Lower Extremity - Tổng 18% mỗi bên ở người lớn)
    "thigh": {
        "name": "Đùi (1 bên)",
        "short_name": "Đùi",
        "adult_tbsa": 9.0,
        "group": "lower_limb",
    },
    "lower_leg": {
        "name": "Cẳng chân (1 bên)",
        "short_name": "Cẳng chân",
        "adult_tbsa": 7.0,
        "group": "lower_limb",
    },
    "foot": {
        "name": "Bàn chân (1 bên)",
        "short_name": "Bàn chân",
        "adult_tbsa": 2.0,
        "group": "lower_limb",
    },
    "left_leg": {
        "name": "Toàn bộ chân trái",
        "short_name": "Chân trái",
        "adult_tbsa": 18.0,
        "group": "lower_limb",
    },
    "right_leg": {
        "name": "Toàn bộ chân phải",
        "short_name": "Chân phải",
        "adult_tbsa": 18.0,
        "group": "lower_limb",
    },
    # Tầng sinh môn (1%)
    "perineum": {
        "name": "Bộ phận sinh dục / Tầng sinh môn",
        "short_name": "Tầng sinh môn",
        "adult_tbsa": 1.0,
        "group": "perineum",
    },
    "ruler_auto": {
        "name": "Đo trực tiếp bằng thước chuẩn (Không giới hạn vùng)",
        "short_name": "Đo thước trực tiếp",
        "adult_tbsa": 100.0,
        "group": "ruler_auto",
    },
    "unknown": {
        "name": "Vùng cơ thể chưa xác định",
        "short_name": "Chưa xác định",
        "adult_tbsa": 9.0,
        "group": "unknown",
    },
}

# Tương thích ngược với hằng số RULE_OF_NINES cũ
RULE_OF_NINES: Dict[str, float] = {
    key: float(info["adult_tbsa"]) for key, info in ANATOMICAL_REGIONS.items()
}


def get_age_adjusted_region_weight(
    body_part: str,
    age: Optional[int] = None
) -> Tuple[float, str, str]:
    """
    Trả về (trọng_số_TBSA_tối_đa, tên_hiển_thị_vùng, tên_phương_pháp_giải_phẫu).
    - Người lớn (>= 15 tuổi hoặc không nhập tuổi): Luật số 9 Wallace + Quy tắc bàn tay 1% (Palmar Rule).
    - Trẻ em (< 15 tuổi): Tự động hiệu chỉnh tỷ lệ Đầu và Chân theo bảng Lund-Browder nhi khoa.
    """
    info = ANATOMICAL_REGIONS.get(body_part, ANATOMICAL_REGIONS["unknown"])
    base_weight = float(info["adult_tbsa"])
    region_name = str(info["name"])
    group = info["group"]

    if body_part == "palm_hand":
        return base_weight, region_name, "Palmar Rule (Quy tắc Bàn tay 1%)"

    if age is not None and 0 <= int(age) < 15:
        age_int = int(age)
        # Bảng Lund-Browder chuẩn Y khoa cho Đầu-Mặt-Cổ và Mỗi Chân (1 bên)
        if age_int < 1:
            ped_head, ped_leg = 19.0, 13.5
        elif age_int <= 4:
            ped_head, ped_leg = 17.0, 14.5
        elif age_int <= 9:
            ped_head, ped_leg = 13.0, 16.5
        else:  # 10 - 14 tuổi
            ped_head, ped_leg = 11.0, 17.5

        if group == "head_neck":
            scale = ped_head / 9.0
            return round(base_weight * scale, 2), region_name, f"Lund-Browder Nhi khoa ({age_int} tuổi)"
        if group == "lower_limb":
            scale = ped_leg / 18.0
            return round(base_weight * scale, 2), region_name, f"Lund-Browder Nhi khoa ({age_int} tuổi)"
        return base_weight, region_name, f"Lund-Browder Nhi khoa ({age_int} tuổi)"

    return base_weight, region_name, "Luật số 9 Wallace (Rule of Nines)"


def estimate_mosteller_bsa_cm2(
    weight_kg: Optional[float] = None,
    age: Optional[int] = None,
    height_cm: Optional[float] = None,
) -> float:
    """
    Tính tổng diện tích bề mặt cơ thể BSA (cm²) theo công thức chuẩn Y khoa Mosteller (1987):
      BSA (m²) = sqrt(Chiều_cao_cm * Cân_nặng_kg / 3600)
    Ưu tiên chiều cao thực tế `height_cm` do người dùng nhập; chỉ ước lượng theo tuổi khi bỏ trống.
    """
    w_kg = float(weight_kg) if (weight_kg and weight_kg > 0) else 65.0
    if height_cm is not None and float(height_cm) > 30.0:
        h_cm = float(height_cm)
    elif age is not None and 0 <= int(age) < 15:
        h_cm = max(55.0, min(162.0, 75.0 + 6.0 * float(age)))
    else:
        h_cm = 168.0
    bsa_m2 = np.sqrt((h_cm * w_kg) / 3600.0)
    return float(round(bsa_m2 * 10000.0, 1))


def segment_visible_skin_mask(
    image_rgb: Optional[np.ndarray],
    burn_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Phân đoạn tự động Vùng da bộ phận cơ thể chụp được trong khung ảnh (bao gồm cả da lành + mô bỏng),
    loại bỏ phông nền ngoại cảnh (ga trải giường, quần áo, tường, nền tối/trắng).

    Thuật toán kết hợp 3 nguồn tín hiệu thị giác máy tính (CS406):
    1. Không gian màu đa kênh YCrCb + HSV cho phổ sắc tố da người (Fitzpatrick I-VI).
    2. Hiệu chuẩn thích nghi vùng da quanh vết bỏng (Adaptive Peri-lesional Chrominance Ring):
       Lấy mẫu trực tiếp dải màu da lành bao quanh biên vết bỏng của chính bệnh nhân trong điều kiện
       ánh sáng thực tế của bức ảnh.
    3. Hợp nhất bất biến mô tổn thương (`burn_mask` luôn thuộc bộ phận cơ thể) + Lọc thành phần
       liên thông gắn kết với vùng tổn thương + Lấp lỗ hình thái học (Silhouette Hole Filling).
    """
    h, w = burn_mask.shape[:2]
    total_pixels = int(h * w)
    burn_bin = (burn_mask > 127).astype(np.uint8)
    burn_pixels = int(np.count_nonzero(burn_bin))

    # Trường hợp không truyền ảnh RGB (chỉ có mask nhị phân): fallback toàn khung ảnh
    if image_rgb is None or image_rgb.size == 0:
        fallback_mask = np.full((h, w), 255, dtype=np.uint8)
        pct_burn = round((burn_pixels / max(1, total_pixels)) * 100.0, 2)
        return {
            "skin_mask": fallback_mask,
            "skin_pixels": total_pixels,
            "pct_skin_in_frame": 100.0,
            "pct_burn_of_skin": float(pct_burn),
            "border_skin_ratio": 1.0,
            "is_closeup_detected": False,
            "segmentation_source": "frame_fallback",
        }

    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Phân tích sắc độ da chặt chẽ theo vật lý quang phổ da người (Kovac + Chai-Ngan YCrCb/HSV)
    # Mọi sắc thái da người (Fitzpatrick I-VI) đều thỏa R > G, R > B, Cr >= 133, Cb <= 128 và Cr - Cb >= 10.
    # Điều kiện này loại bỏ triệt để áo sẫm màu, tã vải trắng ngà dưới đèn vàng và sàn/ghế bệnh viện.
    R = image_rgb[:, :, 0].astype(np.int16)
    G = image_rgb[:, :, 1].astype(np.int16)
    B = image_rgb[:, :, 2].astype(np.int16)

    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    Y = ycrcb[:, :, 0].astype(np.int16)
    Cr = ycrcb[:, :, 1].astype(np.int16)
    Cb = ycrcb[:, :, 2].astype(np.int16)
    H, S, V = cv2.split(hsv)

    chroma_guard = (
        (R > G + 4) &
        (R > B + 6) &
        ((Cr - Cb) >= 10) &
        ((H <= 26) | (H >= 158)) &
        ~((S < 15) & (V > 225))
    )

    ycrcb_skin = chroma_guard & (Y > 30) & (Cr >= 133) & (Cr <= 185) & (Cb >= 75) & (Cb <= 128) & (S >= 15) & (V >= 35)
    prior_skin = ycrcb_skin.astype(np.uint8)

    # 2. Lấy mẫu thích nghi từ vành đai da lành sát rìa vết bỏng (Peri-lesional Ring)
    adaptive_skin = np.zeros((h, w), dtype=np.uint8)
    if burn_pixels > 0:
        k_inner = max(5, int(min(h, w) * 0.02)) | 1
        k_outer = max(15, int(min(h, w) * 0.10)) | 1
        dil_in = cv2.dilate(burn_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_inner, k_inner)))
        dil_out = cv2.dilate(burn_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_outer, k_outer)))
        ring = (dil_out > 0) & (dil_in == 0)

        # Chỉ lấy mẫu trên các pixel vành đai thỏa điều kiện quang phổ da thực sự (nới nhẹ biên Cb/Cr cho da ngăm Fitzpatrick V-VI & đèn LED trắng lạnh)
        ring_chroma_guard = (
            (R > G + 2) &
            (R > B + 3) &
            ((Cr - Cb) >= 6) &
            ((H <= 28) | (H >= 155)) &
            ~((S < 12) & (V > 230))
        )
        valid_ring = ring & ring_chroma_guard & (Cr >= 128) & (Cb <= 135) & (V > 32)
        if int(np.count_nonzero(valid_ring)) >= 40:
            cr_lo, cr_hi = np.percentile(Cr[valid_ring], [10, 90])
            cb_lo, cb_hi = np.percentile(Cb[valid_ring], [10, 90])
            y_lo = max(25.0, float(np.percentile(Y[valid_ring], 5)) - 10.0)
            adaptive_skin = (
                ring_chroma_guard &
                (Cr >= max(127.0, cr_lo - 4.0)) & (Cr <= min(188.0, cr_hi + 4.0)) &
                (Cb >= max(72.0, cb_lo - 4.0)) & (Cb <= min(136.0, cb_hi + 4.0)) &
                (Y >= y_lo)
            ).astype(np.uint8)

    # 3. Hợp nhất da lành + vết bỏng (vết bỏng Độ 3 hoại tử trắng/đen vẫn là diện tích bề mặt cơ thể!)
    combined = ((prior_skin > 0) | (adaptive_skin > 0) | (burn_bin > 0)).astype(np.uint8) * 255

    # Đóng hình thái học nối liền mạch vùng da bị chia cắt nhẹ bởi lông/bóng đổ
    k_close = max(7, int(min(h, w) * 0.03)) | 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # 4. Giữ các thành phần liên thông gắn với vùng bỏng (hoặc lân cận gần vùng bỏng) để loại bỏ đồ gỗ/vật thể nền
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean_body = np.zeros_like(closed)

    if burn_pixels > 0:
        # Mở rộng vùng lân cận vết bỏng để giữ trọn các ngón tay / phần chi cùng bộ phận
        k_prox = max(15, int(min(h, w) * 0.08)) | 1
        burn_proximity = cv2.dilate(
            burn_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_prox, k_prox))
        ) > 0
        for idx in range(1, num_labels):
            comp = labels == idx
            if np.any(comp & burn_proximity):
                clean_body[comp] = 255
    else:
        # Nếu ảnh không có vết bỏng, giữ khối da lớn nhất (>= 5% diện tích ảnh)
        min_comp = int(0.05 * total_pixels)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_comp:
                clean_body[labels == idx] = 255

    # Fallback an toàn nếu ảnh chụp quá sát hoặc ánh sáng đặc biệt làm mất vùng da ngoài vết bỏng
    if np.count_nonzero(clean_body) == 0:
        clean_body = closed.copy()

    # Lấp các lỗ hổng bên trong đường bao bộ phận cơ thể (ví dụ vùng hoại tử sẫm/bóng lóa nội vùng)
    contours, _ = cv2.findContours(clean_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_skin = np.zeros_like(clean_body)
    if contours:
        cv2.drawContours(filled_skin, contours, -1, 255, thickness=cv2.FILLED)
    filled_skin[burn_bin > 0] = 255

    skin_pixels = max(burn_pixels, int(np.count_nonzero(filled_skin > 127)))
    if skin_pixels == 0:
        skin_pixels = total_pixels
        filled_skin = np.full((h, w), 255, dtype=np.uint8)

    pct_skin_in_frame = round((skin_pixels / max(1, total_pixels)) * 100.0, 2)
    pct_burn_of_skin = round((burn_pixels / max(1, skin_pixels)) * 100.0, 2)

    # Kiểm tra mức độ chạm viền khung ảnh để nhận diện góc chụp dí sát (Close-up / Macro)
    top_edge = filled_skin[0, :] > 127
    bottom_edge = filled_skin[-1, :] > 127
    left_edge = filled_skin[:, 0] > 127
    right_edge = filled_skin[:, -1] > 127
    perimeter_px = 2 * (h + w)
    border_skin_px = (
        int(np.count_nonzero(top_edge)) +
        int(np.count_nonzero(bottom_edge)) +
        int(np.count_nonzero(left_edge)) +
        int(np.count_nonzero(right_edge))
    )
    border_skin_ratio = round(border_skin_px / max(1, perimeter_px), 3)

    # Nếu da lấp đầy > 88% khung ảnh và chạm > 82% chu vi 4 cạnh -> camera đang zoom cận cảnh bên trong bộ phận!
    is_closeup_detected = bool(pct_skin_in_frame >= 88.0 and border_skin_ratio >= 0.82)

    return {
        "skin_mask": filled_skin,
        "skin_pixels": int(skin_pixels),
        "pct_skin_in_frame": float(pct_skin_in_frame),
        "pct_burn_of_skin": float(min(100.0, pct_burn_of_skin)),
        "border_skin_ratio": float(border_skin_ratio),
        "is_closeup_detected": is_closeup_detected,
        "segmentation_source": "ycrcb_hsv_adaptive",
    }


def compute_anatomical_tbsa(
    burn_mask: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
    body_part: str = "chest_abdomen",
    region_coverage: float = 1.0,
    calibration_mode: str = "auto",
    pixels_per_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    patient_age: Optional[int] = None,
    height_cm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Chuẩn hóa % TBSA cơ thể theo Vùng giải phẫu (Rule of Nines / Palmar 1% / Lund-Browder)
    kết hợp tỷ lệ Vết bỏng / Diện tích vùng da chụp được hoặc Thước chuẩn vật lý.

    Tham số:
    - burn_mask: Mặt nạ nhị phân vết bỏng (H, W).
    - image_rgb: Ảnh RGB gốc để phân đoạn vùng da bộ phận chụp được.
    - body_part: Mã vùng giải phẫu trong `ANATOMICAL_REGIONS`.
    - region_coverage: Tỷ lệ (0.05 .. 1.0) của vùng giải phẫu đã chọn thực sự xuất hiện trong khung ảnh.
      Ví dụ: Chọn 'left_arm' (9%) nhưng khung ảnh chỉ chụp nửa cánh tay -> region_coverage = 0.5.
    - calibration_mode:
      + 'anatomical': Tính theo (Vết bỏng / Vùng da chụp được) × region_coverage × Trọng số giải phẫu.
      + 'ruler': Tính diện tích thực cm² theo thước chuẩn (pixels_per_cm) chia cho BSA Mosteller.
      + 'auto': Dùng 'ruler' nếu có pixels_per_cm > 0, ngược lại dùng 'anatomical'.
    """
    h, w = burn_mask.shape[:2]
    total_pixels = int(h * w)
    burn_pixels = int(np.count_nonzero(burn_mask > 127))
    pct_frame_area = round((burn_pixels / max(1, total_pixels)) * 100.0, 2)

    # 1. Phân đoạn diện tích vùng da bộ phận chụp được
    skin_info = segment_visible_skin_mask(image_rgb, burn_mask)
    skin_pixels = skin_info["skin_pixels"]
    pct_skin_in_frame = skin_info["pct_skin_in_frame"]
    pct_burn_of_skin = skin_info["pct_burn_of_skin"]
    is_closeup = skin_info["is_closeup_detected"]

    # 2. Lấy trọng số giải phẫu theo Luật số 9 / Palmar 1% / Lund-Browder
    region_weight_pct, region_name, atlas_method = get_age_adjusted_region_weight(body_part, patient_age)
    cov = float(np.clip(region_coverage if region_coverage is not None else 1.0, 0.05, 1.0))
    effective_visible_region_pct = round(region_weight_pct * cov, 3)

    # 3. Tính % TBSA chuẩn hóa theo giải phẫu (không cần thước)
    if burn_pixels == 0:
        anatomical_tbsa_pct = 0.0
    else:
        raw_anat_tbsa = (pct_burn_of_skin / 100.0) * effective_visible_region_pct
        anatomical_tbsa_pct = float(round(min(region_weight_pct, max(0.001, raw_anat_tbsa)), 3))

    # 4. Diện tích bề mặt cơ thể BSA (cm²) theo Mosteller (Chiều cao × Cân nặng / 3600)
    bsa_cm2 = estimate_mosteller_bsa_cm2(weight_kg=weight_kg, age=patient_age, height_cm=height_cm)

    # Xác định chế độ hiệu chuẩn thực tế
    has_valid_ruler = bool(pixels_per_cm is not None and float(pixels_per_cm) > 0)
    mode = (calibration_mode or "auto").lower().strip()
    if mode == "auto":
        active_mode = "ruler" if has_valid_ruler else "anatomical"
    elif mode == "ruler" and has_valid_ruler:
        active_mode = "ruler"
    else:
        active_mode = "anatomical"

    # 5. Quy đổi cm² và % TBSA cuối cùng
    area_cm2: Optional[float] = None
    estimated_area_cm2: Optional[float] = None
    area_is_estimated = False

    if active_mode == "ruler" and has_valid_ruler:
        px_cm = float(pixels_per_cm)
        area_cm2 = float(round(burn_pixels / (px_cm ** 2), 2))
        estimated_area_cm2 = area_cm2
        if burn_pixels == 0:
            estimated_tbsa = 0.0
        else:
            # Khi đã có thước chuẩn vật lý (px/cm), tính trực tiếp % TBSA = Diện tích vết bỏng (cm²) / BSA cơ thể (cm²)
            # Hoàn toàn độc lập với tiêu cự camera (region_coverage) và không bị cắt trần bởi Luật số 9!
            tbsa_from_area = (area_cm2 / bsa_cm2) * 100.0
            estimated_tbsa = float(round(min(95.0, max(0.001, tbsa_from_area)), 3))
        atlas_method = "Thước chuẩn Vật lý & BSA Mosteller"
        tbsa_method_label = f"Thước chuẩn vật lý ({px_cm:.1f} px/cm) & BSA Mosteller"
        formula_summary = (
            f"{area_cm2:.2f} cm² / {bsa_cm2:,.0f} cm² (BSA Mosteller) = {estimated_tbsa:.2f}% TBSA"
        )
    else:
        estimated_tbsa = float(round(anatomical_tbsa_pct, 2 if anatomical_tbsa_pct >= 0.1 else 3))
        if estimated_tbsa > 0:
            estimated_area_cm2 = float(round((estimated_tbsa / 100.0) * bsa_cm2, 1))
            area_is_estimated = True
        cov_pct_str = f" × {int(round(cov * 100))}% khung vùng" if cov < 0.999 else ""
        tbsa_method_label = f"Chuẩn hóa Giải phẫu ({atlas_method})"
        formula_summary = (
            f"{pct_burn_of_skin:.1f}% diện tích da chụp được × {region_weight_pct}% ({region_name})"
            f"{cov_pct_str} = {estimated_tbsa}% TBSA"
        )

    # 6. Cảnh báo lâm sàng thông minh (Cross-check phát hiện chụp dí sát hoặc đặt sai thước ảo)
    framing_warning: Optional[str] = None
    if (
        active_mode == "anatomical"
        and is_closeup
        and region_weight_pct >= 9.0
        and cov >= 0.95
        and burn_pixels > 0
    ):
        framing_warning = (
            f"Phát hiện ảnh chụp cận cảnh (vùng da chiếm {pct_skin_in_frame:.1f}% khung hình, chạm sát 4 viền ảnh) "
            f"nhưng đang chọn toàn bộ '{region_name}' ({region_weight_pct}% TBSA). Nếu camera chỉ chụp một phần nhỏ "
            f"của bộ phận hoặc chụp bàn tay, hãy chọn phân vùng nhỏ hơn (vd: Bàn tay 1%, Cẳng tay 4%) hoặc giảm "
            f"tỷ lệ khung vùng xuống 25%-50% để tránh ước tính thừa thể tích dịch Parkland."
        )
    elif (
        active_mode == "ruler"
        and burn_pixels > 0
        and pct_burn_of_skin >= 30.0
        and estimated_tbsa < 3.5
    ):
        framing_warning = (
            f"Cảnh báo sai số Thước đo vật lý: Vết bỏng chiếm tới {pct_burn_of_skin:.1f}% diện tích da trong ảnh "
            f"nhưng thước chuẩn chỉ tính ra {area_cm2:.1f} cm² ({estimated_tbsa:.1f}% TBSA). Lưu ý: Kéo thước lệch gấp 2 lần "
            f"sẽ làm diện tích bị chia nhỏ 2² = 4 lần! Nếu trong ảnh KHÔNG đặt sẵn thước/CCCD/đồng xu thật, vui lòng "
            f"quay lại Bước 2 và dùng chế độ 'Chuẩn hóa Giải phẫu AI (Không cần thước)'."
        )

    return {
        "total_pixels": total_pixels,
        "burn_pixels": burn_pixels,
        "skin_pixels": int(skin_pixels),
        "pct_frame_area": float(pct_frame_area),
        "pct_skin_in_frame": float(pct_skin_in_frame),
        "pct_burn_of_skin": float(pct_burn_of_skin),
        "body_part": body_part,
        "body_part_name": region_name,
        "region_weight_pct": float(region_weight_pct),
        "region_coverage": float(cov),
        "effective_visible_region_pct": float(effective_visible_region_pct),
        "atlas_method": atlas_method,
        "calibration_mode": active_mode,
        "tbsa_method": tbsa_method_label,
        "formula_summary": formula_summary,
        "bsa_cm2": float(bsa_cm2),
        "area_cm2": area_cm2,
        "estimated_area_cm2": estimated_area_cm2,
        "area_is_estimated": bool(area_is_estimated),
        "anatomical_tbsa_pct": float(anatomical_tbsa_pct),
        "estimated_tbsa_pct": float(estimated_tbsa),
        "is_closeup_detected": bool(is_closeup),
        "framing_warning": framing_warning,
        "skin_mask": skin_info["skin_mask"],
    }
