"""
Unit & Integration Tests: tests/test_anatomical_tbsa.py
Kiểm chứng toàn diện Hướng 7: Chuẩn hóa % TBSA theo Bộ phận Cơ thể (Rule of Nines,
Palmar 1%, Lund-Browder) & Tỷ lệ Vết bỏng / Diện tích Vùng da chụp được trước khi tính Parkland.
"""

import os
import cv2
import numpy as np
import pytest

from src.vision.anatomical_tbsa import (
    ANATOMICAL_REGIONS,
    get_age_adjusted_region_weight,
    segment_visible_skin_mask,
    compute_anatomical_tbsa,
)
from src.vision.burn_analyzer import BurnAnalyzer
from src.rag.simple_advisor import ask_medical_bot


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_synthetic_limb_on_drape(
    h: int = 400,
    w: int = 500,
    skin_fraction: float = 0.50,
    burn_of_skin_fraction: float = 0.60,
):
    """
    Tạo ảnh giả lập bộ phận cơ thể (màu da người thật) đặt trên nền vải y tế màu xanh nhạt,
    trong đó diện tích da chiếm `skin_fraction` khung ảnh và vết bỏng chiếm `burn_of_skin_fraction` vùng da.
    """
    # Nền vải phẫu thuật xanh nhạt (RGB: 185, 225, 235 -> không phải màu da)
    img_rgb = np.full((h, w, 3), [185, 225, 235], dtype=np.uint8)
    burn_mask = np.zeros((h, w), dtype=np.uint8)

    skin_w = int(round(w * skin_fraction))
    x0 = (w - skin_w) // 2
    x1 = x0 + skin_w

    # Vùng da lành ấm (RGB: 228, 178, 148)
    img_rgb[:, x0:x1] = [228, 178, 148]

    # Vùng vết bỏng nằm gọn bên trong vùng da
    burn_h = int(round(h * burn_of_skin_fraction))
    y0 = (h - burn_h) // 2
    y1 = y0 + burn_h
    img_rgb[y0:y1, x0:x1] = [215, 55, 55]
    burn_mask[y0:y1, x0:x1] = 255

    return img_rgb, burn_mask


def test_paradox_hand_vs_back_same_60pct_burn_ratio():
    """
    Kịch bản cốt lõi của bác sĩ chấm bài:
    Cùng 60% diện tích vùng da chụp được bị bỏng Độ 2 trên bệnh nhân 70kg:
    - Ca A (Bàn tay - Quy tắc 1%): TBSA ~ 0.6% -> KHÔNG truyền dịch tĩnh mạch (0 ml).
    - Ca B (Toàn bộ Lưng & Mông - 18%): TBSA ~ 10.8% (>= 10%) -> BẮT BUỘC truyền Parkland (~3024 ml).
    """
    img_rgb, burn_mask = _make_synthetic_limb_on_drape(
        skin_fraction=0.50, burn_of_skin_fraction=0.60
    )
    analyzer = BurnAnalyzer()

    # Ca A: Bàn tay (1% TBSA)
    hand_metrics = analyzer.calculate_metrics(
        mask=burn_mask,
        image_rgb=img_rgb,
        body_part="palm_hand",
        weight_kg=70.0,
        predicted_class_id=1,  # Độ 2
        calibration_mode="anatomical",
    )
    # Lưu ý: Tỷ lệ trên toàn khung ảnh chỉ là 30% (do nền chiếm 50%), nhưng Bỏng/Da là ~60%!
    assert 29.0 <= hand_metrics["pct_frame_area"] <= 31.0
    assert 58.0 <= hand_metrics["pct_burn_of_skin"] <= 62.0
    assert 0.55 <= hand_metrics["estimated_tbsa_pct"] <= 0.65
    assert hand_metrics["parkland_plan"]["indicated"] is False
    assert hand_metrics["parkland_plan"]["total_fluid_24h_ml"] == 0.0

    # Ca B: Lưng & Mông sau (18% TBSA)
    back_metrics = analyzer.calculate_metrics(
        mask=burn_mask,
        image_rgb=img_rgb,
        body_part="back_buttocks",
        weight_kg=70.0,
        predicted_class_id=1,  # Độ 2
        calibration_mode="anatomical",
        burn_time_hours=2.0,
    )
    assert 10.4 <= back_metrics["estimated_tbsa_pct"] <= 11.2
    assert back_metrics["parkland_plan"]["indicated"] is True
    expected_ml = round(4.0 * 70.0 * back_metrics["estimated_tbsa_pct"], 1)
    assert back_metrics["parkland_plan"]["total_fluid_24h_ml"] == pytest.approx(expected_ml, abs=1.0)
    # Đã qua 2h -> 50% đầu tiên phải truyền trong 6h còn lại
    assert back_metrics["parkland_plan"]["remaining_first_8h_hours"] == pytest.approx(6.0)
    assert back_metrics["parkland_plan"]["rate_first_8h_ml_h"] == pytest.approx(
        round((expected_ml * 0.5) / 6.0, 1), abs=0.2
    )


def test_background_removal_on_real_clinical_dataset_images():
    """
    Kiểm tra trực tiếp trên ảnh thật trong tập dữ liệu (img212.jpg: bàn tay bỏng trên ga giường xanh;
    img1101.jpg: cháy nắng lưng trên nền bể bơi xanh):
    Thuật toán phải loại bỏ nền xanh và cho tỷ lệ Vết bỏng / Vùng da cao hơn hẳn Tỷ lệ / Khung ảnh.
    """
    candidates = [
        ("data/processed/images/img212.jpg", "data/processed/masks/img212.png", "palm_hand"),
        ("data/processed/images/img1101.jpg", "data/processed/masks/img1101.png", "upper_back"),
    ]
    tested = 0
    for img_rel, msk_rel, part in candidates:
        img_path = os.path.join(PROJECT_ROOT, img_rel)
        msk_path = os.path.join(PROJECT_ROOT, msk_rel)
        if not (os.path.exists(img_path) and os.path.exists(msk_path)):
            continue
        tested += 1
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        res = compute_anatomical_tbsa(
            burn_mask=msk,
            image_rgb=rgb,
            body_part=part,
            calibration_mode="anatomical",
        )
        assert res["pct_skin_in_frame"] < 75.0, "Phải tách bỏ được phông nền xanh ngoại cảnh"
        assert res["pct_burn_of_skin"] > res["pct_frame_area"] * 1.5
        assert 0.0 < res["estimated_tbsa_pct"] <= ANATOMICAL_REGIONS[part]["adult_tbsa"]

    assert tested >= 1 or not os.path.exists(os.path.join(PROJECT_ROOT, "data/processed/images"))


def test_pediatric_lund_browder_age_adjustment():
    """Trẻ nhỏ 3 tuổi bỏng vùng Đầu-Mặt-Cổ phải được nâng hệ số lên 17% theo Lund-Browder (thay vì 9% người lớn)."""
    w_adult, _, method_adult = get_age_adjusted_region_weight("head_neck", age=30)
    w_child, _, method_child = get_age_adjusted_region_weight("head_neck", age=3)
    w_leg_child, _, _ = get_age_adjusted_region_weight("left_leg", age=3)

    assert w_adult == 9.0
    assert "Wallace" in method_adult
    assert w_child == 17.0
    assert "Lund-Browder" in method_child
    assert w_leg_child == 14.5


def test_closeup_macro_detection_and_region_coverage_scaling():
    """
    Khi ảnh chụp dí sát làm da tràn kín 100% 4 cạnh ảnh trên vùng lớn (chest_abdomen 18%):
    - Phải phát hiện `is_closeup_detected = True` và sinh `framing_warning`.
    - Khi bác sĩ chỉnh `region_coverage = 0.25` (chỉ 1/4 vùng nằm trong ảnh), % TBSA giảm đúng 4 lần.
    """
    h, w = 300, 300
    full_skin_rgb = np.full((h, w, 3), [222, 172, 145], dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[75:225, 75:225] = 255  # 25% diện tích ảnh
    full_skin_rgb[75:225, 75:225] = [210, 45, 45]

    res_full = compute_anatomical_tbsa(
        burn_mask=mask,
        image_rgb=full_skin_rgb,
        body_part="chest_abdomen",
        region_coverage=1.0,
        calibration_mode="anatomical",
    )
    assert res_full["is_closeup_detected"] is True
    assert res_full["framing_warning"] is not None
    assert res_full["estimated_tbsa_pct"] == pytest.approx(4.5, abs=0.1)  # 25% * 18%

    res_quarter = compute_anatomical_tbsa(
        burn_mask=mask,
        image_rgb=full_skin_rgb,
        body_part="chest_abdomen",
        region_coverage=0.25,
        calibration_mode="anatomical",
    )
    assert res_quarter["framing_warning"] is None
    assert res_quarter["estimated_tbsa_pct"] == pytest.approx(1.12, abs=0.05)  # 25% * 18% * 0.25


def test_anatomical_mode_ignores_uncalibrated_default_ruler():
    """
    Đảm bảo khi chọn `calibration_mode='anatomical'`, hệ thống KHÔNG bị ảnh hưởng bởi giá trị
    `pixels_per_cm` mặc định giả định khi trong ảnh không có thước thật.
    """
    img_rgb, burn_mask = _make_synthetic_limb_on_drape(skin_fraction=0.6, burn_of_skin_fraction=0.5)
    res_anat = compute_anatomical_tbsa(
        burn_mask=burn_mask,
        image_rgb=img_rgb,
        body_part="forearm",  # 4% TBSA -> 50% * 4% = 2.0% TBSA
        calibration_mode="anatomical",
        pixels_per_cm=29.2,
    )
    assert res_anat["calibration_mode"] == "anatomical"
    assert res_anat["area_cm2"] is None
    assert res_anat["area_is_estimated"] is True
    assert res_anat["estimated_tbsa_pct"] == pytest.approx(2.0, abs=0.15)


def test_ruler_mode_computes_direct_tbsa_independent_of_region_and_coverage():
    """
    Khi chọn `calibration_mode='ruler'`, hệ thống phải tính thẳng `area_cm2 / BSA_cm2`,
    hoàn toàn độc lập với `body_part` và `region_coverage` (không bị cắt trần bởi Bàn tay 1%).
    """
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[50:450, 50:450] = 255  # 400x400 = 160,000 px
    # Với thước 20 px/cm -> 1 cm² = 400 px -> vết bỏng rộng đúng 400 cm²
    res_ruler = compute_anatomical_tbsa(
        burn_mask=mask,
        body_part="palm_hand",  # Giả sử dropdown cũ đang ở 1%, ruler mode KHÔNG được cắt trần ở 1%!
        region_coverage=0.1,    # Giả sử framing đang 10%, ruler mode KHÔNG được nhân 0.1!
        calibration_mode="ruler",
        pixels_per_cm=20.0,
        weight_kg=68.0,
        patient_age=30,
    )
    assert res_ruler["calibration_mode"] == "ruler"
    assert res_ruler["area_cm2"] == pytest.approx(400.0, abs=0.1)
    expected_tbsa = round((400.0 / res_ruler["bsa_cm2"]) * 100.0, 3)
    assert res_ruler["estimated_tbsa_pct"] == pytest.approx(expected_tbsa, abs=0.01)
    assert res_ruler["estimated_tbsa_pct"] > 2.0


def test_rag_advisor_explains_anatomical_tbsa_and_parkland():
    """Kiểm tra RAG Advisor giải thích rõ chuẩn hóa giải phẫu TBSA và trích xuất thông số ca hiện tại."""
    ctx = {
        "confidence": 92.5,
        "predicted_degree": "Độ 2 (Partial Thickness)",
        "body_part_name": "Bàn tay / Lòng bàn tay (Quy tắc 1%)",
        "pct_frame_area": 58.0,
        "pct_burn_of_skin": 62.0,
        "estimated_tbsa_pct": 0.62,
        "parkland_plan": {
            "indicated": False,
            "clinical_guidance": "Diện tích nhỏ (0.62% TBSA), KHÔNG truyền dịch tĩnh mạch."
        },
    }
    ans_tbsa = ask_medical_bot("Tại sao diện tích TBSA khác với phần trăm khung ảnh?", clinical_context=ctx)
    assert "0.62% TBSA" in ans_tbsa
    assert "62.0%" in ans_tbsa

    ans_parkland = ask_medical_bot("Ca này tính công thức Parkland truyền dịch như thế nào?", clinical_context=ctx)
    assert "KHÔNG PHẢI" in ans_parkland
    assert "0.62% TBSA" in ans_parkland
