"""
Module: src/vision/spatial_uncertainty.py
Mục đích: Lượng hóa Độ bất định Không gian từng Pixel (Spatial / Boundary Uncertainty)
từ 5-Fold Segmentation Ensemble (U-Net ResNet-34) và truyền sai số sang:
  1) Khoảng diện tích bỏng tin cậy (% Frame Area & % TBSA Interval).
  2) Khoảng thể tích dịch truyền hồi sức Parkland 24h (ml Interval).
  3) Cảnh báo an toàn lâm sàng tại ngưỡng ranh giới sốc bỏng (10% TBSA).

Cơ sở lý thuyết:
  - Kendall & Gal (NeurIPS 2017): Epistemic vs Aleatoric Uncertainty in Computer Vision.
  - Houlsby et al. (2011): Bayesian Active Learning by Disagreement (BALD / Mutual Information).
  - Kirscher et al. (MICCAI, nnU-Net group): Cross-Validation Ensemble Uncertainty for Boundary Ambiguity.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import cv2


EPS = 1e-7


def binary_entropy_normalized(p: np.ndarray) -> np.ndarray:
    """
    Tính Entropy nhị phân chuẩn hóa về miền [0.0, 1.0] (chia cho ln(2)):
      H(p) = (-p * ln(p) - (1 - p) * ln(1 - p)) / ln(2)
    Đạt cực đại 1.0 khi p = 0.5 (mơ hồ hoàn toàn) và 0.0 khi p = 0 hoặc p = 1.
    """
    p_clip = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    h = -p_clip * np.log(p_clip) - (1.0 - p_clip) * np.log(1.0 - p_clip)
    return (h / np.log(2.0)).astype(np.float32)


def compute_pixel_uncertainty_maps(prob_stack: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Tính các bản đồ xác suất và độ bất định từng pixel từ K mô hình (5-Fold Ensemble).

    Tham số:
      - prob_stack: Mảng numpy shape (K, H, W) với giá trị xác suất trong [0, 1].
                    Nếu truyền (H, W) đơn lẻ sẽ tự động mở rộng thành (1, H, W).

    Trả về dict gồm các mảng float32 shape (H, W):
      - mean_prob          : Xác suất trung bình P_bar(x, y) thuộc [0, 1].
      - std_prob           : Độ lệch chuẩn xác suất giữa các Fold sigma(x, y) thuộc [0, 0.5].
      - predictive_entropy : Tổng độ bất định H(P_bar) chuẩn hóa thuộc [0, 1].
      - mutual_info        : Độ bất định nhận thức (BALD Epistemic Disagreement) thuộc [0, 1].
    """
    probs = np.asarray(prob_stack, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[np.newaxis, ...]
    if probs.ndim != 3:
        raise ValueError(f"prob_stack phải có shape (K, H, W) hoặc (H, W), nhận được {probs.shape}")

    probs = np.clip(probs, 0.0, 1.0)
    mean_prob = np.mean(probs, axis=0)
    std_prob = np.std(probs, axis=0, ddof=0)

    pred_entropy = binary_entropy_normalized(mean_prob)
    expected_entropy = np.mean(binary_entropy_normalized(probs), axis=0)
    mutual_info = np.clip(pred_entropy - expected_entropy, 0.0, 1.0)

    return {
        "mean_prob": mean_prob.astype(np.float32),
        "std_prob": std_prob.astype(np.float32),
        "predictive_entropy": pred_entropy.astype(np.float32),
        "mutual_info": mutual_info.astype(np.float32),
    }


def summarize_boundary_uncertainty(
    prob_stack: np.ndarray,
    threshold: float = 0.5,
    active_prob_floor: float = 0.10,
) -> Dict[str, float]:
    """
    Tổng hợp bản đồ độ bất định 2D thành các chỉ số vô hướng cấp độ ảnh (Image-level Uncertainty)
    mà KHÔNG bị pha loãng bởi diện tích nền da lành khổng lồ xung quanh.
    """
    maps = compute_pixel_uncertainty_maps(prob_stack)
    mean_p = maps["mean_prob"]
    std_p = maps["std_prob"]
    ent_p = maps["predictive_entropy"]
    mi_p = maps["mutual_info"]

    # Vùng quan tâm hoạt động (Active Region): các pixel có ít nhất một chút nghi ngờ bỏng
    active_mask = mean_p >= active_prob_floor
    if int(np.count_nonzero(active_mask)) < 16:
        # Nếu mô hình dự đoán gần như toàn ảnh là nền, xét top 5% pixel có xác suất cao nhất
        q95 = float(np.quantile(mean_p, 0.95))
        active_mask = mean_p >= max(q95, 1e-4)
        if int(np.count_nonzero(active_mask)) == 0:
            active_mask = np.ones_like(mean_p, dtype=bool)

    # Vành đai biên hình thái học quanh mask nhị phân trung bình
    bin_mask = (mean_p > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(bin_mask, kernel, iterations=1)
    eroded = cv2.erode(bin_mask, kernel, iterations=1)
    boundary_band = (dilated ^ eroded) > 0
    if int(np.count_nonzero(boundary_band)) < 8:
        boundary_band = active_mask

    # Độ phân tán diện tích qua K Folds
    probs = np.asarray(prob_stack, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[np.newaxis, ...]
    fold_areas = [float(np.mean(probs[k] > threshold) * 100.0) for k in range(probs.shape[0])]
    mean_area = float(np.mean(fold_areas))
    std_area = float(np.std(fold_areas, ddof=1)) if len(fold_areas) > 1 else 0.0
    area_cv = float(std_area / (mean_area + 0.5))  # Hệ số biến thiên diện tích ổn định

    return {
        "active_entropy": float(np.mean(ent_p[active_mask])),
        "active_std": float(np.mean(std_p[active_mask])),
        "active_mutual_info": float(np.mean(mi_p[active_mask])),
        "boundary_entropy": float(np.mean(ent_p[boundary_band])),
        "boundary_std": float(np.mean(std_p[boundary_band])),
        "fold_area_mean_pct": round(mean_area, 4),
        "fold_area_std_pct": round(std_area, 4),
        "fold_area_min_pct": round(float(np.min(fold_areas)), 4),
        "fold_area_max_pct": round(float(np.max(fold_areas)), 4),
        "area_cv": round(area_cv, 4),
    }


def compute_clinical_tbsa_and_parkland_interval(
    prob_stack: np.ndarray,
    region_weight: float = 18.0,
    weight_kg: Optional[float] = None,
    predicted_class_id: int = 1,
    pixels_per_cm: Optional[float] = None,
    orig_hw: Optional[tuple] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Truyền độ bất định không gian từ 5-Fold Segmentation sang:
      - Khoảng % diện tích khung hình [pct_low, pct_high]
      - Khoảng % TBSA cơ thể [tbsa_low, tbsa_high]
      - Khoảng thể tích dịch truyền Parkland 24h [fluid_low_ml, fluid_high_ml]
      - Cờ cảnh báo an toàn ranh giới sốc bỏng 10% TBSA (borderline_10pct_warning)
    """
    probs = np.asarray(prob_stack, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[np.newaxis, ...]

    # Resize từng bản đồ xác suất về kích thước gốc nếu được cung cấp
    if orig_hw is not None:
        h_o, w_o = int(orig_hw[0]), int(orig_hw[1])
        resized_list: List[np.ndarray] = []
        for k in range(probs.shape[0]):
            resized_list.append(cv2.resize(probs[k], (w_o, h_o), interpolation=cv2.INTER_LINEAR))
        probs = np.stack(resized_list, axis=0)

    mean_prob = np.mean(probs, axis=0)
    h, w = mean_prob.shape[:2]
    total_pixels = float(h * w)

    # Diện tích theo từng Fold và theo dải xác suất đồng thuận [P_bar > 0.65, P_bar > 0.35]
    fold_pcts = [float(np.count_nonzero(probs[k] > threshold) / total_pixels * 100.0) for k in range(probs.shape[0])]
    ens_pct = float(np.count_nonzero(mean_prob > threshold) / total_pixels * 100.0)
    fold_std_pct = float(np.std(fold_pcts, ddof=1)) if len(fold_pcts) > 1 else 0.0

    if probs.shape[0] > 1:
        pct_low = float(min(np.min(fold_pcts), max(0.0, ens_pct - 1.96 * fold_std_pct)))
        pct_high = float(max(np.max(fold_pcts), min(100.0, ens_pct + 1.96 * fold_std_pct)))
    else:
        # Fallback cho 1-model đơn lẻ: dùng biên độ xác suất mềm [0.65, 0.35]
        pct_low = float(np.count_nonzero(mean_prob > 0.65) / total_pixels * 100.0)
        pct_high = float(np.count_nonzero(mean_prob > 0.35) / total_pixels * 100.0)

    def _pct_to_tbsa(pct_val: float) -> float:
        burn_px = (pct_val / 100.0) * total_pixels
        if pixels_per_cm and pixels_per_cm > 0:
            area_cm2 = burn_px / (float(pixels_per_cm) ** 2)
            pw = float(weight_kg if (weight_kg and weight_kg > 0) else 68.0)
            bsa_cm2 = float(np.sqrt((170.0 * pw) / 3600.0) * 10000.0)
            tbsa = (area_cm2 / bsa_cm2) * 100.0
            return float(round(min(region_weight, max(0.0, tbsa)), 3))
        return float(round(min(region_weight, max(0.0, (pct_val / 100.0) * region_weight)), 2))

    tbsa_ens = _pct_to_tbsa(ens_pct)
    tbsa_low = _pct_to_tbsa(pct_low)
    tbsa_high = _pct_to_tbsa(pct_high)
    tbsa_half_width = round(0.5 * (tbsa_high - tbsa_low), 2)

    is_degree_1 = (predicted_class_id == 0)
    # Cảnh báo ranh giới sốc bỏng 10% TBSA: khi ngưỡng 10% nằm trọn trong khoảng bất định [tbsa_low, tbsa_high]
    borderline_10pct = bool((not is_degree_1) and (tbsa_low < 10.0 <= tbsa_high or (tbsa_low <= 10.0 < tbsa_high)))

    fluid_interval = None
    if weight_kg and weight_kg > 0 and tbsa_high > 0:
        w_kg = float(weight_kg)
        raw_mean_ml = round(4.0 * w_kg * tbsa_ens, 1)
        raw_low_ml = round(4.0 * w_kg * tbsa_low, 1)
        raw_high_ml = round(4.0 * w_kg * tbsa_high, 1)
        fluid_interval = {
            "raw_24h_mean_ml": raw_mean_ml,
            "raw_24h_low_ml": raw_low_ml,
            "raw_24h_high_ml": raw_high_ml,
            "first_8h_low_ml": round(raw_low_ml * 0.5, 1),
            "first_8h_high_ml": round(raw_high_ml * 0.5, 1),
            "borderline_10pct_warning": borderline_10pct,
            "safety_note": (
                f"CẢNH BÁO RANH GIỚI SỐC BỎNG: Khoảng ước lượng diện tích [{tbsa_low}% – {tbsa_high}% TBSA] "
                "cắt ngang ngưỡng truyền dịch tĩnh mạch 10% TBSA. Bác sĩ cần đo kiểm tra lại diện tích trực tiếp!"
                if borderline_10pct else
                f"Khoảng thể tích tham chiếu theo độ bất định biên: {raw_low_ml} – {raw_high_ml} ml / 24h."
            ),
        }

    return {
        "pct_frame_ensemble": round(ens_pct, 2),
        "pct_frame_std": round(fold_std_pct, 2),
        "pct_frame_interval": [round(pct_low, 2), round(pct_high, 2)],
        "tbsa_ensemble_pct": tbsa_ens,
        "tbsa_half_width_pct": tbsa_half_width,
        "tbsa_interval_pct": [tbsa_low, tbsa_high],
        "borderline_10pct_warning": borderline_10pct,
        "parkland_interval": fluid_interval,
    }


def render_uncertainty_heatmap_rgb(
    image_rgb: np.ndarray,
    uncertainty_map: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Trực quan hóa bản đồ độ bất định từng pixel (Predictive Entropy hoặc Normalized Std)
    đè lên ảnh RGB gốc bằng bảng màu TURBO/JET chuẩn y khoa.
    """
    h, w = image_rgb.shape[:2]
    u_resized = cv2.resize(uncertainty_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    u_norm = np.clip(u_resized / (float(np.max(u_resized)) + 1e-6), 0.0, 1.0)
    u_uint8 = (u_norm * 255.0).astype(np.uint8)

    cmap_flag = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(u_uint8, cmap_flag)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Chỉ tô đậm những vùng có độ bất định thực sự (> 0.08) để giữ rõ nền ảnh gốc
    weight = np.clip((u_norm - 0.05) / 0.95, 0.0, 1.0)[..., np.newaxis] * alpha
    blended = (image_rgb.astype(np.float32) * (1.0 - weight) + heatmap_rgb.astype(np.float32) * weight)
    return np.clip(blended, 0, 255).astype(np.uint8)

