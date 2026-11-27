"""
Unit Tests: tests/test_dual_uncertainty_and_ood.py
Kiểm thử tự động toàn bộ nâng cấp Hướng 1 (Spatial Boundary Uncertainty & Parkland Interval),
Hướng 4A (OOD Safety Rejection & Real-World Shift Generalization), và tích hợp BurnAnalyzer.
"""

import os
import numpy as np
import torch
import pytest

from src.vision.spatial_uncertainty import (
    binary_entropy_normalized,
    compute_pixel_uncertainty_maps,
    summarize_boundary_uncertainty,
    compute_clinical_tbsa_and_parkland_interval,
    render_uncertainty_heatmap_rgb,
)
from src.vision.models import build_segmentation_model, build_classification_model
from src.vision.burn_analyzer import BurnAnalyzer
from src.vision.train_cls import get_cls_val_transforms
from tools.evaluate_ood_and_generalization import (
    predict_rgb_batch_ensemble,
    apply_realworld_shift,
)


def test_pixel_uncertainty_maps_math():
    """Kiểm tra toán học của Entropy, Std và BALD Mutual Information trên 5-Fold giả lập."""
    # Vùng trái đồng thuận chắc chắn là bỏng (0.99), vùng phải bất đồng mạnh giữa 5 Fold
    f1 = np.hstack([np.full((32, 16), 0.99), np.full((32, 16), 0.90)])
    f2 = np.hstack([np.full((32, 16), 0.98), np.full((32, 16), 0.10)])
    f3 = np.hstack([np.full((32, 16), 0.99), np.full((32, 16), 0.85)])
    f4 = np.hstack([np.full((32, 16), 0.97), np.full((32, 16), 0.15)])
    f5 = np.hstack([np.full((32, 16), 0.99), np.full((32, 16), 0.50)])
    stack = np.stack([f1, f2, f3, f4, f5], axis=0).astype(np.float32)

    maps = compute_pixel_uncertainty_maps(stack)
    assert maps["mean_prob"].shape == (32, 32)
    assert float(maps["std_prob"][:, :16].mean()) < 0.02
    assert float(maps["std_prob"][:, 16:].mean()) > 0.25
    assert float(maps["predictive_entropy"][:, 16:].mean()) > float(maps["predictive_entropy"][:, :16].mean())
    assert float(maps["mutual_info"][:, 16:].mean()) > 0.15

    summary = summarize_boundary_uncertainty(stack)
    assert 0.0 <= summary["boundary_entropy"] <= 1.0
    assert summary["fold_area_max_pct"] > summary["fold_area_min_pct"]


def test_borderline_10pct_tbsa_parkland_warning():
    """Kiểm tra cờ cảnh báo an toàn khi khoảng diện tích bất định cắt ngang ngưỡng sốc bỏng 10% TBSA."""
    # Với region_weight = 18.0%, 50% khung ảnh = 9.0% TBSA, 65% khung ảnh = 11.7% TBSA -> cắt ngang 10.0%!
    probs = []
    for frac in [0.42, 0.48, 0.52, 0.58, 0.66]:
        m = np.zeros((100, 100), dtype=np.float32)
        n_px = int(10000 * frac)
        m.flat[:n_px] = 0.9
        probs.append(m)
    stack = np.stack(probs, axis=0)

    clin = compute_clinical_tbsa_and_parkland_interval(
        stack, region_weight=18.0, weight_kg=60.0, predicted_class_id=1
    )
    lo, hi = clin["tbsa_interval_pct"]
    assert lo < 10.0 <= hi
    assert clin["borderline_10pct_warning"] is True
    assert clin["parkland_interval"]["raw_24h_high_ml"] > clin["parkland_interval"]["raw_24h_low_ml"]

    # Đối với Bỏng Độ 1 (predicted_class_id=0), tuyệt đối không bật cảnh báo sốc bỏng truyền dịch
    clin_deg1 = compute_clinical_tbsa_and_parkland_interval(
        stack, region_weight=18.0, weight_kg=60.0, predicted_class_id=0
    )
    assert clin_deg1["borderline_10pct_warning"] is False


def test_uncertainty_heatmap_rendering():
    rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
    u_map = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    out = render_uncertainty_heatmap_rgb(rgb, u_map)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_realworld_shifts_and_edl_ensemble_inference():
    img = np.random.randint(30, 220, size=(128, 128, 3), dtype=np.uint8)
    for mode in ["clean", "dim_warm_light", "motion_defocus_blur", "messenger_jpeg"]:
        shifted = apply_realworld_shift(img, mode)
        assert shifted.shape == img.shape
        assert shifted.dtype == np.uint8

    device = torch.device("cpu")
    m1 = build_classification_model("resnet34", num_classes=3, pretrained=False).eval()
    m2 = build_classification_model("resnet34", num_classes=3, pretrained=False).eval()
    tf = get_cls_val_transforms(img_size=224)

    res = predict_rgb_batch_ensemble([m1, m2], [img, img], tf, device, is_edl=True, edl_activation="exp")
    assert res["probs"].shape == (2, 3)
    assert np.allclose(res["probs"].sum(axis=-1), 1.0, atol=1e-4)
    assert np.all((res["uncertainty"] > 0.0) & (res["uncertainty"] <= 1.0))


def test_burn_analyzer_dual_uncertainty_end_to_end(tmp_path):
    """Kiểm tra tích hợp BurnAnalyzer nạp checkpoint thực tế và xuất đủ thông tin Bất định Kép."""
    seg_ckpt = str(tmp_path / "dummy_seg.pth")
    cls_ckpt = str(tmp_path / "dummy_cls.pth")

    seg_m = build_segmentation_model(arch="unet", backbone="resnet34", pretrained=False)
    cls_m = build_classification_model(model_name="resnet34", num_classes=3, pretrained=False)
    torch.save(seg_m.state_dict(), seg_ckpt)
    torch.save(cls_m.state_dict(), cls_ckpt)

    analyzer = BurnAnalyzer(
        seg_checkpoint=seg_ckpt,
        cls_checkpoint=cls_ckpt,
        device="cpu",
        seg_model_key="kfold_unet_resnet34_combo",
        cls_model_key="kfold_cls_resnet34_edl",
    )
    # Tắt TTA trong unit test CPU để chạy tức thì
    analyzer.seg_cfg = dict(analyzer.seg_cfg, use_tta=False, checkpoints=[])
    analyzer.cls_cfg = dict(analyzer.cls_cfg, checkpoints=[])

    dummy_rgb = np.random.randint(40, 220, size=(200, 200, 3), dtype=np.uint8)
    out = analyzer.analyze(dummy_rgb, body_part="chest_abdomen", weight_kg=65.0)

    assert "tbsa_interval_pct" in out["metrics"]
    assert "boundary_entropy" in out["metrics"]
    assert "epistemic_uncertainty" in out["classification"]
    assert out["classification"]["is_edl"] is True
    assert out["visualizations"]["uncertainty_heatmap"] is not None

