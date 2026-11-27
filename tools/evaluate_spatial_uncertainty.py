"""
Module: tools/evaluate_spatial_uncertainty.py
Mục đích (HƯỚNG 1 - Q1/Q2 Upgrade):
  Đánh giá toàn diện Độ bất định Không gian từng Pixel (Spatial Boundary Uncertainty)
  từ 5-Fold Segmentation Ensemble (U-Net ResNet-34 + Combo) trên 187 ảnh Test độc lập:
    1) Tính bản đồ Predictive Entropy, Epistemic Std & Mutual Information (BALD) từng pixel.
    2) Kiểm định thống kê Spearman Correlation (rho, p-value) giữa Độ bất định biên và Sai số phân đoạn (1 - Dice).
    3) Kiểm định Mann-Whitney U & Failure Detection AUROC tách ca phân đoạn Tốt vs Kém.
    4) Đánh giá Khoảng tin cậy Diện tích bỏng (% TBSA Interval) & Khoảng dịch truyền Parkland 24h.
    5) Xuất 2 biểu đồ 300 DPI chuẩn Q1/Q2 + file JSON/CSV tổng kết.

Cách chạy (0 phút train lại):
  python tools/evaluate_spatial_uncertainty.py --arch unet --backbone resnet34 --loss combo
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

from src.vision.models import build_segmentation_model
from src.vision.transforms import get_val_transforms
from src.vision.train_seg import forward_seg_tta
from src.vision.spatial_uncertainty import (
    compute_pixel_uncertainty_maps,
    summarize_boundary_uncertainty,
    compute_clinical_tbsa_and_parkland_interval,
    render_uncertainty_heatmap_rgb,
)


def resolve_kfold_seg_checkpoints(
    ckpt_dir: str,
    run_prefix: str,
    n_folds: int = 5,
) -> List[str]:
    """
    Tìm danh sách 5 file trọng số k-fold phân đoạn (tự động kiểm tra cả thư mục local
    lẫn thư mục sao lưu Google Drive trên Colab nếu cần).
    """
    candidate_dirs = [
        ckpt_dir,
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "kfold_seg"),
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        "/content/drive/MyDrive/burn_research_outputs/checkpoints/kfold_seg",
        "/content/drive/MyDrive/burn_research_outputs/checkpoints",
    ]
    for d in candidate_dirs:
        if not d or not os.path.isdir(d):
            continue
        paths = [os.path.join(d, f"{run_prefix}_fold{k}_best.pth") for k in range(1, n_folds + 1)]
        if all(os.path.exists(p) for p in paths):
            return paths

    raise FileNotFoundError(
        f"Không tìm thấy đủ {n_folds} checkpoints '{run_prefix}_fold{{1..{n_folds}}}_best.pth'.\n"
        f"Hãy tải thư mục 'checkpoints/kfold_seg/' từ Google Drive về 'outputs/checkpoints/kfold_seg/' "
        f"hoặc truyền --ckpt-dir trỏ thẳng tới thư mục chứa checkpoints."
    )


def compute_sample_dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple:
    pred_b = pred_bin.astype(bool)
    gt_b = gt_bin.astype(bool)
    inter = float(np.logical_and(pred_b, gt_b).sum())
    pred_sum = float(pred_b.sum())
    gt_sum = float(gt_b.sum())
    union = pred_sum + gt_sum - inter
    if pred_sum + gt_sum == 0:
        return 1.0, 1.0
    dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


def plot_quantitative_spatial_figure(df: pd.DataFrame, stats_summary: Dict[str, Any], out_path: str):
    """Vẽ biểu đồ đôi 300 DPI: (a) Tương quan Bất định biên vs Sai số (1-Dice) & (b) Selective Segmentation Curve."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)

    # --- Panel (a): Boundary Entropy vs Segmentation Error (1 - Dice) ---
    ax1 = axes[0]
    x = df["boundary_entropy"].values
    y = (1.0 - df["dice"].values) * 100.0
    sev = df["severity_degree"].values
    colors = ["#2ca02c" if s == 0 else ("#ff7f0e" if s == 1 else "#d62728") for s in sev]

    ax1.scatter(x, y, c=colors, alpha=0.72, edgecolors="k", linewidths=0.4, s=36)
    if len(x) > 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax1.plot(xs, np.polyval(coef, xs), "b--", lw=1.8, label="Linear Trend")

    rho = stats_summary["spearman_rho_entropy_vs_err"]
    pval = stats_summary["spearman_p_entropy_vs_err"]
    auroc = stats_summary["failure_detection_auroc"]
    ax1.set_title(
        f"(a) Boundary Uncertainty vs. Segmentation Error\n"
        f"Spearman ρ = {rho:.3f} (p = {pval:.2e}) | Failure AUROC = {auroc*100:.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_xlabel("Boundary Predictive Entropy (5-Fold Disagreement)", fontsize=10)
    ax1.set_ylabel("Segmentation Error: 1 - Dice (%)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Chú thích màu độ bỏng
    for lbl, col in [("Degree 1", "#2ca02c"), ("Degree 2", "#ff7f0e"), ("Degree 3", "#d62728")]:
        ax1.scatter([], [], c=col, edgecolors="k", linewidths=0.4, s=36, label=lbl)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # --- Panel (b): Selective Area/Mask Quality Curve ---
    ax2 = axes[1]
    order = np.argsort(df["area_cv"].values)  # Tăng dần từ chắc chắn nhất -> bất định nhất
    dices_sorted = df["dice"].values[order] * 100.0
    mae_sorted = df["abs_area_err_pct"].values[order]

    covs = np.linspace(1.0, 0.50, 26)
    sel_dices, sel_maes = [], []
    for c in covs:
        k = max(10, int(round(c * len(dices_sorted))))
        sel_dices.append(float(np.mean(dices_sorted[:k])))
        sel_maes.append(float(np.mean(mae_sorted[:k])))

    cov_pct = covs * 100.0
    ax2.plot(cov_pct, sel_dices, color="#1f77b4", lw=2.2, label="Mean Sample Dice (%) ↑")
    ax2.axhline(sel_dices[0], color="gray", linestyle="--", lw=1.2, label=f"100% Coverage ({sel_dices[0]:.2f}%)")
    ax2.fill_between(cov_pct, sel_dices[0], sel_dices, color="#1f77b4", alpha=0.14)
    ax2.set_xlim(100, 50)
    ax2.set_xlabel("Coverage (% Cases Retained Before Manual Boundary Review →)", fontsize=10)
    ax2.set_ylabel("Mean Sample-wise Dice (%)", color="#1f77b4", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax2_r = ax2.twinx()
    ax2_r.plot(cov_pct, sel_maes, color="#d62728", linestyle=":", lw=2.0, label="Frame Area MAE (%) ↓")
    ax2_r.set_ylabel("Mean Absolute Area Error (% Frame)", color="#d62728", fontsize=10)
    ax2_r.tick_params(axis="y", labelcolor="#d62728")

    ax2.set_title(
        "(b) Selective TBSA Estimation Curve\n(Filtering High Area-Uncertainty Cases)",
        fontsize=10.5,
        fontweight="bold",
    )
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative_uncertainty_gallery(samples: List[Dict[str, Any]], out_path: str):
    """
    Xuất bảng trực quan 4 hàng x 4 cột (2 ca tốt nhất + 2 ca ranh giới mơ hồ/thất bại):
      [Ảnh gốc RGB | Ground-Truth Mask | 5-Fold Pred Mask + Interval | Pixel Uncertainty Heatmap]
    Phục vụ trực tiếp cả Hướng 1 lẫn Câu 9 Checklist (Phân tích ca lỗi phân đoạn).
    """
    n_rows = len(samples)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(n_rows, 4, figsize=(14.5, 3.4 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(samples):
        rgb = item["rgb_256"]
        gt = item["gt_256"]
        pred = item["pred_256"]
        u_heat = item["u_heatmap"]

        # Overlay GT (Xanh lá) & Pred (Đỏ)
        gt_vis = rgb.copy()
        gt_vis[gt > 0] = (0.55 * gt_vis[gt > 0] + 0.45 * np.array([0, 220, 90])).astype(np.uint8)

        pred_vis = rgb.copy()
        pred_vis[pred > 0] = (0.55 * pred_vis[pred > 0] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{item['tag']}\nID: {item['image_id']} (Deg {item['deg']+1})", fontsize=9.5, fontweight="bold")

        axes[r, 1].imshow(gt_vis)
        axes[r, 1].set_title(f"MobileSAM Reference\nGT Frame Area: {item['gt_area']:.1f}%", fontsize=9.5)

        axes[r, 2].imshow(pred_vis)
        axes[r, 2].set_title(
            f"5-Fold Ensemble (Dice: {item['dice']*100:.1f}%)\n"
            f"Pred Area: {item['pred_area']:.1f}% [{item['low_area']:.1f}–{item['high_area']:.1f}%]",
            fontsize=9.5,
        )

        axes[r, 3].imshow(u_heat)
        axes[r, 3].set_title(
            f"Pixel Uncertainty Heatmap\nBoundary H={item['bnd_ent']:.3f} | Parkland: {item['parkland_str']}",
            fontsize=9.2,
        )

        for c in range(4):
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-Fold Spatial Boundary Uncertainty & Parkland Intervals (Direction 1)")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "ea_unet", "unetplusplus"])
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--loss", type=str, default="combo")
    parser.add_argument("--run-prefix", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--no-tta", action="store_true", help="Tắt 4x TTA khi trích xuất xác suất từng Fold")
    parser.add_argument("--assumed-weight-kg", type=float, default=65.0, help="Cân nặng chuẩn giả định (65kg) để minh họa khoảng dịch Parkland")
    parser.add_argument("--assumed-region-tbsa", type=float, default=18.0, help="Trọng số vùng giải phẫu chuẩn (18% - Thân trước/Chân) để quy đổi TBSA")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_prefix = args.run_prefix or f"kfold_{args.arch}_{args.backbone}_{args.loss}"
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_paths = resolve_kfold_seg_checkpoints(args.ckpt_dir, run_prefix, n_folds=5)
    print("=" * 84)
    print(f"🔬 ĐÁNH GIÁ ĐỘ BẤT ĐỊNH BIÊN TỪNG PIXEL & KHOẢNG PARKLAND ({run_prefix.upper()})")
    print(f"• Thiết bị: {device} | TTA: {'Bật (4x Flip)' if not args.no_tta else 'Tắt'}")
    print("=" * 84)

    # Nạp đồng thời 5 mô hình phân đoạn
    models = []
    for k, p in enumerate(ckpt_paths, start=1):
        m = build_segmentation_model(arch=args.arch, backbone=args.backbone, in_channels=3, classes=1, pretrained=False)
        sd = torch.load(p, map_location=device)
        m.load_state_dict(sd.get("model_state_dict", sd))
        m.to(device).eval()
        models.append(m)
        print(f"  ✅ Đã nạp Fold {k}: {p}")

    csv_full = args.test_csv if os.path.isabs(args.test_csv) else os.path.join(PROJECT_ROOT, args.test_csv)
    test_df = pd.read_csv(csv_full)
    val_tf = get_val_transforms(img_size=args.img_size)

    records = []
    gallery_cache = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = os.path.join(PROJECT_ROOT, row["image_path"])
            msk_path = os.path.join(PROJECT_ROOT, row["mask_path"])
            if not (os.path.exists(img_path) and os.path.exists(msk_path)):
                continue

            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt_raw = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            gt_256 = cv2.resize((gt_raw > 127).astype(np.uint8), (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
            rgb_256 = cv2.resize(rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)

            inp = val_tf(image=rgb)["image"].unsqueeze(0).to(device)
            fold_probs = []
            for m in models:
                if not args.no_tta:
                    logits = forward_seg_tta(m, inp)
                else:
                    out = m(inp)
                    logits = out[0] if isinstance(out, tuple) else out
                prob_np = torch.sigmoid(logits).squeeze().cpu().numpy()
                fold_probs.append(prob_np)

            prob_stack = np.stack(fold_probs, axis=0)  # (5, H, W)
            u_maps = compute_pixel_uncertainty_maps(prob_stack)
            u_sum = summarize_boundary_uncertainty(prob_stack, threshold=0.5)

            pred_bin = (u_maps["mean_prob"] > 0.5).astype(np.uint8)
            dice, iou = compute_sample_dice_iou(pred_bin, gt_256)
            gt_area_pct = float(np.mean(gt_256 > 0) * 100.0)

            sev = int(row["severity_degree"])
            clin = compute_clinical_tbsa_and_parkland_interval(
                prob_stack,
                region_weight=args.assumed_region_tbsa,
                weight_kg=args.assumed_weight_kg,
                predicted_class_id=sev,
            )

            ens_area_pct = clin["pct_frame_ensemble"]
            low_area, high_area = clin["pct_frame_interval"]
            in_interval = bool((low_area - 1.0) <= gt_area_pct <= (high_area + 1.0))

            pk = clin["parkland_interval"]
            pk_str = f"{pk['raw_24h_low_ml']:.0f}–{pk['raw_24h_high_ml']:.0f}ml" if pk else "0ml"

            rec = {
                "image_id": str(row["image_id"]),
                "severity_degree": sev,
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "gt_frame_area_pct": round(gt_area_pct, 2),
                "pred_frame_area_pct": ens_area_pct,
                "pred_area_low_pct": low_area,
                "pred_area_high_pct": high_area,
                "abs_area_err_pct": round(abs(ens_area_pct - gt_area_pct), 2),
                "gt_within_pred_band": in_interval,
                "tbsa_ens_pct": clin["tbsa_ensemble_pct"],
                "tbsa_low_pct": clin["tbsa_interval_pct"][0],
                "tbsa_high_pct": clin["tbsa_interval_pct"][1],
                "borderline_10pct_warning": clin["borderline_10pct_warning"],
                **u_sum,
            }
            records.append(rec)

            gallery_cache.append({
                "image_id": str(row["image_id"]),
                "deg": sev,
                "dice": dice,
                "gt_area": gt_area_pct,
                "pred_area": ens_area_pct,
                "low_area": low_area,
                "high_area": high_area,
                "bnd_ent": u_sum["boundary_entropy"],
                "parkland_str": pk_str,
                "rgb_256": rgb_256,
                "gt_256": gt_256,
                "pred_256": pred_bin,
                "u_heatmap": render_uncertainty_heatmap_rgb(rgb_256, u_maps["predictive_entropy"]),
            })

    res_df = pd.DataFrame(records)
    err_arr = 1.0 - res_df["dice"].values

    rho_ent, p_ent = spearmanr(res_df["boundary_entropy"].values, err_arr)
    rho_cv, p_cv = spearmanr(res_df["area_cv"].values, err_arr)

    # Kiểm định phát hiện ca phân đoạn thất bại (Dice < 0.50) bằng độ bất định diện tích / biên
    failed_mask = (res_df["dice"].values < 0.50).astype(int)
    if 0 < failed_mask.sum() < len(failed_mask):
        u_score = 0.5 * (res_df["boundary_entropy"].values / (res_df["boundary_entropy"].max() + 1e-6)) + \
                  0.5 * (res_df["area_cv"].values / (res_df["area_cv"].max() + 1e-6))
        fail_auroc = float(roc_auc_score(failed_mask, u_score))
        _, mw_p = mannwhitneyu(u_score[failed_mask == 1], u_score[failed_mask == 0], alternative="greater")
    else:
        fail_auroc, mw_p = 0.5, 1.0

    summary = {
        "run_prefix": run_prefix,
        "n_test_images": int(len(res_df)),
        "mean_sample_dice": round(float(res_df["dice"].mean()), 4),
        "mean_abs_frame_area_err_pct": round(float(res_df["abs_area_err_pct"].mean()), 2),
        "mean_fold_area_std_pct": round(float(res_df["fold_area_std_pct"].mean()), 2),
        "mean_boundary_entropy": round(float(res_df["boundary_entropy"].mean()), 4),
        "spearman_rho_entropy_vs_err": round(float(rho_ent), 4),
        "spearman_p_entropy_vs_err": float(p_ent),
        "spearman_rho_areacv_vs_err": round(float(rho_cv), 4),
        "spearman_p_areacv_vs_err": float(p_cv),
        "n_failed_masks_dice_lt_50": int(failed_mask.sum()),
        "failure_detection_auroc": round(fail_auroc, 4),
        "mann_whitney_p_failed_vs_good": float(mw_p),
        "n_borderline_10pct_tbsa_cases": int(res_df["borderline_10pct_warning"].sum()),
    }

    quant_fig = os.path.join(out_dir, f"{run_prefix}_spatial_uncertainty_q1.png")
    qual_fig = os.path.join(out_dir, f"{run_prefix}_spatial_gallery_q1.png")
    csv_out = os.path.join(out_dir, f"{run_prefix}_spatial_per_image.csv")
    json_out = os.path.join(out_dir, f"{run_prefix}_spatial_summary_q1.json")

    plot_quantitative_spatial_figure(res_df, summary, quant_fig)

    # Chọn 2 ca phân đoạn xuất sắc rõ ràng + 2 ca ranh giới mơ hồ/thất bại có độ bất định cao
    sorted_by_dice = sorted(gallery_cache, key=lambda x: x["dice"], reverse=True)
    good_picks = [c for c in sorted_by_dice if c["gt_area"] > 8.0][:2]
    ambig_picks = sorted([c for c in gallery_cache if c["gt_area"] > 5.0], key=lambda x: x["dice"])[:2]
    for i, g in enumerate(good_picks, 1):
        g["tag"] = f"High-Consensus Case #{i}"
    for i, a in enumerate(ambig_picks, 1):
        a["tag"] = f"Ambiguous/Hard Boundary #{i}"
    plot_qualitative_uncertainty_gallery(good_picks + ambig_picks, qual_fig)

    res_df.to_csv(csv_out, index=False, encoding="utf-8")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n📊 KẾT QUẢ THỐNG KÊ ĐỘ BẤT ĐỊNH KHÔNG GIAN (HƯỚNG 1 - Q1/Q2):")
    print(f"  • Số ảnh Test độc lập              : {summary['n_test_images']} ảnh")
    print(f"  • Sai số diện tích khung tuyệt đối : MAE = {summary['mean_abs_frame_area_err_pct']:.2f}% (Độ lệch giữa 5 Fold: ±{summary['mean_fold_area_std_pct']:.2f}%)")
    print(f"  • Tương quan Bất định biên & Lỗi   : Spearman ρ = {summary['spearman_rho_entropy_vs_err']:.4f} (p = {summary['spearman_p_entropy_vs_err']:.3e})")
    print(f"  • Tương quan Biến thiên diện tích  : Spearman ρ = {summary['spearman_rho_areacv_vs_err']:.4f} (p = {summary['spearman_p_areacv_vs_err']:.3e})")
    print(f"  • Phát hiện ca lỗi phân đoạn (<50%): AUROC = {summary['failure_detection_auroc']*100:.2f}% (Mann-Whitney p = {summary['mann_whitney_p_failed_vs_good']:.3e})")
    print(f"  • Ca cắt ngưỡng sốc 10% TBSA       : {summary['n_borderline_10pct_tbsa_cases']} ca được bật cảnh báo an toàn ranh giới")
    print(f"  📈 Biểu đồ định lượng 300 DPI      : {quant_fig}")
    print(f"  🖼️ Gallery trực quan biên & lỗi    : {qual_fig}")
    print(f"  💾 Báo cáo JSON                    : {json_out}")
    print("=" * 84)


if __name__ == "__main__":
    main()

