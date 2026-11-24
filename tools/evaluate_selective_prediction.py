"""
Module: tools/evaluate_selective_prediction.py
Mục đích: Phân tích chuyên sâu Selective Prediction (Dự đoán Có Chọn lọc),
vẽ biểu đồ Risk-Coverage Curve & Phân phối Độ bất định (KDE / Histogram)
kèm kiểm định thống kê phi tham số Mann-Whitney U (p-value) chuẩn bài báo Scopus / ISI Q1/Q2.

Cách chạy (hỗ trợ cả checkpoint EDL cũ dùng softplus lẫn R-EDL mới dùng exp):
    python tools/evaluate_selective_prediction.py \
        --checkpoint outputs/checkpoints/classifier/run_m1_classifier_edl_best.pth \
        --edl-activation softplus

    python tools/evaluate_selective_prediction.py \
        --checkpoint outputs/checkpoints/classifier/run_m1_classifier_redl_exp_best.pth \
        --edl-activation exp
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.vision.models import build_classification_model
from src.vision.train_cls import BurnROIDataset, get_cls_val_transforms
from src.vision.edl_loss import (
    logits_to_evidence,
    get_edl_metrics,
    calculate_ece,
    compute_selective_prediction_metrics,
)


def plot_selective_prediction_figure(
    sel: Dict[str, Any],
    save_path: str,
    title_prefix: str = "EDL Classifier",
) -> None:
    """
    Xuất biểu đồ đôi chuẩn ấn phẩm Q1/Q2 (300 DPI):
    - Trái: Risk-Coverage & Accuracy-Coverage Curve (kèm đường Baseline 100% Coverage và vùng cải thiện).
    - Phải: Phân phối độ bất định nhận thức u giữa nhóm Chẩn đoán Đúng vs Sai + p-value & Error AUROC.
    """
    covs = np.array(sel["curve_coverages"]) * 100.0
    accs = np.array(sel["curve_accuracies"]) * 100.0
    risks = np.array(sel["curve_risks"]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # --- PANEL A: Selective Accuracy & Risk-Coverage Curve ---
    ax1 = axes[0]
    full_acc = accs[-1]
    ax1.plot(covs, accs, color="#1d4ed8", linewidth=2.5, label="EDL Selective Accuracy (%)")
    ax1.axhline(full_acc, color="#64748b", linestyle="--", linewidth=1.6, label=f"Full-Coverage Baseline ({full_acc:.2f}%)")
    ax1.fill_between(covs, full_acc, accs, where=(accs >= full_acc), color="#93c5fd", alpha=0.28, label="Clinical Safety Gain")

    # Đánh dấu các mốc Triage tiêu biểu (80% và 75% Coverage)
    for row in sel["coverage_table"]:
        if row["target_coverage"] in (0.80, 0.75):
            c_pct = row["actual_coverage"] * 100.0
            a_pct = row["accuracy"] * 100.0
            ax1.scatter([c_pct], [a_pct], color="#dc2626", zorder=5, s=45)
            ax1.annotate(
                f"{c_pct:.0f}% Cov -> {a_pct:.1f}% Acc\n(u <= {row['uncertainty_threshold']:.3f})",
                xy=(c_pct, a_pct),
                xytext=(max(15.0, c_pct - 26.0), min(99.0, a_pct + 3.0)),
                fontsize=8.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0),
            )

    ax1_r = ax1.twinx()
    ax1_r.plot(covs, risks, color="#dc2626", linestyle=":", linewidth=2.0, alpha=0.85, label="Clinical Risk / Error Rate (%)")
    ax1_r.set_ylabel("Clinical Risk / Error Rate (%)", color="#dc2626", fontsize=10.5)
    ax1_r.tick_params(axis="y", labelcolor="#dc2626")
    ax1_r.set_ylim(0.0, max(35.0, float(risks.max()) + 5.0))

    ax1.set_xlim(100.0, 20.0)  # Trục chuẩn từ 100% Coverage giảm dần sang trái (lọc dần ca khó)
    ax1.set_ylim(max(50.0, full_acc - 8.0), 100.5)
    ax1.set_xlabel("Coverage (% Patients Auto-Diagnosed by AI -> Clinician Referral Increases ->)", fontsize=10)
    ax1.set_ylabel("Selective Accuracy (%)", color="#1d4ed8", fontsize=10.5)
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.set_title(
        f"(a) Selective Prediction Curve | AURC={sel['aurc']:.4f}, E-AURC={sel['e_aurc']:.4f}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    # --- PANEL B: Uncertainty Distribution (Correct vs Misclassified + Mann-Whitney p-value) ---
    ax2 = axes[1]
    u_corr = np.array(sel["raw_u_correct"])
    u_err = np.array(sel["raw_u_error"])

    bins = np.linspace(
        min(float(u_corr.min()) if len(u_corr) else 0.0, float(u_err.min()) if len(u_err) else 0.0),
        max(float(u_corr.max()) if len(u_corr) else 1.0, float(u_err.max()) if len(u_err) else 1.0),
        18,
    )
    if len(u_corr) > 0:
        ax2.hist(
            u_corr, bins=bins, density=True, alpha=0.55, color="#16a34a", edgecolor="white",
            label=f"Correct (n={sel['n_correct']}, mean u={sel['u_correct_mean']:.3f}±{sel['u_correct_std']:.3f})",
        )
        ax2.axvline(sel["u_correct_mean"], color="#15803d", linestyle="--", linewidth=2.0)
    if len(u_err) > 0:
        ax2.hist(
            u_err, bins=bins, density=True, alpha=0.55, color="#dc2626", edgecolor="white",
            label=f"Misclassified (n={sel['n_error']}, mean u={sel['u_error_mean']:.3f}±{sel['u_error_std']:.3f})",
        )
        ax2.axvline(sel["u_error_mean"], color="#b91c1c", linestyle="--", linewidth=2.0)

    p_val = sel["p_value"]
    p_str = f"p = {p_val:.2e}" if p_val < 1e-3 else f"p = {p_val:.4f}"
    sig_tag = "Statistically Significant" if p_val < 0.05 else "Not Significant"
    stats_box = (
        f"Mann-Whitney U Test (H1: u_err > u_corr)\n"
        f"• U-statistic = {sel['mann_whitney_u']:.1f}\n"
        f"• {p_str} ({sig_tag})\n"
        f"• Error Detection AUROC = {sel['error_detection_auroc']:.4f}"
    )
    ax2.text(
        0.97, 0.95, stats_box,
        transform=ax2.transAxes,
        fontsize=9.2,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )

    ax2.set_title(f"(b) Epistemic Uncertainty Separation — {title_prefix}", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Epistemic Uncertainty (u = K / S)", fontsize=10.5)
    ax2.set_ylabel("Probability Density", fontsize=10.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_selective_triage_report(sel: Dict[str, Any], ece: float = None) -> None:
    p_val = sel["p_value"]
    p_fmt = f"{p_val:.3e}" if p_val < 1e-3 else f"{p_val:.5f}"
    sig_badge = "✅ CÓ Ý NGHĨA THỐNG KÊ MẠNH (p < 0.01)" if p_val < 0.01 else (
        "✅ CÓ Ý NGHĨA THỐNG KÊ (p < 0.05)" if p_val < 0.05 else "⚠️ Chưa đạt ngưỡng p < 0.05"
    )

    print("\n" + "=" * 82)
    print("🔬 PHÂN TÍCH SELECTIVE PREDICTION & KIỂM ĐỊNH THỐNG KÊ Q1/Q2 (EDL TRIAGE)")
    print("=" * 82)
    print(f"• Độ bất định ca ĐÚNG (n={sel['n_correct']:3d}) : {sel['u_correct_mean']:.4f} ± {sel['u_correct_std']:.4f}")
    print(f"• Độ bất định ca SAI  (n={sel['n_error']:3d}) : {sel['u_error_mean']:.4f} ± {sel['u_error_std']:.4f}")
    print(f"• Mann-Whitney U (u_sai > u_đúng) : U = {sel['mann_whitney_u']:.1f} | p-value = {p_fmt} -> {sig_badge}")
    print(f"• Error Detection AUROC           : {sel['error_detection_auroc'] * 100:.2f}%")
    print(f"• Risk-Coverage AURC / E-AURC     : {sel['aurc']:.4f} / {sel['e_aurc']:.4f}")
    if ece is not None:
        print(f"• Expected Calibration Error (ECE): {ece * 100:.2f}%")

    print("\n📊 BẢNG HIỆU NĂNG PHÂN LUỒNG LÂM SÀNG THEO COVERAGE (SELECTIVE CLINICAL TRIAGE):")
    print("┌───────────┬────────────┬─────────────┬───────────┬───────────┬───────────────┬────────────┐")
    print("│ Coverage  │ AI Tự xử lý│ Chuyển Bác sĩ│ Ngưỡng u  │ Accuracy  │ Macro F1      │ Recall Độ 3│")
    print("├───────────┼────────────┼─────────────┼───────────┼───────────┼───────────────┼────────────┤")
    for r in sel["coverage_table"]:
        rec3_str = f"{r['recall_degree_3'] * 100:6.2f}%" if r["recall_degree_3"] is not None else "   N/A "
        print(
            f"│  {r['actual_coverage'] * 100:5.1f}%   │  {r['n_auto_accepted']:3d} ca    │   {r['n_referred_clinician']:3d} ca    │"
            f" <= {r['uncertainty_threshold']:.4f} │  {r['accuracy'] * 100:6.2f}%  │    {r['macro_f1'] * 100:6.2f}%    │  {rec3_str}   │"
        )
    print("└───────────┴────────────┴─────────────┴───────────┴───────────┴───────────────┴────────────┘")


def evaluate_checkpoint(
    checkpoint_path: str,
    csv_path: str = "data/splits/test_roi.csv",
    model_name: str = "resnet34",
    edl_activation: str = "exp",
    img_size: int = 224,
    batch_size: int = 16,
    output_dir: str = "outputs/figures",
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BurnROIDataset(csv_path=csv_path, transform=get_cls_val_transforms(img_size=img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_classification_model(model_name=model_name, num_classes=3, pretrained=False, drop_rate=0.2).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    all_probs, all_u, all_y = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["severity"].to(device)
            logits = model(images)
            evidence = logits_to_evidence(logits, mode=edl_activation)
            prob, u = get_edl_metrics(evidence, num_classes=3)
            all_probs.append(prob.cpu())
            all_u.append(u.cpu())
            all_y.append(targets.cpu())

    probs_t = torch.cat(all_probs, dim=0)
    u_t = torch.cat(all_u, dim=0)
    y_t = torch.cat(all_y, dim=0)

    ece = calculate_ece(probs_t, y_t)
    sel = compute_selective_prediction_metrics(probs_t, u_t, y_t)
    sel["ece"] = round(ece, 4)
    sel["edl_activation"] = edl_activation
    sel["checkpoint"] = checkpoint_path

    os.makedirs(output_dir, exist_ok=True)
    run_stem = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("_best", "")
    fig_path = os.path.join(output_dir, f"{run_stem}_selective_triage_q1.png")
    json_path = os.path.join(output_dir, f"{run_stem}_selective_triage_q1.json")

    plot_selective_prediction_figure(sel, fig_path, title_prefix=run_stem)
    save_payload = {k: v for k, v in sel.items() if not k.startswith("raw_")}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=4, ensure_ascii=False)

    print_selective_triage_report(sel, ece=ece)
    print(f"\n📈 Biểu đồ Risk-Coverage & Phân phối u (300 DPI): {fig_path}")
    print(f"💾 Số liệu thống kê JSON chuẩn Q1/Q2            : {json_path}")
    return sel


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Selective Prediction & Mann-Whitney p-value cho EDL Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn tới file .pth của mô hình EDL")
    parser.add_argument("--csv", type=str, default="data/splits/test_roi.csv", help="File CSV tập đánh giá")
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34", "convnext_tiny"])
    parser.add_argument("--edl-activation", "--edl_activation", type=str, default="exp", choices=["exp", "softplus"],
                        help="Dùng 'softplus' cho checkpoint M1 cũ hoặc 'exp' cho R-EDL mới")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        model_name=args.model,
        edl_activation=args.edl_activation,
        img_size=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
