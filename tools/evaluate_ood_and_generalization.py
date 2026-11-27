"""
Module: tools/evaluate_ood_and_generalization.py
Mục đích (HƯỚNG 4A + Checklist Câu 9 & Câu 10 - Q1/Q2 Upgrade):
  Đánh giá toàn diện khả năng Tổng quát hóa thực tế (Generalization), Chặn ảnh lạ ngoài miền (OOD Safety Rejection),
  và Phân tích trực quan các ca thất bại (Failure Case Analysis) cho 5-Fold Asymmetric R-EDL (đối chứng 5-Fold Weighted CE):
    1) OOD Non-Burn Rejection Test:
       - Tự động trích xuất vùng Da lành & Nền bệnh viện thật (0% pixel bỏng) từ ảnh Test độc lập.
       - Hỗ trợ nạp thêm thư mục ảnh ngoài đời (--ood-dir: vết bầm, chàm eczema, nốt ruồi ISIC, ảnh thường).
       - Đo OOD Detection AUROC, AUPR, Tỷ lệ từ chối an toàn (Safety Abstention Rate) & Tỷ lệ đoán bừa tự tin cao.
    2) Real-World Camera Shift Robustness (Benchmark nhiễu chụp điện thoại thực tế):
       - Kiểm tra trên Lệch ánh sáng đèn vàng/thiếu sáng, Rung mờ tay chụp (Blur), và Nén ảnh qua mạng (JPEG).
    3) External Burn Dataset Generalization (--external-burn-dir):
       - Chấm điểm Zero-shot trên tập ảnh bỏng ngoại viện (có nhãn degree_1/2/3 hoặc không nhãn).
    4) Failure Case Visual Analysis (Câu 9 Checklist):
       - Phân tích ma trận lỗi, tỷ lệ ca sai được cổng R-EDL cứu (Caught Errors) và xuất bảng ảnh ROI + Grad-CAM 300 DPI.

Cách chạy (0 phút train lại):
  python tools/evaluate_ood_and_generalization.py
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, accuracy_score
from scipy.stats import mannwhitneyu

from src.vision.models import build_classification_model
from src.vision.train_cls import get_cls_val_transforms
from src.vision.edl_loss import logits_to_evidence, get_edl_metrics, calculate_ece
from src.utils.gradcam import GradCAM, overlay_cam_on_image


CLASS_LABELS_SHORT = ["Deg 1", "Deg 2", "Deg 3"]


def resolve_kfold_cls_checkpoints(
    run_prefix: str,
    ckpt_dir: Optional[str] = None,
    n_folds: int = 5,
) -> Optional[List[str]]:
    """Tìm đủ 5 file trọng số k-fold phân loại (hỗ trợ cả local và Google Drive trên Colab)."""
    candidate_dirs = [
        ckpt_dir,
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "kfold_cls"),
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "classifier"),
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints"),
        "/content/drive/MyDrive/burn_research_outputs/checkpoints/kfold_cls",
        "/content/drive/MyDrive/burn_research_outputs/checkpoints",
    ]
    for d in candidate_dirs:
        if not d or not os.path.isdir(d):
            continue
        paths = [os.path.join(d, f"{run_prefix}_fold{k}_best.pth") for k in range(1, n_folds + 1)]
        if all(os.path.exists(p) for p in paths):
            return paths
    return None


def load_ensemble_models(ckpt_paths: List[str], model_name: str, device: torch.device) -> List[torch.nn.Module]:
    models = []
    for p in ckpt_paths:
        m = build_classification_model(model_name=model_name, num_classes=3, pretrained=False, drop_rate=0.0)
        sd = torch.load(p, map_location=device)
        m.load_state_dict(sd.get("model_state_dict", sd))
        m.to(device).eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_rgb_batch_ensemble(
    models: List[torch.nn.Module],
    rgb_list: List[np.ndarray],
    val_tf,
    device: torch.device,
    is_edl: bool = True,
    edl_activation: str = "exp",
    deg3_cost_c3: float = 1.0,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Dự đoán 5-Fold Ensemble cho danh sách ảnh RGB:
      - Với R-EDL: Hợp nhất bằng chứng trung bình (Evidential Fusion) -> probs (N,3) & epistemic u = K/S (N,).
      - Với Weighted CE: Hợp nhất xác suất Softmax trung bình -> probs (N,3) & u = 1 - max(p) (N,).
    """
    if len(rgb_list) == 0:
        return {"probs": np.zeros((0, 3)), "uncertainty": np.zeros((0,)), "preds": np.zeros((0,), dtype=int)}

    tensors = [val_tf(image=img)["image"] for img in rgb_list]
    all_x = torch.stack(tensors, dim=0)

    sum_ev_or_prob = None
    for m in models:
        chunks = []
        for i in range(0, len(all_x), batch_size):
            bx = all_x[i : i + batch_size].to(device)
            logits = m(bx)
            if is_edl:
                ev = logits_to_evidence(logits, mode=edl_activation).cpu()
                chunks.append(ev)
            else:
                pr = F.softmax(logits, dim=-1).cpu()
                chunks.append(pr)
        fold_out = torch.cat(chunks, dim=0)
        sum_ev_or_prob = fold_out if sum_ev_or_prob is None else (sum_ev_or_prob + fold_out)

    mean_ev_or_prob = sum_ev_or_prob / float(len(models))
    if is_edl:
        probs_t, u_t = get_edl_metrics(mean_ev_or_prob, num_classes=3)
    else:
        probs_t = mean_ev_or_prob
        u_t = 1.0 - probs_t.max(dim=-1).values

    probs_np = probs_t.numpy()
    if abs(deg3_cost_c3 - 1.0) > 1e-5:
        adj = probs_np.copy()
        adj[:, 2] *= float(deg3_cost_c3)
        adj = adj / np.clip(adj.sum(axis=-1, keepdims=True), 1e-8, None)
        preds_np = np.argmax(adj, axis=-1)
    else:
        preds_np = np.argmax(probs_np, axis=-1)

    return {
        "probs": probs_np,
        "uncertainty": u_t.numpy(),
        "preds": preds_np,
        "max_conf": np.max(probs_np, axis=-1),
    }


def extract_real_nonburn_patches_from_test(test_seg_csv: str, patch_size: int = 224, max_patches: int = 150) -> List[np.ndarray]:
    """
    Trích xuất tự động các vùng ảnh THẬT 100% KHÔNG CÓ VẾT BỎNG (da người bình thường, quần áo, nền phòng khám)
    từ tập Test độc lập, cách xa ranh giới mask vết bỏng tối thiểu 25 pixel (0% chồng lấp vết bỏng).
    """
    df = pd.read_csv(test_seg_csv)
    patches: List[np.ndarray] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

    for _, row in df.iterrows():
        if len(patches) >= max_patches:
            break
        img_p = os.path.join(PROJECT_ROOT, row["image_path"])
        msk_p = os.path.join(PROJECT_ROOT, row["mask_path"])
        if not (os.path.exists(img_p) and os.path.exists(msk_p)):
            continue

        bgr = cv2.imread(img_p)
        msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        safe_bg = (cv2.dilate((msk > 127).astype(np.uint8), kernel, iterations=1) == 0)

        # Quét lưới tìm mảnh ảnh vuông hoàn toàn sạch bỏng (100% safe_bg) và không phải viền đen/trắng rỗng
        win = min(h, w, max(96, min(h, w) // 2))
        if win < 64:
            continue

        found = False
        for y in range(0, h - win + 1, max(16, win // 2)):
            for x in range(0, w - win + 1, max(16, win // 2)):
                sub_safe = safe_bg[y : y + win, x : x + win]
                if np.all(sub_safe):
                    crop = rgb[y : y + win, x : x + win]
                    # Loại bỏ viền padding đơn sắc vô nghĩa (đảm bảo là da lành/vật thể thật có chi tiết)
                    if float(np.std(crop)) > 14.0 and 25.0 < float(np.mean(crop)) < 235.0:
                        patches.append(cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA))
                        found = True
                        break
            if found:
                break

    return patches


def apply_realworld_shift(img_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Mô phỏng các điều kiện chụp ảnh điện thoại thực tế ngoài phòng cấp cứu (Benchmark chuẩn Hendrycks ImageNet-C)."""
    if mode == "clean":
        return img_rgb
    if mode == "dim_warm_light":
        # Đèn vàng phòng tối / thiếu sáng cấp cứu (Gamma 1.45 + ám vàng nhẹ)
        norm = np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        gamma_img = np.power(norm, 1.45) * 255.0
        gamma_img[..., 0] = np.clip(gamma_img[..., 0] * 1.08, 0, 255)  # Red ấm
        gamma_img[..., 2] = np.clip(gamma_img[..., 2] * 0.86, 0, 255)  # Giảm Blue
        return gamma_img.astype(np.uint8)
    if mode == "motion_defocus_blur":
        # Tay bệnh nhân rung / mất nét camera cận cảnh
        return cv2.GaussianBlur(img_rgb, (7, 7), sigmaX=2.2)
    if mode == "messenger_jpeg":
        # Nén ảnh mạnh khi gửi qua ứng dụng nhắn tin từ tuyến dưới
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 22])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return img_rgb
    raise ValueError(f"Chế độ nhiễu không hợp lệ: {mode}")


def load_images_from_folder(folder: str, max_imgs: int = 300) -> List[Tuple[str, np.ndarray]]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[str] = []
    for ex in exts:
        files.extend(glob.glob(os.path.join(folder, "**", ex), recursive=True))
    files = sorted(set(files))[:max_imgs]
    out = []
    for f in files:
        bgr = cv2.imread(f)
        if bgr is not None:
            out.append((f, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


def plot_ood_and_generalization_figure(
    id_u_edl: np.ndarray,
    ood_u_edl: np.ndarray,
    tau_edl: float,
    ood_metrics: Dict[str, Any],
    shift_records: List[Dict[str, Any]],
    out_path: str,
):
    """Vẽ biểu đồ 3-Panel 300 DPI chuẩn Q1/Q2 cho Hướng 4A (OOD Separation + Model Comparison + Shift Generalization)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)

    # --- Panel (a): Phân phối độ bất định u của R-EDL (In-Distribution Burn vs Real Non-Burn OOD) ---
    ax1 = axes[0]
    bins = np.linspace(0.0, max(float(np.max(ood_u_edl)), float(np.max(id_u_edl)), 0.25), 24)
    ax1.hist(id_u_edl, bins=bins, density=True, alpha=0.62, color="#1f77b4", label=f"ID Burn Test (n={len(id_u_edl)})")
    ax1.hist(ood_u_edl, bins=bins, density=True, alpha=0.62, color="#d62728", label=f"Non-Burn OOD (n={len(ood_u_edl)})")
    ax1.axvline(tau_edl, color="black", linestyle="--", lw=1.8, label=f"Safety Gate τ = {tau_edl:.3f}")
    ax1.set_title(
        f"(a) R-EDL Epistemic Uncertainty on OOD\n"
        f"OOD AUROC = {ood_metrics['edl_ood_auroc']*100:.1f}% | Rejected = {ood_metrics['edl_ood_reject_rate']*100:.1f}%",
        fontsize=10.2,
        fontweight="bold",
    )
    ax1.set_xlabel("Epistemic Uncertainty (u = K / S)", fontsize=9.8)
    ax1.set_ylabel("Probability Density", fontsize=9.8)
    ax1.legend(fontsize=8.3, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- Panel (b): Đối chứng OOD Safety R-EDL vs Weighted CE ---
    ax2 = axes[1]
    if ood_metrics.get("has_wce_comparison", False):
        metrics_names = ["OOD AUROC\n(Higher ↑)", "OOD Safety Reject\n(Higher ↑)", "Overconf. False Burn\n(Lower ↓)"]
        edl_vals = [
            ood_metrics["edl_ood_auroc"] * 100.0,
            ood_metrics["edl_ood_reject_rate"] * 100.0,
            ood_metrics["edl_false_overconf_80_rate"] * 100.0,
        ]
        wce_vals = [
            ood_metrics["wce_ood_auroc"] * 100.0,
            ood_metrics["wce_ood_reject_rate"] * 100.0,
            ood_metrics["wce_false_overconf_80_rate"] * 100.0,
        ]
        x = np.arange(len(metrics_names))
        w = 0.34
        b1 = ax2.bar(x - w / 2, wce_vals, w, label="5-Fold Weighted CE", color="#8c8c8c", edgecolor="k", linewidth=0.5)
        b2 = ax2.bar(x + w / 2, edl_vals, w, label="5-Fold Asymmetric R-EDL", color="#2ca02c", edgecolor="k", linewidth=0.5)
        for bars in (b1, b2):
            for bar in bars:
                h_val = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h_val + 1.2, f"{h_val:.1f}%", ha="center", va="bottom", fontsize=8.2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, fontsize=8.8)
        ax2.set_ylim(0, 112)
        ax2.set_ylabel("Percentage (%)", fontsize=9.8)
        ax2.set_title("(b) Non-Burn Safety Stress Test\n(R-EDL vs. Weighted CE Softmax)", fontsize=10.2, fontweight="bold")
        ax2.legend(fontsize=8.5, frameon=True)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "R-EDL Standalone OOD Mode\n(Weighted CE checkpoints optional)", ha="center", va="center")
        ax2.axis("off")

    # --- Panel (c): Độ bền Tổng quát hóa dưới Nhiễu thực tế (Full vs Selective Triage) ---
    ax3 = axes[2]
    s_names = [r["label_short"] for r in shift_records]
    full_accs = [r["full_acc"] * 100.0 for r in shift_records]
    sel_accs = [r["selective_acc"] * 100.0 for r in shift_records]
    mean_us = [r["mean_u"] * 100.0 for r in shift_records]

    x3 = np.arange(len(s_names))
    w3 = 0.34
    ax3.bar(x3 - w3 / 2, full_accs, w3, label="100% Auto-Diagnosis Acc", color="#4c78a8", edgecolor="k", linewidth=0.5)
    ax3.bar(x3 + w3 / 2, sel_accs, w3, label="Selective Triage Acc (u ≤ τ)", color="#f58518", edgecolor="k", linewidth=0.5)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(s_names, fontsize=8.8)
    ax3.set_ylim(50, 96)
    ax3.set_ylabel("Classification Accuracy (%)", fontsize=9.8)
    ax3.set_title("(c) Generalization Under Real-World Shifts\n(Accuracy & Safety Triage Gain)", fontsize=10.2, fontweight="bold")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax3_r = ax3.twinx()
    ax3_r.plot(x3, mean_us, "r-o", lw=1.8, markersize=5, label="Mean Uncertainty u (×100)")
    ax3_r.set_ylabel("Mean Uncertainty u (×100)", color="#d62728", fontsize=9.2)
    ax3_r.tick_params(axis="y", labelcolor="#d62728")

    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3_r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8.0, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_failure_case_gallery(
    edl_models: List[torch.nn.Module],
    test_rois: List[np.ndarray],
    test_ids: List[str],
    y_true: np.ndarray,
    edl_res: Dict[str, np.ndarray],
    tau_edl: float,
    val_tf,
    device: torch.device,
    out_path: str,
) -> Dict[str, Any]:
    """
    Phân tích toàn bộ các ca dự đoán sai trên tập Test độc lập (phục vụ Câu 9 Checklist):
      - Thống kê cặp nhầm lẫn phổ biến nhất (Độ 1 <-> Độ 2 vs Độ 2 <-> Độ 3).
      - Tính tỷ lệ ca đoán sai ĐÃ ĐƯỢC CỔNG BẤT ĐỊNH BẮT GIỮ (Caught by Triage Gate u > tau).
      - Xuất hình 6 ca lỗi điển hình kèm bản đồ nhiệt Grad-CAM giải thích lý do mô hình nhầm lẫn.
    """
    preds = edl_res["preds"]
    probs = edl_res["probs"]
    u_vals = edl_res["uncertainty"]

    wrong_idx = np.where(preds != y_true)[0]
    caught_mask = u_vals[wrong_idx] > tau_edl if len(wrong_idx) > 0 else np.array([], dtype=bool)
    n_wrong = int(len(wrong_idx))
    n_caught = int(np.sum(caught_mask))

    pair_counts: Dict[str, int] = {}
    for idx in wrong_idx:
        key = f"True {CLASS_LABELS_SHORT[int(y_true[idx])]} -> Pred {CLASS_LABELS_SHORT[int(preds[idx])]}"
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # Chọn tối đa 3 ca lỗi đã được cổng u > tau bắt giữ thành công + 3 ca lỗi khó có u thấp để phân tích trung thực
    caught_indices = [int(i) for i in wrong_idx if u_vals[i] > tau_edl]
    silent_indices = [int(i) for i in wrong_idx if u_vals[i] <= tau_edl]
    caught_indices = sorted(caught_indices, key=lambda i: u_vals[i], reverse=True)[:3]
    silent_indices = sorted(silent_indices, key=lambda i: probs[i, preds[i]], reverse=True)[:3]
    selected = caught_indices + silent_indices

    if len(selected) > 0:
        n_sel = len(selected)
        fig, axes = plt.subplots(n_sel, 3, figsize=(11.5, 2.9 * n_sel), dpi=300)
        if n_sel == 1:
            axes = np.expand_dims(axes, axis=0)

        rep_model = edl_models[0]
        target_layer = rep_model.layer4[-1] if hasattr(rep_model, "layer4") else None

        for r, idx in enumerate(selected):
            roi_rgb = cv2.resize(test_rois[idx], (224, 224), interpolation=cv2.INTER_AREA)
            pred_c = int(preds[idx])
            true_c = int(y_true[idx])
            u_val = float(u_vals[idx])
            is_caught = u_val > tau_edl

            # Grad-CAM trên mô hình đại diện
            cam_vis = roi_rgb.copy()
            if target_layer is not None:
                try:
                    cam_ext = GradCAM(rep_model, target_layer)
                    inp_t = val_tf(image=roi_rgb)["image"].unsqueeze(0).to(device)
                    inp_t.requires_grad_(True)
                    hmap = cam_ext.generate_heatmap(inp_t, class_idx=pred_c)
                    cam_vis = overlay_cam_on_image(roi_rgb, hmap, alpha=0.48)
                    cam_ext.remove_hooks()
                except Exception:
                    pass

            axes[r, 0].imshow(roi_rgb)
            axes[r, 0].set_title(
                f"ID: {test_ids[idx]} | True: {CLASS_LABELS_SHORT[true_c]}\nPred: {CLASS_LABELS_SHORT[pred_c]}",
                fontsize=9.5,
                fontweight="bold",
            )
            axes[r, 0].axis("off")

            axes[r, 1].imshow(cam_vis)
            axes[r, 1].set_title("Grad-CAM Visual Focus\n(Error Region Inspection)", fontsize=9.5)
            axes[r, 1].axis("off")

            # Cột 3: Biểu đồ ngang xác suất Dirichlet + trạng thái cổng an toàn
            ax_bar = axes[r, 2]
            bar_cols = ["#2ca02c" if c == true_c else ("#d62728" if c == pred_c else "#aec7e8") for c in range(3)]
            y_pos = np.arange(3)
            p_pct = probs[idx] * 100.0
            ax_bar.barh(y_pos, p_pct, color=bar_cols, edgecolor="k", height=0.52)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(CLASS_LABELS_SHORT, fontsize=9)
            ax_bar.set_xlim(0, 108)
            for c_i, v_p in enumerate(p_pct):
                ax_bar.text(v_p + 1.5, c_i, f"{v_p:.1f}%", va="center", fontsize=8.5)
            status_txt = "✅ CAUGHT BY R-EDL GATE (Referred to MD)" if is_caught else "⚠️ SILENT ERROR (Borderline Texture)"
            status_col = "#1b7837" if is_caught else "#b2182b"
            ax_bar.set_title(f"u = {u_val:.3f} (Gate τ = {tau_edl:.3f})\n{status_txt}", fontsize=9.0, color=status_col, fontweight="bold")
            ax_bar.set_xlabel("Dirichlet Projected Prob (%)", fontsize=8.5)
            ax_bar.grid(True, axis="x", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "total_misclassified": n_wrong,
        "caught_by_safety_gate_at_85cov": n_caught,
        "safety_catch_rate_of_errors": round(float(n_caught / max(1, n_wrong)), 4),
        "confusion_pair_breakdown": dict(sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "gallery_path": out_path if len(selected) > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD Safety Rejection, Shift Generalization & Failure Cases (Direction 4A)")
    parser.add_argument("--edl-prefix", type=str, default="kfold_cls_resnet34_edl")
    parser.add_argument("--wce-prefix", type=str, default="kfold_cls_resnet34_weighted_ce")
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--edl-activation", type=str, default="exp", choices=["exp", "softplus"])
    parser.add_argument("--target-coverage", type=float, default=0.85, help="Mức Coverage chuẩn lâm sàng (0.85 = AI xử lý 85%, chuyển 15% ca khó)")
    parser.add_argument("--ood-dir", type=str, default="data/external_ood/non_burn", help="Thư mục tùy chọn chứa ảnh ngoài đời không phải bỏng")
    parser.add_argument("--external-burn-dir", type=str, default="data/external_ood/external_burns", help="Thư mục tùy chọn chứa ảnh bỏng nguồn ngoài")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    edl_ckpts = resolve_kfold_cls_checkpoints(args.edl_prefix, ckpt_dir=args.ckpt_dir, n_folds=5)
    if edl_ckpts is None:
        raise FileNotFoundError(
            f"Không tìm thấy đủ 5 checkpoints '{args.edl_prefix}_fold{{1..5}}_best.pth'.\n"
            f"Hãy tải 'checkpoints/kfold_cls/' từ Google Drive về 'outputs/checkpoints/kfold_cls/' hoặc truyền --ckpt-dir."
        )
    wce_ckpts = resolve_kfold_cls_checkpoints(args.wce_prefix, ckpt_dir=args.ckpt_dir, n_folds=5)

    print("=" * 86)
    print("🛡️ ĐÁNH GIÁ TỔNG QUÁT HÓA THỰC TẾ, CHẶN ẢNH LẠ OOD & PHÂN TÍCH CA LỖI (HƯỚNG 4A)")
    print(f"• Thiết bị: {device} | Target Triage Coverage: {args.target_coverage*100:.0f}%")
    print("=" * 86)

    edl_models = load_ensemble_models(edl_ckpts, "resnet34", device)
    print(f"  ✅ Đã nạp 5-Fold Asymmetric R-EDL ({len(edl_models)} models)")
    wce_models = load_ensemble_models(wce_ckpts, "resnet34", device) if wce_ckpts else None
    if wce_models:
        print(f"  ✅ Đã nạp 5-Fold Weighted CE đối chứng ({len(wce_models)} models)")

    val_tf = get_cls_val_transforms(img_size=224)

    # 1. Khóa ngưỡng Triage tau trên tập Validation gốc (Zero Test/OOD Leakage!)
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "val_roi.csv"))
    val_imgs = []
    for _, r in val_df.iterrows():
        p = os.path.join(PROJECT_ROOT, r["roi_path"])
        if os.path.exists(p):
            val_imgs.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    val_edl = predict_rgb_batch_ensemble(edl_models, val_imgs, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
    tau_edl = float(np.quantile(val_edl["uncertainty"], args.target_coverage))

    tau_wce = None
    if wce_models:
        val_wce = predict_rgb_batch_ensemble(wce_models, val_imgs, val_tf, device, is_edl=False)
        tau_wce = float(np.quantile(val_wce["uncertainty"], args.target_coverage))

    # 2. Đọc 187 ảnh Test ROI độc lập
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test_roi.csv"))
    test_rois, test_ids, test_y_list = [], [], []
    for _, r in test_df.iterrows():
        p = os.path.join(PROJECT_ROOT, r["roi_path"])
        if os.path.exists(p):
            test_rois.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            test_ids.append(str(r["image_id"]))
            test_y_list.append(int(r["severity_degree"]))
    test_y = np.array(test_y_list, dtype=int)

    id_edl = predict_rgb_batch_ensemble(edl_models, test_rois, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
    id_wce = predict_rgb_batch_ensemble(wce_models, test_rois, val_tf, device, is_edl=False) if wce_models else None

    # 3. SUITE 1: OOD NON-BURN REJECTION TEST (Tách rõ Da lành bệnh nhân vs Bệnh da liễu ISIC ngoại viện)
    healthy_patches = extract_real_nonburn_patches_from_test(os.path.join(PROJECT_ROOT, "data", "splits", "test.csv"), max_patches=150)
    ood_patches = list(healthy_patches)
    ood_source_note = f"{len(healthy_patches)} real healthy-skin/background patches"

    ext_ood_imgs: List[np.ndarray] = []
    ext_ood_path = args.ood_dir if os.path.isabs(args.ood_dir) else os.path.join(PROJECT_ROOT, args.ood_dir)
    if os.path.isdir(ext_ood_path):
        ext_ood_items = load_images_from_folder(ext_ood_path, max_imgs=250)
        if len(ext_ood_items) > 0:
            ext_ood_imgs = [cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA) for _, im in ext_ood_items]
            ood_patches.extend(ext_ood_imgs)
            ood_source_note += f" + {len(ext_ood_imgs)} external dermatology (ISIC) non-burn lesions"

    ood_edl = predict_rgb_batch_ensemble(edl_models, ood_patches, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
    y_ood_bin = np.concatenate([np.zeros(len(id_edl["uncertainty"])), np.ones(len(ood_edl["uncertainty"]))])
    scores_edl = np.concatenate([id_edl["uncertainty"], ood_edl["uncertainty"]])

    edl_ood_auroc = float(roc_auc_score(y_ood_bin, scores_edl))
    edl_ood_aupr = float(average_precision_score(y_ood_bin, scores_edl))
    _, ood_mw_p = mannwhitneyu(ood_edl["uncertainty"], id_edl["uncertainty"], alternative="greater")

    # Đánh giá riêng trên tập Bệnh da liễu ngoại viện ISIC (nếu có)
    isic_sub_metrics = None
    if len(ext_ood_imgs) > 0:
        isic_edl = predict_rgb_batch_ensemble(edl_models, ext_ood_imgs, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
        y_isic = np.concatenate([np.zeros(len(id_edl["uncertainty"])), np.ones(len(isic_edl["uncertainty"]))])
        s_isic = np.concatenate([id_edl["uncertainty"], isic_edl["uncertainty"]])
        isic_sub_metrics = {
            "n_isic_lesions": len(ext_ood_imgs),
            "edl_isic_auroc": round(float(roc_auc_score(y_isic, s_isic)), 4),
            "edl_isic_reject_rate": round(float(np.mean(isic_edl["uncertainty"] > tau_edl)), 4),
            "edl_isic_false_overconf_80": round(float(np.mean(isic_edl["max_conf"] >= 0.80)), 4),
        }
        if wce_models is not None and id_wce is not None and tau_wce is not None:
            isic_wce = predict_rgb_batch_ensemble(wce_models, ext_ood_imgs, val_tf, device, is_edl=False)
            sw_isic = np.concatenate([id_wce["uncertainty"], isic_wce["uncertainty"]])
            isic_sub_metrics.update({
                "wce_isic_auroc": round(float(roc_auc_score(y_isic, sw_isic)), 4),
                "wce_isic_reject_rate": round(float(np.mean(isic_wce["uncertainty"] > tau_wce)), 4),
                "wce_isic_false_overconf_80": round(float(np.mean(isic_wce["max_conf"] >= 0.80)), 4),
            })

    ood_metrics: Dict[str, Any] = {
        "ood_sample_count": int(len(ood_patches)),
        "ood_sources": ood_source_note,
        "val_locked_tau_edl_85cov": round(tau_edl, 4),
        "id_mean_u_edl": round(float(np.mean(id_edl["uncertainty"])), 4),
        "ood_mean_u_edl": round(float(np.mean(ood_edl["uncertainty"])), 4),
        "mann_whitney_p_ood_gt_id": float(ood_mw_p),
        "edl_ood_auroc": round(edl_ood_auroc, 4),
        "edl_ood_aupr": round(edl_ood_aupr, 4),
        "edl_ood_reject_rate": round(float(np.mean(ood_edl["uncertainty"] > tau_edl)), 4),
        "edl_false_overconf_80_rate": round(float(np.mean(ood_edl["max_conf"] >= 0.80)), 4),
        "isic_dermatology_ood_subset": isic_sub_metrics,
        "has_wce_comparison": bool(wce_models is not None),
    }

    if wce_models is not None and id_wce is not None and tau_wce is not None:
        ood_wce = predict_rgb_batch_ensemble(wce_models, ood_patches, val_tf, device, is_edl=False)
        scores_wce = np.concatenate([id_wce["uncertainty"], ood_wce["uncertainty"]])
        ood_metrics.update({
            "val_locked_tau_wce_85cov": round(tau_wce, 4),
            "wce_ood_auroc": round(float(roc_auc_score(y_ood_bin, scores_wce)), 4),
            "wce_ood_aupr": round(float(average_precision_score(y_ood_bin, scores_wce)), 4),
            "wce_ood_reject_rate": round(float(np.mean(ood_wce["uncertainty"] > tau_wce)), 4),
            "wce_false_overconf_80_rate": round(float(np.mean(ood_wce["max_conf"] >= 0.80)), 4),
        })

    # 4. SUITE 2: REAL-WORLD CAMERA SHIFT GENERALIZATION
    shift_specs = [
        ("clean", "Clean ID\n(Standard)"),
        ("dim_warm_light", "Dim/Warm\nLighting"),
        ("motion_defocus_blur", "Handheld\nBlur"),
        ("messenger_jpeg", "Messenger\nJPEG"),
    ]
    shift_records = []
    for s_key, s_lbl in shift_specs:
        shifted_imgs = [apply_realworld_shift(im, s_key) for im in test_rois]
        s_res = predict_rgb_batch_ensemble(edl_models, shifted_imgs, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
        full_acc = float(accuracy_score(test_y, s_res["preds"]))
        full_f1 = float(f1_score(test_y, s_res["preds"], average="macro", zero_division=0))
        full_rec3 = float(recall_score(test_y, s_res["preds"], labels=[0, 1, 2], average=None, zero_division=0)[2])

        keep = s_res["uncertainty"] <= tau_edl
        if int(np.sum(keep)) >= 5:
            sel_acc = float(accuracy_score(test_y[keep], s_res["preds"][keep]))
            sel_f1 = float(f1_score(test_y[keep], s_res["preds"][keep], average="macro", zero_division=0))
        else:
            sel_acc, sel_f1 = full_acc, full_f1

        shift_records.append({
            "shift_mode": s_key,
            "label_short": s_lbl,
            "full_acc": round(full_acc, 4),
            "full_macro_f1": round(full_f1, 4),
            "full_recall_deg3": round(full_rec3, 4),
            "mean_u": round(float(np.mean(s_res["uncertainty"])), 4),
            "retained_coverage": round(float(np.mean(keep)), 4),
            "selective_acc": round(sel_acc, 4),
            "selective_macro_f1": round(sel_f1, 4),
        })

    # 5. SUITE 3: OPTIONAL EXTERNAL BURN DATASET GENERALIZATION
    external_burn_summary = None
    ext_burn_path = args.external_burn_dir if os.path.isabs(args.external_burn_dir) else os.path.join(PROJECT_ROOT, args.external_burn_dir)
    if os.path.isdir(ext_burn_path):
        labeled_imgs, labeled_y = [], []
        class_dir_map = {"degree_1": 0, "deg1": 0, "0": 0, "degree_2": 1, "deg2": 1, "1": 1, "degree_3": 2, "deg3": 2, "2": 2}
        for sub, cid in class_dir_map.items():
            sub_p = os.path.join(ext_burn_path, sub)
            if os.path.isdir(sub_p):
                items = load_images_from_folder(sub_p, max_imgs=300)
                for _, im in items:
                    labeled_imgs.append(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA))
                    labeled_y.append(cid)

        if len(labeled_imgs) > 0:
            ext_y = np.array(labeled_y, dtype=int)
            ext_res = predict_rgb_batch_ensemble(edl_models, labeled_imgs, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
            keep_ext = ext_res["uncertainty"] <= tau_edl
            rec_per_cls = recall_score(ext_y, ext_res["preds"], labels=[0, 1, 2], average=None, zero_division=0)
            # Xác suất phân biệt nhị phân Bỏng Độ 2 (Dermal) vs Bỏng Độ 3 (Full-thickness) trên tập viện chuyên sâu như BIP_US
            present_classes = sorted(set(ext_y.tolist()))
            binary_2v3_acc = None
            if present_classes == [1, 2]:
                p23_preds = np.where(ext_res["probs"][:, 2] >= ext_res["probs"][:, 1], 2, 1)
                binary_2v3_acc = round(float(accuracy_score(ext_y, p23_preds)), 4)

            external_burn_summary = {
                "mode": "labeled_external_cohort (e.g. BIP_US Hospital Spain)",
                "n_images": int(len(labeled_imgs)),
                "present_classes": present_classes,
                "external_3class_accuracy": round(float(accuracy_score(ext_y, ext_res["preds"])), 4),
                "external_deg2_vs_deg3_accuracy": binary_2v3_acc,
                "external_recall_deg2": round(float(rec_per_cls[1]), 4),
                "external_recall_deg3_full_thickness": round(float(rec_per_cls[2]), 4),
                "external_ece": round(float(calculate_ece(torch.tensor(ext_res["probs"]), torch.tensor(ext_y))), 4),
                "external_mean_u": round(float(np.mean(ext_res["uncertainty"])), 4),
                "id_vs_external_shift_detected_ratio": round(float(np.mean(ext_res["uncertainty"] > tau_edl)), 4),
                "selective_retained_ratio": round(float(np.mean(keep_ext)), 4),
                "selective_3class_accuracy": round(float(accuracy_score(ext_y[keep_ext], ext_res["preds"][keep_ext])), 4) if int(np.sum(keep_ext)) > 0 else None,
            }
        else:
            raw_items = load_images_from_folder(ext_burn_path, max_imgs=200)
            if len(raw_items) > 0:
                raw_imgs = [cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA) for _, im in raw_items]
                ext_res = predict_rgb_batch_ensemble(edl_models, raw_imgs, val_tf, device, is_edl=True, edl_activation=args.edl_activation)
                external_burn_summary = {
                    "mode": "unlabeled_external_cohort",
                    "n_images": int(len(raw_imgs)),
                    "mean_u": round(float(np.mean(ext_res["uncertainty"])), 4),
                    "auto_diagnosed_ratio": round(float(np.mean(ext_res["uncertainty"] <= tau_edl)), 4),
                    "referred_to_clinician_ratio": round(float(np.mean(ext_res["uncertainty"] > tau_edl)), 4),
                }

    # 6. SUITE 4: FAILURE CASE ANALYSIS GALLERY (Checklist Câu 9)
    fail_fig_path = os.path.join(out_dir, "failure_cases_analysis_q1.png")
    failure_summary = generate_failure_case_gallery(
        edl_models, test_rois, test_ids, test_y, id_edl, tau_edl, val_tf, device, fail_fig_path
    )

    ood_fig_path = os.path.join(out_dir, "ood_and_generalization_q1.png")
    plot_ood_and_generalization_figure(
        id_edl["uncertainty"], ood_edl["uncertainty"], tau_edl, ood_metrics, shift_records, ood_fig_path
    )

    full_report = {
        "ood_safety_benchmark": ood_metrics,
        "realworld_shift_generalization": shift_records,
        "external_burn_validation": external_burn_summary,
        "failure_case_analysis": failure_summary,
    }
    json_path = os.path.join(out_dir, "ood_and_generalization_summary_q1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    print("\n📊 1) KẾT QUẢ CHẶN ẢNH LẠ NGOÀI MIỀN (OOD NON-BURN SAFETY TEST):")
    print(f"  • Nguồn mẫu OOD             : {ood_metrics['ood_sources']}")
    print(f"  • Độ bất định u (Bỏng vs Lạ): ID Burn = {ood_metrics['id_mean_u_edl']:.4f} vs OOD = {ood_metrics['ood_mean_u_edl']:.4f} (p = {ood_metrics['mann_whitney_p_ood_gt_id']:.3e})")
    print(f"  • R-EDL OOD AUROC / Reject  : AUROC = {ood_metrics['edl_ood_auroc']*100:.2f}% | Chặn an toàn = {ood_metrics['edl_ood_reject_rate']*100:.2f}% | Đoán bừa tự tin >=80% = {ood_metrics['edl_false_overconf_80_rate']*100:.2f}%")
    if ood_metrics["has_wce_comparison"]:
        print(f"  • Weighted CE OOD đối chứng : AUROC = {ood_metrics['wce_ood_auroc']*100:.2f}% | Chặn an toàn = {ood_metrics['wce_ood_reject_rate']*100:.2f}% | Đoán bừa tự tin >=80% = {ood_metrics['wce_false_overconf_80_rate']*100:.2f}%")

    print("\n📊 2) ĐỘ BỀN TỔNG QUÁT HÓA DƯỚI NHIỄU ĐIỆN THOẠI THỰC TẾ (CHECKLIST CÂU 10):")
    for sr in shift_records:
        lbl_one = sr["label_short"].replace("\n", " ")
        print(f"  • {lbl_one:22s}: Full Acc = {sr['full_acc']*100:5.2f}% -> Selective Acc = {sr['selective_acc']*100:5.2f}% (Mean u = {sr['mean_u']:.4f}, Retained = {sr['retained_coverage']*100:5.1f}%)")

    if external_burn_summary:
        print(f"\n📊 3) KIỂM CHỨNG TẬP BỎNG NGOẠI VIỆN ({external_burn_summary['mode']}): {external_burn_summary}")

    print("\n📊 4) PHÂN TÍCH CA THẤT BẠI TRÊN TẬP TEST (CHECKLIST CÂU 9):")
    print(f"  • Tổng số ca sai / 187 ảnh  : {failure_summary['total_misclassified']} ca (Đã được cổng R-EDL bắt giữ chuyển Bác sĩ: {failure_summary['caught_by_safety_gate_at_85cov']} ca = {failure_summary['safety_catch_rate_of_errors']*100:.1f}%)")
    print(f"  • Phân rã cặp nhầm lẫn      : {failure_summary['confusion_pair_breakdown']}")
    print(f"  📈 Biểu đồ OOD & Độ bền     : {ood_fig_path}")
    print(f"  🖼️ Gallery Phân tích ca lỗi : {fail_fig_path}")
    print(f"  💾 Báo cáo JSON đầy đủ      : {json_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()

