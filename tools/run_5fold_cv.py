"""
Module: tools/run_5fold_cv.py
Mục đích: Huấn luyện và đánh giá chuẩn bài báo Scopus / ISI Q1/Q2 bằng phương pháp
Synchronized Stratified 5-Fold Cross-Validation + 5-Fold Self-Ensemble cho cả 2 giai đoạn:
  1) Giai đoạn 1 (Segmentation): EA-UNet v3 (hoặc U-Net chuẩn) -> Xuất Mean ± Std & 5-Fold Ensemble (+4x TTA).
  2) Giai đoạn 2 (Classification): R-EDL (hoặc Weighted CE) -> Xuất Mean ± Std, 5-Fold Deep Evidential Ensemble,
     Hậu hiệu chuẩn Bayes Out-Of-Fold (OOF Cost Calibration không rò rỉ dữ liệu Test) & Selective Prediction (p-value).

Đặc điểm an toàn cho Google Colab:
  - Đồng bộ 100% danh sách image_id giữa Phân đoạn và Phân loại ở từng Fold (Zero Data Leakage).
  - Khóa cứng 187 ảnh Test độc lập (data/splits/test.csv & test_roi.csv), tuyệt đối không tham gia chia Fold.
  - Hỗ trợ Resume tự động: nếu Colab ngắt kết nối giữa chừng, chạy lại sẽ tự động bỏ qua các Fold đã hoàn tất.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score

from src.vision.dataset import SkinBurnDataset
from src.vision.transforms import get_train_transforms, get_val_transforms
from src.vision.models import build_segmentation_model, build_classification_model, count_parameters
from src.vision.losses import get_segmentation_loss, get_classification_loss
from src.vision.metrics import SegmentationMetricTracker, ClassificationMetricTracker
from src.vision.train_seg import (
    set_seed,
    train_one_epoch as train_seg_one_epoch,
    validate as validate_seg,
    forward_seg_tta,
)
from src.vision.train_cls import (
    BurnROIDataset,
    get_cls_train_transforms,
    get_cls_val_transforms,
    train_one_epoch as train_cls_one_epoch,
    validate as validate_cls,
)
from src.vision.edl_loss import (
    logits_to_evidence,
    get_edl_metrics,
    calculate_ece,
    compute_selective_prediction_metrics,
)
from tools.evaluate_selective_prediction import (
    plot_selective_prediction_figure,
    print_selective_triage_report,
    _apply_deg3_cost_weight,
)


# ==============================================================================
# 1. CHIA TẬP SYNCHRONIZED STRATIFIED 5-FOLD (ZERO DATA LEAKAGE)
# ==============================================================================

def generate_synchronized_5fold_splits(
    n_splits: int = 5,
    seed: int = 42,
    splits_root: str = None,
    out_dir: str = None,
) -> List[Dict[str, str]]:
    """
    Gộp train + val (1.038 ảnh) và chia thành K Folds phân tầng theo severity_degree,
    đồng bộ tuyệt đối image_id giữa Segmentation và ROI Classification.
    Giữ nguyên 187 ảnh Test (test.csv / test_roi.csv) biệt lập hoàn toàn.
    """
    splits_root = splits_root or os.path.join(PROJECT_ROOT, "data", "splits")
    out_dir = out_dir or os.path.join(splits_root, "kfold")
    os.makedirs(out_dir, exist_ok=True)

    seg_train = pd.read_csv(os.path.join(splits_root, "train.csv"))
    seg_val = pd.read_csv(os.path.join(splits_root, "val.csv"))
    seg_test = pd.read_csv(os.path.join(splits_root, "test.csv"))
    seg_dev = pd.concat([seg_train, seg_val], ignore_index=True)

    cls_train = pd.read_csv(os.path.join(splits_root, "train_roi.csv"))
    cls_val = pd.read_csv(os.path.join(splits_root, "val_roi.csv"))
    cls_test = pd.read_csv(os.path.join(splits_root, "test_roi.csv"))
    cls_dev = pd.concat([cls_train, cls_val], ignore_index=True)

    # Kiểm tra nghiêm ngặt không rò rỉ dữ liệu sang tập Test
    test_ids = set(seg_test["image_id"].astype(str)).union(set(cls_test["image_id"].astype(str)))
    dev_ids = set(seg_dev["image_id"].astype(str))
    overlap = dev_ids.intersection(test_ids)
    if overlap:
        raise RuntimeError(f"Phát hiện rò rỉ dữ liệu giữa Dev và Test: {overlap}")

    # Sắp xếp cố định theo image_id trước khi chia StratifiedKFold để tái lập 100%
    seg_dev = seg_dev.sort_values("image_id").reset_index(drop=True)
    cls_map = cls_dev.set_index("image_id")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_manifests: List[Dict[str, str]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(seg_dev, seg_dev["severity_degree"]), start=1):
        seg_tr_df = seg_dev.iloc[tr_idx].reset_index(drop=True)
        seg_va_df = seg_dev.iloc[va_idx].reset_index(drop=True)

        tr_ids = [i for i in seg_tr_df["image_id"] if i in cls_map.index]
        va_ids = [i for i in seg_va_df["image_id"] if i in cls_map.index]
        cls_tr_df = cls_map.loc[tr_ids].reset_index()
        cls_va_df = cls_map.loc[va_ids].reset_index()

        seg_tr_path = os.path.join(out_dir, f"seg_train_fold{fold_idx}.csv")
        seg_va_path = os.path.join(out_dir, f"seg_val_fold{fold_idx}.csv")
        cls_tr_path = os.path.join(out_dir, f"cls_train_fold{fold_idx}.csv")
        cls_va_path = os.path.join(out_dir, f"cls_val_fold{fold_idx}.csv")

        seg_tr_df.to_csv(seg_tr_path, index=False, encoding="utf-8")
        seg_va_df.to_csv(seg_va_path, index=False, encoding="utf-8")
        cls_tr_df.to_csv(cls_tr_path, index=False, encoding="utf-8")
        cls_va_df.to_csv(cls_va_path, index=False, encoding="utf-8")

        fold_manifests.append({
            "fold": fold_idx,
            "seg_train": seg_tr_path,
            "seg_val": seg_va_path,
            "cls_train": cls_tr_path,
            "cls_val": cls_va_path,
        })

    return fold_manifests


# ==============================================================================
# 2. GIAI ĐOẠN 1: 5-FOLD SEGMENTATION + SELF-ENSEMBLE (+4x TTA)
# ==============================================================================

def _predict_seg_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_tta: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_list, masks_list = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            if use_tta:
                logits = forward_seg_tta(model, images)
            else:
                out = model(images)
                logits = out[0] if isinstance(out, tuple) else out
            probs_list.append(torch.sigmoid(logits).cpu())
            masks_list.append(masks.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(masks_list, dim=0)


def run_segmentation_5fold(args, folds: List[Dict[str, str]]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "kfold_seg")
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    run_prefix = args.seg_run_prefix or f"kfold_{args.seg_arch}_{args.seg_backbone}_{args.seg_loss}"
    test_dataset = SkinBurnDataset(
        csv_path=os.path.join(PROJECT_ROOT, "data", "splits", "test.csv"),
        transform=get_val_transforms(img_size=args.seg_img_size),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = get_segmentation_loss(name=args.seg_loss)
    bnd_criterion = get_segmentation_loss(name="combo")

    fold_records = []
    ens_probs_std_sum = None
    ens_probs_tta_sum = None
    test_masks_ref = None

    print("\n" + "=" * 82)
    print(f"🚀 BẮT ĐẦU 5-FOLD PHÂN ĐOẠN: {run_prefix.upper()} ({len(folds)} FOLDS)")
    print("=" * 82)

    for fm in folds:
        k = fm["fold"]
        set_seed(args.seed + k)
        ckpt_path = os.path.join(ckpt_dir, f"{run_prefix}_fold{k}_best.pth")

        train_ds = SkinBurnDataset(csv_path=fm["seg_train"], transform=get_train_transforms(img_size=args.seg_img_size))
        val_ds = SkinBurnDataset(csv_path=fm["seg_val"], transform=get_val_transforms(img_size=args.seg_img_size))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_segmentation_model(
            arch=args.seg_arch,
            backbone=args.seg_backbone,
            in_channels=3,
            classes=1,
            pretrained=True,
        ).to(device)

        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"\n⏩ [Fold {k}/{len(folds)}] Đã có sẵn checkpoint -> Nạp trực tiếp: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            _, best_val_dice, _, _ = validate_seg(model, val_loader, criterion, device, use_tta=False)
        else:
            print(f"\n🔥 [Fold {k}/{len(folds)}] Huấn luyện trên {len(train_ds)} ảnh | Validation {len(val_ds)} ảnh...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.seg_epochs, eta_min=1e-6)
            best_val_dice = 0.0
            patience_cnt = 0

            for epoch in range(1, args.seg_epochs + 1):
                tr_loss, tr_dice, _ = train_seg_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    bnd_criterion=bnd_criterion, bnd_weight=args.bnd_weight,
                )
                va_loss, va_dice, va_iou, _ = validate_seg(model, val_loader, criterion, device, use_tta=False)
                scheduler.step()

                if va_dice > best_val_dice:
                    best_val_dice = va_dice
                    patience_cnt = 0
                    torch.save(model.state_dict(), ckpt_path)
                    tag = " 🔥 [Best]"
                else:
                    patience_cnt += 1
                    tag = f" ({patience_cnt}/{args.patience})"

                if epoch == 1 or epoch % 5 == 0 or tag.startswith(" 🔥"):
                    print(f"  Fold {k} Ep [{epoch:02d}/{args.seg_epochs:02d}] Train Dice: {tr_dice:.4f} | Val Dice: {va_dice:.4f} (IoU: {va_iou:.4f}){tag}")

                if patience_cnt >= args.patience:
                    print(f"  🛑 Fold {k} dừng sớm tại Epoch {epoch} (Best Val Dice = {best_val_dice:.4f}).")
                    break

            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Đánh giá Fold k trên Tập Test 187 ảnh độc lập + tích lũy xác suất cho 5-Fold Ensemble
        probs_std, masks_t = _predict_seg_probs(model, test_loader, device, use_tta=False)
        probs_tta, _ = _predict_seg_probs(model, test_loader, device, use_tta=True)

        trk_std = SegmentationMetricTracker()
        trk_std.update(torch.logit(probs_std.clamp(1e-6, 1.0 - 1e-6)), masks_t)
        m_std = trk_std.compute()

        trk_tta = SegmentationMetricTracker()
        trk_tta.update(torch.logit(probs_tta.clamp(1e-6, 1.0 - 1e-6)), masks_t)
        m_tta = trk_tta.compute()

        ens_probs_std_sum = probs_std if ens_probs_std_sum is None else (ens_probs_std_sum + probs_std)
        ens_probs_tta_sum = probs_tta if ens_probs_tta_sum is None else (ens_probs_tta_sum + probs_tta)
        test_masks_ref = masks_t

        print(
            f"  ✅ Kết quả Fold {k} trên Test (187 ảnh): "
            f"Dice={m_std['dice']*100:.2f}% (IoU={m_std['iou']*100:.2f}%) | "
            f"+TTA Dice={m_tta['dice']*100:.2f}%"
        )
        fold_records.append({
            "fold": k,
            "val_dice": round(best_val_dice, 4),
            "test_dice": m_std["dice"],
            "test_iou": m_std["iou"],
            "test_pixel_acc": m_std["pixel_accuracy"],
            "test_dice_tta": m_tta["dice"],
            "test_iou_tta": m_tta["iou"],
        })

    # Tính 5-Fold Mean ± Std & 5-Fold Self-Ensemble
    n_k = float(len(folds))
    ens_std_logits = torch.logit((ens_probs_std_sum / n_k).clamp(1e-6, 1.0 - 1e-6))
    ens_tta_logits = torch.logit((ens_probs_tta_sum / n_k).clamp(1e-6, 1.0 - 1e-6))

    ens_trk_std = SegmentationMetricTracker()
    ens_trk_std.update(ens_std_logits, test_masks_ref)
    ens_std_metrics = ens_trk_std.compute()

    ens_trk_tta = SegmentationMetricTracker()
    ens_trk_tta.update(ens_tta_logits, test_masks_ref)
    ens_tta_metrics = ens_trk_tta.compute()

    dices = [r["test_dice"] for r in fold_records]
    ious = [r["test_iou"] for r in fold_records]
    paccs = [r["test_pixel_acc"] for r in fold_records]
    dices_tta = [r["test_dice_tta"] for r in fold_records]

    summary = {
        "task": "segmentation_5fold",
        "arch": args.seg_arch,
        "backbone": args.seg_backbone,
        "n_folds": len(folds),
        "folds": fold_records,
        "mean_test_dice": round(float(np.mean(dices)), 4),
        "std_test_dice": round(float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0, 4),
        "mean_test_iou": round(float(np.mean(ious)), 4),
        "std_test_iou": round(float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0, 4),
        "mean_test_pixel_acc": round(float(np.mean(paccs)), 4),
        "std_test_pixel_acc": round(float(np.std(paccs, ddof=1)) if len(paccs) > 1 else 0.0, 4),
        "mean_test_dice_tta": round(float(np.mean(dices_tta)), 4),
        "std_test_dice_tta": round(float(np.std(dices_tta, ddof=1)) if len(dices_tta) > 1 else 0.0, 4),
        "ensemble_test_dice": ens_std_metrics["dice"],
        "ensemble_test_iou": ens_std_metrics["iou"],
        "ensemble_test_pixel_acc": ens_std_metrics["pixel_accuracy"],
        "ensemble_tta_test_dice": ens_tta_metrics["dice"],
        "ensemble_tta_test_iou": ens_tta_metrics["iou"],
        "ensemble_tta_test_pixel_acc": ens_tta_metrics["pixel_accuracy"],
    }

    json_path = os.path.join(fig_dir, f"{run_prefix}_5fold_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 82)
    print(f"🏆 TỔNG KẾT 5-FOLD PHÂN ĐOẠN CHUẨN Q1/Q2 ({run_prefix.upper()})")
    print("=" * 82)
    for r in fold_records:
        print(
            f"  • Fold {r['fold']} : Val Dice={r['val_dice']*100:6.2f}% | "
            f"Test Dice={r['test_dice']*100:6.2f}% (IoU={r['test_iou']*100:6.2f}%) | "
            f"+TTA={r['test_dice_tta']*100:6.2f}%"
        )
    print("-" * 82)
    print(f"  📌 Single-Model (Mean ± Std)   : Dice = {summary['mean_test_dice']*100:6.2f}% ± {summary['std_test_dice']*100:4.2f}% | IoU = {summary['mean_test_iou']*100:6.2f}% ± {summary['std_test_iou']*100:4.2f}%")
    print(f"  📌 5-Fold Self-Ensemble        : Dice = {summary['ensemble_test_dice']*100:6.2f}%          | IoU = {summary['ensemble_test_iou']*100:6.2f}% (Pixel Acc: {summary['ensemble_test_pixel_acc']*100:6.2f}%)")
    print(f"  🔥 5-Fold Self-Ensemble + TTA  : Dice = {summary['ensemble_tta_test_dice']*100:6.2f}%          | IoU = {summary['ensemble_tta_test_iou']*100:6.2f}%")
    print(f"  💾 Đã lưu báo cáo JSON tại     : {json_path}")
    print("=" * 82)
    return summary


# ==============================================================================
# 3. GIAI ĐOẠN 2: 5-FOLD CLASSIFICATION (R-EDL) + DEEP EVIDENTIAL ENSEMBLE
# ==============================================================================

def _infer_cls_evidence_or_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_edl: bool,
    edl_activation: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    ev_or_prob_list, u_list, y_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["severity"]
            logits = model(images)
            if is_edl:
                ev = logits_to_evidence(logits, mode=edl_activation).cpu()
                _, u = get_edl_metrics(ev, num_classes=3)
                ev_or_prob_list.append(ev)
                u_list.append(u.cpu())
            else:
                prob = F.softmax(logits, dim=-1).cpu()
                u = 1.0 - prob.max(dim=-1).values
                ev_or_prob_list.append(prob)
                u_list.append(u)
            y_list.append(targets.cpu())
    return torch.cat(ev_or_prob_list, dim=0), torch.cat(u_list, dim=0), torch.cat(y_list, dim=0)


def run_classification_5fold(args, folds: List[Dict[str, str]]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "kfold_cls")
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    is_edl = (args.cls_loss == "edl")
    run_prefix = args.cls_run_prefix or f"kfold_cls_{args.cls_model}_{args.cls_loss}"

    test_ds = BurnROIDataset(
        csv_path=os.path.join(PROJECT_ROOT, "data", "splits", "test_roi.csv"),
        transform=get_cls_val_transforms(img_size=224),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    fold_records = []
    ens_ev_or_prob_sum = None
    test_y_ref = None
    oof_probs_all, oof_y_all = [], []

    print("\n" + "=" * 82)
    print(f"🚀 BẮT ĐẦU 5-FOLD PHÂN LOẠI: {run_prefix.upper()} ({len(folds)} FOLDS)")
    print("=" * 82)

    for fm in folds:
        k = fm["fold"]
        set_seed(args.seed + k)
        ckpt_path = os.path.join(ckpt_dir, f"{run_prefix}_fold{k}_best.pth")

        train_ds = BurnROIDataset(csv_path=fm["cls_train"], transform=get_cls_train_transforms(img_size=224))
        val_ds = BurnROIDataset(csv_path=fm["cls_val"], transform=get_cls_val_transforms(img_size=224))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        class_counts = [int((train_ds.df["severity_degree"] == c).sum()) for c in range(3)]
        if is_edl:
            total_n = sum(class_counts)
            base_w = [total_n / (3.0 * n) if n > 0 else 1.0 for n in class_counts]
            base_w[2] = base_w[2] * float(args.deg3_boost)
            criterion = get_classification_loss(
                name="edl",
                samples_per_class=class_counts,
                num_classes=3,
                annealing_epochs=10,
                loss_type="digamma",
                kl_weight=args.kl_weight,
                class_weights=torch.tensor(base_w, dtype=torch.float32),
            ).to(device)
        else:
            criterion = get_classification_loss(
                name=args.cls_loss,
                samples_per_class=class_counts,
                label_smoothing=0.05,
            ).to(device)

        model = build_classification_model(
            model_name=args.cls_model,
            num_classes=3,
            pretrained=True,
            drop_rate=0.2,
        ).to(device)

        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"\n⏩ [Fold {k}/{len(folds)}] Đã có sẵn checkpoint -> Nạp trực tiếp: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            _, val_m = validate_cls(model, val_loader, criterion, device, epoch=args.cls_epochs, is_edl=is_edl, edl_activation=args.edl_activation)
            best_val_f1 = val_m["macro_f1"]
        else:
            print(f"\n🔥 [Fold {k}/{len(folds)}] Huấn luyện trên {len(train_ds)} ảnh ROI | Validation {len(val_ds)} ảnh ROI...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cls_epochs, eta_min=1e-6)
            best_score = 0.0
            best_val_f1 = 0.0
            patience_cnt = 0

            for epoch in range(1, args.cls_epochs + 1):
                _, tr_acc, _ = train_cls_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    epoch=epoch, is_edl=is_edl, edl_activation=args.edl_activation,
                )
                _, va_m = validate_cls(
                    model, val_loader, criterion, device,
                    epoch=epoch, is_edl=is_edl, edl_activation=args.edl_activation,
                )
                scheduler.step()

                va_f1 = va_m["macro_f1"]
                va_rec3 = va_m["recall_degree_3"]
                # Cân bằng nhẹ 85% Macro F1 + 15% Recall Độ 3 để không lệch cực đoan
                sel_score = (0.85 * va_f1 + 0.15 * va_rec3) if is_edl else va_f1
                if sel_score > best_score:
                    best_score = sel_score
                    best_val_f1 = max(best_val_f1, va_f1)
                    patience_cnt = 0
                    torch.save(model.state_dict(), ckpt_path)
                    tag = " 🔥 [Best]"
                else:
                    best_val_f1 = max(best_val_f1, va_f1)
                    patience_cnt += 1
                    tag = f" ({patience_cnt}/{args.cls_patience})"

                if epoch == 1 or epoch % 5 == 0 or tag.startswith(" 🔥"):
                    print(
                        f"  Fold {k} Ep [{epoch:02d}/{args.cls_epochs:02d}] "
                        f"Train Acc: {tr_acc*100:5.1f}% | Val F1: {va_f1*100:5.1f}% (Rec Độ 3: {va_rec3*100:5.1f}%){tag}"
                    )

                if patience_cnt >= args.cls_patience:
                    print(f"  🛑 Fold {k} dừng sớm tại Epoch {epoch}.")
                    break

            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Thu thập dự đoán Out-Of-Fold trên tập Val của Fold k (phục vụ hiệu chuẩn Bayes c3 không rò rỉ)
        val_ev_or_p, _, val_y = _infer_cls_evidence_or_probs(model, val_loader, device, is_edl, args.edl_activation)
        val_probs = get_edl_metrics(val_ev_or_p, num_classes=3)[0] if is_edl else val_ev_or_p
        oof_probs_all.append(val_probs)
        oof_y_all.append(val_y)

        # Đánh giá Fold k trên Tập Test 187 ảnh ROI độc lập
        test_ev_or_p, _, test_y = _infer_cls_evidence_or_probs(model, test_loader, device, is_edl, args.edl_activation)
        if is_edl:
            fold_probs, fold_u = get_edl_metrics(test_ev_or_p, num_classes=3)
        else:
            fold_probs = test_ev_or_p
            fold_u = 1.0 - fold_probs.max(dim=-1).values

        cls_trk = ClassificationMetricTracker()
        cls_trk.update(fold_probs, test_y)
        fm_test = cls_trk.compute()
        fold_ece = round(calculate_ece(fold_probs, test_y), 4)

        ens_ev_or_prob_sum = test_ev_or_p if ens_ev_or_prob_sum is None else (ens_ev_or_prob_sum + test_ev_or_p)
        test_y_ref = test_y

        print(
            f"  ✅ Kết quả Fold {k} trên Test (187 ảnh): "
            f"Acc={fm_test['accuracy']*100:.2f}% | F1={fm_test['macro_f1']*100:.2f}% | "
            f"Recall Độ 3={fm_test['recall_degree_3']*100:.2f}% | ECE={fold_ece*100:.2f}%"
        )
        fold_records.append({
            "fold": k,
            "val_macro_f1": round(best_val_f1, 4),
            "test_accuracy": fm_test["accuracy"],
            "test_macro_f1": fm_test["macro_f1"],
            "test_recall_degree_3": fm_test["recall_degree_3"],
            "test_ece": fold_ece,
        })

    # --- HỢP NHẤT 5-FOLD DEEP EVIDENTIAL ENSEMBLE ---
    n_k = float(len(folds))
    mean_ev_or_prob = ens_ev_or_prob_sum / n_k
    if is_edl:
        ens_probs, ens_u = get_edl_metrics(mean_ev_or_prob, num_classes=3)
    else:
        ens_probs = mean_ev_or_prob
        ens_u = 1.0 - ens_probs.max(dim=-1).values

    ens_trk = ClassificationMetricTracker()
    ens_trk.update(ens_probs, test_y_ref)
    ens_metrics = ens_trk.compute()
    ens_ece = round(calculate_ece(ens_probs, test_y_ref), 4)

    # --- HẬU HIỆU CHUẨN BAYES CHI PHÍ ĐỘ 3 TRÊN TẬP OUT-OF-FOLD VALIDATION (1.038 ẢNH, KHÔNG ĐỤNG TEST) ---
    from sklearn.metrics import accuracy_score
    oof_p_cat = torch.cat(oof_probs_all, dim=0)
    oof_y_cat = torch.cat(oof_y_all, dim=0).numpy()
    base_oof_preds = np.argmax(oof_p_cat.numpy(), axis=1)
    base_oof_acc = float(accuracy_score(oof_y_cat, base_oof_preds))

    best_c3 = 1.0
    best_oof_obj = -1.0
    # Quét rộng c3 thuộc [1.0, 4.5] phù hợp với phân phối sắc nét của exp(logits),
    # cân bằng 50% OOF Macro F1 + 50% OOF Recall Độ 3 với ràng buộc an toàn không tụt quá 2.5% OOF Accuracy
    for tau in np.linspace(1.0, 4.5, 71):
        adj_oof = _apply_deg3_cost_weight(oof_p_cat, float(tau)).numpy()
        preds_oof = np.argmax(adj_oof, axis=1)
        oof_acc = float(accuracy_score(oof_y_cat, preds_oof))
        if oof_acc < base_oof_acc - 0.025:
            continue
        oof_f1 = float(f1_score(oof_y_cat, preds_oof, average="macro", zero_division=0))
        oof_rec3 = float(recall_score(oof_y_cat, preds_oof, labels=[0, 1, 2], average=None, zero_division=0)[2])
        obj = 0.50 * oof_f1 + 0.50 * oof_rec3
        if obj > best_oof_obj:
            best_oof_obj = obj
            best_c3 = float(tau)

    ens_probs_cal = _apply_deg3_cost_weight(ens_probs, best_c3)
    ens_cal_trk = ClassificationMetricTracker()
    ens_cal_trk.update(ens_probs_cal, test_y_ref)
    ens_cal_metrics = ens_cal_trk.compute()

    # Phân tích Selective Prediction & Mann-Whitney U p-value cho 5-Fold Ensemble
    sel_ens = compute_selective_prediction_metrics(ens_probs_cal, ens_u, test_y_ref)
    sel_fig_path = os.path.join(fig_dir, f"{run_prefix}_5fold_ensemble_selective_q1.png")
    plot_selective_prediction_figure(sel_ens, sel_fig_path, title_prefix=f"{run_prefix}_5fold_ensemble")

    accs = [r["test_accuracy"] for r in fold_records]
    f1s = [r["test_macro_f1"] for r in fold_records]
    rec3s = [r["test_recall_degree_3"] for r in fold_records]
    eces = [r["test_ece"] for r in fold_records]

    summary = {
        "task": "classification_5fold",
        "model": args.cls_model,
        "loss": args.cls_loss,
        "edl_activation": args.edl_activation if is_edl else None,
        "n_folds": len(folds),
        "folds": fold_records,
        "mean_test_accuracy": round(float(np.mean(accs)), 4),
        "std_test_accuracy": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
        "mean_test_macro_f1": round(float(np.mean(f1s)), 4),
        "std_test_macro_f1": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
        "mean_test_recall_deg3": round(float(np.mean(rec3s)), 4),
        "std_test_recall_deg3": round(float(np.std(rec3s, ddof=1)) if len(rec3s) > 1 else 0.0, 4),
        "mean_test_ece": round(float(np.mean(eces)), 4),
        "std_test_ece": round(float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0, 4),
        "ensemble_standard": {
            "accuracy": ens_metrics["accuracy"],
            "macro_f1": ens_metrics["macro_f1"],
            "recall_degree_3": ens_metrics["recall_degree_3"],
            "ece": ens_ece,
        },
        "oof_calibrated_deg3_cost": round(best_c3, 3),
        "ensemble_oof_calibrated": {
            "accuracy": ens_cal_metrics["accuracy"],
            "macro_f1": ens_cal_metrics["macro_f1"],
            "recall_degree_3": ens_cal_metrics["recall_degree_3"],
            "ece": ens_ece,
        },
        "ensemble_selective_prediction": {k: v for k, v in sel_ens.items() if not k.startswith("raw_")},
    }

    json_path = os.path.join(fig_dir, f"{run_prefix}_5fold_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 82)
    print(f"🏆 TỔNG KẾT 5-FOLD PHÂN LOẠI CHUẨN Q1/Q2 ({run_prefix.upper()})")
    print("=" * 82)
    for r in fold_records:
        print(
            f"  • Fold {r['fold']} : Acc={r['test_accuracy']*100:6.2f}% | "
            f"Macro F1={r['test_macro_f1']*100:6.2f}% | Recall Độ 3={r['test_recall_degree_3']*100:6.2f}% | "
            f"ECE={r['test_ece']*100:5.2f}%"
        )
    print("-" * 82)
    print(
        f"  📌 Single-Model (Mean ± Std)          : "
        f"Acc = {summary['mean_test_accuracy']*100:6.2f}% ± {summary['std_test_accuracy']*100:4.2f}% | "
        f"F1 = {summary['mean_test_macro_f1']*100:6.2f}% ± {summary['std_test_macro_f1']*100:4.2f}% | "
        f"Rec Độ 3 = {summary['mean_test_recall_deg3']*100:6.2f}% ± {summary['std_test_recall_deg3']*100:4.2f}%"
    )
    print(
        f"  📌 5-Fold Evidential Ensemble (c3=1.0): "
        f"Acc = {ens_metrics['accuracy']*100:6.2f}%          | "
        f"F1 = {ens_metrics['macro_f1']*100:6.2f}%          | "
        f"Rec Độ 3 = {ens_metrics['recall_degree_3']*100:6.2f}% (ECE={ens_ece*100:5.2f}%)"
    )
    print(
        f"  🔥 5-Fold Ensemble + OOF Bayes (c3={best_c3:.2f}): "
        f"Acc = {ens_cal_metrics['accuracy']*100:6.2f}%          | "
        f"F1 = {ens_cal_metrics['macro_f1']*100:6.2f}%          | "
        f"Rec Độ 3 = {ens_cal_metrics['recall_degree_3']*100:6.2f}%"
    )
    print_selective_triage_report(sel_ens, ece=ens_ece)
    print(f"  📈 Đã lưu biểu đồ Q1/Q2 300 DPI tại   : {sel_fig_path}")
    print(f"  💾 Đã lưu báo cáo JSON tại            : {json_path}")
    print("=" * 82)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Synchronized 5-Fold Cross-Validation & Self-Ensemble (Q1/Q2 Pipeline)")
    parser.add_argument("--task", type=str, default="both", choices=["cls", "seg", "both"], help="Chạy 5-Fold cho phân loại (cls), phân đoạn (seg), hoặc cả hai (both)")
    parser.add_argument("--n-folds", type=int, default=5, help="Số lượng Folds (mặc định 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--force-retrain", action="store_true", help="Bắt buộc train lại kể cả khi đã có checkpoint của Fold đó")

    # Tham số Segmentation
    parser.add_argument("--seg-arch", type=str, default="ea_unet", choices=["ea_unet", "unet"])
    parser.add_argument("--seg-backbone", type=str, default="resnet34")
    parser.add_argument("--seg-loss", type=str, default="combo", choices=["combo", "focal_tversky"])
    parser.add_argument("--bnd-weight", type=float, default=0.2)
    parser.add_argument("--seg-img-size", type=int, default=256)
    parser.add_argument("--seg-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seg-run-prefix", type=str, default=None)

    # Tham số Classification
    parser.add_argument("--cls-model", type=str, default="resnet34", choices=["resnet34", "convnext_tiny"])
    parser.add_argument("--cls-loss", type=str, default="edl", choices=["edl", "weighted_ce", "cb_focal"])
    parser.add_argument("--edl-activation", type=str, default="exp", choices=["exp", "softplus"])
    parser.add_argument("--kl-weight", type=float, default=0.04)
    parser.add_argument("--deg3-boost", type=float, default=1.0, help="Mặc định 1.0 (thuần công thức nghịch đảo tần suất mẫu, không nhân trùng)")
    parser.add_argument("--cls-epochs", type=int, default=30)
    parser.add_argument("--cls-patience", type=int, default=8)
    parser.add_argument("--cls-run-prefix", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    print("📂 Đang khởi tạo danh sách Synchronized Stratified 5-Fold Splits (khóa riêng 187 ảnh Test)...")
    folds = generate_synchronized_5fold_splits(n_splits=args.n_folds, seed=args.seed)
    print(f"  -> Đã đồng bộ thành công {len(folds)} Folds tại data/splits/kfold/ (Zero Test Leakage).")

    if args.task in ("cls", "both"):
        run_classification_5fold(args, folds)
    if args.task in ("seg", "both"):
        run_segmentation_5fold(args, folds)

    print(f"\n🎉 HOÀN TẤT TOÀN BỘ PIPELINE 5-FOLD TRONG {(time.time() - t0) / 60.0:.2f} PHÚT!")


if __name__ == "__main__":
    main()

