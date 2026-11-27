"""
Unit tests for Synchronized Stratified 5-Fold Cross-Validation split generation (tools/run_5fold_cv.py).
Verifies:
1) Zero data leakage between any Fold (Train/Val) and the locked 187-image Test set.
2) Exact 1-to-1 synchronization of image_id between Segmentation and Classification folds.
3) Disjoint validation sets whose union covers 100% of the 1,038 development images.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_5fold_cv import generate_synchronized_5fold_splits


def test_synchronized_5fold_zero_leakage_and_full_coverage(tmp_path):
    folds = generate_synchronized_5fold_splits(n_splits=5, seed=42, out_dir=str(tmp_path))
    assert len(folds) == 5

    test_seg_ids = set(pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test.csv"))["image_id"])
    test_cls_ids = set(pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test_roi.csv"))["image_id"])
    assert len(test_seg_ids) == 187

    seen_val_ids = set()
    for fm in folds:
        seg_tr = pd.read_csv(fm["seg_train"])
        seg_va = pd.read_csv(fm["seg_val"])
        cls_tr = pd.read_csv(fm["cls_train"])
        cls_va = pd.read_csv(fm["cls_val"])

        seg_tr_ids = set(seg_tr["image_id"])
        seg_va_ids = set(seg_va["image_id"])
        cls_tr_ids = set(cls_tr["image_id"])
        cls_va_ids = set(cls_va["image_id"])

        # 1. Không trùng lặp giữa Train và Val trong cùng 1 Fold
        assert seg_tr_ids.isdisjoint(seg_va_ids)
        assert cls_tr_ids.isdisjoint(cls_va_ids)

        # 2. Đồng bộ tuyệt đối image_id giữa Segmentation và Classification
        assert seg_tr_ids == cls_tr_ids
        assert seg_va_ids == cls_va_ids

        # 3. Tuyệt đối không rò rỉ sang tập Test 187 ảnh
        assert seg_tr_ids.isdisjoint(test_seg_ids)
        assert seg_va_ids.isdisjoint(test_seg_ids)
        assert cls_tr_ids.isdisjoint(test_cls_ids)
        assert cls_va_ids.isdisjoint(test_cls_ids)

        # 4. Các tập Val giữa 5 Folds rời nhau đôi một
        assert seen_val_ids.isdisjoint(seg_va_ids)
        seen_val_ids.update(seg_va_ids)

    # Hợp của 5 tập Val phải phủ đúng toàn bộ 1.038 ảnh phát triển (856 + 182)
    assert len(seen_val_ids) == 1038

