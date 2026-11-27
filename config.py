"""
Centralized configuration for BurnScan Clinical Analysis System.
Manages project directories, database path, and trained model checkpoint registries.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data & Knowledge Base directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
GUIDELINES_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "guidelines")

# Outputs & Weights directories
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
WEB_UPLOADS_DIR = os.path.join(OUTPUTS_DIR, "web_uploads")

# SQLite Database
DB_PATH = os.path.join(PROJECT_ROOT, "burnscan.db")

# Trained Segmentation Model Registry (outputs/checkpoints/)
SEGMENTATION_MODELS = {
    "kfold_unet_resnet34_combo": {
        "name": "5-Fold U-Net Ensemble (ResNet-34 · Combo + Spatial Uncertainty)",
        "arch": "unet",
        "backbone": "resnet34",
        "attention": "none",
        "use_tta": True,
        "checkpoints": [
            os.path.join(CHECKPOINTS_DIR, "kfold_seg", f"kfold_unet_resnet34_combo_fold{k}_best.pth")
            for k in range(1, 6)
        ],
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "unet_resnet34_combo_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "kfold_unet_resnet34_combo_5fold_summary.json"),
    },
    "unet_resnet34_focal_tversky": {
        "name": "U-Net (ResNet-34 · Focal Tversky)",
        "arch": "unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "unet_resnet34_focal_tversky_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "unet_resnet34_focal_tversky_test_results.json"),
    },
    "ea_unet_v3_smp_residual": {
        "name": "EA-UNet v3 (SMP-Anchored Residual · Combo Loss)",
        "arch": "ea_unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "run_ea_unet_v3_smp_residual_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "run_ea_unet_v3_smp_residual_test_results.json"),
    },
    "ea_unet_v2_bnd_supervision": {
        "name": "EA-UNet v2 (Boundary Supervision · Combo Loss)",
        "arch": "ea_unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "run_ea_unet_v2_bnd_supervision_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "run_ea_unet_v2_bnd_supervision_test_results.json"),
    },
    "ea_unet_resnet34_combo": {
        "name": "EA-UNet (ResNet-34 · Combo Loss)",
        "arch": "ea_unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "run_ea_unet_resnet34_combo_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "run_ea_unet_resnet34_combo_test_results.json"),
    },
    "ea_unet_resnet34_focal_tversky": {
        "name": "EA-UNet (ResNet-34 · Focal Tversky)",
        "arch": "ea_unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "run_ea_unet_resnet34_focal_tversky_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "run_ea_unet_resnet34_focal_tversky_test_results.json"),
    },
    "unet_resnet34_combo": {
        "name": "U-Net (ResNet-34 · Combo Loss)",
        "arch": "unet",
        "backbone": "resnet34",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "unet_resnet34_combo_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "unet_resnet34_combo_test_results.json"),
    },
    "unet_convnext_focal_tversky": {
        "name": "U-Net (ConvNeXt-Tiny · Focal Tversky)",
        "arch": "unet",
        "backbone": "tu-convnext_tiny",
        "attention": "none",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "unet_convnext_focal_tversky_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "unet_convnext_focal_tversky_test_results.json"),
    },
    "att_unet_convnext_focal_tversky": {
        "name": "Attention U-Net scSE (ConvNeXt-Tiny · Focal Tversky)",
        "arch": "unet",
        "backbone": "tu-convnext_tiny",
        "attention": "scse",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "att_unet_convnext_focal_tversky_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "att_unet_convnext_focal_tversky_test_results.json"),
    },
}

# Trained Classification Model Registry (outputs/checkpoints/)
CLASSIFICATION_MODELS = {
    "kfold_cls_resnet34_edl": {
        "name": "5-Fold ResNet-34 Ensemble (Asymmetric R-EDL + Selective Triage)",
        "model_name": "resnet34",
        "loss_type": "edl",
        "edl_activation": "exp",
        "uncertainty_threshold": 0.1180,
        "deg3_cost_c3": 1.95,
        "checkpoints": [
            os.path.join(CHECKPOINTS_DIR, "kfold_cls", f"kfold_cls_resnet34_edl_fold{k}_best.pth")
            for k in range(1, 6)
        ],
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "classifier", "run_m1_classifier_redl_exp_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "kfold_cls_resnet34_edl_5fold_summary.json"),
    },
    "cls_resnet34_weighted_ce": {
        "name": "ResNet-34 (Weighted CE)",
        "model_name": "resnet34",
        "loss_type": "softmax",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "cls_resnet34_weighted_ce_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "cls_resnet34_weighted_ce_test_results.json"),
    },
    "cls_resnet34_redl_exp": {
        "name": "ResNet-34 (Relaxed Log-Evidence EDL)",
        "model_name": "resnet34",
        "loss_type": "edl",
        "edl_activation": "exp",
        "uncertainty_threshold": 0.1527,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "classifier", "run_m1_classifier_redl_exp_best.pth"),
        "metrics_file": os.path.join(CHECKPOINTS_DIR, "classifier", "figures", "run_m1_classifier_redl_exp_test_results.json"),
    },
    "cls_resnet34_cb_focal": {
        "name": "ResNet-34 (CB Focal)",
        "model_name": "resnet34",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "cls_resnet34_cb_focal_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "cls_resnet34_cb_focal_test_results.json"),
    },
    "cls_convnext_cb_focal": {
        "name": "ConvNeXt-Tiny (CB Focal)",
        "model_name": "convnext_tiny",
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "cls_convnext_cb_focal_best.pth"),
        "metrics_file": os.path.join(FIGURES_DIR, "cls_convnext_cb_focal_test_results.json"),
    },
}

DEFAULT_SEG_MODEL = "kfold_unet_resnet34_combo"
DEFAULT_CLS_MODEL = "kfold_cls_resnet34_edl"

