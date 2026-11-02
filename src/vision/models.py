"""
Module: src/vision/models.py
Mục đích: Định nghĩa và khởi tạo các mô hình Deep Learning chuẩn cho Hệ thống Phân tích Vết bỏng:
1. Giai đoạn Phân đoạn (Segmentation):
   - U-Net (ResNet-34) -> Mốc Baseline chuẩn mực.
   - U-Net (ConvNeXt-Tiny) -> Nâng cấp Backbone hiện đại.
   - Attention U-Net / Attention U-Net++ -> Tích hợp cổng chú ý scSE (Spatial & Channel Squeeze-and-Excitation).
2. Giai đoạn Phân loại (Classification):
   - ResNet-34 -> Baseline phân loại.
   - ConvNeXt-Tiny -> SOTA phân loại kết cấu da và độ ẩm vết bỏng.
"""

import sys
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm


# ==============================================================================
# PHẦN 1: MÔ HÌNH PHÂN ĐOẠN VẾT BỎNG (SEGMENTATION)
# ==============================================================================

def build_segmentation_model(
    arch: str = "unet",
    backbone: str = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
    attention: str = "none",
    pretrained: bool = True
) -> nn.Module:
    """
    Khởi tạo mô hình phân đoạn vết bỏng với kiến trúc và backbone tùy chọn:
    - arch: 'unet' hoặc 'unetplusplus'
    - backbone: 'resnet34' hoặc 'tu-convnext_tiny' (hoặc 'convnext_tiny')
    - in_channels: 3 (ảnh RGB)
    - classes: 1 (Mask nhị phân 0: da lành, 1: vết bỏng)
    - attention: 'none' hoặc 'scse' (Spatial & Channel Squeeze-and-Excitation Attention Gate)
    - pretrained: True (sử dụng weights ImageNet)
    """
    arch_clean = arch.lower().replace("-", "").replace("_", "")
    
    # Chuẩn hóa tên backbone cho timm / SMP
    if backbone.lower() in ["convnext_tiny", "convnext"]:
        backbone_name = "tu-convnext_tiny"
    else:
        backbone_name = backbone.lower()

    # Cấu hình cơ chế Attention Gate
    attention_type = "scse" if attention.lower() == "scse" else None
    weights = "imagenet" if pretrained else None

    if arch_clean in ["unet", "attentionunet"]:
        model = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=attention_type,
            activation=None  # Trả về Raw Logits cho ComboLoss / FocalTverskyLoss
        )
    elif arch_clean in ["unetplusplus", "unetpp", "attentionunetplusplus"]:
        # Lưu ý: ConvNeXt downsample 4x ngay ở stem nên UnetPlusPlus phù hợp tối ưu với ResNet
        model = smp.UnetPlusPlus(
            encoder_name=backbone_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=attention_type,
            activation=None
        )
    else:
        raise ValueError(
            f"Kiến trúc {arch} không hợp lệ. Vui lòng chọn 'unet' hoặc 'unetplusplus'."
        )

    return model


# ==============================================================================
# PHẦN 2: MÔ HÌNH PHÂN LOẠI CẤP ĐỘ BỎNG (CLASSIFICATION)
# ==============================================================================

def build_classification_model(
    model_name: str = "resnet34",
    num_classes: int = 3,
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> nn.Module:
    """
    Khởi tạo mô hình phân loại cấp độ bỏng (Độ 1, Độ 2, Độ 3):
    - model_name: 'resnet34' hoặc 'convnext_tiny'
    - num_classes: 3
    - pretrained: True (Transfer learning từ ImageNet)
    - drop_rate: 0.2 (Thêm Dropout 20% trước lớp Linear để chống overfit)
    """
    name_clean = model_name.lower().replace("-", "_")
    
    # Hỗ trợ tên thân thiện
    if name_clean in ["convnext", "convnext_t"]:
        name_clean = "convnext_tiny"

    model = timm.create_model(
        name_clean,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    return model


# ==============================================================================
# HÀM BỔ TRỢ: ĐẾM SỐ THAM SỐ (DÙNG CHO BÁO CÁO THỰC NGHIỆM)
# ==============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Đếm số lượng tham số của mô hình:
    Trả về: (tổng_số_tham_số, số_tham_số_cần_train)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ==============================================================================
# KIỂM THỬ ĐƠN VỊ NHANH (SANITY CHECK)
# ==============================================================================
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("--- Testing Unit Tests for src/vision/models.py ---")

    # 1. Test Segmentation Baseline: U-Net (ResNet-34)
    unet_baseline = build_segmentation_model(arch="unet", backbone="resnet34", pretrained=False)
    x_seg = torch.randn(2, 3, 256, 256)
    out_seg = unet_baseline(x_seg)
    total_p, _ = count_parameters(unet_baseline)
    print(f"[OK] U-Net (ResNet-34) output: {out_seg.shape} | Params: {total_p / 1e6:.2f}M")

    # 2. Test Segmentation SOTA: Attention U-Net (ConvNeXt-Tiny)
    att_unet = build_segmentation_model(arch="unet", backbone="tu-convnext_tiny", attention="scse", pretrained=False)
    out_att = att_unet(x_seg)
    total_p_att, _ = count_parameters(att_unet)
    print(f"[OK] Attention U-Net (ConvNeXt-Tiny) output: {out_att.shape} | Params: {total_p_att / 1e6:.2f}M")

    # 3. Test Segmentation SOTA: Attention U-Net++ (ResNet-34)
    att_unetpp = build_segmentation_model(arch="unetplusplus", backbone="resnet34", attention="scse", pretrained=False)
    out_attpp = att_unetpp(x_seg)
    total_p_attpp, _ = count_parameters(att_unetpp)
    print(f"[OK] Attention U-Net++ (ResNet-34) output: {out_attpp.shape} | Params: {total_p_attpp / 1e6:.2f}M")

    # 4. Test Classification Baseline: ResNet-34
    cls_baseline = build_classification_model(model_name="resnet34", num_classes=3, pretrained=False)
    x_cls = torch.randn(2, 3, 224, 224)
    out_cls = cls_baseline(x_cls)
    total_p_cls, _ = count_parameters(cls_baseline)
    print(f"[OK] ResNet-34 Classifier output: {out_cls.shape} | Params: {total_p_cls / 1e6:.2f}M")

    # 5. Test Classification SOTA: ConvNeXt-Tiny
    cls_sota = build_classification_model(model_name="convnext_tiny", num_classes=3, pretrained=False)
    out_sota = cls_sota(x_cls)
    total_p_sota, _ = count_parameters(cls_sota)
    print(f"[OK] ConvNeXt-Tiny Classifier output: {out_sota.shape} | Params: {total_p_sota / 1e6:.2f}M")

    print("[SUCCESS] All model architectures initialized and tested successfully!")
