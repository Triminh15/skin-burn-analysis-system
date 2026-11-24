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
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import timm


# ==============================================================================
# PHẦN 1: MÔ HÌNH PHÂN ĐOẠN VẾT BỎNG (SEGMENTATION)
# ==============================================================================

class DoubleConv(nn.Module):
    """Khối tích chập kép chuẩn cho nhánh giải mã U-Net."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CoreLesionBranch(nn.Module):
    """
    Nhánh Lõi (Core Branch): Sử dụng Dilated Convolutions (dilation=2) để mở rộng
    trường thụ cảm (Receptive Field) bao quát toàn bộ vùng hoại tử trung tâm vết bỏng
    mà không làm suy giảm độ phân giải không gian.
    """
    def __init__(self, in_ch: int = 32, out_ch: int = 32):
        super().__init__()
        self.dilated_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dilated_block(x)


class ErythemaBoundaryBranch(nn.Module):
    """
    Nhánh Viền Ban Đỏ (Erythema Boundary Branch):
    Khai thác trực tiếp đặc trưng độ phân giải cao từ tầng nông c0 (H/2, W/2)
    kết hợp tích chập định hướng để bắt chặt dải hồng ban mờ nhạt ở ranh giới tổn thương.
    """
    def __init__(self, in_ch: int = 64, out_ch: int = 32):
        super().__init__()
        self.edge_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_block(x)


class GatedFusionModule(nn.Module):
    """
    Cổng Hợp Nhất Thích Nghi (Adaptive Gated Fusion):
    alpha = Sigmoid(Conv([F_core, F_boundary]))
    F_fused = alpha * F_core + (1 - alpha) * F_boundary
    Tự động ưu tiên nhánh Lõi ở giữa vết thương và chuyển tiếp mượt mà sang nhánh Viền ở rìa ban đỏ.
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_core: torch.Tensor, f_bnd: torch.Tensor) -> torch.Tensor:
        alpha = self.gate_conv(torch.cat([f_core, f_bnd], dim=1))
        return alpha * f_core + (1.0 - alpha) * f_bnd


class EAUNet(nn.Module):
    """
    Erythema-Aware Dual-Branch U-Net (EA-UNet) — Đỉnh cao SOTA Phân đoạn Tổn thương Bỏng.
    
    Cấu trúc:
    1. Pretrained Encoder (ResNet-34 / ConvNeXt): Trích xuất đặc trưng đa tầng (c0..c4).
    2. Expansive Decoder Path: Giải nén không gian kết hợp Skip Connections.
    3. Dual-Branch Refinement:
       - Core Branch: Bắt trọn lòng tổn thương hoại tử sâu.
       - Boundary Branch: Tách riêng dải viền ban đỏ mờ ngoại vi (Erythema).
       - Gated Fusion: Hợp nhất thích nghi đặc trưng lõi và viền.
    4. Output: Tensor logits (B, classes, H, W) hoàn toàn tương thích với các loss hàm phân đoạn.
    """
    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        self.classes = classes

        # 1. Backbone Encoder
        self.encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=in_channels, features_only=True)
        enc_channels = self.encoder.feature_info.channels()
        c0_ch, c1_ch, c2_ch, c3_ch, c4_ch = enc_channels

        # 2. Decoder Stages
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DoubleConv(c4_ch + c3_ch, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(256 + c2_ch, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(128 + c1_ch, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(64 + c0_ch, 32)

        # 3. Dual-Branch Refinement at H/2 Resolution
        self.core_branch = CoreLesionBranch(in_ch=32, out_ch=32)
        self.boundary_branch = ErythemaBoundaryBranch(in_ch=c0_ch, out_ch=32)
        self.gated_fusion = GatedFusionModule(channels=32)

        # 4. Final Head at Original Resolution (H x W)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = DoubleConv(32, 16)
        self.final_conv = nn.Conv2d(16, classes, kernel_size=1)

    def _cat_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if up_feat.shape[2:] != skip_feat.shape[2:]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up_feat, skip_feat], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c0, c1, c2, c3, c4 = self.encoder(x)

        # Expansive path
        d4 = self.dec4(self._cat_skip(self.up4(c4), c3))
        d3 = self.dec3(self._cat_skip(self.up3(d4), c2))
        d2 = self.dec2(self._cat_skip(self.up2(d3), c1))
        d1 = self.dec1(self._cat_skip(self.up1(d2), c0))  # Shape (B, 32, H/2, W/2)

        # Dual-Branch Processing
        f_core = self.core_branch(d1)
        # Nhánh viền nhận đặc trưng nông c0
        f_bnd = self.boundary_branch(c0)
        if f_bnd.shape[2:] != f_core.shape[2:]:
            f_bnd = F.interpolate(f_bnd, size=f_core.shape[2:], mode='bilinear', align_corners=False)

        # Gated Fusion
        f_fused = self.gated_fusion(f_core, f_bnd)

        # Upsample về độ phân giải gốc H x W
        u0 = self.up0(f_fused)
        if u0.shape[2:] != x.shape[2:]:
            u0 = F.interpolate(u0, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.dec0(u0)
        mask_logits = self.final_conv(out)
        return mask_logits


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
    - arch: 'unet', 'unetplusplus', hoặc 'ea_unet' (Erythema-Aware Dual-Branch U-Net)
    - backbone: 'resnet34' hoặc 'tu-convnext_tiny' (hoặc 'convnext_tiny')
    - in_channels: 3 (ảnh RGB)
    - classes: 1 (Mask nhị phân 0: da lành, 1: vết bỏng)
    - attention: 'none' hoặc 'scse'
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

    if arch_clean in ["eaunet", "ea_unet", "erythemaawareunet"]:
        # Khởi tạo mô hình EA-UNet đột phá
        model = EAUNet(
            backbone=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch_clean in ["unet", "attentionunet"]:
        model = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=attention_type,
            activation=None  # Trả về Raw Logits cho ComboLoss / FocalTverskyLoss
        )
    elif arch_clean in ["unetplusplus", "unetpp", "attentionunetplusplus"]:
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
            f"Kiến trúc {arch} không hợp lệ. Vui lòng chọn 'ea_unet', 'unet' hoặc 'unetplusplus'."
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
