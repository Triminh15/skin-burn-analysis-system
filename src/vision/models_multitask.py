"""
Module: src/vision/models_multitask.py
Mục đích: Định nghĩa kiến trúc mạng đa nhiệm MT-EANet-EDL (Multi-Task Erythema-Aware Network
với Evidential Deep Learning) theo đặc tả kỹ thuật tại Section 6 & Section 9.2 của
docs/RESEARCH_ARCHITECTURE_AND_ROADMAP.md.

Kiến trúc bao gồm:
1. Shared Encoder: ResNet-34 qua thư viện timm (features_only=True) trích xuất đặc trưng đa mức (c0..c4).
2. Dual-Branch Decoder:
   - Expansive path U-Net khôi phục độ phân giải không gian với skip connections.
   - Nhánh mask_head: Dự đoán toàn bộ mặt nạ tổn thương (Burn Lesion Mask).
   - Nhánh boundary_head: Dự đoán viền ban đỏ mờ ngoại vi (Peripheral Erythema Boundary).
3. Mask-Guided Evidential Classifier Head:
   - MaskGuidedSpatialAttention: Tập trung đặc trưng bottleneck vào vùng tổn thương dựa trên xác suất mask.
   - Global Average Pooling & Fully Connected layers.
   - Đầu ra Evidence >= 0 (qua hàm softplus) đại diện cho tham số tập trung phân phối Dirichlet.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """
    Khối tích chập kép (Double Convolution Block) chuẩn cho U-Net:
    Conv2d(3x3) -> BatchNorm -> ReLU -> Conv2d(3x3) -> BatchNorm -> ReLU
    """
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


class MaskGuidedSpatialAttention(nn.Module):
    """
    Cơ chế chú ý không gian có điều hướng bởi mặt nạ (Mask-Guided Spatial Attention):
    Sử dụng mặt nạ phân đoạn mềm để hội tụ đặc trưng vào vùng tổn thương vết bỏng.
    Attended Feature = Feat * Attention_Map + Feat (Residual connection bảo toàn ngữ cảnh).
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Feature map bottleneck từ encoder, shape (B, C, H_feat, W_feat)
            mask: Soft mask probability (0..1), shape (B, 1, H, W) hoặc (B, 1, H_feat, W_feat)
        Returns:
            attended_feat: Feature map đã được lọc chú ý, shape (B, C, H_feat, W_feat)
        """
        # Downsample hoặc nội suy mask về cùng kích thước không gian với feature map
        if mask.shape[2:] != feat.shape[2:]:
            down_mask = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=False)
        else:
            down_mask = mask

        attn = self.conv(torch.cat([feat, down_mask], dim=1))
        attended_feat = feat * attn + feat  # Residual attention
        return attended_feat


class MTEANet(nn.Module):
    """
    Mạng Đa Nhiệm Multi-Task Erythema-Aware Network (MT-EANet-EDL).

    Hỗ trợ 2 chế độ kiến trúc cho Ablation Study:
    1. Multi-Task Thuần (M2 / use_boundary=False): Shared ResNet-34 + U-Net Decoder + EDL Head.
    2. MT-EANet-EDL (M3 / use_boundary=True): Tích hợp Dual-Branch Boundary Head giám sát viền ban đỏ.

    Đầu vào:
        x: Tensor ảnh RGB toàn cảnh (B, 3, H, W)
    Đầu ra Dict:
        - mask_logits: Tensor (B, 1, H, W) - Logits phân đoạn toàn bộ vùng bỏng
        - boundary_logits: Tensor (B, 1, H, W) nếu use_boundary=True, ngược lại None
        - evidence: Tensor (B, num_classes) >= 0 - Bằng chứng Dirichlet cho EDL
        - cls_logits: Tensor (B, num_classes) - Logits phân loại độ nghiêm trọng trước softplus
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        encoder_name: str = "resnet34",
        use_boundary: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_boundary = use_boundary

        # 1. SHARED ENCODER (ResNet-34 trích xuất 5 mức phân giải)
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        # Kênh đặc trưng mặc định của ResNet-34: [64, 64, 128, 256, 512]
        enc_channels = self.encoder.feature_info.channels()
        c0_ch, c1_ch, c2_ch, c3_ch, c4_ch = enc_channels

        # 2. NHÁNH DECODER PHÂN ĐOẠN (U-Net expansive path)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = ConvBlock(c4_ch + c3_ch, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(256 + c2_ch, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(128 + c1_ch, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(64 + c0_ch, 32)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = ConvBlock(32, 16)

        # Đầu ra phân đoạn (Lesion Mask Head & Peripheral Boundary Head nếu bật Dual-Branch)
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)
        if self.use_boundary:
            self.boundary_head = nn.Conv2d(16, 1, kernel_size=1)
        else:
            self.boundary_head = None

        # 3. NHÁNH PHÂN LOẠI EVIDENTIAL (EDL CLASSIFIER HEAD)
        self.mask_guided_attn = MaskGuidedSpatialAttention(in_channels=c4_ch)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(c4_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def _cat_skip(self, up_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """Đảm bảo kích thước không gian khớp hoàn toàn trước khi nối."""
        if up_feat.shape[2:] != skip_feat.shape[2:]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up_feat, skip_feat], dim=1)

    def forward(self, x: torch.Tensor, detach_mask: bool = True) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass của mạng MT-EANet.

        Args:
            x: Tensor (B, 3, H, W)
            detach_mask: Nếu True (mặc định), gradient từ nhánh phân loại không lan truyền
                         ngược vào decoder phân đoạn qua attention map, tránh xung đột gradient.
        Returns:
            Dict chứa "mask_logits", "boundary_logits", "evidence", "cls_logits"
        """
        feats = self.encoder(x)
        c0, c1, c2, c3, c4 = feats  # c4: (B, 512, H/32, W/32)

        # --- DECODER PHÂN ĐOẠN ---
        d4 = self.dec4(self._cat_skip(self.up4(c4), c3))
        d3 = self.dec3(self._cat_skip(self.up3(d4), c2))
        d2 = self.dec2(self._cat_skip(self.up2(d3), c1))
        d1 = self.dec1(self._cat_skip(self.up1(d2), c0))

        u0 = self.up0(d1)
        if u0.shape[2:] != x.shape[2:]:
            u0 = F.interpolate(u0, size=x.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(u0)

        pred_mask_logits = self.mask_head(d0)          # (B, 1, H, W)
        pred_boundary_logits = self.boundary_head(d0) if self.boundary_head is not None else None

        pred_mask_prob = torch.sigmoid(pred_mask_logits)

        # --- NHÁNH PHÂN LOẠI MASK-GUIDED + EDL ---
        mask_for_attn = pred_mask_prob.detach() if detach_mask else pred_mask_prob
        c4_attended = self.mask_guided_attn(c4, mask_for_attn)
        pooled = self.global_pool(c4_attended).flatten(1)  # (B, 512)
        cls_logits = self.cls_head(pooled)                 # (B, num_classes)
        cls_evidence = F.softplus(cls_logits)              # Biến đổi thành Evidence >= 0

        return {
            "mask_logits": pred_mask_logits,
            "boundary_logits": pred_boundary_logits,
            "evidence": cls_evidence,
            "cls_logits": cls_logits,
        }


def build_multitask_model(
    num_classes: int = 3,
    pretrained: bool = True,
    encoder_name: str = "resnet34",
    use_boundary: bool = True,
) -> MTEANet:
    """
    Hàm khởi tạo tiện ích cho mô hình MT-EANet-EDL hoặc Multi-Task thuần.
    """
    return MTEANet(
        num_classes=num_classes,
        pretrained=pretrained,
        encoder_name=encoder_name,
        use_boundary=use_boundary,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Đếm số lượng tham số của mô hình:
    Trả về: (tổng_số_tham_số, số_tham_số_cần_train)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("--- Testing Unit Tests for src/vision/models_multitask.py ---")
    model = build_multitask_model(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 256, 256, requires_grad=True)

    outputs = model(x)
    print(f"[OK] mask_logits shape: {outputs['mask_logits'].shape}")
    print(f"[OK] boundary_logits shape: {outputs['boundary_logits'].shape}")
    print(f"[OK] evidence shape: {outputs['evidence'].shape}")
    print(f"[OK] min evidence (should be >= 0): {outputs['evidence'].min().item():.4f}")

    total_p, train_p = count_parameters(model)
    print(f"[OK] Total parameters: {total_p / 1e6:.2f}M | Trainable: {train_p / 1e6:.2f}M")

    # Kiểm tra Gradient Flow qua Shared Encoder
    loss = outputs["mask_logits"].sum() + outputs["boundary_logits"].sum() + outputs["evidence"].sum()
    loss.backward()

    has_grad = model.encoder.conv1.weight.grad is not None
    grad_norm = model.encoder.conv1.weight.grad.norm().item() if has_grad else 0.0
    print(f"[OK] Shared encoder gradient flow check: has_grad={has_grad}, norm={grad_norm:.4f}")
    assert has_grad and grad_norm > 0, "Gradient flow to shared encoder failed!"
    print("--- All Sanity Checks Passed Successfully! ---")
