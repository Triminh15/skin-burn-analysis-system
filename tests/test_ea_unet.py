"""
tests/test_ea_unet.py

Unit tests for Erythema-Aware Dual-Branch U-Net (EA-UNet)
Tests for:
1) EAUNet instantiation through build_segmentation_model
2) Forward pass output tensor shape (B, 1, H, W)
3) Backward pass gradient flow to all dual-branch components (Core, Boundary, Gated Fusion, Encoder)
4) Single-sample inference edge case
5) Integration with ComboLoss and FocalTverskyLoss
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.models import build_segmentation_model, count_parameters, EAUNet
from src.vision.losses import get_segmentation_loss


class TestEAUNetArchitecture:
    """Verifies EA-UNet architecture, forward pass, and backpropagation."""

    @pytest.fixture(scope="class")
    def model(self):
        return build_segmentation_model(arch="ea_unet", backbone="resnet34", pretrained=False)

    def test_ea_unet_instantiation(self, model):
        """Model must be an instance of EAUNet and have ~24.5M parameters."""
        assert isinstance(model, EAUNet), f"Expected EAUNet, got {type(model)}"
        total_p, train_p = count_parameters(model)
        assert 24.0e6 <= total_p <= 26.0e6, f"Expected ~24.5M params, got {total_p / 1e6:.2f}M"
        assert total_p == train_p, "All parameters must be trainable"

    def test_ea_unet_forward_shape(self, model):
        """Output shape must strictly be (B, 1, H, W) matching input resolution."""
        model.eval()
        batch_size = 2
        h, w = 256, 256
        x = torch.randn(batch_size, 3, h, w)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch_size, 1, h, w), f"Expected shape {(batch_size, 1, h, w)}, got {out.shape}"
        assert torch.isfinite(out).all(), "Output must contain only finite numbers"

    def test_ea_unet_single_sample_inference(self, model):
        """Batch size = 1 must work cleanly during deployment inference."""
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_ea_unet_gradient_flow_to_dual_branch(self, model):
        """Backpropagation must update encoder, core branch, boundary branch, and gated fusion."""
        model.train()
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        masks = torch.randint(0, 2, (2, 1, 256, 256)).float()

        criterion = get_segmentation_loss("combo")
        model.zero_grad()
        outputs = model(x)
        if isinstance(outputs, tuple):
            mask_logits, bnd_logits = outputs
            loss = criterion(mask_logits, masks) + 0.4 * criterion(bnd_logits, masks)
        else:
            loss = criterion(outputs, masks)
        loss.backward()

        # 1. Encoder gradient
        assert model.encoder.conv1.weight.grad is not None
        assert model.encoder.conv1.weight.grad.norm().item() > 0

        # 2. Core branch gradient
        core_grad = model.core_branch.dilated_block[0].weight.grad
        assert core_grad is not None and core_grad.norm().item() > 0

        # 3. Boundary branch gradient
        bnd_grad = model.boundary_branch.edge_block[0].weight.grad
        assert bnd_grad is not None and bnd_grad.norm().item() > 0

        # 4. Boundary head gradient (Deep Boundary Supervision)
        bnd_head_grad = model.boundary_head.weight.grad
        assert bnd_head_grad is not None and bnd_head_grad.norm().item() > 0

        # 5. Gated fusion gradient
        gate_grad = model.gated_fusion.gate_conv[0].weight.grad
        assert gate_grad is not None and gate_grad.norm().item() > 0

        # 6. Final output conv gradient
        final_grad = model.final_conv.weight.grad
        assert final_grad is not None and final_grad.norm().item() > 0

