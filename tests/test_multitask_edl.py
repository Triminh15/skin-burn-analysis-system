"""
tests/test_multitask_edl.py

Comprehensive verification test suite for Multi-Task Erythema-Aware Network with Evidential Deep Learning (MT-EANet-EDL).
Covers:
1) EDLLoss and uncertainty calculation shapes and bounds
2) MTEANet forward pass tensor shapes (mask_logits, boundary_logits, evidence)
3) Backward pass gradient flow through shared encoder
4) Dataset output format and boundary extraction
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.edl_loss import EDLLoss, get_edl_metrics, calculate_ece
from src.vision.models_multitask import MTEANet, build_multitask_model, count_parameters
from src.vision.dataset_multitask import (
    SkinBurnMultiTaskDataset,
    derive_boundary_mask,
    get_multitask_dataloaders,
)


# ==============================================================================
# SECTION 1: EDLLoss and Uncertainty Calculation Shapes & Bounds
# ==============================================================================

class TestEDLLossAndUncertainty:
    """Verifies EDLLoss computation, shapes, annealing schedule, and uncertainty bounds."""

    def test_edl_loss_output_shape_and_scalar(self):
        """EDLLoss must return a 0-dim scalar tensor that is positive and finite."""
        batch_size = 4
        num_classes = 3
        loss_fn = EDLLoss(num_classes=num_classes, annealing_epochs=10)

        # Evidence >= 0 from softplus
        evidence = F.softplus(torch.randn(batch_size, num_classes, requires_grad=True))
        targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        loss = loss_fn(evidence, targets, epoch=0)
        assert loss.ndim == 0, f"Expected scalar loss (ndim=0), got ndim={loss.ndim}"
        assert torch.isfinite(loss), "Loss must be finite (not NaN or Inf)"
        assert loss.item() > 0, f"Loss must be strictly positive, got {loss.item()}"

    def test_edl_loss_annealing_progression(self):
        """KL divergence regularization weight increases with epoch according to annealing schedule."""
        num_classes = 3
        loss_fn = EDLLoss(num_classes=num_classes, annealing_epochs=10)

        # Consistent non-zero logits
        evidence = F.softplus(torch.tensor([
            [3.0, 0.5, 0.1],
            [0.2, 2.5, 0.3],
            [0.1, 0.2, 4.0]
        ]))
        targets = torch.tensor([0, 1, 2], dtype=torch.long)

        loss_ep0 = loss_fn(evidence, targets, epoch=0).item()
        loss_ep5 = loss_fn(evidence, targets, epoch=5).item()
        loss_ep10 = loss_fn(evidence, targets, epoch=10).item()
        loss_ep20 = loss_fn(evidence, targets, epoch=20).item()

        assert loss_ep0 <= loss_ep5 <= loss_ep10, (
            f"Loss should increase as annealing coefficient increases: "
            f"ep0={loss_ep0:.4f}, ep5={loss_ep5:.4f}, ep10={loss_ep10:.4f}"
        )
        assert abs(loss_ep10 - loss_ep20) < 1e-5, (
            f"Annealing coefficient should saturate at min(1.0, epoch/10) for epoch >= 10: "
            f"ep10={loss_ep10:.4f}, ep20={loss_ep20:.4f}"
        )

    def test_get_edl_metrics_shapes_and_bounds(self):
        """Probability sums to 1.0, probabilities in [0, 1], uncertainty in (0, 1]."""
        batch_size = 16
        num_classes = 3
        evidence = torch.rand(batch_size, num_classes) * 20.0

        probs, uncertainty = get_edl_metrics(evidence, num_classes=num_classes)

        # 1. Shapes
        assert probs.shape == (batch_size, num_classes), (
            f"Expected probs shape {(batch_size, num_classes)}, got {probs.shape}"
        )
        assert uncertainty.shape == (batch_size,), (
            f"Expected uncertainty shape {(batch_size,)}, got {uncertainty.shape}"
        )

        # 2. Probability bounds and sum
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), "All probabilities must be within [0, 1]"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5), (
            "Probabilities must sum to 1.0 along the class dimension"
        )

        # 3. Uncertainty bounds (0, 1]
        assert (uncertainty > 0.0).all(), "Uncertainty must be strictly greater than 0"
        assert (uncertainty <= 1.0).all(), "Uncertainty must be <= 1.0"

    def test_edl_metrics_edge_cases(self):
        """Test zero evidence yields max uncertainty (1.0) and high evidence yields low uncertainty."""
        num_classes = 3

        # Case A: Zero evidence -> S = 3, u = 3/3 = 1.0, uniform probability [1/3, 1/3, 1/3]
        zero_evidence = torch.zeros(1, num_classes)
        p_zero, u_zero = get_edl_metrics(zero_evidence, num_classes=num_classes)
        assert abs(u_zero.item() - 1.0) < 1e-5, f"Zero evidence must yield u=1.0, got {u_zero.item()}"
        assert torch.allclose(p_zero, torch.full((1, num_classes), 1.0 / num_classes), atol=1e-5)

        # Case B: High confident evidence -> u -> 0, high class probability
        high_evidence = torch.tensor([[100.0, 0.0, 0.0]])
        p_high, u_high = get_edl_metrics(high_evidence, num_classes=num_classes)
        assert u_high.item() < 0.05, f"High evidence must yield low uncertainty < 0.05, got {u_high.item()}"
        assert p_high[0, 0].item() > 0.95, f"Class 0 prob should dominate, got {p_high[0, 0].item()}"

    def test_kl_divergence_non_negativity(self):
        """KL divergence to uniform Dirichlet prior is 0 at uniform prior and non-negative elsewhere."""
        num_classes = 3
        loss_fn = EDLLoss(num_classes=num_classes)

        # Prior alpha = [1, 1, 1]
        alpha_uniform = torch.ones((3, num_classes))
        kl_uniform = loss_fn.kl_divergence(alpha_uniform)
        assert torch.allclose(kl_uniform, torch.zeros_like(kl_uniform), atol=1e-5), (
            "KL divergence of uniform Dirichlet prior to itself must be 0"
        )

        # Arbitrary alpha >= 1
        alpha_arbitrary = torch.rand(10, num_classes) * 15.0 + 1.0
        kl_arbitrary = loss_fn.kl_divergence(alpha_arbitrary)
        assert (kl_arbitrary >= -1e-6).all(), "KL divergence must be non-negative"


# ==============================================================================
# SECTION 2: MTEANet Forward Pass Tensor Shapes
# ==============================================================================

@pytest.fixture(scope="module")
def shared_model():
    """Create an un-pretrained MTEANet instance for fast execution."""
    return build_multitask_model(num_classes=3, pretrained=False)


class TestMTEANetForward:
    """Verifies MT-EANet architecture forward output shapes and properties."""

    def test_mteanet_forward_dict_keys(self, shared_model):
        """Model forward must return dict with mask_logits, boundary_logits, evidence."""
        shared_model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            outputs = shared_model(x)

        assert isinstance(outputs, dict), "Forward pass must return a dictionary"
        required_keys = {"mask_logits", "boundary_logits", "evidence"}
        missing_keys = required_keys - set(outputs.keys())
        assert not missing_keys, f"Missing required output keys: {missing_keys}"

    def test_mteanet_forward_tensor_shapes(self, shared_model):
        """Checks spatial resolution matching and channel counts for all output heads."""
        shared_model.eval()
        batch_size = 2
        height, width = 256, 256
        x = torch.randn(batch_size, 3, height, width)

        with torch.no_grad():
            outputs = shared_model(x)

        # 1. Mask logits: (B, 1, H, W)
        assert outputs["mask_logits"].shape == (batch_size, 1, height, width), (
            f"Expected mask_logits shape {(batch_size, 1, height, width)}, got {outputs['mask_logits'].shape}"
        )

        # 2. Boundary logits: (B, 1, H, W)
        assert outputs["boundary_logits"].shape == (batch_size, 1, height, width), (
            f"Expected boundary_logits shape {(batch_size, 1, height, width)}, got {outputs['boundary_logits'].shape}"
        )

        # 3. Evidence: (B, num_classes)
        assert outputs["evidence"].shape == (batch_size, 3), (
            f"Expected evidence shape {(batch_size, 3)}, got {outputs['evidence'].shape}"
        )

    def test_mteanet_evidence_non_negativity(self, shared_model):
        """Evidence values must be strictly non-negative (>= 0.0) via softplus."""
        shared_model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            outputs = shared_model(x)

        evidence = outputs["evidence"]
        assert (evidence >= 0.0).all(), (
            f"Evidence must be >= 0.0 everywhere, minimum value found: {evidence.min().item()}"
        )

    def test_mteanet_batch_size_one(self, shared_model):
        """Single-sample batch (B=1) inference must work properly in eval mode."""
        shared_model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            outputs = shared_model(x)

        assert outputs["mask_logits"].shape == (1, 1, 256, 256)
        assert outputs["boundary_logits"].shape == (1, 1, 256, 256)
        assert outputs["evidence"].shape == (1, 3)

    def test_pure_multitask_without_boundary(self):
        """Verify M2 configuration (Multi-Task thuần không có Dual-Branch viền)."""
        model_m2 = build_multitask_model(num_classes=3, pretrained=False, use_boundary=False)
        model_m2.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            outputs = model_m2(x)

        assert outputs["mask_logits"].shape == (2, 1, 256, 256)
        assert outputs["boundary_logits"] is None, "In pure multitask (M2), boundary_logits must be None"
        assert outputs["evidence"].shape == (2, 3)


# ==============================================================================
# SECTION 3: Backward Pass Gradient Flow Through Shared Encoder
# ==============================================================================

class TestGradientFlow:
    """Verifies end-to-end backpropagation through all task heads and shared encoder."""

    def test_multi_task_joint_gradient_flow_to_encoder(self):
        """
        Backpropagating joint loss L_total = w_seg*L_seg + w_bnd*L_bnd + w_cls*L_cls
        must populate non-zero gradients in the shared encoder (e.g., conv1, layer1..layer4).
        """
        model = build_multitask_model(num_classes=3, pretrained=False)
        model.train()

        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        outputs = model(x, detach_mask=True)

        l_seg = outputs["mask_logits"].mean()
        l_bnd = outputs["boundary_logits"].mean()
        l_cls = outputs["evidence"].mean()

        total_loss = 1.0 * l_seg + 0.3 * l_bnd + 0.5 * l_cls
        model.zero_grad()
        total_loss.backward()

        # 1. Verify shared encoder initial layer gradient
        assert model.encoder.conv1.weight.grad is not None, "Encoder conv1 weight grad must not be None"
        conv1_grad_norm = model.encoder.conv1.weight.grad.norm().item()
        assert conv1_grad_norm > 0, f"Encoder conv1 gradient norm must be > 0, got {conv1_grad_norm}"

        # 2. Verify shared encoder bottleneck layer gradients (layer4 has parameters with non-zero grad)
        layer4_grads = [p.grad.norm().item() for p in model.encoder.layer4.parameters() if p.grad is not None]
        assert len(layer4_grads) > 0, "Encoder layer4 must have parameters with computed gradients"
        total_layer4_grad = sum(layer4_grads)
        assert total_layer4_grad > 0, f"Encoder layer4 total gradient norm must be > 0, got {total_layer4_grad}"

        # 3. Verify task head gradients
        assert model.mask_head.weight.grad is not None and model.mask_head.weight.grad.norm() > 0, (
            "Mask head weight gradient must be non-zero"
        )
        assert model.boundary_head.weight.grad is not None and model.boundary_head.weight.grad.norm() > 0, (
            "Boundary head weight gradient must be non-zero"
        )
        assert model.cls_head[-1].weight.grad is not None and model.cls_head[-1].weight.grad.norm() > 0, (
            "Classifier head output layer weight gradient must be non-zero"
        )

    def test_segmentation_branch_alone_updates_encoder(self):
        """Segmentation loss alone must propagate gradients back into shared encoder."""
        model = build_multitask_model(num_classes=3, pretrained=False)
        model.train()

        x = torch.randn(2, 3, 256, 256)
        outputs = model(x)
        l_seg = outputs["mask_logits"].mean()

        model.zero_grad()
        l_seg.backward()

        assert model.encoder.conv1.weight.grad is not None
        assert model.encoder.conv1.weight.grad.norm().item() > 0, "Seg loss alone must propagate to conv1"

    def test_classifier_branch_alone_updates_encoder(self):
        """Classifier EDL loss alone must propagate gradients back into shared encoder c4 stage."""
        model = build_multitask_model(num_classes=3, pretrained=False)
        model.train()

        x = torch.randn(2, 3, 256, 256)
        outputs = model(x)
        l_cls = outputs["evidence"].mean()

        model.zero_grad()
        l_cls.backward()

        # Encoder layer4 produces c4 feature map used by classifier
        layer4_grads = [p.grad.norm().item() for p in model.encoder.layer4.parameters() if p.grad is not None]
        assert len(layer4_grads) > 0, "Encoder layer4 must have parameters with computed gradients"
        total_layer4_grad = sum(layer4_grads)
        assert total_layer4_grad > 0, (
            f"Classifier loss alone must propagate to encoder layer4, got total grad norm {total_layer4_grad}"
        )


# ==============================================================================
# SECTION 4: Dataset Output Format and Boundary Mask Derivation
# ==============================================================================

class TestMultiTaskDataset:
    """Verifies dataset output format, keys, tensor dimensions, and boundary generation."""

    def test_derive_boundary_mask_morphology(self):
        """
        Morphological boundary derivation B = Dilation(M) - Erosion(M)
        must identify the outer contour of a binary object.
        """
        # Create 64x64 synthetic mask with 20x20 solid white square at center [22:42, 22:42]
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[22:42, 22:42] = 1.0

        bnd = derive_boundary_mask(mask, kernel_size=5, iterations=1)

        assert isinstance(bnd, np.ndarray), "Boundary output should be a numpy array"
        assert bnd.shape == (64, 64), f"Boundary shape mismatch: {bnd.shape}"
        assert set(np.unique(bnd)).issubset({0.0, 1.0}), f"Boundary must be binary {0.0, 1.0}, got {np.unique(bnd)}"

        # Boundary must have non-zero pixels
        assert bnd.sum() > 0, "Boundary mask should contain boundary pixels"

        # Far outside (e.g. corner [0, 0]) must be 0
        assert bnd[0, 0] == 0.0, "Background pixels should have 0 boundary"

        # Center of square [32, 32] is eroded and should be 0 (interior)
        assert bnd[32, 32] == 0.0, "Center interior of solid square should have 0 boundary"

        # Border region around [22, 22] should have 1.0 boundary
        assert bnd[21:24, 21:24].sum() > 0, "Perimeter of square must have active boundary pixels"

    def test_dataset_item_keys_and_shapes(self):
        """SkinBurnMultiTaskDataset must yield image, mask, boundary, severity, and burn_percentage."""
        train_csv = os.path.join(PROJECT_ROOT, "data/splits/train.csv")
        if not os.path.exists(train_csv):
            pytest.skip(f"Data split {train_csv} not found for live dataset test")

        dataset = SkinBurnMultiTaskDataset(
            csv_path=train_csv,
            root_dir=PROJECT_ROOT,
            transform=None,
            boundary_kernel_size=5,
        )

        assert len(dataset) > 0, "Dataset must not be empty"
        item = dataset[0]

        # 1. Required keys
        required_keys = {"image", "mask", "boundary", "severity", "burn_percentage", "image_id"}
        assert required_keys.issubset(item.keys()), f"Item missing required keys: {required_keys - set(item.keys())}"

        # 2. Tensor shapes & types
        image = item["image"]
        mask = item["mask"]
        boundary = item["boundary"]
        severity = item["severity"]

        assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor"
        assert image.ndim == 3 and image.shape[0] == 3, f"Image shape must be (3, H, W), got {image.shape}"

        assert isinstance(mask, torch.Tensor), "Mask must be a torch.Tensor"
        assert mask.ndim == 3 and mask.shape[0] == 1, f"Mask shape must be (1, H, W), got {mask.shape}"

        assert isinstance(boundary, torch.Tensor), "Boundary must be a torch.Tensor"
        assert boundary.ndim == 3 and boundary.shape[0] == 1, f"Boundary shape must be (1, H, W), got {boundary.shape}"
        assert mask.shape[1:] == boundary.shape[1:], "Mask and boundary spatial sizes must match"

        assert isinstance(severity, torch.Tensor), "Severity must be a torch.Tensor"
        assert severity.ndim == 0, f"Severity must be scalar tensor, got ndim={severity.ndim}"
        assert severity.item() in {0, 1, 2}, f"Severity degree must be in {{0, 1, 2}}, got {severity.item()}"

    def test_dataloader_batch_dimensions(self):
        """Batch collation from get_multitask_dataloaders must produce correct tensor shapes."""
        train_csv = os.path.join(PROJECT_ROOT, "data/splits/train.csv")
        val_csv = os.path.join(PROJECT_ROOT, "data/splits/val.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            pytest.skip("Train or Val split CSV missing")

        batch_size = 4
        img_size = 256
        train_loader, val_loader = get_multitask_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=None,
            root_dir=PROJECT_ROOT,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,
        )

        batch = next(iter(train_loader))

        assert batch["image"].shape == (batch_size, 3, img_size, img_size), (
            f"Expected image batch shape {(batch_size, 3, img_size, img_size)}, got {batch['image'].shape}"
        )
        assert batch["mask"].shape == (batch_size, 1, img_size, img_size), (
            f"Expected mask batch shape {(batch_size, 1, img_size, img_size)}, got {batch['mask'].shape}"
        )
        assert batch["boundary"].shape == (batch_size, 1, img_size, img_size), (
            f"Expected boundary batch shape {(batch_size, 1, img_size, img_size)}, got {batch['boundary'].shape}"
        )
        assert batch["severity"].shape == (batch_size,), (
            f"Expected severity batch shape {(batch_size,)}, got {batch['severity'].shape}"
        )
