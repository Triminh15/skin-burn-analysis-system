"""
Unit Tests for Evidential Deep Learning (EDL) Loss and Metrics
Tests for: src/vision/edl_loss.py and integration with src/vision/losses.py
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.edl_loss import EDLLoss, get_edl_metrics, calculate_ece
from src.vision.losses import get_classification_loss


def test_edl_loss_forward_and_backward():
    """Kiểm tra: EDLLoss tính toán đúng shape, ra loss dương và lan truyền gradient ngược thành công"""
    batch_size = 8
    num_classes = 3
    loss_fn = EDLLoss(num_classes=num_classes, annealing_epochs=10)

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    evidence = F.softplus(logits)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(evidence, targets, epoch=1)
    assert loss.dim() == 0, "Loss phải là scalar tensor"
    assert loss.item() > 0, "Loss phải có giá trị dương"

    loss.backward()
    assert logits.grad is not None, "Gradient phải lan truyền về logits"
    assert not torch.isnan(logits.grad).any(), "Gradient không được chứa NaN"


def test_edl_loss_annealing_schedule():
    """Kiểm tra: Annealing schedule hoạt động đúng (lambda tăng dần theo epoch)"""
    num_classes = 3
    loss_fn = EDLLoss(num_classes=num_classes, annealing_epochs=10)

    logits = torch.tensor([
        [2.0, -1.0, -1.0],
        [-1.0, 3.0, -1.0],
        [-1.0, -1.0, 2.5]
    ], requires_grad=False)
    evidence = F.softplus(logits)
    targets = torch.tensor([0, 1, 2])

    loss_ep0 = loss_fn(evidence, targets, epoch=0).item()
    loss_ep5 = loss_fn(evidence, targets, epoch=5).item()
    loss_ep10 = loss_fn(evidence, targets, epoch=10).item()
    loss_ep20 = loss_fn(evidence, targets, epoch=20).item()

    assert loss_ep0 <= loss_ep5 <= loss_ep10, "Loss phải tăng khi hệ số phạt KL tăng theo epoch"
    assert abs(loss_ep10 - loss_ep20) < 1e-5, "Hệ số annealing bão hòa tại epoch >= annealing_epochs (1.0)"


def test_kl_divergence_properties():
    """Kiểm tra: KL divergence với uniform prior = 0 khi alpha = [1, 1, ..., 1] và luôn >= 0"""
    num_classes = 3
    loss_fn = EDLLoss(num_classes=num_classes)

    # alpha = [1, 1, 1] (Uniform prior)
    alpha_uniform = torch.ones((2, num_classes))
    kl_zero = loss_fn.kl_divergence(alpha_uniform)
    assert torch.allclose(kl_zero, torch.zeros_like(kl_zero), atol=1e-5), "KL divergence với chính uniform prior phải bằng 0"

    # alpha bất kỳ >= 1
    alpha_random = torch.rand((10, num_classes)) * 10.0 + 1.0
    kl_random = loss_fn.kl_divergence(alpha_random)
    assert (kl_random >= 0.0).all(), "KL divergence phải luôn không âm"


def test_get_edl_metrics_properties():
    """Kiểm tra: get_edl_metrics trả về đúng shape, xác suất chuẩn hóa tổng = 1 và độ bất định u in (0, 1]"""
    num_classes = 3
    batch_size = 12
    evidence = torch.rand(batch_size, num_classes) * 10.0

    probs, uncertainty = get_edl_metrics(evidence, num_classes=num_classes)

    assert probs.shape == (batch_size, num_classes)
    assert uncertainty.shape == (batch_size,)

    # Xác suất kỳ vọng cộng lại bằng 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), "Tổng xác suất kỳ vọng mỗi mẫu phải bằng 1"

    # Bounded uncertainty: u in (0, 1]
    assert (uncertainty > 0.0).all() and (uncertainty <= 1.0).all(), "Độ bất định phải thuộc (0, 1]"

    # Trường hợp không có bằng chứng (evidence = 0) -> u = 1.0, prob = 1/K
    zero_ev = torch.zeros(1, num_classes)
    p_zero, u_zero = get_edl_metrics(zero_ev, num_classes=num_classes)
    assert abs(u_zero.item() - 1.0) < 1e-5, "Khi evidence = 0, uncertainty phải đạt cực đại u = 1.0"
    assert torch.allclose(p_zero, torch.full((1, num_classes), 1.0 / num_classes)), "Khi evidence = 0, xác suất phải là đều 1/K"

    # Trường hợp bằng chứng rất cao -> u -> 0
    high_ev = torch.tensor([[100.0, 0.0, 0.0]])
    p_high, u_high = get_edl_metrics(high_ev, num_classes=num_classes)
    assert u_high.item() < 0.05, "Khi evidence cao, uncertainty phải rất nhỏ"
    assert p_high[0, 0].item() > 0.95, "Xác suất của lớp có bằng chứng cao phải vượt trội"


def test_calculate_ece():
    """Kiểm tra: Độ đo định chuẩn ECE (Expected Calibration Error)"""
    # Hoàn hảo: 100% tự tin và 100% chính xác -> ECE = 0
    probs_perfect = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets_perfect = torch.tensor([0, 1])
    ece_perf = calculate_ece(probs_perfect, targets_perfect, n_bins=5)
    assert abs(ece_perf) < 1e-5, "Mô hình định chuẩn hoàn hảo phải có ECE = 0"

    # Hoàn toàn tự tin sai: 100% tự tin nhưng 0% chính xác -> ECE = 1
    probs_wrong = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets_wrong = torch.tensor([1, 1])
    ece_wrong = calculate_ece(probs_wrong, targets_wrong, n_bins=5)
    assert abs(ece_wrong - 1.0) < 1e-5, "Mô hình overconfident hoàn toàn sai phải có ECE = 1"

    # Tensor numpy input test
    probs_np = np.array([[0.8, 0.2], [0.3, 0.7]])
    targets_np = np.array([0, 1])
    ece_np = calculate_ece(probs_np, targets_np, n_bins=5)
    assert 0.0 <= ece_np <= 1.0


def test_get_classification_loss_factory():
    """Kiểm tra: Factory get_classification_loss nạp đúng EDLLoss"""
    edl_fn = get_classification_loss("edl", num_classes=3, annealing_epochs=8)
    assert isinstance(edl_fn, EDLLoss)
    assert edl_fn.num_classes == 3
    assert edl_fn.annealing_epochs == 8

    # Test alias
    edl_fn2 = get_classification_loss("edl_loss", num_classes=3)
    assert isinstance(edl_fn2, EDLLoss)
