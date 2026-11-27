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

from src.vision.edl_loss import (
    EDLLoss,
    get_edl_metrics,
    calculate_ece,
    logits_to_evidence,
    compute_selective_prediction_metrics,
)
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
    assert edl_fn2.loss_type == "digamma"
    assert edl_fn2.class_weights is not None


def test_edl_loss_types_and_class_weights():
    """Kiểm tra cả 3 chế độ digamma/log/mse, class_weights và kl_weight capping"""
    logits = torch.tensor([[2.0, 0.5, -0.5], [-0.5, 0.2, 1.5], [0.1, 1.2, 0.3]], requires_grad=True)
    evidence = F.softplus(logits)
    targets = torch.tensor([0, 2, 1])
    cw = torch.tensor([0.8, 0.8, 2.0])

    for lt in ("digamma", "log", "mse"):
        fn = EDLLoss(num_classes=3, annealing_epochs=10, loss_type=lt, kl_weight=0.2, class_weights=cw)
        loss = fn(evidence, targets, epoch=5)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0

    # KL weight nhỏ hơn phải tạo ra loss nhỏ hơn khi có bằng chứng sai lệch ở epoch >= annealing_epochs
    fn_low_kl = EDLLoss(num_classes=3, annealing_epochs=10, loss_type="digamma", kl_weight=0.1)
    fn_high_kl = EDLLoss(num_classes=3, annealing_epochs=10, loss_type="digamma", kl_weight=1.0)
    assert fn_low_kl(evidence, targets, epoch=10).item() < fn_high_kl(evidence, targets, epoch=10).item()


def test_logits_to_evidence_exp_non_vanishing_gradient():
    """Kiểm tra: Log-Evidence ('exp') duy trì gradient mạnh hơn 'softplus' khi logit lớp đúng cao."""
    fn = EDLLoss(num_classes=3, annealing_epochs=10, loss_type="digamma", kl_weight=0.0)
    targets = torch.tensor([0])

    z_exp = torch.tensor([[5.0, -2.0, -2.0]], requires_grad=True)
    loss_exp = fn(logits_to_evidence(z_exp, mode="exp"), targets, epoch=0)
    loss_exp.backward()

    z_sp = torch.tensor([[5.0, -2.0, -2.0]], requires_grad=True)
    loss_sp = fn(logits_to_evidence(z_sp, mode="softplus"), targets, epoch=0)
    loss_sp.backward()

    assert torch.isfinite(z_exp.grad).all()
    p_exp, u_exp = get_edl_metrics(logits_to_evidence(z_exp.detach(), mode="exp"), num_classes=3)
    p_sp, u_sp = get_edl_metrics(logits_to_evidence(z_sp.detach(), mode="softplus"), num_classes=3)
    # Với cùng logit z_0 = 5.0, chế độ 'exp' cho xác suất hiệu chuẩn cao hơn hẳn 'softplus' (vốn bị kẹt ~75%)
    assert p_exp[0, 0].item() > 0.95 and p_sp[0, 0].item() < 0.80
    assert u_exp.item() < u_sp.item()


def test_selective_prediction_and_mann_whitney_pvalue():
    """Kiểm tra: Risk-Coverage Curve giảm rủi ro khi giảm Coverage và Mann-Whitney U cho p < 0.01."""
    # Giả lập 40 mẫu: 30 ca đúng có u thấp (0.05..0.20), 10 ca sai có u cao (0.45..0.85)
    probs_correct = torch.tensor([[0.85, 0.10, 0.05]] * 10 + [[0.05, 0.85, 0.10]] * 10 + [[0.05, 0.10, 0.85]] * 10)
    targets_correct = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    u_correct = torch.linspace(0.05, 0.20, 30)

    probs_error = torch.tensor([[0.40, 0.35, 0.25]] * 10)  # Dự đoán lớp 0 nhưng thực tế lớp 1/2
    targets_error = torch.tensor([1] * 5 + [2] * 5)
    u_error = torch.linspace(0.45, 0.85, 10)

    probs = torch.cat([probs_correct, probs_error], dim=0)
    targets = torch.cat([targets_correct, targets_error], dim=0)
    u = torch.cat([u_correct, u_error], dim=0)

    sel = compute_selective_prediction_metrics(probs, u, targets)
    assert sel["n_total"] == 40
    assert sel["n_correct"] == 30 and sel["n_error"] == 10
    assert sel["u_error_mean"] > sel["u_correct_mean"]
    assert sel["p_value"] < 0.001
    assert sel["statistically_significant_p001"] is True
    assert sel["error_detection_auroc"] > 0.95
    # Tại Coverage 75% (giữ 30 ca tự tin nhất), Accuracy phải đạt 100% (so với 75% ở 100% Coverage)
    cov_100 = [r for r in sel["coverage_table"] if r["target_coverage"] == 1.0][0]
    cov_75 = [r for r in sel["coverage_table"] if r["target_coverage"] == 0.75][0]
    assert cov_75["accuracy"] > cov_100["accuracy"]
    assert cov_75["accuracy"] == 1.0


def test_asymmetric_kl_protects_minority_class():
    """Kiểm tra: Asymmetric Class-Balanced KL phạt nhẹ hơn với bằng chứng nhầm sang lớp hiếm Độ 3 (cw cao)."""
    cw = torch.tensor([0.8, 0.8, 2.5])
    fn = EDLLoss(num_classes=3, annealing_epochs=10, loss_type="digamma", kl_weight=0.2, class_weights=cw)

    # Cùng nhãn thật là lớp 1 (Độ 2): mẫu A nghi ngờ sang lớp 0 (đa số), mẫu B nghi ngờ sang lớp 2 (Độ 3 hiếm)
    ev_spill_to_maj = torch.tensor([[8.0, 10.0, 0.0]])
    ev_spill_to_min = torch.tensor([[0.0, 10.0, 8.0]])
    target = torch.tensor([1])

    loss_maj = fn(ev_spill_to_maj, target, epoch=10).item()
    loss_min = fn(ev_spill_to_min, target, epoch=10).item()
    # Phạt KL khi nghi ngờ sang Độ 3 phải nhỏ hơn phạt khi nghi ngờ sang Độ 1 để không dập tắt Recall Độ 3
    assert loss_min < loss_maj
