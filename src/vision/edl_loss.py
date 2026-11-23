"""
Module: src/vision/edl_loss.py
Mục đích: Định nghĩa hàm mất mát Evidential Deep Learning (EDL) dựa trên phân phối Dirichlet
và các hàm tiện ích tính toán độ bất định (Uncertainty) và xác suất kỳ vọng (Expected Probability).
Tài liệu tham khảo:
- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018.
- CS406 Research Architecture & Roadmap (Section 5 & 9.1).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EDLLoss(nn.Module):
    """
    Evidential Deep Learning Loss (Dirichlet Distribution)
    Dựa trên: Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018.

    Hàm mất mát gồm 2 thành phần:
    1. L_data: Expected Mean Squared Error giữa phân phối Dirichlet và One-hot Ground Truth.
    2. L_KL: KL Divergence regularization phạt bằng chứng sai lệch, kéo về phân phối đều Uniform Prior.
             Đi kèm hệ số annealing schedule: lambda_t = min(1.0, epoch / annealing_epochs).
    """
    def __init__(self, num_classes: int = 3, annealing_epochs: int = 10, from_logits: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.from_logits = from_logits

    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Tính toán KL Divergence giữa phân phối Dirichlet Alpha và Uniform Prior Beta = [1, 1, ..., 1].
        alpha: Tensor (B, num_classes) >= 1.0
        Returns:
            kl: Tensor (B, 1)
        """
        # Uniform prior Dirichlet: beta = [1, 1, ..., 1]
        beta = torch.ones((1, self.num_classes), dtype=alpha.dtype, device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def forward(self, evidence: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """
        evidence: Tensor (B, num_classes) >= 0 (đã qua F.softplus hoặc kích hoạt không âm)
                  Nếu from_logits=True hoặc có phần tử âm, tự động qua F.softplus.
        target: Tensor (B,) chứa class labels [0, 1, ..., num_classes - 1]
        epoch: Epoch hiện tại (0-based hoặc 1-based) dùng cho Annealing Schedule
        """
        if self.from_logits or torch.any(evidence < 0):
            evidence = F.softplus(evidence)

        # Chuyển nhãn target về dạng one-hot
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Tham số Dirichlet alpha_k = e_k + 1
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # 1. Expected MSE Loss (Khớp dữ liệu: (y - p)^2 + Var(p))
        loss_data = torch.sum((y_one_hot - p) ** 2, dim=1, keepdim=True)
        variance = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0)), dim=1, keepdim=True)
        loss_fit = torch.mean(loss_data + variance)
        
        # 2. KL Regularization với Annealing Schedule: lambda_t = min(1.0, epoch / annealing_epochs)
        if self.annealing_epochs > 0:
            annealing_coef = min(1.0, float(epoch) / float(self.annealing_epochs))
        else:
            annealing_coef = 1.0
            
        # alpha_tilde: loại bỏ phạt với lớp ground truth
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        loss_kl = torch.mean(self.kl_divergence(alpha_tilde))
        
        return loss_fit + annealing_coef * loss_kl


def get_edl_metrics(evidence: torch.Tensor, num_classes: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính Expected Probabilities và Epistemic Uncertainty từ evidence:
    - Expected Probability: p_k = alpha_k / S = (e_k + 1) / sum(e_j + 1)
    - Epistemic Uncertainty (vacuity): u = K / S in (0, 1]
    
    evidence: Tensor (B, num_classes) >= 0 (hoặc raw logits nếu có giá trị âm)
    num_classes: số lớp (mặc định 3)
    Returns:
        prob: Tensor (B, num_classes) - tổng mỗi hàng bằng 1
        uncertainty: Tensor (B,) - độ bất định nhận thức thuộc (0, 1]
    """
    if torch.any(evidence < 0):
        evidence = F.softplus(evidence)

    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    uncertainty = (num_classes / S).squeeze(-1)
    return prob, uncertainty


def calculate_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Tính Expected Calibration Error (ECE) - Độ đo định chuẩn xác suất chuẩn y tế.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    probs: Tensor (N, num_classes) hoặc numpy array chứa xác suất dự đoán
    targets: Tensor (N,) hoặc numpy array chứa nhãn thực tế [0, 1, ..., C-1]
    n_bins: Số lượng bin chia đều khoảng confidence [0, 1] (mặc định 10)
    Returns:
        ece: Giá trị ECE (float) thuộc [0, 1]
    """
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    probs = probs.float()
    targets = targets.long()

    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = in_bin.float().mean()
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item())


# ==============================================================================
# KIỂM THỬ ĐƠN VỊ NHANH (SANITY CHECK)
# ==============================================================================
if __name__ == "__main__":
    print("--- Testing Unit Tests for src/vision/edl_loss.py ---")
    
    # 1. Test EDLLoss shapes and backward
    loss_fn = EDLLoss(num_classes=3, annealing_epochs=10)
    
    dummy_logits = torch.randn(8, 3, requires_grad=True)
    dummy_evidence = F.softplus(dummy_logits)
    dummy_targets = torch.randint(0, 3, (8,))
    
    # Epoch 0 (no KL penalty)
    loss_ep0 = loss_fn(dummy_evidence, dummy_targets, epoch=0)
    loss_ep0.backward(retain_graph=True)
    print(f"[OK] EDLLoss (epoch 0): {loss_ep0.item():.4f} (Grad OK)")
    
    # Epoch 10 (full KL penalty)
    loss_ep10 = loss_fn(dummy_evidence, dummy_targets, epoch=10)
    print(f"[OK] EDLLoss (epoch 10): {loss_ep10.item():.4f}")
    assert loss_ep10.item() >= loss_ep0.item(), "Loss with KL should be >= Loss without KL"
    
    # 2. Test get_edl_metrics
    probs, uncertainty = get_edl_metrics(dummy_evidence, num_classes=3)
    print(f"[OK] Probs shape: {probs.shape}, Uncertainty shape: {uncertainty.shape}")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8)), "Probabilities must sum to 1"
    assert torch.all(uncertainty > 0.0) and torch.all(uncertainty <= 1.0), "Uncertainty must be in (0, 1]"
    
    # Test high evidence -> low uncertainty
    high_ev = torch.tensor([[50.0, 1.0, 0.0]])
    p_high, u_high = get_edl_metrics(high_ev, num_classes=3)
    print(f"[OK] High evidence uncertainty: {u_high.item():.4f} (expected ~0.055)")
    assert u_high.item() < 0.1, "High evidence should yield low uncertainty"
    
    # Test zero evidence -> max uncertainty
    zero_ev = torch.tensor([[0.0, 0.0, 0.0]])
    p_zero, u_zero = get_edl_metrics(zero_ev, num_classes=3)
    print(f"[OK] Zero evidence uncertainty: {u_zero.item():.4f} (expected 1.0)")
    assert abs(u_zero.item() - 1.0) < 1e-5, "Zero evidence should yield uncertainty = 1.0"
    
    # 3. Test calculate_ece
    dummy_probs = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.3, 0.4, 0.3],
        [0.1, 0.1, 0.8]
    ])
    dummy_labels = torch.tensor([0, 0, 1, 2])
    ece_val = calculate_ece(dummy_probs, dummy_labels, n_bins=5)
    print(f"[OK] ECE test: {ece_val:.4f}")
    assert 0.0 <= ece_val <= 1.0, "ECE must be in [0, 1]"
    
    print("[SUCCESS] All EDLLoss and metrics sanity checks passed!")
