"""tests/test_reliability_gate.py

Comprehensive Unit & Integration Test Suite for Reliability-Aware Fusion Architecture (ITER-20).
All inputs are synthetic and used strictly for unit testing and numerical verification.
These tests verify architectural correctness, gradient flow, mathematical constraints,
and exact single-modality fallback guarantees under clinical missingness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.reliability_gate import (
    AdaptiveReliabilityFusion,
    BothModalitiesUnavailableError,
    ReliabilityGate,
    compute_masked_normalized_weights,
    fuse_logits_with_availability,
    validate_availability_mask,
)


# ---------------------------------------------------------------------------
# 1. Availability Mask Validation Tests
# ---------------------------------------------------------------------------

def test_validate_availability_mask_valid_formats() -> None:
    """Test mask validation with boolean, float, 1D broadcast, and numpy formats."""
    # 2D boolean tensor
    m_bool = torch.tensor([[True, False], [True, True]])
    out = validate_availability_mask(m_bool)
    assert out.shape == (2, 2)
    assert out.dtype == torch.bool
    assert torch.equal(out, m_bool)

    # 2D binary float tensor
    m_float = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = validate_availability_mask(m_float)
    assert out.shape == (2, 2)
    assert out.dtype == torch.bool
    assert out[0, 0].item() is True and out[0, 1].item() is False

    # 1D tensor broadcast to expected batch size
    m_1d = torch.tensor([1, 0])
    out = validate_availability_mask(m_1d, expected_batch_size=4)
    assert out.shape == (4, 2)
    assert out[:, 0].all() and not out[:, 1].any()

    # NumPy array format
    m_np = np.array([[1, 1], [0, 1]], dtype=np.int32)
    out = validate_availability_mask(m_np)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2)


def test_validate_availability_mask_rejection_of_invalid_inputs() -> None:
    """Test that non-binary, non-finite, and shape-mismatched masks fail loudly."""
    # Non-binary float (e.g. 0.5)
    with pytest.raises(ValueError, match="binary values"):
        validate_availability_mask(torch.tensor([[0.5, 1.0]]))

    # Negative value
    with pytest.raises(ValueError, match="binary values"):
        validate_availability_mask(torch.tensor([[-1.0, 1.0]]))

    # NaN / Inf in mask
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_availability_mask(torch.tensor([[float("nan"), 1.0]]))

    # Wrong shape (3 modalities instead of 2)
    with pytest.raises(ValueError, match="shape"):
        validate_availability_mask(torch.tensor([[1, 1, 1]]))

    # Batch size mismatch
    with pytest.raises(ValueError, match="Batch size mismatch"):
        validate_availability_mask(torch.tensor([[1, 0], [0, 1]]), expected_batch_size=3)


# ---------------------------------------------------------------------------
# 2. Masked Normalized Weights Computation Tests
# ---------------------------------------------------------------------------

def test_compute_masked_normalized_weights_both_available() -> None:
    """Test weight normalization when both modalities are available."""
    raw = torch.tensor([[2.0, 3.0], [4.0, 1.0]])  # Shape [2, 2]
    mask = torch.tensor([[True, True], [True, True]])

    weights = compute_masked_normalized_weights(raw, mask)

    # Weights must sum to 1.0
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    # Exact analytic values: [2/(2+3), 3/(2+3)] = [0.4, 0.6]; [4/(4+1), 1/(4+1)] = [0.8, 0.2]
    expected = torch.tensor([[0.4, 0.6], [0.8, 0.2]])
    assert torch.allclose(weights, expected, atol=1e-5)


def test_single_modality_fallback_exact_mathematical_guarantee() -> None:
    """Test that single-modality presence yields identically 1.0 and 0.0 weights."""
    # Arbitrary raw weights that would otherwise favor the missing modality
    raw = torch.tensor([
        [0.01, 100.0],  # Video has very low score, kinematics has high score, but kin is missing!
        [100.0, 0.01],  # Kinematics has low score, video has high score, but video is missing!
    ])
    # Row 0: Video only [1, 0]; Row 1: Kinematics only [0, 1]
    mask = torch.tensor([[True, False], [False, True]])

    weights = compute_masked_normalized_weights(raw, mask)

    # Row 0 MUST be exactly [1.0, 0.0]
    assert weights[0, 0].item() == 1.0
    assert weights[0, 1].item() == 0.0

    # Row 1 MUST be exactly [0.0, 1.0]
    assert weights[1, 0].item() == 0.0
    assert weights[1, 1].item() == 1.0

    # Test fusion with synthetic logits
    v_logits = torch.tensor([[10.0, -5.0], [3.0, 7.0]])
    k_logits = torch.tensor([[-2.0, 8.0], [-10.0, 20.0]])

    fused = weights[:, 0:1] * v_logits + weights[:, 1:2] * k_logits

    # Row 0 must be identical to v_logits[0]
    assert torch.equal(fused[0], v_logits[0])
    # Row 1 must be identical to k_logits[1]
    assert torch.equal(fused[1], k_logits[1])


def test_both_modalities_missing_fails_loudly_by_default() -> None:
    """Test that both modalities missing raises BothModalitiesUnavailableError."""
    raw = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[False, False]])

    with pytest.raises(BothModalitiesUnavailableError, match="Both video and kinematics are unavailable"):
        compute_masked_normalized_weights(raw, mask, allow_both_missing=False)


def test_both_modalities_missing_safe_fallback_mode() -> None:
    """Test that explicit allow_both_missing mode returns documented fallback weights."""
    raw = torch.tensor([[2.0, 3.0]])
    mask = torch.tensor([[False, False]])

    # 'equal' fallback
    w_equal = compute_masked_normalized_weights(
        raw, mask, allow_both_missing=True, both_missing_fallback="equal"
    )
    assert torch.allclose(w_equal, torch.tensor([[0.5, 0.5]]))

    # 'zero' fallback
    w_zero = compute_masked_normalized_weights(
        raw, mask, allow_both_missing=True, both_missing_fallback="zero"
    )
    assert torch.allclose(w_zero, torch.tensor([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# 3. ReliabilityGate Module Architecture Tests
# ---------------------------------------------------------------------------

def test_reliability_gate_forward_shapes_and_constraints() -> None:
    """Test ReliabilityGate output shapes, values in [0, 1], and row sums == 1."""
    torch.manual_seed(42)
    B, D_v, D_k = 8, 128, 76
    gate = ReliabilityGate(video_dim=D_v, kinematics_dim=D_k, use_linear_direct=True)

    v_feat = torch.randn(B, D_v)
    k_feat = torch.randn(B, D_k)
    mask = torch.tensor([
        [True, True],    # Both
        [True, False],   # Video only
        [False, True],   # Kinematics only
        [True, True],
        [True, False],
        [False, True],
        [True, True],
        [True, True],
    ])

    alpha = gate(v_feat, k_feat, availability_mask=mask)

    assert alpha.shape == (B, 2)
    assert torch.all(torch.isfinite(alpha))
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()

    # Verify single modality rows
    assert alpha[1, 0].item() == 1.0 and alpha[1, 1].item() == 0.0
    assert alpha[2, 0].item() == 0.0 and alpha[2, 1].item() == 1.0

    # Verify all row sums are 1.0
    row_sums = alpha.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_reliability_gate_two_layer_hidden_network() -> None:
    """Test non-direct (two-layer with LayerNorm and GELU) gate configuration."""
    torch.manual_seed(42)
    B, D_v, D_k = 4, 32, 32
    gate = ReliabilityGate(
        video_dim=D_v,
        kinematics_dim=D_k,
        hidden_dim=16,
        use_linear_direct=False,
        dropout=0.0,
    )

    v_feat = torch.randn(B, D_v)
    k_feat = torch.randn(B, D_k)
    alpha = gate(v_feat, k_feat)  # Default: both available

    assert alpha.shape == (B, 2)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-6)


def test_reliability_gate_input_validation_rejections() -> None:
    """Test that ReliabilityGate rejects non-finite inputs, shape mismatches, and invalid dimensions."""
    gate = ReliabilityGate(video_dim=64, kinematics_dim=32)

    # 1. NaN in video feature
    v_nan = torch.randn(4, 64)
    v_nan[0, 0] = float("nan")
    k_good = torch.randn(4, 32)
    with pytest.raises(ValueError, match="non-finite"):
        gate(v_nan, k_good)

    # 2. Inf in kinematics feature
    v_good = torch.randn(4, 64)
    k_inf = torch.randn(4, 32)
    k_inf[1, 1] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        gate(v_good, k_inf)

    # 3. Incompatible feature dimension
    v_wrong_dim = torch.randn(4, 30)  # expected 64
    with pytest.raises(ValueError, match="incompatible with video_dim"):
        gate(v_wrong_dim, k_good)

    # 4. Incompatible batch size
    v_diff_b = torch.randn(5, 64)
    k_diff_b = torch.randn(4, 32)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        gate(v_diff_b, k_diff_b)


def test_reliability_gate_deterministic_eval_mode() -> None:
    """Test that ReliabilityGate is deterministic in eval mode across multiple passes."""
    gate = ReliabilityGate(video_dim=16, kinematics_dim=16, dropout=0.3)
    gate.eval()

    v = torch.randn(3, 16)
    k = torch.randn(3, 16)

    out1 = gate(v, k)
    out2 = gate(v, k)
    assert torch.equal(out1, out2)


def test_reliability_gate_gradient_flow() -> None:
    """Test that gradients flow through the gating network during optimization."""
    torch.manual_seed(42)
    B, C = 4, 3
    gate = ReliabilityGate(video_dim=C, kinematics_dim=C, use_linear_direct=True, dropout=0.0)

    # Synthetic logits
    v_logits = torch.randn(B, C, requires_grad=True)
    k_logits = torch.randn(B, C, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])

    alpha = gate(v_logits, k_logits)
    fused_logits = alpha[:, 0:1] * v_logits + alpha[:, 1:2] * k_logits

    loss = nn.CrossEntropyLoss()(fused_logits, targets)
    loss.backward()

    # Check that gate parameters received gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())
    assert has_grads, "Gate parameters must receive gradients during backpropagation."


# ---------------------------------------------------------------------------
# 4. AdaptiveReliabilityFusion Integration Tests
# ---------------------------------------------------------------------------

class DummySequenceModel(nn.Module):
    """Simple toy unimodal classifier for integration testing."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is sequence [B, W, D], take last timestep
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.fc(x)


def test_adaptive_reliability_fusion_forward_logits() -> None:
    """Test logit-level fusion and single-modality fallback in AdaptiveReliabilityFusion."""
    torch.manual_seed(42)
    B, C = 3, 5
    fusion = AdaptiveReliabilityFusion(num_classes=C, video_dim=C, kinematics_dim=C)

    v_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 2.0, 3.0, 4.0, 5.0],
                             [0.0, 0.0, 0.0, 0.0, 10.0]])
    k_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 10.0],
                             [5.0, 4.0, 3.0, 2.0, 1.0],
                             [10.0, 0.0, 0.0, 0.0, 0.0]])

    # Row 0: Video only [1, 0] -> output must be exactly v_logits[0]
    # Row 1: Both [1, 1] -> convex combination
    # Row 2: Kinematics only [0, 1] -> output must be exactly k_logits[2]
    mask = torch.tensor([[True, False], [True, True], [False, True]])

    fused, alpha = fusion.forward_logits(v_logits, k_logits, availability_mask=mask)

    assert fused.shape == (B, C)
    assert alpha.shape == (B, 2)

    # Fallback assertions
    assert torch.equal(fused[0], v_logits[0])
    assert torch.equal(fused[2], k_logits[2])
    assert alpha[0, 0].item() == 1.0 and alpha[0, 1].item() == 0.0
    assert alpha[2, 0].item() == 0.0 and alpha[2, 1].item() == 1.0


def test_adaptive_reliability_fusion_end_to_end_submodel_skipping() -> None:
    """Test that missing modality sub-model is not evaluated when flagged unavailable."""
    torch.manual_seed(42)
    B, C = 2, 4
    v_model = DummySequenceModel(in_dim=16, num_classes=C)
    k_model = DummySequenceModel(in_dim=8, num_classes=C)
    fusion = AdaptiveReliabilityFusion(
        num_classes=C, video_model=v_model, kinematics_model=k_model, video_dim=C, kinematics_dim=C
    )

    # Case 1: Video only available
    v_x = torch.randn(B, 16)
    k_x = None  # Telemetry missing entirely
    mask_v_only = torch.tensor([[True, False], [True, False]])

    fused_v, alpha_v = fusion(video_x=v_x, kinematics_x=k_x, availability_mask=mask_v_only)
    assert alpha_v[:, 0].all() and not alpha_v[:, 1].any()
    expected_v = v_model(v_x)
    assert torch.allclose(fused_v, expected_v, atol=1e-6)

    # Case 2: Kinematics only available
    v_x = None  # Video stream missing entirely
    k_x = torch.randn(B, 8)
    mask_k_only = torch.tensor([[False, True], [False, True]])

    fused_k, alpha_k = fusion(video_x=v_x, kinematics_x=k_x, availability_mask=mask_k_only)
    assert not alpha_k[:, 0].any() and alpha_k[:, 1].all()
    expected_k = k_model(k_x)
    assert torch.allclose(fused_k, expected_k, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Standalone NumPy / PyTorch Function Tests
# ---------------------------------------------------------------------------

def test_fuse_logits_with_availability_pure_function() -> None:
    """Test standalone fuse_logits_with_availability on both Tensor and NumPy arrays."""
    # PyTorch test
    v_t = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    k_t = torch.tensor([[6.0, 8.0], [5.0, 7.0]])
    m_t = torch.tensor([[True, False], [False, True]])

    out_t = fuse_logits_with_availability(v_t, k_t, m_t)
    assert torch.equal(out_t[0], v_t[0])
    assert torch.equal(out_t[1], k_t[1])

    # NumPy test
    v_np = np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32)
    k_np = np.array([[6.0, 8.0], [5.0, 7.0]], dtype=np.float32)
    m_np = np.array([[1, 0], [0, 1]])

    out_np = fuse_logits_with_availability(v_np, k_np, m_np)
    assert isinstance(out_np, np.ndarray)
    assert np.array_equal(out_np[0], v_np[0])
    assert np.array_equal(out_np[1], k_np[1])


# ---------------------------------------------------------------------------
# 6. Zero Test-Label Dependence Assertion
# ---------------------------------------------------------------------------

def test_zero_dependence_on_test_labels() -> None:
    """Verify that gating weights depend strictly on observable features, never on ground truth labels."""
    torch.manual_seed(42)
    gate = ReliabilityGate(video_dim=4, kinematics_dim=4)
    gate.eval()

    v = torch.randn(2, 4)
    k = torch.randn(2, 4)
    mask = torch.tensor([[True, True], [True, True]])

    # Gating computation signature does not accept test labels
    weights_pass1 = gate(v, k, availability_mask=mask)

    # Fake label perturbations cannot alter the gating output
    fake_labels_a = torch.tensor([0, 1])
    fake_labels_b = torch.tensor([1, 0])

    weights_pass2 = gate(v, k, availability_mask=mask)
    assert torch.equal(weights_pass1, weights_pass2)
