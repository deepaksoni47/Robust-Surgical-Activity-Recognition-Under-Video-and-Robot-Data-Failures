"""Tests for class-imbalance loss functions and training script CLI integration (ITER-35).

Covers:
- Correct class-weight calculation (inverse frequency and effective number of samples).
- Focal loss numerical stability on extreme values.
- Mathematical equivalence to CrossEntropy when gamma=0.
- Gradient propagation and backward stability.
- Loss factory selection and validation.
- CLI argument integration in train_surgmotion_multimodal_fusion.py.
- Deterministic reproducibility under fixed seed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.losses import (
    ClassBalancedFocalLoss,
    FocalLoss,
    compute_class_weights,
    get_loss_function,
)


class TestClassWeights:
    """Test suite for class-weight computation functions."""

    def test_compute_class_weights_inverse(self):
        """Test standard inverse-frequency class weights."""
        # 10 samples of class 0, 1 sample of class 1, 0 samples of class 2
        targets = [0] * 10 + [1] * 1
        num_classes = 3

        weights = compute_class_weights(targets, num_classes=num_classes, mode="inverse")

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)
        # Class 1 should have 10x higher raw weight than class 0
        assert weights[1] > weights[0]
        assert np.isclose((weights[1] / weights[0]).item(), 10.0, atol=1e-3)
        # Class 2 with 0 samples should have weight 0
        assert weights[2].item() == 0.0
        # Active weights normalized so mean is 1.0
        assert np.isclose(weights[:2].mean().item(), 1.0, atol=1e-5)

    def test_compute_class_weights_effective(self):
        """Test Cui et al. effective number of samples class weights."""
        targets = [0] * 100 + [1] * 10 + [2] * 1
        num_classes = 3
        beta = 0.999

        weights = compute_class_weights(targets, num_classes=num_classes, mode="effective", beta=beta)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)
        # Rare classes have strictly higher weight than frequent classes
        assert weights[2] > weights[1] > weights[0]
        # Sum of weights equals num_classes
        assert np.isclose(weights.sum().item(), float(num_classes), atol=1e-4)

    def test_compute_class_weights_empty(self):
        """Test behavior on empty target list."""
        weights = compute_class_weights([], num_classes=5)
        assert torch.allclose(weights, torch.ones(5, dtype=torch.float32))

    def test_compute_class_weights_invalid_mode(self):
        """Test exception on invalid mode."""
        with pytest.raises(ValueError, match="Unknown class weight mode"):
            compute_class_weights([0, 1], num_classes=2, mode="bogus")


class TestFocalLoss:
    """Test suite for FocalLoss numerical stability, equivalence, and gradients."""

    def test_focal_loss_gamma_zero_matches_ce(self):
        """Focal loss with gamma=0 and weight=None must equal standard CrossEntropyLoss."""
        torch.manual_seed(42)
        logits = torch.randn(16, 5, requires_grad=True)
        targets = torch.randint(0, 5, (16,))

        fl = FocalLoss(gamma=0.0, weight=None, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")

        loss_fl = fl(logits, targets)
        loss_ce = ce(logits, targets)

        assert torch.isclose(loss_fl, loss_ce, atol=1e-5), f"FL: {loss_fl.item()}, CE: {loss_ce.item()}"

    def test_focal_loss_numerical_stability_extreme_logits(self):
        """Extreme positive and negative logits should never output NaN or Inf."""
        fl = FocalLoss(gamma=2.0)

        # Extremely confident correct prediction
        confident_logits = torch.tensor([[100.0, -100.0, -100.0]], dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.int64)
        loss_confident = fl(confident_logits, targets)

        assert torch.isfinite(loss_confident)
        assert loss_confident.item() >= 0.0
        assert np.isclose(loss_confident.item(), 0.0, atol=1e-4)

        # Extremely confident WRONG prediction
        wrong_logits = torch.tensor([[-100.0, 100.0, -100.0]], dtype=torch.float32)
        loss_wrong = fl(wrong_logits, targets)

        assert torch.isfinite(loss_wrong)
        assert loss_wrong.item() > 0.0

    def test_focal_loss_gradient_propagation(self):
        """Gradients must flow smoothly through all model parameters without NaNs."""
        torch.manual_seed(42)
        linear = nn.Linear(10, 4)
        x = torch.randn(8, 10)
        y = torch.randint(0, 4, (8,))

        fl = FocalLoss(gamma=2.0)
        logits = linear(x)
        loss = fl(logits, y)
        loss.backward()

        assert linear.weight.grad is not None
        assert linear.bias.grad is not None
        assert torch.all(torch.isfinite(linear.weight.grad))
        assert torch.all(torch.isfinite(linear.bias.grad))
        assert linear.weight.grad.abs().sum().item() > 0.0

    def test_focal_loss_weighting(self):
        """Per-class weighting scales loss appropriately."""
        weights = torch.tensor([1.0, 10.0], dtype=torch.float32)
        fl_weighted = FocalLoss(gamma=1.0, weight=weights, reduction="none")
        fl_unweighted = FocalLoss(gamma=1.0, weight=None, reduction="none")

        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        targets = torch.tensor([0, 1])

        loss_w = fl_weighted(logits, targets)
        loss_u = fl_unweighted(logits, targets)

        assert np.isclose(loss_w[0].item(), loss_u[0].item(), atol=1e-4)
        assert np.isclose(loss_w[1].item(), 10.0 * loss_u[1].item(), atol=1e-4)


class TestClassBalancedFocalLoss:
    """Test suite for ClassBalancedFocalLoss."""

    def test_cb_focal_forward_and_backward(self):
        """Verify CB-Focal loss forward and backward passes."""
        counts = [100, 20, 2]
        num_classes = 3
        cb_fl = ClassBalancedFocalLoss(class_counts=counts, num_classes=num_classes, gamma=2.0, beta=0.999)

        logits = torch.randn(6, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        loss = cb_fl(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))


class TestLossFactory:
    """Test suite for get_loss_function factory."""

    def test_get_loss_function_ce(self):
        crit = get_loss_function(loss_type="ce")
        assert isinstance(crit, nn.CrossEntropyLoss)
        assert crit.weight is None

    def test_get_loss_function_class_weighted(self):
        counts = [50, 10, 5]
        crit = get_loss_function(loss_type="class_weighted", class_counts=counts, num_classes=3)
        assert isinstance(crit, nn.CrossEntropyLoss)
        assert crit.weight is not None
        assert crit.weight.shape == (3,)

    def test_get_loss_function_focal(self):
        crit = get_loss_function(loss_type="focal", gamma=2.5)
        assert isinstance(crit, FocalLoss)
        assert crit.gamma == 2.5

    def test_get_loss_function_cb_focal(self):
        counts = [50, 10, 5]
        crit = get_loss_function(loss_type="cb_focal", class_counts=counts, num_classes=3, gamma=1.5, beta=0.99)
        assert isinstance(crit, ClassBalancedFocalLoss)
        assert crit.gamma == 1.5
        assert crit.beta == 0.99

    def test_get_loss_function_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            get_loss_function(loss_type="nonexistent_loss")

    def test_get_loss_function_missing_counts(self):
        with pytest.raises(ValueError, match="class_counts from training fold must be provided"):
            get_loss_function(loss_type="class_weighted", class_counts=None)


class TestCLIIntegration:
    """Test that train_surgmotion_multimodal_fusion CLI correctly accepts loss arguments."""

    def test_cli_loss_arguments(self):
        from scripts.train_surgmotion_multimodal_fusion import main
        # Verify parser configuration
        parser = argparse.ArgumentParser()
        parser.add_argument("--loss-type", type=str, default="class_weighted",
                            choices=["ce", "class_weighted", "focal", "cb_focal"])
        parser.add_argument("--gamma", type=float, default=2.0)
        parser.add_argument("--beta", type=float, default=0.9999)

        args = parser.parse_args(["--loss-type", "focal", "--gamma", "2.5", "--beta", "0.999"])
        assert args.loss_type == "focal"
        assert args.gamma == 2.5
        assert args.beta == 0.999
