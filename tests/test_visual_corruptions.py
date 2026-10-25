"""tests/test_visual_corruptions.py

Comprehensive Unit & Integration Test Suite for Visual Corruptions (ITER-22).
Verifies:
1. Clean input identity: severity=0 returns an identical, unperturbed copy.
2. Bitwise determinism: same seed produces identical corrupted outputs.
3. Configurable severity: increasing severity yields strictly stronger perturbation.
4. Tensor shapes, data types, and finiteness (zero NaNs or Infs).
5. Immutability: input arrays, tensors, and labels are NEVER modified in-place.
6. Explicit distinction between frame-level optical corruptions and feature-space perturbations.
7. End-to-end robustness evaluation and aggregation on synthetic multimodal batches.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.robustness.evaluation import (
    audit_robustness_readiness,
    compute_robustness_aggregates,
    evaluate_model_under_corruption,
)
from src.robustness.video_corruptions import (
    FEATURE_CORRUPTIONS,
    FRAME_CORRUPTIONS,
    BaseVisualCorruption,
    CauterySmokeCorruption,
    DefocusBlurCorruption,
    FeatureDropoutCorruption,
    FeatureGaussianNoiseCorruption,
    FeatureScaleDriftCorruption,
    MotionBlurCorruption,
    PacketLossCorruption,
    ShotNoiseCorruption,
    TemporalFrameDropoutCorruption,
    get_visual_corruption,
    list_available_corruptions,
)
from src.utils.reproducibility import load_config


# ---------------------------------------------------------------------------
# 1. Base Interface & Registry Tests
# ---------------------------------------------------------------------------

def test_list_available_corruptions() -> None:
    """Verify registry lists all 5 frame-level and 4 feature-level corruptions."""
    corr_dict = list_available_corruptions()
    assert "frame_level" in corr_dict
    assert "feature_level" in corr_dict

    assert set(corr_dict["frame_level"]) == {
        "defocus_blur", "motion_blur", "shot_noise", "cautery_smoke", "packet_loss"
    }
    assert set(corr_dict["feature_level"]) == {
        "feature_gaussian_noise", "feature_dropout", "temporal_frame_dropout", "feature_scale_drift"
    }


def test_get_visual_corruption_factory() -> None:
    """Test instantiating corruptions by name and invalid name rejection."""
    op = get_visual_corruption("defocus_blur", severity=3, seed=42)
    assert isinstance(op, DefocusBlurCorruption)
    assert op.level == "frame"
    assert op.severity == 3

    feat_op = get_visual_corruption("feature_gaussian_noise", severity=2, seed=123)
    assert isinstance(feat_op, FeatureGaussianNoiseCorruption)
    assert feat_op.level == "feature"

    with pytest.raises(KeyError, match="Unknown corruption"):
        get_visual_corruption("nonexistent_corruption")


def test_severity_range_validation() -> None:
    """Verify severity must be integer within [0, 5]."""
    # Valid
    DefocusBlurCorruption(severity=0)
    DefocusBlurCorruption(severity=5)

    # Invalid
    with pytest.raises(ValueError, match="Severity must be in range"):
        DefocusBlurCorruption(severity=-1)

    with pytest.raises(ValueError, match="Severity must be in range"):
        DefocusBlurCorruption(severity=6)


# ---------------------------------------------------------------------------
# 2. Frame-Level Optical Corruptions Tests (Endo-C6 Inspired)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_frame() -> np.ndarray:
    """Synthetic RGB frame [H, W, 3] in uint8."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_rgb_clip() -> np.ndarray:
    """Synthetic RGB clip [T, H, W, 3] in uint8."""
    np.random.seed(42)
    return np.random.randint(0, 256, (4, 48, 48, 3), dtype=np.uint8)


def test_frame_corruptions_clean_input_identity(sample_rgb_frame: np.ndarray) -> None:
    """Verify that severity=0 returns an identical, unperturbed copy for all frame operators."""
    for c_name, cls in FRAME_CORRUPTIONS.items():
        op = cls(severity=0)
        out = op.apply(sample_rgb_frame)
        assert np.array_equal(out, sample_rgb_frame), f"{c_name} at severity=0 must be clean identity."


def test_frame_corruptions_determinism(sample_rgb_frame: np.ndarray) -> None:
    """Verify that applying with the same seed produces bitwise identical results."""
    for c_name, cls in FRAME_CORRUPTIONS.items():
        op1 = cls(severity=3, seed=999)
        op2 = cls(severity=3, seed=999)
        out1 = op1.apply(sample_rgb_frame)
        out2 = op2.apply(sample_rgb_frame)
        assert np.array_equal(out1, out2), f"{c_name} must be deterministic for identical seed."


def test_frame_corruptions_immutability(sample_rgb_frame: np.ndarray) -> None:
    """Verify that input image array is never modified in-place."""
    frame_backup = copy.deepcopy(sample_rgb_frame)
    for c_name, cls in FRAME_CORRUPTIONS.items():
        op = cls(severity=4, seed=42)
        _ = op.apply(sample_rgb_frame)
        assert np.array_equal(sample_rgb_frame, frame_backup), f"{c_name} mutated input in-place!"


def test_frame_corruptions_clip_support(sample_rgb_clip: np.ndarray) -> None:
    """Verify that all frame corruptions support 4D temporal video clips [T, H, W, C]."""
    for c_name, cls in FRAME_CORRUPTIONS.items():
        op = cls(severity=2, seed=42)
        out = op.apply(sample_rgb_clip)
        assert out.shape == sample_rgb_clip.shape, f"{c_name} altered 4D clip shape."
        assert np.all(np.isfinite(out)), f"{c_name} produced non-finite values in 4D clip."


def test_frame_corruptions_torch_support(sample_rgb_frame: np.ndarray) -> None:
    """Verify that frame corruptions support PyTorch tensors seamlessly."""
    t_frame = torch.from_numpy(sample_rgb_frame)
    op = DefocusBlurCorruption(severity=3)
    out_t = op.apply(t_frame)
    assert isinstance(out_t, torch.Tensor)
    assert out_t.shape == t_frame.shape
    assert out_t.dtype == t_frame.dtype


def test_frame_corruptions_monotonic_severity_diff(sample_rgb_frame: np.ndarray) -> None:
    """Verify that higher severity produces strictly greater perturbation from clean."""
    for c_name in ["defocus_blur", "shot_noise", "cautery_smoke"]:
        op_mild = get_visual_corruption(c_name, severity=1, seed=42)
        op_severe = get_visual_corruption(c_name, severity=5, seed=42)

        out_mild = op_mild.apply(sample_rgb_frame).astype(np.float32)
        out_severe = op_severe.apply(sample_rgb_frame).astype(np.float32)
        clean = sample_rgb_frame.astype(np.float32)

        diff_mild = np.mean(np.abs(out_mild - clean))
        diff_severe = np.mean(np.abs(out_severe - clean))

        assert diff_severe > diff_mild, f"{c_name}: severity 5 must alter frame more than severity 1."


# ---------------------------------------------------------------------------
# 3. Feature-Space Perturbations Tests (Post-Encoder)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_window() -> np.ndarray:
    """Synthetic windowed visual feature representation [W=16, D=512]."""
    np.random.seed(42)
    return np.random.randn(16, 512).astype(np.float32)


def test_feature_corruptions_clean_input_identity(sample_feature_window: np.ndarray) -> None:
    """Verify that severity=0 returns an identical copy for all feature operators."""
    for c_name, cls in FEATURE_CORRUPTIONS.items():
        op = cls(severity=0)
        out = op.apply(sample_feature_window)
        assert np.array_equal(out, sample_feature_window), f"{c_name} at severity=0 must be clean identity."


def test_feature_corruptions_determinism(sample_feature_window: np.ndarray) -> None:
    """Verify that feature operators with fixed seed are bitwise deterministic."""
    for c_name, cls in FEATURE_CORRUPTIONS.items():
        op1 = cls(severity=3, seed=777)
        op2 = cls(severity=3, seed=777)
        out1 = op1.apply(sample_feature_window)
        out2 = op2.apply(sample_feature_window)
        assert np.array_equal(out1, out2), f"{c_name} must be deterministic for identical seed."


def test_feature_corruptions_immutability(sample_feature_window: np.ndarray) -> None:
    """Verify that source feature array is never mutated in-place."""
    feat_backup = copy.deepcopy(sample_feature_window)
    for c_name, cls in FEATURE_CORRUPTIONS.items():
        op = cls(severity=3, seed=42)
        _ = op.apply(sample_feature_window)
        assert np.array_equal(sample_feature_window, feat_backup), f"{c_name} mutated input in-place!"


def test_feature_corruptions_torch_tensor_support(sample_feature_window: np.ndarray) -> None:
    """Verify that feature operators work directly with torch.Tensor (e.g. in PyTorch DataLoader)."""
    t_feat = torch.from_numpy(sample_feature_window)
    for c_name, cls in FEATURE_CORRUPTIONS.items():
        op = cls(severity=2, seed=42)
        out_t = op.apply(t_feat)
        assert isinstance(out_t, torch.Tensor)
        assert out_t.shape == t_feat.shape
        assert torch.all(torch.isfinite(out_t)), f"{c_name} produced non-finite values in torch tensor."


def test_temporal_frame_dropout_zeros_entire_timesteps(sample_feature_window: np.ndarray) -> None:
    """Verify temporal frame dropout zeros entire D-dimensional vectors at dropped timesteps."""
    op = TemporalFrameDropoutCorruption(severity=5, seed=42)
    corrupted = op.apply(sample_feature_window)

    # Some rows must be identically zero
    row_norms = np.linalg.norm(corrupted, axis=-1)
    zero_rows = np.sum(row_norms == 0.0)
    assert zero_rows > 0, "Severity 5 temporal dropout must drop at least one timestep."


# ---------------------------------------------------------------------------
# 4. Robustness Evaluation Runner & Metrics Aggregation Tests
# ---------------------------------------------------------------------------

class ToyDataset(Dataset):
    """Simple toy dataset for evaluation integration test."""

    def __init__(self) -> None:
        torch.manual_seed(42)
        self.video = torch.randn(10, 16, 512)
        self.kinematics = torch.randn(10, 16, 76)
        self.target = torch.randint(0, 3, (10,))

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "video": self.video[idx],
            "kinematics": self.kinematics[idx],
            "target": self.target[idx],
        }


class ToyClassifier(nn.Module):
    """Toy classifier for evaluation integration test."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.fc(x)


def test_evaluate_model_under_corruption_clean_vs_corrupted() -> None:
    """Test evaluate_model_under_corruption on clean vs corrupted inputs."""
    device = torch.device("cpu")
    model = ToyClassifier().to(device)
    dataset = ToyDataset()
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    class_names = {0: "G1", 1: "G2", 2: "G3"}

    # 1. Clean run
    clean_res = evaluate_model_under_corruption(
        model=model,
        dataloader=loader,
        num_classes=3,
        class_names=class_names,
        device=device,
        corruption=None,
        model_type="video_only",
    )
    assert "metrics" in clean_res
    assert clean_res["corruption"]["is_identity"] is True
    assert len(clean_res["y_pred"]) == 10

    # 2. Corrupted run
    op = get_visual_corruption("feature_gaussian_noise", severity=5, seed=42)
    corrupted_res = evaluate_model_under_corruption(
        model=model,
        dataloader=loader,
        num_classes=3,
        class_names=class_names,
        device=device,
        corruption=op,
        model_type="video_only",
    )
    assert corrupted_res["corruption"]["name"] == "feature_gaussian_noise"
    assert corrupted_res["corruption"]["severity"] == 5
    assert len(corrupted_res["y_pred"]) == 10


def test_compute_robustness_aggregates_calculation() -> None:
    """Test standard Endo-C6 aggregate calculation (Clean, Mean-C, Worst-C, Drop)."""
    clean_f1 = 0.80
    corrupted_f1s = {
        "defocus_blur": 0.70,
        "motion_blur": 0.60,
        "shot_noise": 0.50,
    }

    aggs = compute_robustness_aggregates(clean_metric=clean_f1, corrupted_metrics=corrupted_f1s)
    assert aggs["clean"] == 0.80
    assert pytest.approx(aggs["mean_c"], abs=1e-5) == (0.70 + 0.60 + 0.50) / 3.0
    assert aggs["worst_c"] == 0.50
    # Relative drop = (0.80 - 0.60) / 0.80 = 0.25 (25%)
    assert pytest.approx(aggs["relative_drop"], abs=1e-5) == 0.25


def test_audit_robustness_readiness_detects_blocked_status() -> None:
    """Verify that audit_robustness_readiness cleanly flags all tasks as BLOCKED."""
    cfg = load_config("configs/robustness.yaml")

    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        audit = audit_robustness_readiness(task=task, cfg=cfg)
        assert audit["task"] == task
        assert audit["total_folds"] == 8
        assert audit["status"] == "BLOCKED"
        assert len(audit["ready_folds"]) == 0
