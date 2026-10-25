"""tests/test_adaptive_training.py

Unit and integration tests for reproducible adaptive reliability fusion training,
fold readiness auditing, checkpoint reproducibility, and single-modality fallback verification (ITER-21).
All training tests use synthetic data to verify algorithmic and architectural properties
without fabricating benchmark numbers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_adaptive_fusion import (
    build_adaptive_model,
    check_adaptive_fusion_readiness,
    evaluate_adaptive_model,
    run_adaptive_fusion_benchmark,
    train_one_epoch,
    verify_checkpoint_reproducibility,
)
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import (
    AdaptiveReliabilityFusion,
    BothModalitiesUnavailableError,
    ReliabilityGate,
)
from src.models.video_head import VideoGRU
from src.utils.reproducibility import load_config


# ---------------------------------------------------------------------------
# Synthetic Dataset for Reproducible Training Tests
# ---------------------------------------------------------------------------

class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset generating synchronized multimodal windows for testing."""

    def __init__(
        self,
        n_samples: int = 40,
        window_size: int = 16,
        video_dim: int = 512,
        kin_dim: int = 76,
        num_classes: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.video = torch.randn(n_samples, window_size, video_dim)
        self.kinematics = torch.randn(n_samples, window_size, kin_dim)
        self.targets = torch.randint(0, num_classes, (n_samples,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "video": self.video[idx],
            "kinematics": self.kinematics[idx],
            "target": self.targets[idx],
        }


class SpySubmodel(nn.Module):
    """Spy submodel to verify execution/skipping of modalities."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.fc(x)


# ---------------------------------------------------------------------------
# 1. Audit Current Training Readiness Tests
# ---------------------------------------------------------------------------

def test_check_adaptive_fusion_readiness_all_tasks() -> None:
    """Test that check_adaptive_fusion_readiness accurately audits all 3 tasks and 8 folds."""
    cfg = load_config("configs/adaptive_fusion.yaml")

    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        audit = check_adaptive_fusion_readiness(task, cfg)
        assert audit["task"] == task
        assert audit["total_folds"] == 8
        assert len(audit["fold_status"]) == 8
        assert not audit["all_ready"], f"Task {task} should be BLOCKED since feature extraction is incomplete."
        assert len(audit["ready_folds"]) == 0

        # Verify missing cache counts match real file system
        for f_num, f_info in audit["fold_status"].items():
            assert not f_info["ready"]
            assert f_info["missing_cache_count"] > 0
            assert f_info["missing_kinematics_count"] == 0  # 76D kinematics exist for all trials


def test_benchmark_runner_blocked_status() -> None:
    """Test that run_adaptive_fusion_benchmark cleanly reports BLOCKED when caches are incomplete."""
    res = run_adaptive_fusion_benchmark(
        task="Suturing",
        config_path="configs/adaptive_fusion.yaml",
        dry_run=False,
    )
    assert res["status"] == "BLOCKED"
    assert res["task"] == "Suturing"
    assert len(res["ready_folds"]) == 0


# ---------------------------------------------------------------------------
# 2. Model Builder & Architecture Tests
# ---------------------------------------------------------------------------

def test_build_adaptive_model() -> None:
    """Test building AdaptiveReliabilityFusion from configuration."""
    cfg = load_config("configs/adaptive_fusion.yaml")
    device = torch.device("cpu")
    num_classes = 10

    model = build_adaptive_model(cfg, num_classes=num_classes, device=device)

    assert isinstance(model, AdaptiveReliabilityFusion)
    assert isinstance(model.video_model, VideoGRU)
    assert isinstance(model.kinematics_model, KinematicsGRU)
    assert isinstance(model.gate, ReliabilityGate)
    assert model.num_classes == num_classes


# ---------------------------------------------------------------------------
# 3. Fallback Integration & Submodel Skipping Verification
# ---------------------------------------------------------------------------

def test_single_modality_fallback_skips_unobserved_submodels() -> None:
    """Verify that unobserved submodels are NOT executed when modality is absent."""
    C = 4
    v_spy = SpySubmodel(in_dim=16, num_classes=C)
    k_spy = SpySubmodel(in_dim=8, num_classes=C)
    fusion = AdaptiveReliabilityFusion(
        num_classes=C,
        video_model=v_spy,
        kinematics_model=k_spy,
        video_dim=C,
        kinematics_dim=C,
    )

    # 1. Video-Only: Kinematics submodel must NOT be executed
    v_x = torch.randn(2, 16)
    mask_v = torch.tensor([[True, False], [True, False]])
    fused_v, alpha_v = fusion(video_x=v_x, kinematics_x=None, availability_mask=mask_v)

    assert v_spy.call_count == 1
    assert k_spy.call_count == 0, "Kinematics model must not be called when kinematics is missing."
    assert alpha_v[:, 0].all() and not alpha_v[:, 1].any()
    assert fused_v.shape == (2, C)

    # 2. Kinematics-Only: Video submodel must NOT be executed
    k_x = torch.randn(2, 8)
    mask_k = torch.tensor([[False, True], [False, True]])
    fused_k, alpha_k = fusion(video_x=None, kinematics_x=k_x, availability_mask=mask_k)

    assert v_spy.call_count == 1, "Video model must not be called when video is missing."
    assert k_spy.call_count == 1
    assert not alpha_k[:, 0].any() and alpha_k[:, 1].all()
    assert fused_k.shape == (2, C)

    # 3. Both Available: Both submodels executed
    mask_both = torch.tensor([[True, True], [True, True]])
    fused_both, alpha_both = fusion(video_x=v_x, kinematics_x=k_x, availability_mask=mask_both)

    assert v_spy.call_count == 2
    assert k_spy.call_count == 2
    assert fused_both.shape == (2, C)

    # 4. Both Missing: Fails loudly with BothModalitiesUnavailableError
    mask_none = torch.tensor([[False, False], [False, False]])
    with pytest.raises(BothModalitiesUnavailableError, match="entirely unobserved"):
        fusion(video_x=None, kinematics_x=None, availability_mask=mask_none)

    assert v_spy.call_count == 2  # No calls executed on failure
    assert k_spy.call_count == 2


def test_fallback_input_validation_errors() -> None:
    """Verify informative errors when modality marked available is None."""
    C = 4
    v_spy = SpySubmodel(in_dim=16, num_classes=C)
    k_spy = SpySubmodel(in_dim=8, num_classes=C)
    fusion = AdaptiveReliabilityFusion(num_classes=C, video_model=v_spy, kinematics_model=k_spy)

    # Video marked available but video_x is None
    mask_v_avail = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="video_x is None"):
        fusion(video_x=None, kinematics_x=torch.randn(1, 8), availability_mask=mask_v_avail)

    # Kinematics marked available but kinematics_x is None
    mask_k_avail = torch.tensor([[False, True]])
    with pytest.raises(ValueError, match="kinematics_x is None"):
        fusion(video_x=torch.randn(1, 16), kinematics_x=None, availability_mask=mask_k_avail)


# ---------------------------------------------------------------------------
# 4. Reproducible Training & Evaluation Tests
# ---------------------------------------------------------------------------

def test_clean_training_one_epoch_synthetic() -> None:
    """Test training one epoch with simulated missingness on synthetic data."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    num_classes = 4

    v_model = SpySubmodel(in_dim=32, num_classes=num_classes)
    k_model = SpySubmodel(in_dim=16, num_classes=num_classes)
    gate = ReliabilityGate(video_dim=num_classes, kinematics_dim=num_classes)
    model = AdaptiveReliabilityFusion(
        num_classes=num_classes, video_model=v_model, kinematics_model=k_model, gate=gate
    ).to(device)

    dataset = SyntheticMultimodalDataset(
        n_samples=24, window_size=8, video_dim=32, kin_dim=16, num_classes=num_classes
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Epoch 1
    loss1 = train_one_epoch(model, dataloader, criterion, optimizer, device, missing_prob=0.2)
    assert np.isfinite(loss1)
    assert loss1 > 0.0

    # Epoch 2
    loss2 = train_one_epoch(model, dataloader, criterion, optimizer, device, missing_prob=0.2)
    assert np.isfinite(loss2)


def test_evaluate_adaptive_model_fallbacks() -> None:
    """Test evaluate_adaptive_model under clean, video-only, and kin-only conditions."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    num_classes = 3

    v_model = SpySubmodel(in_dim=16, num_classes=num_classes)
    k_model = SpySubmodel(in_dim=8, num_classes=num_classes)
    gate = ReliabilityGate(video_dim=num_classes, kinematics_dim=num_classes)
    model = AdaptiveReliabilityFusion(
        num_classes=num_classes, video_model=v_model, kinematics_model=k_model, gate=gate
    ).to(device)

    dataset = SyntheticMultimodalDataset(
        n_samples=16, window_size=4, video_dim=16, kin_dim=8, num_classes=num_classes
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    inv_vocab = {0: "G1", 1: "G2", 2: "G3"}

    # 1. Clean
    m_clean, y_t, y_p, w = evaluate_adaptive_model(
        model, loader, num_classes, inv_vocab, device, fixed_mask=(True, True)
    )
    assert "accuracy" in m_clean
    assert "macro_f1" in m_clean
    assert len(y_p) == 16
    assert w.shape == (16, 2)

    # 2. Video fallback
    m_v, _, _, _ = evaluate_adaptive_model(
        model, loader, num_classes, inv_vocab, device, fixed_mask=(True, False)
    )
    assert "accuracy" in m_v
    assert np.isfinite(m_v["accuracy"])

    # 3. Kinematics fallback
    m_k, _, _, _ = evaluate_adaptive_model(
        model, loader, num_classes, inv_vocab, device, fixed_mask=(False, True)
    )
    assert "accuracy" in m_k
    assert np.isfinite(m_k["accuracy"])


# ---------------------------------------------------------------------------
# 5. Checkpoint Reload Reproducibility Verification
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_reload_exact_reproducibility() -> None:
    """Test saving checkpoint and verifying exact reload reproducibility within tolerance."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    cfg = load_config("configs/adaptive_fusion.yaml")
    num_classes = 5

    # Build model and dataset
    model = build_adaptive_model(cfg, num_classes=num_classes, device=device)
    dataset = SyntheticMultimodalDataset(
        n_samples=16, window_size=16, video_dim=512, kin_dim=76, num_classes=num_classes
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_adaptive_ckpt.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Verify reproducibility function succeeds
        verified = verify_checkpoint_reproducibility(
            checkpoint_path=ckpt_path,
            original_model=model,
            dataloader=dataloader,
            cfg=cfg,
            num_classes=num_classes,
            device=device,
            tolerance=1e-6,
        )
        assert verified is True
