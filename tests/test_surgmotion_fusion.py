"""Unit and integration tests for SurgMotion + Kinematics Multimodal Fusion (ITER-28)."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from src.models.concat_fusion import ConcatFusionGRU
from src.models.reliability_gate import ReliabilityGate
from scripts.train_surgmotion_multimodal_fusion import (
    SurgMotionAdaptiveFusionModel,
    run_multimodal_benchmark,
)


def test_concat_fusion_gru_surgmotion_dimensions():
    """Verify ConcatFusionGRU forward pass and output shapes with 1024D SurgMotion + 76D kinematics."""
    batch_size = 4
    seq_len = 8
    num_classes = 11

    model = ConcatFusionGRU(
        video_dim=1024,
        kinematics_dim=76,
        proj_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
    )

    v_in = torch.randn(batch_size, seq_len, 1024)
    k_in = torch.randn(batch_size, seq_len, 76)

    logits = model(v_in, k_in)
    assert logits.shape == (batch_size, num_classes)
    assert torch.all(torch.isfinite(logits))

    # Sequential inference check
    seq_logits = model.forward_sequence(v_in, k_in)
    assert seq_logits.shape == (batch_size, seq_len, num_classes)
    assert torch.all(torch.isfinite(seq_logits))


def test_surgmotion_adaptive_fusion_single_modality_fallback():
    """Verify AdaptiveReliabilityFusion exact fallback when optical or robot telemetry drops."""
    batch_size = 2
    seq_len = 8
    num_classes = 7

    model = SurgMotionAdaptiveFusionModel(
        num_classes=num_classes,
        video_dim=1024,
        kinematics_dim=76,
        hidden_dim=64,
    )
    model.eval()

    v_in = torch.randn(batch_size, seq_len, 1024)
    k_in = torch.randn(batch_size, seq_len, 76)

    # 1. Clean condition: both available [1, 1]
    mask_both = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    _, alpha_both = model(v_in, k_in, availability_mask=mask_both)
    assert alpha_both.shape == (batch_size, 2)
    assert torch.allclose(alpha_both.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    # 2. Video missing condition: [0, 1] -> g_video identically 0.0, g_kin identically 1.0
    mask_no_v = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
    _, alpha_no_v = model(v_in, k_in, availability_mask=mask_no_v)
    assert torch.allclose(alpha_no_v[:, 0], torch.zeros(batch_size), atol=1e-6)
    assert torch.allclose(alpha_no_v[:, 1], torch.ones(batch_size), atol=1e-6)

    # 3. Kinematics missing condition: [1, 0] -> g_kin identically 0.0, g_video identically 1.0
    mask_no_k = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
    _, alpha_no_k = model(v_in, k_in, availability_mask=mask_no_k)
    assert torch.allclose(alpha_no_k[:, 0], torch.ones(batch_size), atol=1e-6)
    assert torch.allclose(alpha_no_k[:, 1], torch.zeros(batch_size), atol=1e-6)


def test_surgmotion_fusion_smoke_benchmark(tmp_path):
    """Verify smoke cross-validation execution of ConcatFusionGRU."""
    from scripts.train_surgmotion_multimodal_fusion import run_fixed_late_fusion_fold

    # Test late fusion fold
    res = run_fixed_late_fusion_fold(
        task="Suturing",
        fold=1,
        device=torch.device("cpu"),
        sequence_length=8,
        batch_size=32,
        save_dir=tmp_path,
    )
    assert res["is_active"]
    assert "accuracy" in res
    assert "best_alpha" in res
    assert 0.0 <= res["best_alpha"] <= 1.0
    assert (tmp_path / "preds_late_fusion_Suturing_fold1.json").is_file()
