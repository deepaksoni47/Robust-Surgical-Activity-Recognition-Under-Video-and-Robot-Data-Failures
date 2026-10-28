"""tests/test_noise_ablation.py

Unit and integration tests for ITER-31 Workstream A:
- Noise injection operator properties and determinism
- Normalization order and variance scaling
- SurgMotionAdaptiveFusionModel forward and sequential inference with gate logging
- Modality availability mask exact fallback guarantees
- GateTrajectoryLogger serialization and aggregation
- Ablation CLI and preflight checks
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.surgmotion_fusion import (
    GateTrajectoryLogger,
    SurgMotionAdaptiveFusionModel,
    inject_feature_noise,
)
from scripts.train_surgmotion_noise_ablation import (
    compress_sequence,
    compute_edit_score,
    levenshtein_distance,
    run_preflight_check,
)


def test_inject_feature_noise_zero_std_is_identity():
    """Verify zero noise std returns original features unchanged."""
    feats = torch.randn(8, 8, 1024)
    noisy = inject_feature_noise(feats, noise_std=0.0)
    assert torch.equal(feats, noisy)


def test_inject_feature_noise_variance_scaling():
    """Verify empirical variance of injected Gaussian noise matches noise_std^2."""
    sigma = 1.2
    zeros = torch.zeros(100, 10, 512)
    noisy = inject_feature_noise(zeros, noise_std=sigma)
    emp_std = float(noisy.std())
    assert abs(emp_std - sigma) < 0.05, f"Expected std ~ {sigma}, got {emp_std}"


def test_inject_feature_noise_determinism_with_generator():
    """Verify torch.Generator reproduces identical noise patterns."""
    feats = torch.randn(4, 8, 128)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    noisy1 = inject_feature_noise(feats.clone(), noise_std=0.75, generator=g1)
    noisy2 = inject_feature_noise(feats.clone(), noise_std=0.75, generator=g2)
    assert torch.allclose(noisy1, noisy2, atol=1e-6)


def test_surgmotion_adaptive_fusion_forward_shapes_and_sum_to_one():
    """Test batch classification forward pass output shapes and gate weight normalization."""
    B, S, C = 4, 8, 10
    model = SurgMotionAdaptiveFusionModel(
        num_classes=C,
        video_dim=1024,
        kinematics_dim=76,
        hidden_dim=64,
        dropout=0.0,
    )
    model.eval()

    v_in = torch.randn(B, S, 1024)
    k_in = torch.randn(B, S, 76)

    logits, alpha = model(v_in, k_in)
    assert logits.shape == (B, C)
    assert alpha.shape == (B, 2)
    assert torch.all(alpha >= 0.0)
    # Sum over video + kinematics weights must equal 1.0
    alpha_sum = alpha.sum(dim=-1)
    assert torch.allclose(alpha_sum, torch.ones(B), atol=1e-5)


def test_surgmotion_adaptive_fusion_forward_sequence_return_alpha():
    """Test sequential forward inference and time-resolved gate weight extraction."""
    B, T, C = 2, 16, 10
    model = SurgMotionAdaptiveFusionModel(
        num_classes=C,
        video_dim=1024,
        kinematics_dim=76,
        hidden_dim=64,
        dropout=0.0,
    )
    model.eval()

    v_seq = torch.randn(B, T, 1024)
    k_seq = torch.randn(B, T, 76)

    fused_seq, alpha_seq = model.forward_sequence(v_seq, k_seq, return_alpha=True)
    assert fused_seq.shape == (B, T, C)
    assert alpha_seq.shape == (B, T, 2)
    # Check normalization at every timestep
    step_sums = alpha_seq.sum(dim=-1)
    assert torch.allclose(step_sums, torch.ones(B, T), atol=1e-5)


def test_surgmotion_adaptive_fusion_exact_fallback_on_missing_mask():
    """Test that availability mask [0, 1] routes 100% mass to kinematics and [1, 0] to video."""
    B, S, C = 3, 8, 8
    model = SurgMotionAdaptiveFusionModel(
        num_classes=C,
        video_dim=1024,
        kinematics_dim=76,
        hidden_dim=64,
        dropout=0.0,
    )
    model.eval()

    v_in = torch.randn(B, S, 1024)
    k_in = torch.randn(B, S, 76)

    # Missing video: mask = [0, 1]
    mask_no_vid = torch.tensor([[0, 1]], dtype=torch.bool).expand(B, 2)
    _, alpha_no_vid = model(v_in, k_in, availability_mask=mask_no_vid)
    assert torch.allclose(alpha_no_vid[:, 0], torch.zeros(B), atol=1e-7)
    assert torch.allclose(alpha_no_vid[:, 1], torch.ones(B), atol=1e-7)

    # Missing kinematics: mask = [1, 0]
    mask_no_kin = torch.tensor([[1, 0]], dtype=torch.bool).expand(B, 2)
    _, alpha_no_kin = model(v_in, k_in, availability_mask=mask_no_kin)
    assert torch.allclose(alpha_no_kin[:, 0], torch.ones(B), atol=1e-7)
    assert torch.allclose(alpha_no_kin[:, 1], torch.zeros(B), atol=1e-7)


def test_gate_trajectory_logger(tmp_path: Path):
    """Test logging trial trajectories, computing condition aggregates, and JSON saving."""
    logger_obj = GateTrajectoryLogger()
    T = 5
    alphas = np.array([
        [0.6, 0.4],
        [0.55, 0.45],
        [0.5, 0.5],
        [0.7, 0.3],
        [0.65, 0.35],
    ])
    centers = np.arange(10, 10 + 16 * T, 16)
    gt = np.array([1, 1, 2, 2, 3])
    preds = np.array([1, 1, 2, 2, 2])

    logger_obj.log_trial_trajectory(
        fold=1,
        task="Suturing",
        trial_id="Suturing_B001",
        condition="clean",
        noise_std=0.0,
        alpha_weights=alphas,
        center_frames=centers,
        ground_truth=gt,
        predictions=preds,
    )

    assert len(logger_obj.records) == T
    first = logger_obj.records[0]
    assert first["trial_id"] == "Suturing_B001"
    assert first["alpha_video"] == 0.6
    assert first["alpha_kinematics"] == 0.4

    summary = logger_obj.compute_condition_aggregates()
    assert "clean" in summary
    assert summary["clean"]["count"] == T
    assert summary["clean"]["mean_alpha_video"] == 0.6

    out_file = tmp_path / "trajectories.json"
    logger_obj.save_json(out_file)
    assert out_file.is_file()
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded["total_records"] == T


def test_edit_score_calculation_helpers():
    """Verify Levenshtein distance, sequence compression, and Edit Score."""
    assert compress_sequence([1, 1, 1, 2, 2, 3]) == [1, 2, 3]
    assert compress_sequence([]) == []
    assert levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein_distance([1, 2], [1, 3]) == 1

    # Identical sequences: 100%
    score = compute_edit_score([[1, 1, 2, 2]], [[1, 2]])
    assert score == 100.0


def test_preflight_check_runs():
    """Verify preflight environment checker runs and reports status."""
    status = run_preflight_check()
    assert isinstance(status, bool)


def test_second_dataset_feasibility_json_schema():
    """Verify Workstream B dataset comparison JSON structure, candidates, and recommendation."""
    feasibility_path = Path("artifacts/iter31_dataset_comparison.json")
    assert feasibility_path.is_file(), f"Missing feasibility JSON: {feasibility_path}"

    with open(feasibility_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "candidates" in data
    assert "study_design_analysis" in data
    assert "recommended_pilot" in data

    candidates = data["candidates"]
    for c_name in ["SurgSync", "ROSMAG40", "MISAW"]:
        assert c_name in candidates, f"Candidate {c_name} missing from feasibility analysis"
        cand = candidates[c_name]
        assert "verified_urls" in cand
        assert "feasibility_assessment" in cand
        assert "compatibility_score" in cand["feasibility_assessment"]
        assert "verdict" in cand["feasibility_assessment"]

    # Assert SurgSync is the top recommendation
    surgsync = candidates["SurgSync"]
    assert "RECOMMENDED" in surgsync["feasibility_assessment"]["verdict"]
    assert surgsync["feasibility_assessment"]["compatibility_score"] >= 9.0

    # Assert pilot details
    pilot = data["recommended_pilot"]
    assert pilot["dataset"] == "SurgSync"
    assert "Suturing" in pilot["tasks"]
    assert "Knot Tying" in pilot["tasks"]

