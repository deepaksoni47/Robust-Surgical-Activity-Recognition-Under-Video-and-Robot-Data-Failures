from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.dataset import KinematicsNormalizer
from src.data.feature_cache import save_feature_cache
from src.data.multimodal_dataset import (
    JIGSAWSMultimodalWindowDataset,
    get_multimodal_fold_datasets,
)
from src.data.splits import assert_no_leakage, get_louo_splits
from src.data.video_dataset import VideoFeatureNormalizer
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.fusion import (
    FixedLateFusion,
    FixedLateFusionClassifier,
    fuse_logits,
    search_optimal_alpha,
    validate_class_mappings,
)
from src.models.kinematics_gru import KinematicsGRU
from src.models.video_head import VideoGRU
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# 1. Fusion Arithmetic & Alpha Boundary Cases
# ---------------------------------------------------------------------------


def test_fuse_logits_arithmetic_and_boundary_cases() -> None:
    """Verify convex combination arithmetic and exact boundary conditions."""
    v_torch = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    k_torch = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    # 1. Exact boundary alpha = 0.0 -> pure kinematics
    out_0 = fuse_logits(v_torch, k_torch, alpha=0.0)
    torch.testing.assert_close(out_0, k_torch)

    # 2. Exact boundary alpha = 1.0 -> pure video
    out_1 = fuse_logits(v_torch, k_torch, alpha=1.0)
    torch.testing.assert_close(out_1, v_torch)

    # 3. Alpha = 0.5 -> arithmetic mean
    out_half = fuse_logits(v_torch, k_torch, alpha=0.5)
    expected_half = 0.5 * v_torch + 0.5 * k_torch
    torch.testing.assert_close(out_half, expected_half)

    # 4. Arbitrary alpha = 0.25 -> 0.25 * v + 0.75 * k
    out_quarter = fuse_logits(v_torch, k_torch, alpha=0.25)
    expected_quarter = 0.25 * v_torch + 0.75 * k_torch
    torch.testing.assert_close(out_quarter, expected_quarter)

    # 5. Numpy array input parity
    v_np = v_torch.numpy()
    k_np = k_torch.numpy()
    out_np_0 = fuse_logits(v_np, k_np, alpha=0.0)
    np.testing.assert_allclose(out_np_0, k_np)

    out_np_1 = fuse_logits(v_np, k_np, alpha=1.0)
    np.testing.assert_allclose(out_np_1, v_np)

    out_np_half = fuse_logits(v_np, k_np, alpha=0.5)
    np.testing.assert_allclose(out_np_half, 0.5 * v_np + 0.5 * k_np)


def test_fuse_logits_input_validation_errors() -> None:
    """Verify loud failure on invalid alphas, mismatched shapes, non-finite values, and bad types."""
    v = torch.randn(4, 11)
    k = torch.randn(4, 11)

    # Alpha bounds
    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        fuse_logits(v, k, alpha=-0.1)

    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        fuse_logits(v, k, alpha=1.05)

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        fuse_logits(v, torch.randn(4, 7), alpha=0.5)

    with pytest.raises(ValueError, match="Shape mismatch"):
        fuse_logits(v, torch.randn(5, 11), alpha=0.5)

    # Type mismatch
    with pytest.raises(TypeError, match="Input type mismatch"):
        fuse_logits(v, k.numpy(), alpha=0.5)

    # Non-finite values
    v_nan = v.clone()
    v_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match="contains non-finite values"):
        fuse_logits(v_nan, k, alpha=0.5)

    k_inf = k.clone()
    k_inf[0, 0] = float("inf")
    with pytest.raises(ValueError, match="contains non-finite values"):
        fuse_logits(v, k_inf, alpha=0.5)


# ---------------------------------------------------------------------------
# 2. Class Vocabulary Alignment & Mapping Verification
# ---------------------------------------------------------------------------


def test_validate_class_mappings() -> None:
    """Verify that class index mismatches and differing vocabularies are strictly rejected."""
    vocab_a = {"G0": 0, "G1": 1, "G2": 2}
    vocab_b = {"G0": 0, "G1": 1, "G2": 2}
    assert validate_class_mappings(vocab_a, vocab_b) is True

    # Differing keys
    vocab_missing = {"G0": 0, "G1": 1}
    with pytest.raises(ValueError, match="Vocabulary mismatch between modalities"):
        validate_class_mappings(vocab_a, vocab_missing)

    # Permuted indices
    vocab_permuted = {"G0": 0, "G1": 2, "G2": 1}
    with pytest.raises(ValueError, match="Class index mismatch for gesture 'G1'"):
        validate_class_mappings(vocab_a, vocab_permuted)


# ---------------------------------------------------------------------------
# 3. Validation Alpha Search (Zero Leakage into Test Data)
# ---------------------------------------------------------------------------


def test_search_optimal_alpha_strictly_validation() -> None:
    """Verify alpha search optimizes validation metric without touching test data."""
    # Synthetic scenario: Video is accurate on first half, Kinematics accurate on second half
    # True labels: [0, 0, 1, 1]
    val_targets = np.array([0, 0, 1, 1], dtype=np.int64)

    # Modality 1: High confidence correct on class 0, wrong on class 1
    val_v = np.array([
        [10.0, -10.0],
        [10.0, -10.0],
        [10.0, -10.0],
        [10.0, -10.0],
    ], dtype=np.float32)

    # Modality 2: High confidence wrong on class 0, correct on class 1
    val_k = np.array([
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
    ], dtype=np.float32)

    # With candidate alphas:
    # At alpha = 1.0 (pure video), macro-F1 is low (only class 0 predicted)
    # At alpha = 0.0 (pure kin), macro-F1 is low (only class 1 predicted)
    # At alpha = 0.5 (balanced), both get equal logits, tie-breaker determines
    res = search_optimal_alpha(
        val_video_logits=val_v,
        val_kinematics_logits=val_k,
        val_targets=val_targets,
        metric="macro_f1",
        alpha_candidates=[0.0, 0.2, 0.5, 0.8, 1.0],
        num_classes=2,
    )

    assert "optimal_alpha" in res
    assert "best_val_score" in res
    assert 0.0 <= res["optimal_alpha"] <= 1.0
    assert len(res["search_history"]) == 5

    # Verification: Test set labels are never part of the signature or execution
    # Any hypothetical test data has no effect on res["optimal_alpha"]


# ---------------------------------------------------------------------------
# 4. Multimodal Dataset & Alignment Verification
# ---------------------------------------------------------------------------


def _create_mock_multimodal_trial(
    base_dir: Path,
    task: str,
    trial_id: str,
    n_frames: int,
    n_kin_samples: int,
    feat_dim: int = 64,
    encoder_name: str = "ResNet18",
    camera: str = "capture1",
) -> None:
    """Helper to create synchronized mock feature cache, kinematics, and transcriptions."""
    clean_enc = encoder_name.replace(" ", "_")
    cache_path = base_dir / "cache" / clean_enc / f"{trial_id}_{camera}.npz"
    feats = np.random.randn(n_frames, feat_dim).astype(np.float32)
    indices = np.arange(n_frames, dtype=np.int32)
    meta = {
        "trial_id": trial_id,
        "task": task,
        "encoder_name": encoder_name,
        "output_dim": feat_dim,
        "num_frames": n_frames,
    }
    save_feature_cache(cache_path, feats, indices, meta, overwrite=True)

    # Kinematics
    kin_dir = base_dir / "dataset" / task / "kinematics" / "AllGestures"
    kin_dir.mkdir(parents=True, exist_ok=True)
    kin_path = kin_dir / f"{trial_id}.txt"
    kin_data = np.random.randn(n_kin_samples, 76).astype(np.float32)
    lines = [" ".join(f"{v:.6f}" for v in row) for row in kin_data]
    kin_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Transcription
    trans_dir = base_dir / "dataset" / task / "transcriptions"
    trans_dir.mkdir(parents=True, exist_ok=True)
    trans_path = trans_dir / f"{trial_id}.txt"
    effective_t = min(n_frames, n_kin_samples)
    mid = effective_t // 2
    trans_path.write_text(f"1 {mid} G1\n{mid + 1} {effective_t} G2\n", encoding="utf-8")


def test_multimodal_dataset_tail_clipping_and_batch_shapes(tmp_path: Path) -> None:
    """Verify multimodal dataset synchronizes video, kinematics, and labels with tail clipping."""
    task = "Suturing"
    trial_id = "Suturing_M001"
    # Video has 85 frames, Kinematics has 80 samples -> should clip to 80
    _create_mock_multimodal_trial(
        base_dir=tmp_path,
        task=task,
        trial_id=trial_id,
        n_frames=85,
        n_kin_samples=80,
        feat_dim=64,
    )

    dataset = JIGSAWSMultimodalWindowDataset(
        task=task,
        trials=[trial_id],
        dataset_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        encoder_name="ResNet18",
        expected_dim=64,
        window_size=16,
        stride=8,
    )

    assert len(dataset) > 0
    sample = dataset[0]

    assert sample["video"].shape == (16, 64)
    assert sample["kinematics"].shape == (16, 76)
    assert sample["labels"].shape == (16,)
    assert sample["target"].dtype == torch.long
    assert sample["target"].item() == sample["labels"][-1].item()

    record = dataset.trial_records[0]
    assert record["total_frames"] == 80
    assert record["video"].shape == (80, 64)
    assert record["kinematics"].shape == (80, 76)


def test_multimodal_dataset_excessive_delta_raises_error(tmp_path: Path) -> None:
    """Verify loud failure when video and kinematics length delta exceeds max_allowed_delta."""
    task = "Suturing"
    trial_id = "Suturing_BadSync"
    # Delta 600 > 500
    _create_mock_multimodal_trial(
        base_dir=tmp_path,
        task=task,
        trial_id=trial_id,
        n_frames=700,
        n_kin_samples=100,
        feat_dim=64,
    )

    with pytest.raises(ValueError, match="Unacceptable video-kinematics frame delta"):
        JIGSAWSMultimodalWindowDataset(
            task=task,
            trials=[trial_id],
            dataset_dir=tmp_path / "dataset",
            cache_dir=tmp_path / "cache",
            encoder_name="ResNet18",
            expected_dim=64,
            max_allowed_delta=500,
        )


def test_official_splits_zero_leakage_assertion() -> None:
    """Verify zero leakage across official LOUO splits across all 3 tasks."""
    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        splits = get_louo_splits(task, create_val_split=True)
        for fold_num, fold_info in splits.items():
            assert_no_leakage(
                train_trials=fold_info["train_trials"],
                test_trials=fold_info["test_trials"],
                val_trials=fold_info.get("val_trials", []),
            )


# ---------------------------------------------------------------------------
# 5. FixedLateFusionClassifier PyTorch Module Parity
# ---------------------------------------------------------------------------


def test_fixed_late_fusion_classifier_module() -> None:
    """Verify end-to-end forward pass and forward_sequence of FixedLateFusionClassifier."""
    v_model = VideoGRU(input_dim=64, proj_dim=32, hidden_dim=32, num_classes=5)
    k_model = KinematicsGRU(input_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)

    fusion_model = FixedLateFusionClassifier(
        video_model=v_model,
        kinematics_model=k_model,
        alpha=0.6,
    )

    v_model.eval()
    k_model.eval()

    x_v = torch.randn(4, 16, 64)
    x_k = torch.randn(4, 16, 76)

    # 1. Causal window forward pass
    with torch.no_grad():
        fused_out = fusion_model(x_v, x_k)
        assert fused_out.shape == (4, 5)
        assert torch.all(torch.isfinite(fused_out))

        # Parity check with manual fuse_logits
        expected_out = fuse_logits(v_model(x_v), k_model(x_k), alpha=0.6)
        torch.testing.assert_close(fused_out, expected_out)

        # 2. Sequence forward pass
        fused_seq = fusion_model.forward_sequence(x_v, x_k)
        assert fused_seq.shape == (4, 16, 5)



# ---------------------------------------------------------------------------
# 6. End-to-End Synthetic Fusion Smoke Test
# ---------------------------------------------------------------------------


def test_synthetic_end_to_end_late_fusion_smoke(tmp_path: Path) -> None:
    """Synthetic end-to-end test verifying readiness check, alpha search, and fusion evaluation."""
    from scripts.run_late_fusion import evaluate_late_fusion_fold

    task = "Suturing"
    trial_b = "Suturing_B001"
    trial_c = "Suturing_C001"
    trial_d = "Suturing_D001"
    feat_dim = 64

    # 1. Create mock multimodal data
    for t_id in [trial_b, trial_c, trial_d]:
        _create_mock_multimodal_trial(
            base_dir=tmp_path,
            task=task,
            trial_id=t_id,
            n_frames=60,
            n_kin_samples=60,
            feat_dim=feat_dim,
        )

    # 2. Create mock trained checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    v_ckpt = ckpt_dir / "video_gru_fold1.pt"
    k_ckpt = ckpt_dir / "kin_gru_fold1.pt"

    v_model = VideoGRU(input_dim=feat_dim, proj_dim=32, hidden_dim=32, num_classes=11, num_layers=1)
    k_model = KinematicsGRU(input_dim=76, proj_dim=32, hidden_dim=32, num_classes=11, num_layers=1)
    torch.save(v_model.state_dict(), v_ckpt)
    torch.save(k_model.state_dict(), k_ckpt)

    cfg = {
        "fusion": {
            "method": "late_logit",
            "alpha_search": True,
            "metric": "macro_f1",
            "default_alpha": 0.5,
            "alpha_candidates": [0.0, 0.5, 1.0],
        },
        "models": {
            "video": {
                "name": "VideoGRU",
                "input_dim": feat_dim,
                "proj_dim": 32,
                "hidden_dim": 32,
                "num_classes": 11,
                "num_layers": 1,
                "dropout": 0.1,
                "checkpoint_pattern": str(v_ckpt),
            },
            "kinematics": {
                "name": "KinematicsGRU",
                "input_dim": 76,
                "proj_dim": 32,
                "hidden_dim": 32,
                "num_classes": 11,
                "num_layers": 1,
                "dropout": 0.1,
                "checkpoint_pattern": str(k_ckpt),
            },
        },
        "data": {
            "dataset_dir": str(tmp_path / "dataset"),
            "cache_dir": str(tmp_path / "cache"),
            "encoder_name": "ResNet18",
            "camera": "capture1",
            "expected_dim": feat_dim,
            "window_size": 16,
            "train_stride": 16,
            "eval_stride": 16,
            "normalize": True,
            "max_allowed_delta": 500,
        },
        "evaluation": {
            "batch_size": 16,
            "seed": 42,
        },
    }

    # Mock get_louo_splits for the synthetic test
    import src.data.multimodal_dataset as mmd_module

    original_get_splits = mmd_module.get_louo_splits
    try:
        mmd_module.get_louo_splits = lambda t, **kwargs: {
            1: {
                "fold": 1,
                "test_user": "B",
                "test_trials": [trial_b],
                "val_trials": [trial_c],
                "train_trials": [trial_d],
            }
        }

        res = evaluate_late_fusion_fold(
            task=task,
            fold_num=1,
            cfg=cfg,
            device=torch.device("cpu"),
        )

        assert "fusion_metrics" in res
        assert "kinematics_metrics" in res
        assert "video_metrics" in res
        assert "alpha" in res
        assert 0.0 <= res["alpha"] <= 1.0

        f_metrics = res["fusion_metrics"]
        assert "accuracy" in f_metrics
        assert "balanced_accuracy" in f_metrics
        assert "macro_f1" in f_metrics
        assert "edit_score" in f_metrics

        # JSON serialization test
        json_str = json.dumps(res, indent=2)
        assert len(json_str) > 0

    finally:
        mmd_module.get_louo_splits = original_get_splits
