from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.feature_cache import save_feature_cache
from src.data.multimodal_dataset import (
    JIGSAWSMultimodalWindowDataset,
    get_multimodal_fold_datasets,
)
from src.data.splits import assert_no_leakage, get_louo_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.concat_fusion import ConcatFusionGRU, ConcatFusionMLP
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# 1. Forward Shapes, Dimensions, and Feature Concatenation
# ---------------------------------------------------------------------------


def test_concat_fusion_gru_and_mlp_forward_shapes() -> None:
    """Test 2D, 3D, and pre-concatenated inputs for ConcatFusionGRU and ConcatFusionMLP."""
    b, w, d_v, d_k, c = 4, 16, 512, 76, 11
    gru = ConcatFusionGRU(video_dim=d_v, kinematics_dim=d_k, proj_dim=64, hidden_dim=64, num_classes=c, num_layers=2)
    mlp = ConcatFusionMLP(video_dim=d_v, kinematics_dim=d_k, hidden_dim=64, num_classes=c)

    x_v_3d = torch.randn(b, w, d_v)
    x_k_3d = torch.randn(b, w, d_k)
    x_v_2d = torch.randn(b, d_v)
    x_k_2d = torch.randn(b, d_k)
    x_preconcat = torch.randn(b, w, d_v + d_k)

    # 1. ConcatFusionGRU tests
    out_gru_3d = gru(x_v_3d, x_k_3d)
    assert out_gru_3d.shape == (b, c)
    assert torch.all(torch.isfinite(out_gru_3d))

    out_gru_2d = gru(x_v_2d, x_k_2d)
    assert out_gru_2d.shape == (b, c)

    out_gru_pre = gru(x_preconcat)
    assert out_gru_pre.shape == (b, c)

    seq_gru = gru.forward_sequence(x_v_3d, x_k_3d)
    assert seq_gru.shape == (b, w, c)

    feats_gru = gru.extract_features(x_v_3d, x_k_3d)
    assert feats_gru.shape == (b, 64)

    # 2. ConcatFusionMLP tests
    out_mlp_3d = mlp(x_v_3d, x_k_3d)
    assert out_mlp_3d.shape == (b, c)

    out_mlp_2d = mlp(x_v_2d, x_k_2d)
    assert out_mlp_2d.shape == (b, c)

    out_mlp_pre = mlp(x_preconcat)
    assert out_mlp_pre.shape == (b, c)

    seq_mlp = mlp.forward_sequence(x_v_3d, x_k_3d)
    assert seq_mlp.shape == (b, w, c)

    feats_mlp = mlp.extract_features(x_v_3d, x_k_3d)
    assert feats_mlp.shape == (b, 64)


def test_concat_fusion_input_validation_errors() -> None:
    """Verify informative errors on mismatched batch sizes, lengths, dims, or NaNs."""
    gru = ConcatFusionGRU(video_dim=512, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)

    x_v = torch.randn(4, 16, 512)
    x_k = torch.randn(4, 16, 76)

    # Dimension mismatch
    with pytest.raises(ValueError, match="Video feature dimension mismatch"):
        gru(torch.randn(4, 16, 256), x_k)

    with pytest.raises(ValueError, match="Kinematics dimension mismatch"):
        gru(x_v, torch.randn(4, 16, 50))

    # Batch size mismatch
    with pytest.raises(ValueError, match="Batch size mismatch"):
        gru(torch.randn(2, 16, 512), x_k)

    # Sequence length mismatch
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        gru(torch.randn(4, 8, 512), x_k)

    # Non-finite inputs
    x_v_nan = x_v.clone()
    x_v_nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        gru(x_v_nan, x_k)

    x_k_inf = x_k.clone()
    x_k_inf[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite values"):
        gru(x_v, x_k_inf)

    # Bad type
    with pytest.raises(TypeError, match="Inputs must be torch.Tensor"):
        gru(x_v.numpy(), x_k)


# ---------------------------------------------------------------------------
# 2. Gradient Flow & Memorization Capacity
# ---------------------------------------------------------------------------


def test_concat_fusion_gradient_flow() -> None:
    """Verify backward pass through concatenated projection trunk and GRU cells."""
    model = ConcatFusionGRU(video_dim=64, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x_v = torch.randn(4, 8, 64)
    x_k = torch.randn(4, 8, 76)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x_v, x_k)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"

    optimizer.step()


def test_concat_fusion_overfit_tiny_batch() -> None:
    """Verify model capacity to memorize multimodal sequence targets on a small batch."""
    set_seed(42)
    model = ConcatFusionGRU(
        video_dim=32,
        kinematics_dim=76,
        proj_dim=32,
        hidden_dim=32,
        num_classes=4,
        num_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()

    x_v = torch.randn(4, 8, 32)
    x_k = torch.randn(4, 8, 76)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    initial_loss = criterion(model(x_v, x_k), target).item()

    for _ in range(70):
        optimizer.zero_grad()
        logits = model(x_v, x_k)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x_v, x_k), target).item()
    assert final_loss < initial_loss
    assert final_loss < 0.15


def test_concat_fusion_deterministic_inference() -> None:
    """Verify identical outputs under deterministic seeding in eval mode."""
    set_seed(123)
    model1 = ConcatFusionGRU(video_dim=64, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
    model1.eval()

    set_seed(123)
    model2 = ConcatFusionGRU(video_dim=64, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
    model2.eval()

    x_v = torch.randn(2, 8, 64)
    x_k = torch.randn(2, 8, 76)

    with torch.no_grad():
        out1 = model1(x_v, x_k)
        out2 = model2(x_v, x_k)

    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 3. Checkpoint Saving & Loading
# ---------------------------------------------------------------------------


def test_concat_fusion_checkpoint_saving_and_loading(tmp_path: Path) -> None:
    """Verify state dict saving, loading, and output parity."""
    model = ConcatFusionGRU(video_dim=64, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
    ckpt_path = tmp_path / "concat_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.is_file()

    loaded_model = ConcatFusionGRU(video_dim=64, kinematics_dim=76, proj_dim=32, hidden_dim=32, num_classes=5)
    loaded_model.load_state_dict(torch.load(ckpt_path))

    x_v = torch.randn(2, 8, 64)
    x_k = torch.randn(2, 8, 76)
    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        out1 = model(x_v, x_k)
        out2 = loaded_model(x_v, x_k)

    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 4. End-to-End Synthetic Training Smoke Test
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


def test_synthetic_end_to_end_concat_fusion_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: mock multimodal data, train 1 epoch, evaluate metrics, serialize."""
    from scripts.train_concat_fusion import train_and_eval_fold

    task = "Suturing"
    trial_b = "Suturing_B001"
    trial_c = "Suturing_C001"
    trial_d = "Suturing_D001"
    feat_dim = 64

    # 1. Create mock multimodal trials
    for t_id in [trial_b, trial_c, trial_d]:
        _create_mock_multimodal_trial(
            base_dir=tmp_path,
            task=task,
            trial_id=t_id,
            n_frames=60,
            n_kin_samples=60,
            feat_dim=feat_dim,
        )

    cfg = {
        "model": {
            "name": "ConcatFusionGRU",
            "video_dim": feat_dim,
            "kinematics_dim": 76,
            "proj_dim": 32,
            "hidden_dim": 32,
            "num_classes": 11,
            "num_layers": 1,
            "dropout": 0.1,
            "bidirectional": False,
        },
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "use_class_weights": False,
            "patience": 1,
            "seed": 42,
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
    }

    ckpt_path = tmp_path / "concat_test.pt"

    # Mock get_louo_splits for synthetic smoke test
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

        metrics = train_and_eval_fold(
            task=task,
            fold_num=1,
            cfg=cfg,
            device=torch.device("cpu"),
            save_checkpoint_path=ckpt_path,
        )

        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "macro_f1" in metrics
        assert "edit_score" in metrics
        assert metrics["fold"] == 1
        assert metrics["test_user"] == "B"
        assert metrics["video_dim"] == feat_dim
        assert metrics["kinematics_dim"] == 76
        assert ckpt_path.is_file()

        # Serialization test
        json_str = json.dumps(metrics, indent=2)
        assert len(json_str) > 0

    finally:
        mmd_module.get_louo_splits = original_get_splits
