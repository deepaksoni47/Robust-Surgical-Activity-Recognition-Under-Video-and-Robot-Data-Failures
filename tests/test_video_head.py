from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.feature_cache import save_feature_cache
from src.data.splits import assert_no_leakage, get_louo_splits
from src.data.video_dataset import (
    JIGSAWSVideoWindowDataset,
    VideoFeatureNormalizer,
    get_video_fold_dataloaders,
    get_video_fold_datasets,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.video_head import VideoGRU, VideoMLP
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# 1. Model Forward Shapes, Dimensions, and Gradient Flow
# ---------------------------------------------------------------------------


def test_video_gru_and_mlp_forward_shapes() -> None:
    """Test 2D and 3D input handling, forward shapes, sequence forward, and feature extraction."""
    b, w, d, c = 4, 16, 512, 11
    gru = VideoGRU(input_dim=d, proj_dim=64, hidden_dim=64, num_classes=c, num_layers=2)
    mlp = VideoMLP(input_dim=d, hidden_dim=64, num_classes=c)

    x_3d = torch.randn(b, w, d)
    x_2d = torch.randn(b, d)

    # 1. VideoGRU tests
    out_gru_3d = gru(x_3d)
    assert out_gru_3d.shape == (b, c)
    assert torch.all(torch.isfinite(out_gru_3d))

    out_gru_2d = gru(x_2d)
    assert out_gru_2d.shape == (b, c)

    seq_gru = gru.forward_sequence(x_3d)
    assert seq_gru.shape == (b, w, c)

    feats_gru = gru.extract_features(x_3d)
    assert feats_gru.shape == (b, 64)

    # 2. VideoMLP tests
    out_mlp_3d = mlp(x_3d)
    assert out_mlp_3d.shape == (b, c)

    out_mlp_2d = mlp(x_2d)
    assert out_mlp_2d.shape == (b, c)

    seq_mlp = mlp.forward_sequence(x_3d)
    assert seq_mlp.shape == (b, w, c)

    feats_mlp = mlp.extract_features(x_3d)
    assert feats_mlp.shape == (b, 64)

    # 3. Dimension mismatch checks
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        gru(torch.randn(b, w, 76))

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        mlp(torch.randn(b, w, 76))


def test_video_gru_gradient_flow() -> None:
    """Test backward pass through VideoGRU recurrent parameters."""
    model = VideoGRU(input_dim=128, proj_dim=32, hidden_dim=32, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 8, 128)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"

    optimizer.step()


def test_video_gru_overfit_tiny_batch() -> None:
    """Verify VideoGRU capacity to memorize sequence targets on a small batch."""
    set_seed(42)
    model = VideoGRU(
        input_dim=64,
        proj_dim=32,
        hidden_dim=32,
        num_classes=4,
        num_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 8, 64)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    initial_loss = criterion(model(x), target).item()

    for _ in range(70):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss
    assert final_loss < 0.15


def test_deterministic_inference_and_reproducible_seeds() -> None:
    """Verify identical outputs when evaluated with deterministic seeding in eval mode."""
    set_seed(123)
    model1 = VideoGRU(input_dim=64, proj_dim=32, hidden_dim=32, num_classes=5)
    model1.eval()

    set_seed(123)
    model2 = VideoGRU(input_dim=64, proj_dim=32, hidden_dim=32, num_classes=5)
    model2.eval()

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    torch.testing.assert_close(out1, out2)



# ---------------------------------------------------------------------------
# 2. Normalization & Zero-Leakage Checks
# ---------------------------------------------------------------------------


def test_video_feature_normalizer() -> None:
    """Test standard z-score normalization, zero variance protection, and serialization."""
    np.random.seed(42)
    t1 = np.random.randn(100, 64) * 4.0 + 3.0
    t2 = np.random.randn(150, 64) * 2.0 - 1.0
    # Channel 0 constant at 7.0 (zero-variance test)
    t1[:, 0] = 7.0
    t2[:, 0] = 7.0

    normalizer = VideoFeatureNormalizer()
    normalizer.fit([t1, t2])

    assert normalizer.is_fitted
    assert normalizer.mean.shape == (64,)
    assert normalizer.std.shape == (64,)

    # Channel 0 std protected to 1.0
    assert normalizer.std[0] == 1.0

    norm_t1 = normalizer.transform(t1)
    norm_t2 = normalizer.transform(t2)
    assert np.all(np.isfinite(norm_t1))
    assert np.all(np.isfinite(norm_t2))
    np.testing.assert_allclose(norm_t1[:, 0], 0.0, atol=1e-5)

    # Invertibility check on non-constant channels
    inv_t2 = normalizer.inverse_transform(norm_t2)
    np.testing.assert_allclose(inv_t2[:, 1:], t2[:, 1:], rtol=1e-5, atol=1e-5)

    # Serialization roundtrip
    meta = normalizer.to_dict()
    restored = VideoFeatureNormalizer.from_dict(meta)
    np.testing.assert_allclose(normalizer.mean, restored.mean)
    np.testing.assert_allclose(normalizer.std, restored.std)


# ---------------------------------------------------------------------------
# 3. Cache-to-Label Alignment, Missing Errors, and Edge Cases
# ---------------------------------------------------------------------------


def _create_mock_jigsaws_trial(
    base_dir: Path,
    task: str,
    trial_id: str,
    n_frames: int,
    feat_dim: int = 64,
    encoder_name: str = "ResNet18",
    camera: str = "capture1",
    n_kin_samples: Optional[int] = None,
) -> Path:
    """Helper to create synchronized mock trial cache, kinematics, and transcriptions."""
    # 1. Feature cache
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

    # 2. Transcription
    trans_dir = base_dir / "dataset" / task / "transcriptions"
    trans_dir.mkdir(parents=True, exist_ok=True)
    trans_file = trans_dir / f"{trial_id}.txt"
    mid = n_frames // 2
    trans_content = f"1 {mid} G1\n{mid + 1} {n_frames} G2\n"
    trans_file.write_text(trans_content, encoding="utf-8")

    # 3. Kinematics (optional)
    if n_kin_samples is not None:
        kin_dir = base_dir / "dataset" / task / "kinematics" / "AllGestures"
        kin_dir.mkdir(parents=True, exist_ok=True)
        kin_file = kin_dir / f"{trial_id}.txt"
        kin_data = np.random.randn(n_kin_samples, 76).astype(np.float32)
        lines = [" ".join(f"{v:.6f}" for v in row) for row in kin_data]
        kin_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return cache_path


def test_video_dataset_cache_to_label_alignment(tmp_path: Path) -> None:
    """Verify feature alignment, sliding window creation, and causal target."""
    task = "Suturing"
    trial_id = "Suturing_T001"
    n_frames = 60
    feat_dim = 64

    _create_mock_jigsaws_trial(
        base_dir=tmp_path,
        task=task,
        trial_id=trial_id,
        n_frames=n_frames,
        feat_dim=feat_dim,
    )

    dataset = JIGSAWSVideoWindowDataset(
        task=task,
        trials=[trial_id],
        dataset_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        encoder_name="ResNet18",
        expected_dim=feat_dim,
        window_size=16,
        stride=8,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["video"].shape == (16, feat_dim)
    assert sample["labels"].shape == (16,)
    assert sample["target"].dtype == torch.long
    assert sample["target"].item() == sample["labels"][-1].item()


def test_video_dataset_tail_clipping_with_kinematics(tmp_path: Path) -> None:
    """Verify tail clipping when video and kinematics lengths differ slightly."""
    task = "Suturing"
    trial_id = "Suturing_T002"
    # Video has 70 frames, kinematics has 65 rows -> effective frames should be 65
    _create_mock_jigsaws_trial(
        base_dir=tmp_path,
        task=task,
        trial_id=trial_id,
        n_frames=70,
        feat_dim=64,
        n_kin_samples=65,
    )

    dataset = JIGSAWSVideoWindowDataset(
        task=task,
        trials=[trial_id],
        dataset_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        encoder_name="ResNet18",
        expected_dim=64,
        window_size=16,
        stride=8,
    )

    record = dataset.trial_records[0]
    assert record["total_frames"] == 65
    assert record["video"].shape == (65, 64)
    assert len(record["labels"]) == 65


def test_video_dataset_excessive_mismatch_raises_error(tmp_path: Path) -> None:
    """Verify error is raised when alignment delta exceeds max_allowed_delta."""
    task = "Suturing"
    trial_id = "Suturing_T003"
    # Video has 700 frames, kinematics has 100 rows -> delta 600 > 500
    _create_mock_jigsaws_trial(
        base_dir=tmp_path,
        task=task,
        trial_id=trial_id,
        n_frames=700,
        feat_dim=64,
        n_kin_samples=100,
    )

    with pytest.raises(ValueError, match="Unacceptable frame alignment delta"):
        JIGSAWSVideoWindowDataset(
            task=task,
            trials=[trial_id],
            dataset_dir=tmp_path / "dataset",
            cache_dir=tmp_path / "cache",
            encoder_name="ResNet18",
            expected_dim=64,
            max_allowed_delta=500,
        )


def test_video_dataset_missing_cache_informative_error(tmp_path: Path) -> None:
    """Verify missing cache raises FileNotFoundError explaining how to generate it."""
    with pytest.raises(FileNotFoundError, match="Missing visual feature cache for trial 'Suturing_Missing'"):
        JIGSAWSVideoWindowDataset(
            task="Suturing",
            trials=["Suturing_Missing"],
            dataset_dir=tmp_path / "dataset",
            cache_dir=tmp_path / "cache",
            encoder_name="ResNet18",
        )


def test_video_dataset_encoder_mismatch_fails_loudly(tmp_path: Path) -> None:
    """Verify error is raised when cache was generated with a different encoder."""
    task = "Suturing"
    trial_id = "Suturing_T004"
    # Save cache into ResNet18 directory, but with encoder_name in metadata set to 'SurgMotion'
    clean_enc = "ResNet18"
    cache_path = tmp_path / "cache" / clean_enc / f"{trial_id}_capture1.npz"
    feats = np.random.randn(50, 64).astype(np.float32)
    indices = np.arange(50, dtype=np.int32)
    meta = {
        "trial_id": trial_id,
        "task": task,
        "encoder_name": "SurgMotion",  # Deliberate mismatch with requested ResNet18
        "output_dim": 64,
        "num_frames": 50,
    }
    save_feature_cache(cache_path, feats, indices, meta, overwrite=True)

    # Transcription
    trans_dir = tmp_path / "dataset" / task / "transcriptions"
    trans_dir.mkdir(parents=True, exist_ok=True)
    (trans_dir / f"{trial_id}.txt").write_text("1 25 G1\n26 50 G2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Encoder mismatch in cache"):
        JIGSAWSVideoWindowDataset(
            task=task,
            trials=[trial_id],
            dataset_dir=tmp_path / "dataset",
            cache_dir=tmp_path / "cache",
            encoder_name="ResNet18",
            expected_dim=64,
        )


# ---------------------------------------------------------------------------
# 4. Official Fold Isolation, Checkpoint Saving, and End-to-End Smoke Test
# ---------------------------------------------------------------------------


def test_official_fold_isolation_and_zero_leakage() -> None:
    """Verify official LOUO split isolation and zero user/trial leakage across all 8 folds."""
    splits = get_louo_splits("Suturing", create_val_split=True)
    assert len(splits) == 8

    for fold_num, fold_info in splits.items():
        assert_no_leakage(
            train_trials=fold_info["train_trials"],
            test_trials=fold_info["test_trials"],
            val_trials=fold_info.get("val_trials", []),
        )



def test_checkpoint_saving_and_loading(tmp_path: Path) -> None:
    """Verify state dict checkpoint saving, loading, and output parity."""
    model = VideoGRU(input_dim=64, proj_dim=32, hidden_dim=32, num_classes=5)
    ckpt_path = tmp_path / "video_gru.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.is_file()

    loaded_model = VideoGRU(input_dim=64, proj_dim=32, hidden_dim=32, num_classes=5)
    loaded_model.load_state_dict(torch.load(ckpt_path))

    x = torch.randn(2, 8, 64)
    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        out1 = model(x)
        out2 = loaded_model(x)

    torch.testing.assert_close(out1, out2)


def test_synthetic_end_to_end_training_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: mock multi-trial fold, train 1 epoch, evaluate metrics, serialize."""
    from scripts.train_video_baseline import train_and_eval_fold

    task = "Suturing"
    # Create mock trials for User B (test) and Users C-I (val/train)
    trial_b = "Suturing_B001"
    trial_c = "Suturing_C001"
    trial_d = "Suturing_D001"
    feat_dim = 64

    for t_id in [trial_b, trial_c, trial_d]:
        _create_mock_jigsaws_trial(
            base_dir=tmp_path,
            task=task,
            trial_id=t_id,
            n_frames=60,
            feat_dim=feat_dim,
            encoder_name="ResNet18",
            n_kin_samples=60,
        )

    # Create dummy split files for Fold 1
    split_dir = tmp_path / "splits" / task
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_users0.txt").write_text(f"{trial_d}\n", encoding="utf-8")
    (split_dir / "test_users0.txt").write_text(f"{trial_b}\n", encoding="utf-8")

    cfg = {
        "model": {
            "name": "VideoGRU",
            "input_dim": feat_dim,
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
        },
    }

    ckpt_path = tmp_path / "test_model.pt"

    # We mock get_louo_splits for this synthetic test
    import src.data.video_dataset as vd_module

    original_get_splits = vd_module.get_louo_splits
    try:
        vd_module.get_louo_splits = lambda t, **kwargs: {
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
        assert metrics["encoder"] == "ResNet18"
        assert ckpt_path.is_file()

        # Serialization test
        json_str = json.dumps(metrics, indent=2)
        assert len(json_str) > 0

    finally:
        vd_module.get_louo_splits = original_get_splits
