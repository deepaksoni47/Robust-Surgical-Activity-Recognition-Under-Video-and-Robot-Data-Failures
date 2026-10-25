from pathlib import Path
import json
import numpy as np
import pytest
import torch

from src.data.feature_cache import (
    extract_and_cache_trial,
    get_cache_path,
    load_feature_cache,
    save_feature_cache,
    validate_feature_cache,
)
from src.models.visual_encoder import (
    BaseVisualEncoder,
    ResNet18VisualEncoder,
    SurgMotionVisualEncoder,
)


class DummyVisualEncoder(BaseVisualEncoder):
    """Deterministic dummy visual encoder for fast, offline unit testing without GPU."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(device="cpu")
        self._dim = dim
        self._name = "DummyEncoder"

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def encoder_name(self) -> str:
        return self._name

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        b = frames.shape[0]
        # Return deterministic linear projection based on mean of pixels
        mean_val = frames.mean(dim=[-3, -2, -1], keepdim=True).squeeze(-1).squeeze(-1) # [B, 1]
        weight = torch.linspace(0.1, 1.0, self._dim).unsqueeze(0) # [1, D]
        return mean_val * weight


def test_get_cache_path(tmp_path: Path) -> None:
    """Verify deterministic cache filepath formatting."""
    path = get_cache_path(tmp_path, "Suturing_B001", "ResNet18", camera="capture1")
    assert path == tmp_path / "ResNet18" / "Suturing_B001_capture1.npz"


def test_save_and_load_feature_cache(tmp_path: Path) -> None:
    """Verify atomic saving, loading, and metadata roundtrip."""
    cache_path = tmp_path / "test_trial_capture1.npz"
    n_frames = 50
    feat_dim = 64

    features = np.random.randn(n_frames, feat_dim).astype(np.float32)
    indices = np.arange(n_frames, dtype=np.int32)
    meta = {
        "trial_id": "Suturing_B001",
        "task": "Suturing",
        "encoder_name": "DummyEncoder",
        "output_dim": feat_dim,
        "num_frames": n_frames,
    }

    saved = save_feature_cache(cache_path, features, indices, meta)
    assert saved.is_file()
    assert cache_path.is_file()

    loaded = load_feature_cache(cache_path, validate=True, expected_dim=feat_dim)
    assert np.allclose(loaded["features"], features)
    assert np.array_equal(loaded["frame_indices"], indices)
    assert loaded["metadata"]["trial_id"] == "Suturing_B001"
    assert loaded["output_dim"] == feat_dim


def test_validate_feature_cache_corrupt_key(tmp_path: Path) -> None:
    """Verify validation fails if required arrays are missing."""
    bad_path = tmp_path / "bad_cache.npz"
    np.savez(bad_path, dummy_array=np.zeros(10))

    with pytest.raises(ValueError, match="missing required arrays"):
        validate_feature_cache(bad_path)


def test_validate_feature_cache_nan_detection(tmp_path: Path) -> None:
    """Verify validation rejects non-finite feature values."""
    bad_path = tmp_path / "nan_cache.npz"
    feats = np.ones((10, 32), dtype=np.float32)
    feats[5, 5] = np.nan
    indices = np.arange(10, dtype=np.int32)

    with pytest.raises(ValueError, match="Cannot save cache containing NaN or Inf"):
        save_feature_cache(bad_path, feats, indices, {"trial_id": "test"})


def test_validate_feature_cache_dimension_mismatch(tmp_path: Path) -> None:
    """Verify validation flags expected dimension mismatch."""
    cache_path = tmp_path / "dim_test.npz"
    feats = np.ones((10, 32), dtype=np.float32)
    indices = np.arange(10, dtype=np.int32)
    save_feature_cache(cache_path, feats, indices, {"trial_id": "test"})

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        validate_feature_cache(cache_path, expected_dim=64)


def test_validate_feature_cache_non_monotonic_indices(tmp_path: Path) -> None:
    """Verify non-monotonic frame indices are rejected."""
    bad_path = tmp_path / "bad_indices.npz"
    feats = np.ones((5, 32), dtype=np.float32)
    indices = np.array([0, 1, 3, 2, 4], dtype=np.int32) # out of order

    np.savez_compressed(bad_path, features=feats, frame_indices=indices)
    with pytest.raises(ValueError, match="not strictly monotonically increasing"):
        validate_feature_cache(bad_path)


def test_atomic_overwrite_and_skip(tmp_path: Path) -> None:
    """Verify overwrite control and skipping behavior."""
    cache_path = tmp_path / "overwrite_test.npz"
    feats = np.ones((10, 32), dtype=np.float32)
    indices = np.arange(10, dtype=np.int32)

    save_feature_cache(cache_path, feats, indices, {"trial_id": "test"}, overwrite=False)
    assert cache_path.is_file()

    with pytest.raises(FileExistsError):
        save_feature_cache(cache_path, feats, indices, {"trial_id": "test"}, overwrite=False)

    # Overwrite=True succeeds
    new_feats = np.ones((10, 32), dtype=np.float32) * 2.0
    save_feature_cache(cache_path, new_feats, indices, {"trial_id": "test"}, overwrite=True)
    loaded = load_feature_cache(cache_path)
    assert np.allclose(loaded["features"], new_feats)


def test_surgmotion_missing_checkpoint_fails_loudly() -> None:
    """Verify SurgMotion fails with informative error when checkpoint is not present."""
    with pytest.raises(FileNotFoundError, match="SurgMotion checkpoint not found"):
        SurgMotionVisualEncoder(checkpoint_path="checkpoints/missing_weights.pt")


def test_extract_and_cache_real_video_smoke(tmp_path: Path) -> None:
    """Integration test: extract and cache 16 frames from a real Suturing video using ResNet-18."""
    video_path = Path("dataset/Suturing/video/Suturing_B001_capture1.avi")
    if not video_path.is_file():
        pytest.skip(f"Video file {video_path} not available.")

    trial_row = {
        "trial_id": "Suturing_B001",
        "task": "Suturing",
        "video_capture1_path": str(video_path),
        "kinematics_path": "dataset/Suturing/kinematics/AllGestures/Suturing_B001.txt",
        "transcription_path": "dataset/Suturing/transcriptions/Suturing_B001.txt",
    }

    encoder = ResNet18VisualEncoder(pretrained=False)
    cache_dir = tmp_path / "features"

    # 1. First extraction
    res1 = extract_and_cache_trial(
        trial_row=trial_row,
        encoder=encoder,
        cache_dir=cache_dir,
        camera="capture1",
        batch_size=8,
        max_frames=16,
        force=False,
    )

    assert res1["status"] == "EXTRACTED"
    assert res1["num_frames"] == 16
    assert res1["output_dim"] == 512
    cache_file = Path(res1["cache_path"])
    assert cache_file.is_file()

    # 2. Re-running with force=False should skip
    res2 = extract_and_cache_trial(
        trial_row=trial_row,
        encoder=encoder,
        cache_dir=cache_dir,
        camera="capture1",
        batch_size=8,
        max_frames=16,
        force=False,
    )
    assert res2["status"] == "SKIPPED"
    assert res2["num_frames"] == 16

    # 3. Re-running with force=True should re-extract
    res3 = extract_and_cache_trial(
        trial_row=trial_row,
        encoder=encoder,
        cache_dir=cache_dir,
        camera="capture1",
        batch_size=8,
        max_frames=16,
        force=True,
    )
    assert res3["status"] == "EXTRACTED"

    # 4. Validate cache content and provenance
    cache_data = load_feature_cache(cache_file, validate=True, expected_dim=512)
    assert cache_data["features"].shape == (16, 512)
    assert cache_data["metadata"]["trial_id"] == "Suturing_B001"
    assert cache_data["metadata"]["encoder_name"] == "ResNet18"
    assert cache_data["metadata"]["n_kinematics_samples"] is not None
    assert np.all(np.isfinite(cache_data["features"]))
