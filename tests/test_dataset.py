import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    JIGSAWSWindowDataset,
    KinematicsNormalizer,
    get_fold_dataloaders,
    get_fold_datasets,
)
from src.data.splits import get_louo_splits


def test_kinematics_normalizer_fit_transform() -> None:
    """Test standard z-score normalization and exact invertibility."""
    np.random.seed(42)
    # Generate 3 dummy trials with 76 features
    t1 = np.random.randn(200, 76) * 5.0 + 10.0
    t2 = np.random.randn(300, 76) * 2.0 - 4.0
    t3 = np.random.randn(250, 76) * 0.5 + 1.0

    normalizer = KinematicsNormalizer()
    normalizer.fit([t1, t2, t3])

    assert normalizer.is_fitted
    assert normalizer.mean.shape == (76,)
    assert normalizer.std.shape == (76,)

    # Transform
    norm_t1 = normalizer.transform(t1)
    norm_t2 = normalizer.transform(t2)
    norm_t3 = normalizer.transform(t3)

    all_norm = np.vstack([norm_t1, norm_t2, norm_t3])
    # Overall mean should be ~0 and std should be ~1
    np.testing.assert_allclose(all_norm.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(all_norm.std(axis=0), 1.0, atol=1e-5)

    # Inverse transform
    inv_t1 = normalizer.inverse_transform(norm_t1)
    np.testing.assert_allclose(inv_t1, t1, rtol=1e-5, atol=1e-5)


def test_kinematics_normalizer_zero_variance() -> None:
    """Test that constant channels do not produce division by zero or NaN."""
    t1 = np.ones((100, 76), dtype=np.float32)
    t1[:, 0] = 5.0  # Channel 0 constant at 5.0

    normalizer = KinematicsNormalizer()
    normalizer.fit([t1])

    norm = normalizer.transform(t1)
    assert np.all(np.isfinite(norm))
    # Constant channels should normalize to 0
    np.testing.assert_allclose(norm[:, 0], 0.0, atol=1e-6)


def test_kinematics_normalizer_serialization() -> None:
    """Test to_dict and from_dict roundtrip."""
    t = np.random.randn(150, 76).astype(np.float32)
    norm1 = KinematicsNormalizer().fit([t])

    data = norm1.to_dict()
    assert "mean" in data
    assert "std" in data
    assert len(data["mean"]) == 76

    norm2 = KinematicsNormalizer.from_dict(data)
    assert norm2.is_fitted
    np.testing.assert_allclose(norm1.mean, norm2.mean)
    np.testing.assert_allclose(norm1.std, norm2.std)


def test_window_dataset_shapes_and_types() -> None:
    """Test that JIGSAWSWindowDataset produces expected PyTorch tensors and window slices."""
    trials = ["Suturing_B001", "Suturing_B002"]
    dataset = JIGSAWSWindowDataset(
        task="Suturing",
        trials=trials,
        dataset_dir="dataset",
        window_size=16,
        stride=8,
    )

    assert len(dataset) > 0

    sample = dataset[0]
    assert "kinematics" in sample
    assert "labels" in sample
    assert "target" in sample
    assert "trial_id" in sample

    assert sample["kinematics"].shape == (16, 76)
    assert sample["kinematics"].dtype == torch.float32

    assert sample["labels"].shape == (16,)
    assert sample["labels"].dtype == torch.int64

    assert sample["target"].dtype == torch.int64
    assert sample["target"].item() == sample["labels"][-1].item()


def test_window_dataset_class_weights() -> None:
    """Test inverse-frequency class weight computation."""
    trials = ["Suturing_B001", "Suturing_B002"]
    dataset = JIGSAWSWindowDataset(
        task="Suturing",
        trials=trials,
        dataset_dir="dataset",
        window_size=16,
        stride=8,
    )

    weights = dataset.compute_class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.dim() == 1
    assert weights.shape[0] == len(dataset.vocab)
    assert torch.all(weights >= 0)


def test_get_fold_datasets_suturing() -> None:
    """Test full LOUO fold construction for Suturing with strict train-only normalization."""
    fold_data = get_fold_datasets(
        task="Suturing",
        fold_num=1,
        window_size=16,
        train_stride=8,
        eval_stride=8,
        dataset_dir="dataset",
    )

    assert "train" in fold_data
    assert "val" in fold_data
    assert "test" in fold_data
    assert "normalizer" in fold_data
    assert fold_data["normalizer"].is_fitted

    train_ds = fold_data["train"]
    test_ds = fold_data["test"]

    # Verify user B is strictly test and not in train
    for record in train_ds.trial_records:
        assert not record["trial_id"].startswith("Suturing_B"), "Test user leaked to train!"

    for record in test_ds.trial_records:
        assert record["trial_id"].startswith("Suturing_B"), "Test trial must be User B!"

    assert len(train_ds) > 0
    assert len(test_ds) > 0


def test_get_fold_dataloaders_batch_execution() -> None:
    """Test that PyTorch DataLoader produces complete batches with correct dimensions."""
    loaders = get_fold_dataloaders(
        task="Suturing",
        fold_num=1,
        batch_size=16,
        window_size=16,
        train_stride=8,
        eval_stride=8,
        num_workers=0,
        dataset_dir="dataset",
    )

    train_loader = loaders["train_loader"]
    assert isinstance(train_loader, DataLoader)

    # Fetch one batch
    batch = next(iter(train_loader))
    assert batch["kinematics"].shape == (16, 16, 76)
    assert batch["labels"].shape == (16, 16)
    assert batch["target"].shape == (16,)
    assert len(batch["trial_id"]) == 16


def test_get_fold_dataloaders_all_three_tasks() -> None:
    """Test that all three tasks (Suturing, Knot_Tying, Needle_Passing) build and load cleanly."""
    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        loaders = get_fold_dataloaders(
            task=task,
            fold_num=1,
            batch_size=8,
            window_size=16,
            train_stride=16,
            eval_stride=16,
            num_workers=0,
            dataset_dir="dataset",
        )
        batch = next(iter(loaders["train_loader"]))
        assert batch["kinematics"].shape == (8, 16, 76)
        assert batch["target"].shape == (8,)
        assert loaders["normalizer"].is_fitted
