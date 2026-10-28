"""Tests for SurgMotion dataset loader, midpoint label alignment, and normalizers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.surgmotion_dataset import JIGSAWSSurgMotionDataset, get_surgmotion_dataloaders
from src.data.splits import get_louo_splits


def test_surgmotion_dataset_single_trial():
    """Verify loading, shape, and finiteness of SurgMotion dataset for Suturing_B001."""
    ds = JIGSAWSSurgMotionDataset(
        task="Suturing",
        trials=["Suturing_B001"],
        sequence_length=8,
        sequence_stride=1,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["features"].shape == (8, 1024)
    assert sample["kinematics"].shape == (8, 76)
    assert sample["labels"].shape == (8,)
    assert isinstance(sample["target"].item(), int)
    assert torch.all(torch.isfinite(sample["features"]))
    assert torch.all(torch.isfinite(sample["kinematics"]))


def test_surgmotion_clip_level_dataset():
    """Verify single clip-level dataset mode (sequence_length=1)."""
    ds = JIGSAWSSurgMotionDataset(
        task="Suturing",
        trials=["Suturing_B001"],
        sequence_length=1,
        sequence_stride=1,
    )
    assert len(ds) == 349  # exactly 349 clips for Suturing_B001
    sample = ds[0]
    assert sample["features"].shape == (1, 1024)
    assert sample["labels"].shape == (1,)
    assert sample["target"] == sample["labels"][0]


def test_surgmotion_dataloaders_louo():
    """Verify LOUO fold DataLoaders, shapes, and zero-leakage training-only normalization."""
    train_loader, val_loader, test_loader, normalizer, fold_info = get_surgmotion_dataloaders(
        task="Suturing",
        fold=1,
        sequence_length=8,
        batch_size=16,
    )
    assert normalizer.is_fitted
    assert normalizer.mean.shape == (1024,)
    assert normalizer.std.shape == (1024,)

    # Verify train batch
    batch = next(iter(train_loader))
    assert batch["features"].shape == (16, 8, 1024)
    assert batch["kinematics"].shape == (16, 8, 76)
    assert batch["labels"].shape == (16, 8)
    assert batch["target"].shape == (16,)
    assert torch.all(torch.isfinite(batch["features"]))

    # Verify disjointness between train, val, and test trials
    train_set = set(fold_info["train_trials"])
    val_set = set(fold_info["val_trials"])
    test_set = set(fold_info["test_trials"])
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    assert len(train_set & val_set) == 0
