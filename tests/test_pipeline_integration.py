from pathlib import Path
import pytest
import torch

from scripts.prepare_data import prepare_sanity_subset
from src.data.dataset import get_fold_dataloaders


def test_pipeline_end_to_end_suturing(tmp_path: Path) -> None:
    """Test full data pipeline from raw files to saved sanity diagnostics for Suturing."""
    summary = prepare_sanity_subset(
        task="Suturing",
        fold_num=1,
        output_dir=str(tmp_path),
        batch_size=16,
        window_size=16,
        stride=8,
    )

    assert summary["task"] == "Suturing"
    assert summary["fold_num"] == 1
    assert summary["test_user"] == "B"
    assert summary["num_classes"] == 11  # G0 + G1-G6, G8-G11
    assert summary["batch_shapes"]["train_kinematics"] == [16, 16, 76]
    assert summary["batch_shapes"]["train_labels"] == [16, 16]
    assert summary["batch_shapes"]["train_target"] == [16]
    assert summary["num_train_windows"] > 5000
    assert summary["num_test_windows"] > 500

    summary_file = tmp_path / "Suturing_fold1_summary.json"
    assert summary_file.is_file()


def test_pipeline_end_to_end_knot_tying(tmp_path: Path) -> None:
    """Test full data pipeline for Knot Tying."""
    summary = prepare_sanity_subset(
        task="Knot_Tying",
        fold_num=1,
        output_dir=str(tmp_path),
        batch_size=16,
        window_size=16,
        stride=8,
    )

    assert summary["task"] == "Knot_Tying"
    assert summary["num_classes"] == 7  # G0 + G1, G11-G15
    assert summary["batch_shapes"]["train_kinematics"] == [16, 16, 76]
    summary_file = tmp_path / "Knot_Tying_fold1_summary.json"
    assert summary_file.is_file()


def test_pipeline_end_to_end_needle_passing(tmp_path: Path) -> None:
    """Test full data pipeline for Needle Passing."""
    summary = prepare_sanity_subset(
        task="Needle_Passing",
        fold_num=1,
        output_dir=str(tmp_path),
        batch_size=16,
        window_size=16,
        stride=8,
    )

    assert summary["task"] == "Needle_Passing"
    assert summary["num_classes"] == 11
    assert summary["batch_shapes"]["train_kinematics"] == [16, 16, 76]
    summary_file = tmp_path / "Needle_Passing_fold1_summary.json"
    assert summary_file.is_file()


def test_dataloader_batch_bounds_and_validity() -> None:
    """Verify multiple batches iterate without error and all labels stay strictly in-bounds."""
    loader_dict = get_fold_dataloaders(
        task="Suturing",
        fold_num=1,
        batch_size=32,
        window_size=16,
        train_stride=8,
        eval_stride=8,
        num_workers=0,
    )

    num_classes = len(loader_dict["vocab"])
    train_loader = loader_dict["train_loader"]

    batches_tested = 0
    for batch in train_loader:
        kin = batch["kinematics"]
        lbl = batch["labels"]
        tgt = batch["target"]

        assert kin.shape == (32, 16, 76)
        assert torch.all(torch.isfinite(kin))
        assert torch.all(tgt >= 0)
        assert torch.all(tgt < num_classes)
        assert torch.all(lbl >= 0)
        assert torch.all(lbl < num_classes)

        batches_tested += 1
        if batches_tested >= 5:
            break

    assert batches_tested == 5


def test_sliding_window_overlap_consistency() -> None:
    """Verify that consecutive windows in a dataset share exact expected overlap."""
    loader_dict = get_fold_dataloaders(
        task="Suturing",
        fold_num=1,
        batch_size=32,
        window_size=16,
        train_stride=8,
        eval_stride=8,
    )

    ds = loader_dict["datasets"]["train"]
    # Check two adjacent windows from the same trial
    w0 = ds[0]
    w1 = ds[1]

    if w0["trial_id"] == w1["trial_id"]:
        # With window_size=16 and stride=8, w0[8:16] must exactly match w1[0:8]
        overlap_w0 = w0["kinematics"][8:16]
        overlap_w1 = w1["kinematics"][0:8]
        assert torch.allclose(overlap_w0, overlap_w1, atol=1e-6)

        overlap_labels_w0 = w0["labels"][8:16]
        overlap_labels_w1 = w1["labels"][0:8]
        assert torch.equal(overlap_labels_w0, overlap_labels_w1)
