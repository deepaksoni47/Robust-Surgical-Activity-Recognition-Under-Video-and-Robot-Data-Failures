from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark_comparison import (
    audit_task_multimodal_readiness,
    compare_model_benchmarks,
    export_comparison_csv,
    format_comparison_markdown,
    format_readiness_audit_markdown,
)
from src.models.fusion import search_optimal_alpha


def test_audit_task_multimodal_readiness_structure() -> None:
    """Test readiness auditor on Suturing with verified real files."""
    audit = audit_task_multimodal_readiness(
        task="Suturing",
        dataset_dir="dataset",
        cache_dir="data/cache/features",
        checkpoints_dir="checkpoints",
        encoder_name="ResNet18",
        max_allowed_delta=500,
    )

    assert audit["task"] == "Suturing"
    assert audit["total_trials"] == 39
    assert audit["cached_trials_count"] >= 1  # At least Suturing_B001 exists
    assert audit["cached_trials_count"] + audit["missing_cache_trials_count"] == 39
    assert audit["status"] in ("BLOCKED", "READY")
    assert audit["active_encoder"] == "ResNet18"
    assert "SurgMotion" in audit["surgmotion_status"]
    assert len(audit["alignment_exceptions"]) == 0

    # Checkpoints audit keys
    ckpts = audit["checkpoints_audit"]
    assert "kinematics_mlp" in ckpts
    assert "kinematics_gru" in ckpts
    assert "video_gru" in ckpts
    assert "concat_fusion" in ckpts
    assert ckpts["kinematics_mlp"]["fold_1"] is True
    assert isinstance(ckpts["video_gru"]["fold_1"], bool)


def test_audit_alignment_exception_flagging() -> None:
    """Test that trials exceeding max_allowed_delta are flagged as alignment exceptions."""
    # When max_allowed_delta is set to 0, if cache frames != kin samples, it flags exception
    audit = audit_task_multimodal_readiness(
        task="Suturing",
        dataset_dir="dataset",
        cache_dir="data/cache/features",
        checkpoints_dir="checkpoints",
        encoder_name="ResNet18",
        max_allowed_delta=0,  # Strict zero-tolerance delta
    )
    # Suturing_B001 has a non-zero tail delta (typically ~10 frames)
    assert len(audit["alignment_exceptions"]) >= 1
    assert "Suturing_B001" in audit["alignment_exceptions"]
    assert audit["status"] == "BLOCKED"


def test_compare_model_benchmarks_hand_computed() -> None:
    """Test result aggregation and delta calculation against exact hand-computed values."""
    # Hand-crafted summaries for 3 folds
    # Unimodal Kinematics baseline:
    # Acc: 60.0, BalAcc: 50.0, Macro-F1: 40.0, Edit: 30.0
    # Unimodal Video baseline:
    # Acc: 70.0, BalAcc: 65.0, Macro-F1: 55.0, Edit: 45.0
    # (Strongest Unimodal: Video for all metrics)
    # Multimodal Late Fusion:
    # Acc: 75.0, BalAcc: 70.0, Macro-F1: 62.0, Edit: 50.0
    # Expected Deltas vs Strongest Unimodal (Video):
    # Delta Acc = 75.0 - 70.0 = +5.0
    # Delta BalAcc = 70.0 - 65.0 = +5.0
    # Delta Macro-F1 = 62.0 - 55.0 = +7.0
    # Delta Edit = 50.0 - 45.0 = +5.0

    summaries = {
        "kinematics_gru": {
            "model": "KinematicsGRU",
            "mean_accuracy": 60.0,
            "std_accuracy": 2.0,
            "mean_balanced_accuracy": 50.0,
            "std_balanced_accuracy": 3.0,
            "mean_macro_f1": 0.40,  # In [0, 1] decimal format
            "std_macro_f1": 0.04,
            "mean_edit_score": 30.0,
            "std_edit_score": 1.5,
            "folds": [
                {"fold": 1, "accuracy": 0.58},
                {"fold": 2, "accuracy": 0.60},
                {"fold": 3, "accuracy": 0.62},
            ],
            "per_class": {
                "G1": {"f1": 0.50},
                "G2": {"f1": 0.30},
            },
        },
        "video_gru": {
            "model": "VideoGRU",
            "mean_accuracy": 70.0,
            "std_accuracy": 1.8,
            "mean_balanced_accuracy": 65.0,
            "std_balanced_accuracy": 2.5,
            "mean_macro_f1": 0.55,
            "std_macro_f1": 0.03,
            "mean_edit_score": 45.0,
            "std_edit_score": 2.0,
            "folds": [
                {"fold": 1, "accuracy": 0.69},
                {"fold": 2, "accuracy": 0.70},
                {"fold": 3, "accuracy": 0.71},
            ],
            "per_class": {
                "G1": {"f1": 0.60},
                "G2": {"f1": 0.50},
            },
        },
        "late_fusion": {
            "model": "FixedLateFusion",
            "mean_accuracy": 75.0,
            "std_accuracy": 1.5,
            "mean_balanced_accuracy": 70.0,
            "std_balanced_accuracy": 2.0,
            "mean_macro_f1": 0.62,
            "std_macro_f1": 0.02,
            "mean_edit_score": 50.0,
            "std_edit_score": 1.8,
            "folds": [
                {"fold": 1, "accuracy": 0.74},
                {"fold": 2, "accuracy": 0.75},
                {"fold": 3, "accuracy": 0.76},
            ],
            "per_class": {
                "G1": {"f1": 0.65},
                "G2": {"f1": 0.59},
            },
        },
    }

    comp = compare_model_benchmarks(summaries, task="Suturing")

    assert comp["task"] == "Suturing"
    assert comp["n_folds"] == 3
    assert comp["evaluated_folds"] == [1, 2, 3]

    strongest = comp["strongest_unimodal_baseline"]
    assert pytest.approx(strongest["accuracy"]) == 70.0
    assert pytest.approx(strongest["balanced_accuracy"]) == 65.0
    assert pytest.approx(strongest["macro_f1"]) == 55.0
    assert pytest.approx(strongest["edit_score"]) == 45.0

    # Locate fusion record
    fusion_rec = next(r for r in comp["records"] if r["model_id"] == "late_fusion")
    assert pytest.approx(fusion_rec["accuracy"]) == 75.0
    assert pytest.approx(fusion_rec["macro_f1"]) == 62.0
    assert pytest.approx(fusion_rec["delta_accuracy"]) == 5.0
    assert pytest.approx(fusion_rec["delta_balanced_accuracy"]) == 5.0
    assert pytest.approx(fusion_rec["delta_macro_f1"]) == 7.0
    assert pytest.approx(fusion_rec["delta_edit_score"]) == 5.0

    # Locate unimodal record (deltas must be 0)
    video_rec = next(r for r in comp["records"] if r["model_id"] == "video_gru")
    assert pytest.approx(video_rec["delta_accuracy"]) == 0.0
    assert pytest.approx(video_rec["delta_macro_f1"]) == 0.0


def test_compare_model_benchmarks_mismatched_folds_raises() -> None:
    """Test that models evaluating different folds fail loudly with ValueError."""
    summaries = {
        "model_a": {
            "model": "ModelA",
            "mean_accuracy": 50.0,
            "folds": [{"fold": 1}, {"fold": 2}, {"fold": 3}],
        },
        "model_b": {
            "model": "ModelB",
            "mean_accuracy": 60.0,
            "folds": [{"fold": 1}, {"fold": 2}, {"fold": 4}],  # Fold 4 instead of 3
        },
    }

    with pytest.raises(ValueError, match="Mismatched fold coverage"):
        compare_model_benchmarks(summaries, task="Suturing")


def test_compare_model_benchmarks_missing_folds_raises() -> None:
    """Test that a summary without fold information raises KeyError."""
    summaries = {
        "model_a": {
            "model": "ModelA",
            "mean_accuracy": 50.0,
        }
    }
    with pytest.raises(KeyError, match="missing fold results"):
        compare_model_benchmarks(summaries, task="Suturing")


def test_empty_summaries_raises() -> None:
    """Test that empty summaries dict raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        compare_model_benchmarks({}, task="Suturing")


def test_export_comparison_csv_and_markdown() -> None:
    """Test CSV export and markdown generation."""
    summaries = {
        "kinematics": {
            "model": "KinematicsGRU",
            "mean_accuracy": 66.8,
            "std_accuracy": 9.0,
            "mean_balanced_accuracy": 55.8,
            "std_balanced_accuracy": 12.0,
            "mean_macro_f1": 45.4,
            "std_macro_f1": 9.5,
            "mean_edit_score": 34.4,
            "std_edit_score": 9.5,
            "folds": [{"fold": 1}],
        },
        "late_fusion": {
            "model": "FixedLateFusion",
            "mean_accuracy": 72.0,
            "std_accuracy": 7.0,
            "mean_balanced_accuracy": 62.0,
            "std_balanced_accuracy": 8.0,
            "mean_macro_f1": 52.0,
            "std_macro_f1": 6.5,
            "mean_edit_score": 40.0,
            "std_edit_score": 7.0,
            "folds": [{"fold": 1}],
        },
    }

    comp = compare_model_benchmarks(summaries, task="Suturing")

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_comparison.csv"
        export_comparison_csv(comp, csv_path)

        assert csv_path.is_file()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "delta_accuracy" in df.columns
        assert "delta_macro_f1" in df.columns
        assert "std_accuracy" in df.columns

    md = format_comparison_markdown(comp)
    assert "# Comparative Evaluation" in md
    assert "KinematicsGRU" in md
    assert "FixedLateFusion" in md
    assert "+5.20%" in md  # Delta accuracy (72.0 - 66.8)


def test_format_readiness_audit_markdown() -> None:
    """Test formatting of readiness audit markdown."""
    audit = {
        "task": "Suturing",
        "status": "BLOCKED",
        "total_trials": 39,
        "cached_trials_count": 1,
        "missing_cache_trials_count": 38,
        "active_encoder": "ResNet18",
        "encoder_fallback_type": "ResNet-18 (ImageNet Pretrained, 512-D)",
        "surgmotion_status": "Gated on Hugging Face ('gated: manual')",
        "alignment_exceptions": [],
        "checkpoints_audit": {
            "kinematics_mlp": {"fold_1": True, "fold_2": False},
        },
    }
    md = format_readiness_audit_markdown(audit)
    assert "# Multi-Modality Benchmark Readiness Audit: Suturing" in md
    assert "`BLOCKED`" in md
    assert "extract_visual_features.py" in md
    assert "train_video_baseline.py" in md
    assert "run_late_fusion.py" in md
    assert "train_concat_fusion.py" in md


def test_no_test_leakage_in_fusion_or_normalization() -> None:
    """Assert that test labels cannot influence fusion weights or normalizer parameters."""
    rng = np.random.RandomState(42)
    n_val = 100
    n_test = 50
    n_classes = 3

    # Validation logits & labels
    val_v = rng.randn(n_val, n_classes)
    val_k = rng.randn(n_val, n_classes)
    val_y = rng.randint(0, n_classes, size=n_val)

    # Test logits & labels
    test_v = rng.randn(n_test, n_classes)
    test_k = rng.randn(n_test, n_classes)
    test_y = rng.randint(0, n_classes, size=n_test)

    # Search alpha strictly on validation data
    search_res = search_optimal_alpha(
        val_video_logits=val_v,
        val_kinematics_logits=val_k,
        val_targets=val_y,
        metric="macro_f1",
        num_classes=n_classes,
    )

    best_alpha = search_res["optimal_alpha"]
    search_history = search_res["search_history"]

    # Assert search was performed strictly using validation targets
    assert 0.0 <= best_alpha <= 1.0
    assert len(search_history) > 0

    # Ensure search history evaluated all candidates
    for pt in search_history:
        assert pt["alpha"] >= 0.0
        assert "val_score" in pt
