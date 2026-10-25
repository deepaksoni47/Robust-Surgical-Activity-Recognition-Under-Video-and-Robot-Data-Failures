import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    collapse_to_segments,
    compute_classification_metrics,
    compute_edit_score,
    compute_levenshtein_distance,
    format_metrics_table,
    to_numpy_labels,
)


def test_to_numpy_labels_varied_inputs() -> None:
    """Test label conversion for 1D lists, 1D arrays, 1D tensors, and 2D logits."""
    # 1D list
    assert np.array_equal(to_numpy_labels([0, 1, 2]), np.array([0, 1, 2]))

    # 1D PyTorch tensor
    t = torch.tensor([2, 1, 0])
    assert np.array_equal(to_numpy_labels(t), np.array([2, 1, 0]))

    # 2D logits tensor: shape (3, 4)
    logits = torch.tensor([
        [0.1, 0.9, 0.0, 0.0],  # argmax 1
        [0.8, 0.1, 0.0, 0.1],  # argmax 0
        [0.0, 0.1, 0.7, 0.2],  # argmax 2
    ])
    assert np.array_equal(to_numpy_labels(logits), np.array([1, 0, 2]))


def test_classification_metrics_perfect_score() -> None:
    """Test metric computation on identical true and predicted labels."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    metrics = compute_classification_metrics(y, y, num_classes=4)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["total_samples"] == 8
    assert metrics["num_classes"] == 4


def test_classification_metrics_hand_calculated_toy_example() -> None:
    """Test against an analytically calculated 3-class toy confusion case.
    
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    
    Class 0: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
    Class 1: TP=2, FP=1, FN=0 -> Prec=2/3, Rec=1.0, F1=0.8
    Class 2: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=2/3 (~0.6667)
    
    Accuracy = 4/6 = 0.666667
    Balanced Accuracy = (0.5 + 1.0 + 0.5) / 3 = 0.666667
    Macro-F1 = (0.5 + 0.8 + 2/3) / 3 = 0.655556
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    class_names = {0: "G0", 1: "G1", 2: "G2"}

    metrics = compute_classification_metrics(
        y_true, y_pred, num_classes=3, class_names=class_names
    )

    assert pytest.approx(metrics["accuracy"], rel=1e-4) == 4.0 / 6.0
    assert pytest.approx(metrics["balanced_accuracy"], rel=1e-4) == 2.0 / 3.0
    assert pytest.approx(metrics["macro_f1"], rel=1e-4) == (0.5 + 0.8 + (2.0 / 3.0)) / 3.0

    per_class = metrics["per_class"]
    assert pytest.approx(per_class["G0"]["precision"], rel=1e-4) == 0.5
    assert pytest.approx(per_class["G0"]["recall"], rel=1e-4) == 0.5
    assert pytest.approx(per_class["G0"]["f1"], rel=1e-4) == 0.5

    assert pytest.approx(per_class["G1"]["precision"], rel=1e-4) == 2.0 / 3.0
    assert pytest.approx(per_class["G1"]["recall"], rel=1e-4) == 1.0
    assert pytest.approx(per_class["G1"]["f1"], rel=1e-4) == 0.8

    assert pytest.approx(per_class["G2"]["precision"], rel=1e-4) == 1.0
    assert pytest.approx(per_class["G2"]["recall"], rel=1e-4) == 0.5
    assert pytest.approx(per_class["G2"]["f1"], rel=1e-4) == 2.0 / 3.0

    # Confusion matrix: rows=true, cols=pred
    expected_cm = [
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ]
    assert metrics["confusion_matrix"] == expected_cm


def test_zero_support_class_safety() -> None:
    """Test that absent classes in ground truth or predictions evaluate gracefully with 0.0."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    # Evaluate with 4 classes (classes 2 and 3 have 0 support)
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=4)

    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["Class_2"]["support"] == 0
    assert metrics["per_class"]["Class_2"]["f1"] == 0.0


def test_collapse_to_segments() -> None:
    """Test merging of consecutive identical gesture frames into discrete segments."""
    frames = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1]
    expected_segments = [1, 2, 3, 1]
    assert collapse_to_segments(frames) == expected_segments

    # Empty frames
    assert collapse_to_segments([]) == []

    # Single segment
    assert collapse_to_segments([5, 5, 5]) == [5]


def test_compute_levenshtein_distance() -> None:
    """Test Levenshtein distance on gesture sequences."""
    # Insertion
    assert compute_levenshtein_distance([1, 2], [1, 2, 3]) == 1
    # Substitution
    assert compute_levenshtein_distance([1, 2, 3], [1, 4, 3]) == 1
    # Deletion
    assert compute_levenshtein_distance([1, 2, 3], [1, 3]) == 1
    # Equal
    assert compute_levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0


def test_compute_edit_score() -> None:
    """Test Segmental Edit Score computation."""
    # 1. Exact match -> 100%
    true_seqs = [[1, 1, 2, 2, 3, 3]]
    pred_seqs = [[1, 2, 3]]
    assert compute_edit_score(true_seqs, pred_seqs) == 100.0

    # 2. 1 edit out of 3 max segments -> (1 - 1/3) * 100 = 66.67%
    true_seqs = [[1, 1, 2, 2, 3, 3]]  # segments: [1, 2, 3] (len 3)
    pred_seqs = [[1, 1, 4, 4, 3, 3]]  # segments: [1, 4, 3] (dist 1, len 3)
    assert pytest.approx(compute_edit_score(true_seqs, pred_seqs), rel=1e-4) == 66.6667


def test_format_metrics_table() -> None:
    """Test formatting output table."""
    metrics = compute_classification_metrics([0, 1], [0, 1], num_classes=2)
    table_str = format_metrics_table(metrics)
    assert "Accuracy:" in table_str
    assert "Macro-F1:" in table_str
    assert "| Class | ID |" in table_str
