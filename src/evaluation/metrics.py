from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def to_numpy_labels(
    y: Union[np.ndarray, torch.Tensor, List[int], List[float]]
) -> np.ndarray:
    """Convert input predictions or targets to a 1D int64 numpy array.
    
    If y has 2 or more dimensions (e.g. logits of shape (N, C)), takes argmax along the last dimension.
    """
    if isinstance(y, torch.Tensor):
        if y.ndim >= 2:
            y = torch.argmax(y, dim=-1)
        y = y.detach().cpu().numpy()
    elif isinstance(y, list):
        y = np.array(y)

    if not isinstance(y, np.ndarray):
        raise TypeError(f"Expected numpy array, torch Tensor, or list, got {type(y)}")

    if y.ndim >= 2:
        y = np.argmax(y, axis=-1)

    return y.astype(np.int64).flatten()


def collapse_to_segments(frame_labels: List[int] | np.ndarray) -> List[int]:
    """Collapse continuous framewise labels into discrete gesture segment sequences.
    
    Example:
        [1, 1, 1, 2, 2, 1, 1] -> [1, 2, 1]
    """
    if len(frame_labels) == 0:
        return []

    segments = [int(frame_labels[0])]
    for lbl in frame_labels[1:]:
        lbl_int = int(lbl)
        if lbl_int != segments[-1]:
            segments.append(lbl_int)

    return segments


def compute_levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute minimum number of single-element edits (insertions, deletions, substitutions)
    to transform seq1 into seq2.
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return int(dp[n][m])


def compute_edit_score(
    y_true_seqs: List[List[int] | np.ndarray],
    y_pred_seqs: List[List[int] | np.ndarray],
) -> float:
    """Compute standard normalized Segmental Edit Score (%) used in JIGSAWS benchmarks.
    
    Formula per trial:
        Score = (1 - Levenshtein(S_pred, S_true) / max(|S_pred|, |S_true|)) * 100
    
    Args:
        y_true_seqs: List of framewise ground truth sequences (one per trial).
        y_pred_seqs: List of framewise predicted sequences (one per trial).
        
    Returns:
        Mean edit score across all trials (0.0 to 100.0).
    """
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(f"Mismatched trial counts: {len(y_true_seqs)} true vs {len(y_pred_seqs)} pred.")

    if not y_true_seqs:
        return 0.0

    scores = []
    for y_t, y_p in zip(y_true_seqs, y_pred_seqs):
        s_true = collapse_to_segments(y_t)
        s_pred = collapse_to_segments(y_p)

        max_len = max(len(s_true), len(s_pred))
        if max_len == 0:
            scores.append(100.0)
            continue

        dist = compute_levenshtein_distance(s_true, s_pred)
        score = (1.0 - (dist / max_len)) * 100.0
        scores.append(max(0.0, score))

    return float(np.mean(scores))


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
    y_pred: Union[np.ndarray, torch.Tensor, List[int]],
    num_classes: Optional[int] = None,
    class_names: Optional[Union[List[str], Dict[int, str]]] = None,
    include_background: bool = True,
) -> Dict[str, Any]:
    """Calculate comprehensive surgical gesture evaluation metrics.
    
    Metrics:
        - Accuracy (overall framewise accuracy)
        - Balanced Accuracy (macro-recall across all classes)
        - Macro-F1 (unweighted average of F1 across classes)
        - Weighted-F1 (support-weighted average of F1)
        - Per-class Precision, Recall, F1, and Support
        - Confusion Matrix (C x C integer array)
        
    Args:
        y_true: Ground truth labels (1D array or tensor).
        y_pred: Predicted labels or logits.
        num_classes: Total number of classes. Inferred if not provided.
        class_names: Optional mapping from integer class index to readable string name.
        include_background: Whether to include class 0 (G0) in the report.
        
    Returns:
        Dict of computed scalar metrics, per-class details, and confusion matrix.
    """
    true_arr = to_numpy_labels(y_true)
    pred_arr = to_numpy_labels(y_pred)

    if len(true_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch: {len(true_arr)} true labels vs {len(pred_arr)} predictions."
        )

    if len(true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Determine unique class labels
    if num_classes is not None:
        labels = list(range(num_classes))
    else:
        max_cls = max(int(np.max(true_arr)), int(np.max(pred_arr)))
        labels = list(range(max_cls + 1))

    # Standard metrics
    acc = float(accuracy_score(true_arr, pred_arr))
    bal_acc = float(balanced_accuracy_score(true_arr, pred_arr))
    macro_f1 = float(f1_score(true_arr, pred_arr, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_arr, pred_arr, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(true_arr, pred_arr, labels=labels)

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        true_arr, pred_arr, labels=labels, zero_division=0
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, cls_id in enumerate(labels):
        name = class_names[cls_id] if class_names and cls_id in class_names else f"Class_{cls_id}"
        per_class[name] = {
            "class_id": int(cls_id),
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "support": int(supports[idx]),
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_samples": int(len(true_arr)),
        "num_classes": len(labels),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format per-class metrics into a clean GitHub-style markdown table."""
    lines = [
        f"**Summary Metrics:**",
        f"- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"- Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        f"- Weighted-F1: {metrics['weighted_f1']:.4f}",
        f"- Total Samples: {metrics['total_samples']:,}",
        "",
        "| Class | ID | Precision | Recall | F1-Score | Support |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, data in metrics["per_class"].items():
        lines.append(
            f"| {name} | {data['class_id']} | {data['precision']:.4f} | "
            f"{data['recall']:.4f} | {data['f1']:.4f} | {data['support']:,} |"
        )

    return "\n".join(lines)
