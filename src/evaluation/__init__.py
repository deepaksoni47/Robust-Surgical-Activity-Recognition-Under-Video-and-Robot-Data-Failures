"""Evaluation metrics, prediction export, and cross-validation runners."""

from src.evaluation.benchmark_comparison import (
    audit_task_multimodal_readiness,
    compare_model_benchmarks,
    export_comparison_csv,
    format_comparison_markdown,
    format_readiness_audit_markdown,
)
from src.evaluation.metrics import (
    collapse_to_segments,
    compute_classification_metrics,
    compute_edit_score,
    compute_levenshtein_distance,
    format_metrics_table,
    to_numpy_labels,
)

__all__ = [
    "compute_classification_metrics",
    "compute_edit_score",
    "compute_levenshtein_distance",
    "collapse_to_segments",
    "to_numpy_labels",
    "format_metrics_table",
    "audit_task_multimodal_readiness",
    "format_readiness_audit_markdown",
    "compare_model_benchmarks",
    "format_comparison_markdown",
    "export_comparison_csv",
]
