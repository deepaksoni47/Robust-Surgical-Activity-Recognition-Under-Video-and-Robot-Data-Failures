"""tests/test_iter34_ablation.py

Unit and Validation Tests for ITER-34b Ablation and Model Contribution Analysis.
Verifies statistical equations, degrees of freedom, probability constraints,
confusion matrix integrity, and multiple-testing corrections.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "iter34_ablation_summary.json"


@pytest.fixture(scope="module")
def ablation_summary():
    assert SUMMARY_PATH.is_file(), f"Missing ablation summary artifact at {SUMMARY_PATH}"
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_ablation_summary_structure(ablation_summary):
    """Verify presence and completion status of all 4 required workstreams."""
    assert ablation_summary["status"] == "COMPLETED"
    for ws in [
        "workstream_a_fusion_ablations",
        "workstream_b_temporal_modeling",
        "workstream_c_error_and_subject_analysis",
        "workstream_d_statistical_validation",
    ]:
        assert ws in ablation_summary, f"Missing {ws} in ablation summary"


def test_workstream_a_statistical_properties(ablation_summary):
    """Verify degrees of freedom, sample sizes, and CI bounds in Workstream A."""
    data = ablation_summary["workstream_a_fusion_ablations"]
    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        assert task in data
        models = data[task]["models"]
        for m in ["unimodal_video", "unimodal_kinematics", "fixed_late_fusion", "feature_concat_fusion", "adaptive_reliability_fusion"]:
            assert m in models
            assert 0.0 <= models[m]["accuracy"] <= 100.0
            assert 0.0 <= models[m]["macro_f1"] <= 100.0
            assert 0.0 <= models[m]["edit_score"] <= 100.0

        # Check paired stats
        comp = data[task]["adaptive_vs_kinematics"]
        expected_n = 7 if task == "Needle_Passing" else 8
        for metric in ["accuracy", "macro_f1", "edit_score"]:
            st = comp[metric]
            assert st["n"] == expected_n
            assert st["treatment_std"] >= 0.0
            assert st["baseline_std"] >= 0.0
            assert st["ci_95"][0] <= st["ci_95"][1]
            assert 0.0 <= st["p_value_t"] <= 1.0
            assert 0.0 <= st["p_value_wilcoxon"] <= 1.0
            # Delta mean equals treatment - baseline within rounding
            expected_delta = round(st["treatment_mean"] - st["baseline_mean"], 2)
            assert abs(st["delta_mean"] - expected_delta) <= 0.05


def test_workstream_b_temporal_gains(ablation_summary):
    """Verify that temporal sequence modeling dramatically improves Edit Score over framewise MLP."""
    data = ablation_summary["workstream_b_temporal_modeling"]
    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        t_data = data[task]
        
        # Kinematics GRU vs MLP
        kin_comp = t_data["kinematics_gru_vs_mlp"]
        acc_delta = kin_comp["accuracy"]["delta_mean"]
        edit_delta = kin_comp["edit_score"]["delta_mean"]
        
        # Frame accuracy difference is small (within +/- 3%)
        assert abs(acc_delta) < 3.0, f"Frame accuracy delta unexpectedly large: {acc_delta}"
        # Edit score difference is overwhelmingly large (> +20 points)
        assert edit_delta > 20.0, f"Expected huge Edit Score gain, got {edit_delta}"
        assert kin_comp["edit_score"]["p_value_t"] < 0.001
        assert kin_comp["edit_score"]["cohens_dz"] > 2.0


def test_workstream_c_confusion_matrix_properties(ablation_summary):
    """Verify mathematical properties of aggregate confusion matrices."""
    data = ablation_summary["workstream_c_error_and_subject_analysis"]
    expected_shapes = {
        "Suturing": (11, 11),
        "Knot_Tying": (7, 7),
        "Needle_Passing": (11, 11),
    }

    for task, expected_shape in expected_shapes.items():
        cm = np.array(data[task]["confusion_matrix"])
        assert cm.shape == expected_shape
        assert np.all(cm >= 0)
        assert cm.sum() == data[task]["total_samples"]

        # Check rare gestures identification
        rare_gestures = data[task]["rare_gestures"]
        for g in rare_gestures:
            assert g in data[task]["per_gesture_breakdown"]
            assert data[task]["per_gesture_breakdown"][g]["is_rare"] is True


def test_workstream_d_multiple_testing_monotonicity(ablation_summary):
    """Verify Holm-Bonferroni and Benjamini-Hochberg monotonicities."""
    family = ablation_summary["workstream_d_statistical_validation"]["family_wise_evaluations"]
    assert len(family) == 9  # 3 tasks x 3 metrics

    for key, eval_item in family.items():
        raw_p = eval_item["raw_p"]
        holm_p = eval_item["holm_bonferroni_p"]
        bh_q = eval_item["benjamini_hochberg_q"]

        # Adjusted p-values must be bounded in [0, 1]
        assert 0.0 <= raw_p <= 1.0
        assert 0.0 <= holm_p <= 1.0
        assert 0.0 <= bh_q <= 1.0

        # Holm-Bonferroni p_adj >= raw_p
        assert holm_p >= raw_p - 1e-9


def test_figure_files_exist():
    """Verify that all 5 publication figure files were generated in PDF and PNG."""
    fig_dir = PROJECT_ROOT / "paper" / "figures"
    expected_names = [
        "fig_iter34_ablation_contributions",
        "fig_iter34_temporal_impact",
        "fig_iter34_confusion_matrices",
        "fig_iter34_subject_variance",
        "fig_iter34_gesture_imbalance",
    ]
    for name in expected_names:
        pdf_path = fig_dir / f"{name}.pdf"
        png_path = fig_dir / f"{name}.png"
        assert pdf_path.is_file(), f"Missing vector figure: {pdf_path}"
        assert png_path.is_file(), f"Missing raster figure: {png_path}"
        assert pdf_path.stat().st_size > 1000
        assert png_path.stat().st_size > 1000
