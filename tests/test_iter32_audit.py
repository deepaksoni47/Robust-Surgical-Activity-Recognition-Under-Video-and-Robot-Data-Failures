"""tests/test_iter32_audit.py

Automated test suite verifying the integrity, completeness, and statistical consistency
of the ITER-31 mentor GPU ablation results and ITER-32 scientific audit.
"""

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "iter31_noise_ablation"
SUMMARY_FILE = PROJECT_ROOT / "artifacts" / "iter32_results_summary.json"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def test_iter32_summary_file_exists_and_valid():
    """Verify artifacts/iter32_results_summary.json exists and parses correctly."""
    assert SUMMARY_FILE.is_file(), f"Missing audit summary: {SUMMARY_FILE}"
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "file_inventory" in data
    assert "fold_coverage" in data
    assert "clean_comparison" in data
    assert "robustness_comparison" in data
    assert "gate_trajectory_audit" in data


def test_all_12_mentor_gpu_artifacts_exist_and_non_empty():
    """Verify all 6 results summaries and 6 gate trajectory files exist with valid sizes."""
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    conditions = ["continuous_noise", "existing"]

    for cond in conditions:
        for t in tasks:
            res_file = ARTIFACTS_DIR / f"surgmotion_ablation_{cond}_{t}_results.json"
            traj_file = ARTIFACTS_DIR / f"gate_trajectories_{cond}_{t}.json"

            assert res_file.is_file(), f"Missing result file: {res_file}"
            assert traj_file.is_file(), f"Missing trajectory file: {traj_file}"

            assert res_file.stat().st_size > 4000, f"Result file suspiciously small: {res_file}"
            assert traj_file.stat().st_size > 10_000_000, f"Trajectory file suspiciously small: {traj_file}"


def test_fold_coverage_exact_23_active_folds():
    """Verify exact 23 active LOUO folds evaluated per condition, with Subject G skipped on Needle Passing."""
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cov = data["fold_coverage"]
    # Suturing: 8 active folds
    assert cov["continuous_noise_Suturing"]["n_active_folds"] == 8
    assert cov["existing_Suturing"]["n_active_folds"] == 8

    # Knot Tying: 8 active folds
    assert cov["continuous_noise_Knot_Tying"]["n_active_folds"] == 8
    assert cov["existing_Knot_Tying"]["n_active_folds"] == 8

    # Needle Passing: 7 active folds (Subject G skipped)
    assert cov["continuous_noise_Needle_Passing"]["n_active_folds"] == 7
    assert cov["existing_Needle_Passing"]["n_active_folds"] == 7

    assert "G" not in cov["continuous_noise_Needle_Passing"]["test_users"]
    assert "G" not in cov["existing_Needle_Passing"]["test_users"]


def test_gate_weights_alpha_video_flat_response():
    """Verify that empirical alpha_v shift across noise levels sigma in [0.0, 2.0] is under 0.02."""
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rob = data["robustness_comparison"]
    for t in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        clean_alpha_v = rob[t]["clean"]["continuous_alpha_v"]
        noisy_alpha_v = rob[t]["video_noise_2.0"]["continuous_alpha_v"]
        delta_alpha = abs(noisy_alpha_v - clean_alpha_v)
        # Shift must be less than 0.02 (empirically < 0.009)
        assert delta_alpha < 0.02, f"Task {t} unexpected alpha shift: {delta_alpha}"


def test_deterministic_fallback_gate_weights():
    """Verify that missing video routes strictly 0.0 to video and missing kin routes 1.0 to video."""
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rob = data["robustness_comparison"]
    for t in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        assert rob[t]["missing_video"]["continuous_alpha_v"] == 0.0
        assert rob[t]["missing_video"]["existing_alpha_v"] == 0.0

        assert rob[t]["missing_kinematics"]["continuous_alpha_v"] == 1.0
        assert rob[t]["missing_kinematics"]["existing_alpha_v"] == 1.0


def test_iter32_publication_figures_exist():
    """Verify all 3 regenerated publication figures exist in PDF and PNG."""
    expected_figures = [
        "fig_iter32_real_gate_noise_response",
        "fig_iter32_real_gate_temporal_dynamics",
        "fig_iter32_real_ablation_performance",
    ]
    for fig_name in expected_figures:
        pdf_file = FIGURES_DIR / f"{fig_name}.pdf"
        png_file = FIGURES_DIR / f"{fig_name}.png"
        assert pdf_file.is_file(), f"Missing figure PDF: {pdf_file}"
        assert png_file.is_file(), f"Missing figure PNG: {png_file}"
        assert pdf_file.stat().st_size > 1000
        assert png_file.stat().st_size > 1000
