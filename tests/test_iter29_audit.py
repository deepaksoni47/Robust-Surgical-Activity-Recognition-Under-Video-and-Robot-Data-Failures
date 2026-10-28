"""tests/test_iter29_audit.py

Automated verification test suite for ITER-29:
- Validates integrity of all 9 ITER-28 multimodal fusion JSON artifacts.
- Asserts correct active fold counts (8 Suturing, 8 Knot Tying, 7 Needle Passing).
- Asserts strict finite values and mathematical precision of metrics.
- Validates unified comparison and robustness JSON artifacts.
- Verifies statistical analysis output schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tasks() -> list[str]:
    return ["Suturing", "Knot_Tying", "Needle_Passing"]


@pytest.fixture
def expected_folds() -> dict[str, int]:
    return {
        "Suturing": 8,
        "Knot_Tying": 8,
        "Needle_Passing": 7,
    }


def test_iter28_raw_artifacts_exist_and_valid(tasks, expected_folds):
    """Verify that all 9 ITER-28 fusion result artifacts exist and are valid JSON."""
    fusion_types = ["late", "concat", "adaptive"]

    for task in tasks:
        for ftype in fusion_types:
            fpath = PROJECT_ROOT / f"artifacts/surgmotion_fusion_{ftype}_{task}_results.json"
            assert fpath.is_file(), f"Missing artifact: {fpath.name}"

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "fold_results" in data, f"{fpath.name} missing fold_results"
            assert len(data["fold_results"]) == expected_folds[task], (
                f"{fpath.name} has {len(data['fold_results'])} folds, expected {expected_folds[task]}"
            )
            assert data.get("active_folds") == expected_folds[task]

            # Verify finite values and mathematical consistency of means
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]:
                vals = [f[metric] for f in data["fold_results"]]
                assert np.all(np.isfinite(vals)), f"{fpath.name} contains non-finite values in {metric}"
                calc_mean = round(float(np.mean(vals)), 2)
                doc_mean = data["metrics"][metric]["mean"]
                assert doc_mean == calc_mean, (
                    f"{fpath.name} {metric} mismatch: documented {doc_mean} vs calculated {calc_mean}"
                )


def test_empty_needle_passing_fold_excluded():
    """Verify Needle Passing fold 6 (User G) is properly omitted due to empty test split."""
    for ftype in ["late", "concat", "adaptive"]:
        fpath = PROJECT_ROOT / f"artifacts/surgmotion_fusion_{ftype}_Needle_Passing_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        users = [f["test_user"] for f in data["fold_results"]]
        assert "G" not in users, f"User G should be excluded from Needle Passing ({ftype})"
        assert len(users) == 7, f"Needle Passing should have 7 active folds ({ftype})"
        assert users == ["B", "C", "D", "E", "F", "H", "I"]


def test_unified_comparison_artifact():
    """Verify artifacts/iter29_unified_comparison.json schema and completeness."""
    fpath = PROJECT_ROOT / "artifacts/iter29_unified_comparison.json"
    assert fpath.is_file(), "Unified comparison artifact missing"

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "tasks" in data

    expected_models = [
        "majority_baseline",
        "resnet18_videogru",
        "surgmotion_videomlp",
        "surgmotion_videogru",
        "kinematics_gru",
        "surgmotion_fusion_late",
        "surgmotion_fusion_concat",
        "surgmotion_fusion_adaptive",
    ]

    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        assert task in data["tasks"]
        t_data = data["tasks"][task]
        assert "models" in t_data
        assert "strongest_unimodal_baselines" in t_data
        assert "fusion_comparisons" in t_data

        for m_id in expected_models:
            assert m_id in t_data["models"], f"Model {m_id} missing in {task}"
            m_info = t_data["models"][m_id]
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]:
                assert metric in m_info["metrics"]
                assert np.isfinite(m_info["metrics"][metric]["mean"])


def test_robustness_comparison_artifact():
    """Verify artifacts/iter29_robustness_comparison.json schema and conditions."""
    fpath = PROJECT_ROOT / "artifacts/iter29_robustness_comparison.json"
    assert fpath.is_file(), "Robustness comparison artifact missing"

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_conditions = [
        "clean",
        "video_corrupted",
        "kinematics_corrupted",
        "video_missing",
        "kinematics_missing",
    ]

    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        assert task in data["tasks"]
        for ftype in ["concat", "adaptive"]:
            assert ftype in data["tasks"][task]
            conds = data["tasks"][task][ftype]["conditions"]
            for cond in expected_conditions:
                assert cond in conds, f"Condition {cond} missing in {task} {ftype}"
                assert "accuracy" in conds[cond]
                assert "macro_f1" in conds[cond]
                assert "edit_score" in conds[cond]
                assert "degradation_vs_clean" in conds[cond]


def test_statistical_analysis_artifact():
    """Verify artifacts/iter29_statistical_analysis.json exists and has valid tests."""
    fpath = PROJECT_ROOT / "artifacts/iter29_statistical_analysis.json"
    assert fpath.is_file(), "Statistical analysis artifact missing"

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for task in ["Suturing", "Knot_Tying", "Needle_Passing"]:
        assert task in data
        assert "adaptive_vs_kin" in data[task]
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]:
            entry = data[task]["adaptive_vs_kin"][metric]
            assert "mean_diff" in entry
            assert "p_value_t" in entry
            assert "p_value_t_holm_bonferroni" in entry
            assert "cohen_d" in entry
            assert np.isfinite(entry["p_value_t"])
