"""tests/test_iter36_analysis.py

Test Suite for ITER-36 Independent Scientific Audit.
Verifies:
1. Integrity of all 4 ITER-35 JSON result artifacts and summary file.
2. Complete 8-fold coverage for each loss formulation on Suturing.
3. Correctness of paired statistical tests (t-test, Wilcoxon, Cohen's dz).
4. Correctness of step-down Holm-Bonferroni p-value adjustment.
5. Presence and validity of generated publication figures in paper/figures/.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.analyze_iter36_results import (
    compute_paired_stats,
    holm_bonferroni,
    audit_iter34b_optional,
    analyze_iter35_experiments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestArtifactIntegrity:
    """Verify that all required JSON artifacts exist and have 8 complete folds."""

    @pytest.mark.parametrize("cond", ["ce", "class_weighted", "focal", "cb_focal"])
    def test_condition_json_exists_and_valid(self, cond: str):
        if cond == "class_weighted":
            path = PROJECT_ROOT / "artifacts/surgmotion_fusion_adaptive_Suturing_results.json"
        else:
            path = PROJECT_ROOT / f"artifacts/surgmotion_fusion_adaptive_Suturing_{cond}_results.json"

        assert path.is_file(), f"Missing artifact for {cond}: {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["task"] == "Suturing"
        assert data["fusion_type"] == "adaptive"
        assert data["active_folds"] == 8
        assert len(data["fold_results"]) == 8

        # Verify all 8 users B-I
        expected_users = ["B", "C", "D", "E", "F", "G", "H", "I"]
        actual_users = [fr["test_user"] for fr in data["fold_results"]]
        assert actual_users == expected_users

        # Check finite metric ranges
        for fr in data["fold_results"]:
            assert 0.0 <= fr["accuracy"] <= 100.0
            assert 0.0 <= fr["balanced_accuracy"] <= 100.0
            assert 0.0 <= fr["macro_f1"] <= 100.0
            assert 0.0 <= fr["edit_score"] <= 100.0

    def test_integrated_results_json(self):
        path = PROJECT_ROOT / "artifacts/iter36_integrated_results.json"
        assert path.is_file()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audit_version"] == "ITER-36"
        assert "iter34b_optional_audit" in data
        assert "iter35_class_imbalance_analysis" in data
        assert len(data["iter35_class_imbalance_analysis"]["fold_breakdown"]) == 8


class TestStatisticalMethods:
    """Verify paired comparison math and Holm-Bonferroni correction."""

    def test_paired_stats_zero_difference(self):
        v = [70.0, 75.0, 80.0, 65.0, 85.0, 60.0, 78.0, 72.0]
        res = compute_paired_stats(v, v)
        assert res["delta_mean"] == 0.0
        assert res["cohens_dz"] == 0.0
        assert res["p_value_t"] == 1.0

    def test_paired_stats_known_difference(self):
        base = [70.0, 70.0, 70.0, 70.0]
        treat = [75.0, 75.0, 75.0, 75.0]
        res = compute_paired_stats(treat, base)
        assert res["delta_mean"] == 5.0
        assert res["delta_std"] == 0.0

    def test_holm_bonferroni_ordering(self):
        # 3 p-values: smallest multiplied by 3, middle by 2, largest by 1
        raw = [0.01, 0.04, 0.20]
        adj = holm_bonferroni(raw)
        assert adj[0] == pytest.approx(0.03, abs=1e-5)
        assert adj[1] == pytest.approx(0.08, abs=1e-5)
        assert adj[2] == pytest.approx(0.20, abs=1e-5)

    def test_holm_bonferroni_monotonicity(self):
        raw = [0.04, 0.03, 0.50]
        adj = holm_bonferroni(raw)
        # Smallest is 0.03 (rank 0, mult 3) -> 0.09
        # Middle is 0.04 (rank 1, mult 2) -> 0.08, but enforced max(cum_max) -> 0.09
        # Largest is 0.50 (rank 2, mult 1) -> 0.50
        assert adj[1] == pytest.approx(0.09, abs=1e-5)
        assert adj[0] == pytest.approx(0.09, abs=1e-5)
        assert adj[2] == pytest.approx(0.50, abs=1e-5)


class TestFigureArtifacts:
    """Verify that all 4 figure pairs (.png and .pdf) were generated."""

    @pytest.mark.parametrize("fig_name", [
        "fig_iter36_loss_comparison",
        "fig_iter36_subject_variance",
        "fig_iter36_robustness_profiles",
        "fig_iter36_imbalance_tradeoff",
    ])
    def test_figure_files_exist_and_nonempty(self, fig_name: str):
        png_path = PROJECT_ROOT / f"paper/figures/{fig_name}.png"
        pdf_path = PROJECT_ROOT / f"paper/figures/{fig_name}.pdf"
        assert png_path.is_file(), f"Missing PNG figure: {png_path}"
        assert pdf_path.is_file(), f"Missing PDF figure: {pdf_path}"
        assert png_path.stat().st_size > 1000
        assert pdf_path.stat().st_size > 1000
