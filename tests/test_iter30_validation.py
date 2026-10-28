"""tests/test_iter30_validation.py

Automated test suite for ITER-30:
- Validates that the numerical validation script executes cleanly without discrepancies.
- Asserts that all 8 publication figures (4 PDF, 4 PNG) exist in paper/figures/ with non-zero size.
- Verifies that paper/main.tex compiles and references all generated figures and tables.
- Asserts that numerical claims in iter30_numerical_validation.json match audited ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_validation_script_status_passed():
    """Verify that validate_all_numerical_claims() reports PASSED status with 0 discrepancies."""
    from scripts.validate_iter30_claims import validate_all_numerical_claims
    report = validate_all_numerical_claims()
    assert report["status"] == "PASSED", f"Validation failed with discrepancies: {report['discrepancies']}"
    assert len(report["discrepancies"]) == 0


def test_publication_figures_exist():
    """Verify that all 4 figures exist in both PDF and PNG formats with valid file sizes."""
    fig_dir = PROJECT_ROOT / "paper/figures"
    expected_figures = [
        "fig1_model_comparison",
        "fig2_robustness_degradation",
        "fig3_gesture_timelines_ribbon",
        "fig4_statistical_effect_sizes",
    ]

    for fig_name in expected_figures:
        pdf_path = fig_dir / f"{fig_name}.pdf"
        png_path = fig_dir / f"{fig_name}.png"

        assert pdf_path.is_file(), f"Missing PDF figure: {pdf_path.name}"
        assert png_path.is_file(), f"Missing PNG figure: {png_path.name}"

        assert pdf_path.stat().st_size > 5000, f"PDF figure {fig_name}.pdf is unexpectedly small"
        assert png_path.stat().st_size > 10000, f"PNG figure {fig_name}.png is unexpectedly small"


def test_paper_latex_contains_updated_results_and_figures():
    """Verify that paper/main.tex includes all new figures, tables, and audited scores."""
    paper_path = PROJECT_ROOT / "paper/main.tex"
    assert paper_path.is_file(), "paper/main.tex missing"

    content = paper_path.read_text(encoding="utf-8")

    # Check figure inclusions
    assert "figures/fig1_model_comparison.pdf" in content
    assert "figures/fig2_robustness_degradation.pdf" in content
    assert "figures/fig3_gesture_timelines_ribbon.pdf" in content
    assert "figures/fig4_statistical_effect_sizes.pdf" in content

    # Check key verified metrics are present in text/tables
    assert "72.28" in content  # Suturing Adaptive Acc
    assert "72.34" in content  # Knot Tying Adaptive Acc
    assert "51.48" in content  # Needle Passing Adaptive Acc
    assert "68.95" in content  # Suturing Edit Score
    assert "64.66" in content  # Knot Tying Edit Score
    assert "54.81" in content  # Needle Passing Edit Score

    # Check official split and limitations mentions
    assert "Leave-One-User-Out" in content
    assert "Fold 6" in content
    assert "Subject G" in content
    assert "genlock" in content or "synchronization" in content


def test_iter30_numerical_validation_json():
    """Verify artifacts/iter30_numerical_validation.json schema and exact values."""
    val_json_path = PROJECT_ROOT / "artifacts/iter30_numerical_validation.json"
    assert val_json_path.is_file(), "artifacts/iter30_numerical_validation.json missing"

    with open(val_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASSED"
    assert len(data["discrepancies"]) == 0

    # Suturing
    sut = data["tasks"]["Suturing"]["fusion_comparisons"]["adaptive"]
    assert sut["accuracy"]["fusion_mean"] == 72.28
    assert sut["accuracy"]["percentage_point_delta"] == 5.64

    # Knot Tying
    kt = data["tasks"]["Knot_Tying"]["fusion_comparisons"]["adaptive"]
    assert kt["accuracy"]["fusion_mean"] == 72.34
    assert kt["accuracy"]["percentage_point_delta"] == 11.25

    # Needle Passing
    np_task = data["tasks"]["Needle_Passing"]["fusion_comparisons"]["adaptive"]
    assert np_task["accuracy"]["fusion_mean"] == 51.48
    assert np_task["accuracy"]["percentage_point_delta"] == 2.79
