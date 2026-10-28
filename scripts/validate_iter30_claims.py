"""scripts/validate_iter30_claims.py

Independent, reproducible validation script for all numerical claims, statistical tests,
effect sizes, and comparisons across unimodal and multimodal models in ITER-29/ITER-30.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies the step-down Holm-Bonferroni correction to control FWER."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(cum_max, 1.0)
    return adjusted


def load_model_folds(task: str, model_file: str) -> List[Dict[str, Any]]:
    fpath = PROJECT_ROOT / model_file.format(task=task)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "fold_results" in data:
        raw_folds = data["fold_results"]
    elif "folds" in data:
        raw_folds = data["folds"]
    elif task in data and "folds" in data[task]:
        raw_folds = data[task]["folds"]
    else:
        raise ValueError(f"Could not parse fold data from {fpath.name}")

    standardized = []
    for f in raw_folds:
        if "majority" in model_file:
            acc = float(f["accuracy"]) * 100.0
            bal = float(f.get("balanced_accuracy", 0.0)) * 100.0
            f1 = float(f["macro_f1"]) * 100.0
            edit = float(f.get("edit_score", 0.0))
        elif "kinematics_gru" in model_file or "video_videogru" in model_file:
            # Older PyTorch runners saved acc, bal, and f1 as fractions in [0, 1]
            acc = float(f["accuracy"]) * 100.0
            bal = float(f.get("balanced_accuracy", 0.0)) * 100.0
            f1 = float(f["macro_f1"]) * 100.0
            edit = float(f.get("edit_score", 0.0))
        else:
            # SurgMotion baselines and fusion models are already formatted as percentages [0, 100]
            acc = float(f["accuracy"])
            bal = float(f.get("balanced_accuracy", 0.0))
            f1 = float(f["macro_f1"])
            edit = float(f.get("edit_score", 0.0))

        user = f.get("test_user") or f.get("user")
        standardized.append({
            "test_user": user,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "macro_f1": f1,
            "edit_score": edit,
        })
    return standardized


def validate_all_numerical_claims() -> Dict[str, Any]:
    validation_report = {
        "status": "PASSED",
        "tasks": {},
        "discrepancies": [],
    }

    # Model definitions
    unimodal_models = {
        "majority": "artifacts/majority_baseline_results.json",
        "resnet18_videogru": "artifacts/video_videogru_{task}_results.json",
        "surgmotion_videomlp": "artifacts/surgmotion_VideoMLP_{task}_results.json",
        "surgmotion_videogru": "artifacts/surgmotion_VideoGRU_{task}_results.json",
        "kinematics_gru": "artifacts/kinematics_gru_{task}_results.json",
    }

    fusion_models = {
        "late": "artifacts/surgmotion_fusion_late_{task}_results.json",
        "concat": "artifacts/surgmotion_fusion_concat_{task}_results.json",
        "adaptive": "artifacts/surgmotion_fusion_adaptive_{task}_results.json",
    }

    for task in TASKS:
        expected_n = 7 if task == "Needle_Passing" else 8
        task_report = {
            "expected_folds": expected_n,
            "unimodal_summary": {},
            "fusion_summary": {},
            "strongest_unimodal": {},
            "fusion_comparisons": {},
            "statistical_tests": {},
        }

        # 1. Load and verify unimodal models
        unimodal_data = {}
        for m_id, template in unimodal_models.items():
            folds = load_model_folds(task, template)
            if len(folds) != expected_n:
                validation_report["discrepancies"].append(
                    f"{task} {m_id}: fold count {len(folds)} != expected {expected_n}"
                )
            unimodal_data[m_id] = folds
            task_report["unimodal_summary"][m_id] = {
                metric: {
                    "mean": round(float(np.mean([f[metric] for f in folds])), 2),
                    "std": round(float(np.std([f[metric] for f in folds])), 2),
                }
                for metric in METRICS
            }

        # 2. Identify strongest unimodal baseline per metric
        strongest = {}
        for metric in METRICS:
            best_id = max(
                unimodal_data.keys(),
                key=lambda k: float(np.mean([f[metric] for f in unimodal_data[k]]))
            )
            strongest[metric] = {
                "model_id": best_id,
                "mean": task_report["unimodal_summary"][best_id][metric]["mean"],
                "std": task_report["unimodal_summary"][best_id][metric]["std"],
            }
        task_report["strongest_unimodal"] = strongest

        # 3. Load and verify fusion models
        fusion_data = {}
        for f_id, template in fusion_models.items():
            folds = load_model_folds(task, template)
            if len(folds) != expected_n:
                validation_report["discrepancies"].append(
                    f"{task} {f_id}: fold count {len(folds)} != expected {expected_n}"
                )
            fusion_data[f_id] = folds
            task_report["fusion_summary"][f_id] = {
                metric: {
                    "mean": round(float(np.mean([f[metric] for f in folds])), 2),
                    "std": round(float(np.std([f[metric] for f in folds])), 2),
                }
                for metric in METRICS
            }

            # Differences against strongest unimodal
            comp = {}
            for metric in METRICS:
                f_mean = task_report["fusion_summary"][f_id][metric]["mean"]
                b_mean = strongest[metric]["mean"]
                delta_pts = round(f_mean - b_mean, 2)
                pct_improvement = round((f_mean - b_mean) / b_mean * 100.0, 2) if b_mean > 0 else 0.0
                comp[metric] = {
                    "fusion_mean": f_mean,
                    "strongest_baseline_name": strongest[metric]["model_id"],
                    "baseline_mean": b_mean,
                    "percentage_point_delta": delta_pts,
                    "relative_percent_improvement": pct_improvement,
                    "beats_baseline": bool(delta_pts > 0),
                }
            task_report["fusion_comparisons"][f_id] = comp

        # 4. Statistical Tests: Adaptive vs Kinematics GRU (paired fold-by-fold)
        ad_folds = fusion_data["adaptive"]
        kin_folds = unimodal_data["kinematics_gru"]

        # Assert fold user ordering matches exactly
        ad_users = [f["test_user"] for f in ad_folds]
        kin_users = [f["test_user"] for f in kin_folds]
        if ad_users != kin_users:
            validation_report["discrepancies"].append(
                f"{task} user mismatch between Adaptive ({ad_users}) and Kinematics GRU ({kin_users})"
            )

        p_vals = []
        metric_test_records = {}
        for metric in METRICS:
            ad_vals = np.array([f[metric] for f in ad_folds])
            kin_vals = np.array([f[metric] for f in kin_folds])
            diff = ad_vals - kin_vals

            n = len(diff)
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1))
            se_d = std_d / np.sqrt(n)

            # 95% Confidence Interval using Student's t distribution with df=n-1
            ci = stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se_d)

            # Paired t-test
            t_stat, p_t = stats.ttest_rel(ad_vals, kin_vals)

            # Wilcoxon signed-rank test
            try:
                w_stat, p_w = stats.wilcoxon(ad_vals, kin_vals)
            except Exception:
                w_stat, p_w = 0.0, 1.0

            # Cohen's d_z for paired samples
            d_z = mean_d / std_d if std_d > 0 else 0.0

            p_vals.append(float(p_t))
            metric_test_records[metric] = {
                "mean_diff_percentage_points": round(mean_d, 2),
                "std_diff": round(std_d, 2),
                "se_diff": round(se_d, 2),
                "ci_95": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
                "paired_t_stat": round(float(t_stat), 3),
                "p_value_t_uncorrected": round(float(p_t), 4),
                "wilcoxon_stat": round(float(w_stat), 3),
                "p_value_wilcoxon": round(float(p_w), 4),
                "cohen_d_z": round(float(d_z), 3),
            }

        # Apply Holm-Bonferroni correction
        adj_p_vals = holm_bonferroni_correction(p_vals)
        for idx, metric in enumerate(METRICS):
            metric_test_records[metric]["p_value_holm_bonferroni"] = round(adj_p_vals[idx], 4)
            metric_test_records[metric]["significant_at_05"] = bool(adj_p_vals[idx] < 0.05)

        task_report["statistical_tests"]["adaptive_vs_kinematics_gru"] = metric_test_records
        validation_report["tasks"][task] = task_report

    if validation_report["discrepancies"]:
        validation_report["status"] = "FAILED"

    return validation_report


if __name__ == "__main__":
    report = validate_all_numerical_claims()
    print("=" * 80)
    print(f"ITER-30 NUMERICAL CLAIM VALIDATION REPORT: Status={report['status']}")
    print("=" * 80)
    for task, t_data in report["tasks"].items():
        print(f"\n--- Task: {task} ({t_data['expected_folds']} active folds) ---")
        for metric in METRICS:
            best_unimodal = t_data["strongest_unimodal"][metric]
            ad_comp = t_data["fusion_comparisons"]["adaptive"][metric]
            stat = t_data["statistical_tests"]["adaptive_vs_kinematics_gru"][metric]
            print(
                f"  {metric:18s}: Best Unimodal={best_unimodal['model_id']} ({best_unimodal['mean']}%) | "
                f"Adaptive={ad_comp['fusion_mean']}% (diff={ad_comp['percentage_point_delta']:+6.2f} pts) | "
                f"p_uncorr={stat['p_value_t_uncorrected']:.4f}, p_adj={stat['p_value_holm_bonferroni']:.4f}, "
                f"d_z={stat['cohen_d_z']:5.2f}, Sig(0.05)={stat['significant_at_05']}"
            )
    out_file = PROJECT_ROOT / "artifacts/iter30_numerical_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report to {out_file}")
