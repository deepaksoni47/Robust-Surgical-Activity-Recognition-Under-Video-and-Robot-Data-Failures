"""scripts/statistical_analysis_iter29.py
Statistical significance and effect size analysis for ITER-29.
Compares multimodal fusion models with the strongest unimodal baselines across folds.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
metrics_list = ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni step-down procedure to a list of p-values."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted_p[orig_idx] = min(cum_max, 1.0)
    return adjusted_p


def run_statistical_tests():
    results = {}

    for task in tasks:
        results[task] = {}

        # Load adaptive fusion
        with open(PROJECT_ROOT / f"artifacts/surgmotion_fusion_adaptive_{task}_results.json") as f:
            ad_data = json.load(f)
        # Load concat fusion
        with open(PROJECT_ROOT / f"artifacts/surgmotion_fusion_concat_{task}_results.json") as f:
            concat_data = json.load(f)
        # Load late fusion
        with open(PROJECT_ROOT / f"artifacts/surgmotion_fusion_late_{task}_results.json") as f:
            late_data = json.load(f)
        # Load Kinematics GRU
        with open(PROJECT_ROOT / f"artifacts/kinematics_gru_{task}_results.json") as f:
            kin_data = json.load(f)
        # Load Video GRU
        with open(PROJECT_ROOT / f"artifacts/surgmotion_VideoGRU_{task}_results.json") as f:
            vid_data = json.load(f)

        kin_folds = kin_data.get("folds", kin_data.get("fold_results", []))
        vid_folds = vid_data["fold_results"]
        ad_folds = ad_data["fold_results"]
        concat_folds = concat_data["fold_results"]
        late_folds = late_data["fold_results"]

        n_folds = len(ad_folds)

        # Baseline fold arrays
        kin_metrics = {
            "accuracy": np.array([f["accuracy"] * (100 if f["accuracy"] <= 1 else 1) for f in kin_folds]),
            "balanced_accuracy": np.array([f.get("balanced_accuracy", 0.0) * (100 if f.get("balanced_accuracy", 0.0) <= 1 else 1) for f in kin_folds]),
            "macro_f1": np.array([f["macro_f1"] * (100 if f["macro_f1"] <= 1 else 1) for f in kin_folds]),
            "edit_score": np.array([f.get("edit_score", 0.0) for f in kin_folds]),
        }
        vid_metrics = {
            "accuracy": np.array([f["accuracy"] for f in vid_folds]),
            "balanced_accuracy": np.array([f["balanced_accuracy"] for f in vid_folds]),
            "macro_f1": np.array([f["macro_f1"] for f in vid_folds]),
            "edit_score": np.array([f["edit_score"] for f in vid_folds]),
        }

        # Compare Adaptive Fusion vs Kinematics GRU and vs Video GRU
        for model_name, model_folds in [
            ("adaptive_vs_kin", ad_folds),
            ("concat_vs_kin", concat_folds),
            ("late_vs_kin", late_folds),
        ]:
            results[task][model_name] = {}
            p_vals_task = []
            test_keys = []

            for m in metrics_list:
                m_arr = np.array([f[m] for f in model_folds])
                baseline_arr = kin_metrics[m]

                diff = m_arr - baseline_arr
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1)) if n_folds > 1 else 0.0
                se_diff = std_diff / np.sqrt(n_folds) if n_folds > 0 else 0.0

                # 95% Confidence Interval using t-distribution
                ci_95 = stats.t.interval(0.95, df=n_folds - 1, loc=mean_diff, scale=se_diff) if se_diff > 0 else (mean_diff, mean_diff)

                # Paired t-test
                t_stat, p_t = stats.ttest_rel(m_arr, baseline_arr)

                # Wilcoxon signed-rank test
                try:
                    w_stat, p_w = stats.wilcoxon(m_arr, baseline_arr)
                except Exception as e:
                    w_stat, p_w = 0.0, 1.0

                # Cohen's d_z for paired samples
                cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0

                results[task][model_name][m] = {
                    "model_mean": float(np.mean(m_arr)),
                    "baseline_mean": float(np.mean(baseline_arr)),
                    "mean_diff": round(mean_diff, 2),
                    "std_diff": round(std_diff, 2),
                    "se_diff": round(se_diff, 2),
                    "ci_95": [round(float(ci_95[0]), 2), round(float(ci_95[1]), 2)],
                    "per_fold_diff": [round(float(x), 2) for x in diff],
                    "t_stat": round(float(t_stat), 3),
                    "p_value_t": round(float(p_t), 4),
                    "wilcoxon_stat": round(float(w_stat), 3),
                    "p_value_wilcoxon": round(float(p_w), 4),
                    "cohen_d": round(float(cohen_d), 3),
                }
                p_vals_task.append(float(p_t))
                test_keys.append(m)

            # Apply Holm-Bonferroni correction across the 4 metrics for this comparison
            adj_p = holm_bonferroni(p_vals_task)
            for k, adj in zip(test_keys, adj_p):
                results[task][model_name][k]["p_value_t_holm_bonferroni"] = round(float(adj), 4)
                results[task][model_name][k]["statistically_significant_05"] = bool(adj < 0.05)

        # Also compare Adaptive Fusion vs SurgMotion VideoGRU for Edit Score
        ad_arr_edit = np.array([f["edit_score"] for f in ad_folds])
        vid_arr_edit = vid_metrics["edit_score"]
        diff_edit = ad_arr_edit - vid_arr_edit
        mean_d = float(np.mean(diff_edit))
        std_d = float(np.std(diff_edit, ddof=1))
        se_d = std_d / np.sqrt(n_folds)
        t_stat, p_t = stats.ttest_rel(ad_arr_edit, vid_arr_edit)
        try:
            w_stat, p_w = stats.wilcoxon(ad_arr_edit, vid_arr_edit)
        except Exception:
            w_stat, p_w = 0.0, 1.0

        results[task]["adaptive_vs_video_edit"] = {
            "model_mean": float(np.mean(ad_arr_edit)),
            "baseline_mean": float(np.mean(vid_arr_edit)),
            "mean_diff": round(mean_d, 2),
            "std_diff": round(std_d, 2),
            "se_diff": round(se_d, 2),
            "ci_95": [round(float(mean_d - 2.365 * se_d), 2), round(float(mean_d + 2.365 * se_d), 2)],
            "t_stat": round(float(t_stat), 3),
            "p_value_t": round(float(p_t), 4),
            "p_value_wilcoxon": round(float(p_w), 4),
            "cohen_d": round(float(mean_d / std_d), 3),
        }

    return results


if __name__ == "__main__":
    stat_report = run_statistical_tests()
    print(json.dumps(stat_report, indent=2))
    with open(PROJECT_ROOT / "artifacts/iter29_statistical_analysis.json", "w", encoding="utf-8") as f:
        json.dump(stat_report, f, indent=2)
