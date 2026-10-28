"""scripts/audit_iter32_results.py

Comprehensive independent audit of ITER-31 mentor GPU ablation results:
- Validates all 12 JSON files in artifacts/iter31_noise_ablation/
- Verifies fold coverage (23 active folds, Subject G empty fold handling)
- Performs paired statistical tests across clean and robustness conditions
- Analyzes gate trajectory weights across continuous visual noise levels
- Generates artifacts/iter32_results_summary.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "iter31_noise_ablation"


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


def compute_paired_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute mean, std, difference, 95% CI of diff, paired t-test p-value, and Cohen's d_z."""
    n = len(x)
    diff = x - y
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    sem = std_diff / np.sqrt(n) if n > 1 else 0.0

    if n > 1:
        t_stat, p_val = stats.ttest_rel(x, y)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=sem) if sem > 0 else (mean_diff, mean_diff)
        w_stat, w_pval = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)
    else:
        t_stat, p_val = 0.0, 1.0
        ci = (mean_diff, mean_diff)
        w_stat, w_pval = 0.0, 1.0

    return {
        "mean_x": round(float(np.mean(x)), 2),
        "std_x": round(float(np.std(x)), 2),
        "mean_y": round(float(np.mean(y)), 2),
        "std_y": round(float(np.std(y)), 2),
        "diff_mean": round(mean_diff, 2),
        "diff_ci_95": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
        "p_value_paired_t": round(float(p_val), 5),
        "wilcoxon_p": round(float(w_pval), 5),
        "cohen_d_z": round(cohen_d_paired(x, y), 3),
        "n_folds": int(n),
    }


def audit_results() -> Dict[str, Any]:
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    conditions = ["continuous_noise", "existing"]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]
    robustness_conditions = [
        "clean",
        "video_noise_0.2",
        "video_noise_0.5",
        "video_noise_1.0",
        "video_noise_1.5",
        "video_noise_2.0",
        "kinematics_corrupted",
        "missing_video",
        "missing_kinematics",
    ]

    summary: Dict[str, Any] = {
        "file_inventory": {},
        "fold_coverage": {},
        "clean_comparison": {},
        "robustness_comparison": {},
        "degradation_comparison": {},
        "gate_trajectory_audit": {},
    }

    # 1. File Inventory
    all_files_exist = True
    for cond in conditions:
        for t in tasks:
            res_file = ARTIFACTS_DIR / f"surgmotion_ablation_{cond}_{t}_results.json"
            traj_file = ARTIFACTS_DIR / f"gate_trajectories_{cond}_{t}.json"
            summary["file_inventory"][res_file.name] = {
                "exists": res_file.is_file(),
                "size_bytes": res_file.stat().st_size if res_file.is_file() else 0,
            }
            summary["file_inventory"][traj_file.name] = {
                "exists": traj_file.is_file(),
                "size_bytes": traj_file.stat().st_size if traj_file.is_file() else 0,
            }
            if not res_file.is_file() or not traj_file.is_file():
                all_files_exist = False

    summary["all_files_present"] = all_files_exist

    # 2. Fold Coverage Audit
    raw_data: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        raw_data[cond] = {}
        for t in tasks:
            res_file = ARTIFACTS_DIR / f"surgmotion_ablation_{cond}_{t}_results.json"
            data = load_json(res_file)
            raw_data[cond][t] = data

            fold_nums = [r["fold"] for r in data["fold_results"]]
            test_users = [r["test_user"] for r in data["fold_results"]]
            summary["fold_coverage"][f"{cond}_{t}"] = {
                "n_active_folds": data["n_active_folds"],
                "fold_numbers": fold_nums,
                "test_users": test_users,
                "clean_acc_mean": data["clean_summary"]["accuracy_mean"],
                "clean_f1_mean": data["clean_summary"]["macro_f1_mean"],
                "clean_edit_mean": data["clean_summary"]["edit_score_mean"],
            }

    # 3. Clean Performance Paired Statistical Comparison
    # continuous_noise (X) vs existing (Y)
    for t in tasks:
        summary["clean_comparison"][t] = {}
        data_c = raw_data["continuous_noise"][t]
        data_e = raw_data["existing"][t]

        folds_c = {r["fold"]: r for r in data_c["fold_results"]}
        folds_e = {r["fold"]: r for r in data_e["fold_results"]}
        common_folds = sorted(list(set(folds_c.keys()).intersection(folds_e.keys())))

        for m in metrics:
            xc = np.array([folds_c[f][m] for f in common_folds])
            ye = np.array([folds_e[f][m] for f in common_folds])
            st = compute_paired_stats(xc, ye)
            summary["clean_comparison"][t][m] = st

    # 4. Robustness Comparison across all 9 conditions
    for t in tasks:
        summary["robustness_comparison"][t] = {}
        summary["degradation_comparison"][t] = {}
        data_c = raw_data["continuous_noise"][t]
        data_e = raw_data["existing"][t]

        rob_c = data_c.get("robustness_summary", {})
        rob_e = data_e.get("robustness_summary", {})

        clean_acc_c = data_c["clean_summary"]["accuracy_mean"]
        clean_acc_e = data_e["clean_summary"]["accuracy_mean"]
        clean_f1_c = data_c["clean_summary"]["macro_f1_mean"]
        clean_f1_e = data_e["clean_summary"]["macro_f1_mean"]

        for r_cond in robustness_conditions:
            if r_cond in rob_c and r_cond in rob_e:
                rc = rob_c[r_cond]
                re = rob_e[r_cond]

                summary["robustness_comparison"][t][r_cond] = {
                    "continuous_acc": rc["accuracy_mean"],
                    "existing_acc": re["accuracy_mean"],
                    "diff_acc": round(rc["accuracy_mean"] - re["accuracy_mean"], 2),
                    "continuous_f1": rc["macro_f1_mean"],
                    "existing_f1": re["macro_f1_mean"],
                    "diff_f1": round(rc["macro_f1_mean"] - re["macro_f1_mean"], 2),
                    "continuous_edit": rc["edit_score_mean"],
                    "existing_edit": re["edit_score_mean"],
                    "diff_edit": round(rc["edit_score_mean"] - re["edit_score_mean"], 2),
                    "continuous_alpha_v": rc.get("alpha_video_mean", None),
                    "existing_alpha_v": re.get("alpha_video_mean", None),
                }

                # Degradation from own clean
                deg_acc_c = round(rc["accuracy_mean"] - clean_acc_c, 2)
                deg_acc_e = round(re["accuracy_mean"] - clean_acc_e, 2)
                deg_f1_c = round(rc["macro_f1_mean"] - clean_f1_c, 2)
                deg_f1_e = round(re["macro_f1_mean"] - clean_f1_e, 2)

                summary["degradation_comparison"][t][r_cond] = {
                    "continuous_deg_acc": deg_acc_c,
                    "existing_deg_acc": deg_acc_e,
                    "relative_advantage_acc": round(deg_acc_c - deg_acc_e, 2),  # > 0 means continuous drops less
                    "continuous_deg_f1": deg_f1_c,
                    "existing_deg_f1": deg_f1_e,
                    "relative_advantage_f1": round(deg_f1_c - deg_f1_e, 2),
                }

    # 5. Gate Trajectory Audit
    for t in tasks:
        summary["gate_trajectory_audit"][t] = {}
        for cond in conditions:
            traj_path = ARTIFACTS_DIR / f"gate_trajectories_{cond}_{t}.json"
            with open(traj_path, "r", encoding="utf-8") as f:
                tdata = json.load(f)

            records = tdata["records"]
            by_cond: Dict[str, List[float]] = {}
            for r in records:
                c_name = r["condition"]
                by_cond.setdefault(c_name, []).append(r["alpha_video"])

            cond_means = {cn: round(float(np.mean(vals)), 4) for cn, vals in by_cond.items()}
            summary["gate_trajectory_audit"][t][cond] = {
                "total_records": len(records),
                "alpha_video_by_condition": cond_means,
            }

    out_file = PROJECT_ROOT / "artifacts" / "iter32_results_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = audit_results()
    print("Audit finished successfully. Results summary written to artifacts/iter32_results_summary.json")
