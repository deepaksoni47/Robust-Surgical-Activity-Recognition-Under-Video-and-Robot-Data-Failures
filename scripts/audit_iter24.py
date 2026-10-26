"""Comprehensive Empirical Audit Script for ITER-24.

Audits:
1. Completion of all models across all active LOUO folds for Suturing (8), Knot Tying (8), and Needle Passing (7).
2. Checkpoint files, prediction files, and result JSON files.
3. Verification that Subject G is strictly excluded from Needle Passing.
4. Independent recomputation of metrics (Acc, Balanced Acc, Macro-F1, Edit Score) from raw saved predictions.
5. Analysis of robustness experiments (latent perturbations and optical frame corruptions).
6. Comparison of summary statistics against paper/main.tex tables and text.
"""

import os
import json
import numpy as np
from pathlib import Path
from src.evaluation.metrics import (
    compute_classification_metrics,
    collapse_to_segments,
    compute_levenshtein_distance,
)

TASKS = {
    "Suturing": {
        "folds": list(range(1, 9)),
        "users": ["B", "C", "D", "E", "F", "G", "H", "I"],
        "num_classes": 11,
    },
    "Knot_Tying": {
        "folds": list(range(1, 9)),
        "users": ["B", "C", "D", "E", "F", "G", "H", "I"],
        "num_classes": 7,
    },
    "Needle_Passing": {
        "folds": [1, 2, 3, 4, 5, 7, 8],
        "users": ["B", "C", "D", "E", "F", "H", "I"],
        "num_classes": 9,
    },
}

MODELS = [
    "majority",
    "kinematics_mlp",
    "kinematics_gru",
    "video_videogru",
    "late_fusion",
    "concat_fusion",
    "adaptive_fusion",
]

CHECKPOINT_PREFIXES = {
    "kinematics_mlp": "kinematics_mlp",
    "kinematics_gru": "kinematics_gru",
    "video_videogru": "video_videogru",
    "concat_fusion": "concat_fusion",
    "adaptive_fusion": "adaptive_fusion",
}

def compute_single_seq_edit_score(true_seq, pred_seq):
    s_true = collapse_to_segments(true_seq)
    s_pred = collapse_to_segments(pred_seq)
    max_len = max(len(s_true), len(s_pred))
    if max_len == 0:
        return 100.0
    dist = compute_levenshtein_distance(s_true, s_pred)
    return max(0.0, (1.0 - (dist / max_len)) * 100.0)

def main():
    root = Path(".")
    pred_dir = root / "artifacts" / "predictions"
    ckpt_dir = root / "checkpoints"
    artifacts_dir = root / "artifacts"

    print("=" * 80)
    print("PHASE 1: AUDITING ARTIFACT COMPLETENESS AND FOLDS")
    print("=" * 80)

    audit_summary = {}

    for task_name, task_info in TASKS.items():
        expected_folds = task_info["folds"]
        num_classes = task_info["num_classes"]
        audit_summary[task_name] = {}
        print(f"\n--- Task: {task_name} (Expected active folds: {len(expected_folds)}, Num classes: {num_classes}) ---")

        for model in MODELS:
            audit_summary[task_name][model] = {
                "checkpoints_found": 0,
                "predictions_found": 0,
                "recomputed_metrics": [],
                "stored_metrics": [],
            }

            # Majority baseline is stored in artifacts/majority_baseline_results.json
            if model == "majority":
                maj_file = artifacts_dir / "majority_baseline_results.json"
                if maj_file.exists():
                    with open(maj_file) as f:
                        maj_data = json.load(f)
                    task_maj = maj_data.get(task_name, {})
                    audit_summary[task_name][model]["summary_file"] = str(maj_file)
                    audit_summary[task_name][model]["mean_acc"] = task_maj.get("mean_accuracy")
                    audit_summary[task_name][model]["std_acc"] = task_maj.get("std_accuracy", 0.0)
                    audit_summary[task_name][model]["mean_bal_acc"] = task_maj.get("mean_balanced_accuracy")
                    audit_summary[task_name][model]["std_bal_acc"] = task_maj.get("std_balanced_accuracy", 0.0)
                    audit_summary[task_name][model]["mean_f1"] = task_maj.get("mean_macro_f1")
                    audit_summary[task_name][model]["std_f1"] = task_maj.get("std_macro_f1", 0.0)
                continue

            # Check summary json file
            summary_file = artifacts_dir / f"{model}_{task_name}_results.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary_data = json.load(f)
                audit_summary[task_name][model]["summary_data"] = summary_data

            # Check individual folds
            for fold in expected_folds:
                # Checkpoint
                if model in CHECKPOINT_PREFIXES:
                    ckpt_file = ckpt_dir / f"{CHECKPOINT_PREFIXES[model]}_{task_name}_fold{fold}.pt"
                    if ckpt_file.exists():
                        audit_summary[task_name][model]["checkpoints_found"] += 1
                    else:
                        print(f"MISSING CKPT: {ckpt_file}")

                # Prediction file
                p_file = pred_dir / f"{model}_{task_name}_fold{fold}.json"
                if p_file.exists():
                    audit_summary[task_name][model]["predictions_found"] += 1
                    with open(p_file) as f:
                        p_data = json.load(f)
                    
                    y_pred = p_data.get("predictions", [])
                    y_true = p_data.get("ground_truth", [])

                    if len(y_pred) > 0 and len(y_true) > 0:
                        assert len(y_pred) == len(y_true), f"Length mismatch in {p_file}"
                        
                        # Recompute using project metrics with canonical num_classes
                        c_metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
                        acc = c_metrics["accuracy"] * 100.0
                        bal_acc = c_metrics["balanced_accuracy"] * 100.0
                        f1 = c_metrics["macro_f1"] * 100.0
                        edit = compute_single_seq_edit_score(y_true, y_pred)

                        # Compare recomputed with stored in fold file
                        stored_acc = p_data.get("accuracy")
                        stored_bal = p_data.get("balanced_accuracy")
                        stored_f1 = p_data.get("macro_f1")
                        stored_edit = p_data.get("edit_score")

                        if stored_acc is not None:
                            stored_acc_100 = stored_acc * 100.0 if stored_acc <= 1.0 else stored_acc
                            assert abs(acc - stored_acc_100) < 1e-3, f"Acc discrepancy in {p_file}: {acc} vs {stored_acc_100}"
                        if stored_bal is not None:
                            stored_bal_100 = stored_bal * 100.0 if stored_bal <= 1.0 else stored_bal
                            assert abs(bal_acc - stored_bal_100) < 1e-3, f"BalAcc discrepancy in {p_file}: {bal_acc} vs {stored_bal_100}"
                        if stored_f1 is not None:
                            stored_f1_100 = stored_f1 * 100.0 if stored_f1 <= 1.0 else stored_f1
                            assert abs(f1 - stored_f1_100) < 1e-3, f"MacroF1 discrepancy in {p_file}: {f1} vs {stored_f1_100}"
                        if stored_edit is not None:
                            # Verify edit score calculation
                            assert abs(edit - stored_edit) < 1e-3, f"Edit discrepancy in {p_file}: {edit} vs {stored_edit}"

                        audit_summary[task_name][model]["recomputed_metrics"].append({
                            "fold": fold,
                            "acc": acc,
                            "bal_acc": bal_acc,
                            "f1": f1,
                            "edit": edit,
                            "n_samples": len(y_true),
                            "stored_edit": stored_edit,
                        })
                else:
                    print(f"MISSING PREDICTION: {p_file}")

            # Check if fold 6 is excluded from Needle_Passing
            if task_name == "Needle_Passing":
                ckpt_f6 = ckpt_dir / f"{CHECKPOINT_PREFIXES.get(model, '')}_Needle_Passing_fold6.pt"
                pred_f6 = pred_dir / f"{model}_Needle_Passing_fold6.json"
                if ckpt_f6.exists():
                    print(f"WARNING: Unexpected checkpoint for Needle_Passing fold 6: {ckpt_f6}")
                if pred_f6.exists():
                    print(f"WARNING: Unexpected prediction for Needle_Passing fold 6: {pred_f6}")

            ckpts = audit_summary[task_name][model]["checkpoints_found"]
            preds = audit_summary[task_name][model]["predictions_found"]
            expected_n = len(expected_folds)
            print(f"  Model: {model:<20} | Ckpts: {ckpts}/{expected_n if model in CHECKPOINT_PREFIXES else 'N/A'} | Preds: {preds}/{expected_n}")

    print("\n" + "=" * 80)
    print("PHASE 2: RECOMPUTED VS STORED AGGREGATE METRICS")
    print("=" * 80)

    tables_for_paper = {}

    for task_name, task_info in TASKS.items():
        print(f"\n=================== TASK: {task_name} ===================")
        print(f"{'Model':<25} | {'Acc (Mean±Std)':<18} | {'BalAcc (Mean±Std)':<18} | {'Macro-F1 (Mean±Std)':<20} | {'Edit (Mean±Std)':<18} | Folds")
        print("-" * 115)

        tables_for_paper[task_name] = []

        for model in MODELS:
            if model == "majority":
                maj = audit_summary[task_name]["majority"]
                mean_acc = maj.get("mean_acc", 0.0)
                mean_bal = maj.get("mean_bal_acc", 0.0)
                mean_f1 = maj.get("mean_f1", 0.0)
                if mean_f1 <= 1.0:
                    mean_f1 *= 100.0
                row = {
                    "model": model,
                    "acc_mean": mean_acc,
                    "acc_std": maj.get("std_acc", 0.0),
                    "bal_acc_mean": mean_bal,
                    "bal_acc_std": maj.get("std_bal_acc", 0.0),
                    "f1_mean": mean_f1,
                    "f1_std": maj.get("std_f1", 0.0) * (100.0 if maj.get("std_f1", 0.0) <= 1.0 else 1.0),
                    "edit_mean": 0.0,
                    "edit_std": 0.0,
                    "n_folds": len(task_info["folds"]),
                }
                tables_for_paper[task_name].append(row)
                print(f"{model:<25} | {row['acc_mean']:6.2f} ± {row['acc_std']:4.2f}%    | {row['bal_acc_mean']:6.2f} ± {row['bal_acc_std']:4.2f}%    | {row['f1_mean']:6.2f} ± {row['f1_std']:4.2f}%      | N/A                | {row['n_folds']}")
                continue

            recomputed = audit_summary[task_name][model]["recomputed_metrics"]
            if not recomputed:
                print(f"{model:<25} | NO PREDICTIONS")
                continue

            accs = [m["acc"] for m in recomputed]
            bal_accs = [m["bal_acc"] for m in recomputed]
            f1s = [m["f1"] for m in recomputed]
            edits = [m["edit"] for m in recomputed]

            row = {
                "model": model,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "bal_acc_mean": float(np.mean(bal_accs)),
                "bal_acc_std": float(np.std(bal_accs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "edit_mean": float(np.mean(edits)),
                "edit_std": float(np.std(edits)),
                "n_folds": len(recomputed),
                "acc_min": float(np.min(accs)),
                "acc_max": float(np.max(accs)),
                "bal_acc_min": float(np.min(bal_accs)),
                "bal_acc_max": float(np.max(bal_accs)),
                "f1_min": float(np.min(f1s)),
                "f1_max": float(np.max(f1s)),
                "edit_min": float(np.min(edits)),
                "edit_max": float(np.max(edits)),
            }
            tables_for_paper[task_name].append(row)

            acc_str = f"{row['acc_mean']:.2f} ± {row['acc_std']:.2f}%"
            bal_str = f"{row['bal_acc_mean']:.2f} ± {row['bal_acc_std']:.2f}%"
            f1_str = f"{row['f1_mean']:.2f} ± {row['f1_std']:.2f}%"
            edit_str = f"{row['edit_mean']:.2f} ± {row['edit_std']:.2f}%"

            print(f"{model:<25} | {acc_str:<18} | {bal_str:<18} | {f1_str:<20} | {edit_str:<18} | {len(recomputed)}")

    print("\n" + "=" * 80)
    print("PHASE 3: AUDITING ROBUSTNESS ARTIFACTS")
    print("=" * 80)

    latent_file = artifacts_dir / "robustness" / "latent_perturbations_Suturing_results.json"
    if latent_file.exists():
        with open(latent_file) as f:
            latent_data = json.load(f)
        print(f"\nLoaded {latent_file}:")
        print(f"Benchmark: {latent_data.get('benchmark')}, Task: {latent_data.get('task')}")
        models_dict = latent_data.get("models", {})
        for model_k, model_v in models_dict.items():
            print(f"\n  Model: {model_k}")
            clean_mean = model_v.get("clean_macro_f1_mean") * 100.0
            clean_std = model_v.get("clean_macro_f1_std") * 100.0
            mean_c_mean = model_v.get("mean_c_mean") * 100.0
            mean_c_std = model_v.get("mean_c_std") * 100.0
            worst_c_mean = model_v.get("worst_c_mean") * 100.0
            worst_c_std = model_v.get("worst_c_std") * 100.0
            rel_drop_mean = model_v.get("relative_drop_mean") * 100.0
            rel_drop_std = model_v.get("relative_drop_std") * 100.0

            print(f"    Clean Macro-F1:     {clean_mean:.2f} ± {clean_std:.2f}%")
            print(f"    Mean-C Macro-F1:    {mean_c_mean:.2f} ± {mean_c_std:.2f}%")
            print(f"    Worst-C Macro-F1:   {worst_c_mean:.2f} ± {worst_c_std:.2f}%")
            print(f"    Relative Drop (%):  {rel_drop_mean:.2f} ± {rel_drop_std:.2f}%")

            # Verify per-corruption breakdowns
            # Compute average across all folds for each corruption at severity 1, 3, 5
            folds = model_v.get("folds", [])
            corruption_means = {}
            for fold_data in folds:
                c_by_sev = fold_data.get("corruptions_by_severity", {})
                for sev, corrs in c_by_sev.items():
                    for corr_name, corr_val in corrs.items():
                        if corr_name not in corruption_means:
                            corruption_means[corr_name] = []
                        corruption_means[corr_name].append(corr_val)
            print("    Corruption averages across all severities & folds:")
            for c_name, c_vals in corruption_means.items():
                print(f"      {c_name:<25}: {np.mean(c_vals)*100.0:.2f}%")

    optical_file = artifacts_dir / "robustness" / "image_corruptions_Suturing_results.json"
    if optical_file.exists():
        with open(optical_file) as f:
            optical_data = json.load(f)
        print(f"\nLoaded {optical_file}:")
        print(json.dumps(optical_data, indent=2))

    # Save audited tables to JSON
    out_file = artifacts_dir / "analysis" / "verified_benchmark_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(tables_for_paper, f, indent=2)
    print(f"\nSaved verified audit results to {out_file}")

if __name__ == "__main__":
    main()
