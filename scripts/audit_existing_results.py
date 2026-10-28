"""Audit existing unimodal baselines and SurgMotion VideoGRU results for ITER-28."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import get_louo_splits, FOLD_USERS


def audit_splits_and_metrics():
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    report = {}

    for task in tasks:
        splits = get_louo_splits(task, create_val_split=True)
        sm_path = PROJECT_ROOT / f"artifacts/surgmotion_VideoGRU_{task}_results.json"
        rn_path = PROJECT_ROOT / f"artifacts/video_videogru_{task}_results.json"
        km_path = PROJECT_ROOT / f"artifacts/kinematics_gru_{task}_results.json"

        sm_data = json.load(open(sm_path))
        rn_data = json.load(open(rn_path))
        km_data = json.load(open(km_path))

        sm_folds = {f["fold"]: f for f in sm_data["fold_results"]}

        task_audit = {
            "official_total_folds": len(splits),
            "surgmotion_active_folds": len(sm_folds),
            "resnet_folds": rn_data.get("n_folds_evaluated"),
            "kinematics_folds": km_data.get("n_folds"),
            "fold_user_alignment": True,
            "metrics_recomputed_match": True,
            "fold_discrepancies": [],
        }

        # Check fold-by-fold user assignment
        for fold_num, split_info in splits.items():
            test_user = split_info["test_user"]
            test_trials = split_info["test_trials"]
            if len(test_trials) == 0:
                # Needle passing fold 6 User G has no trials
                if fold_num in sm_folds:
                    task_audit["fold_user_alignment"] = False
                    task_audit["fold_discrepancies"].append(f"Fold {fold_num} (User {test_user}) was evaluated but should be empty!")
                continue

            if fold_num not in sm_folds:
                task_audit["fold_user_alignment"] = False
                task_audit["fold_discrepancies"].append(f"Fold {fold_num} (User {test_user}) missing from SurgMotion results!")
                continue

            sm_fold = sm_folds[fold_num]
            if sm_fold["test_user"] != test_user:
                task_audit["fold_user_alignment"] = False
                task_audit["fold_discrepancies"].append(
                    f"Fold {fold_num} test_user mismatch: expected {test_user}, got {sm_fold['test_user']}"
                )

        # Check aggregate metrics calculation
        accs = [f["accuracy"] for f in sm_folds.values()]
        bal_accs = [f["balanced_accuracy"] for f in sm_folds.values()]
        f1s = [f["macro_f1"] for f in sm_folds.values()]
        edits = [f["edit_score"] for f in sm_folds.values()]

        mean_acc = round(float(np.mean(accs)), 2)
        mean_bal = round(float(np.mean(bal_accs)), 2)
        mean_f1 = round(float(np.mean(f1s)), 2)
        mean_edit = round(float(np.mean(edits)), 2)

        doc_acc = sm_data["metrics"]["accuracy"]["mean"]
        doc_bal = sm_data["metrics"]["balanced_accuracy"]["mean"]
        doc_f1 = sm_data["metrics"]["macro_f1"]["mean"]
        doc_edit = sm_data["metrics"]["edit_score"]["mean"]

        if (mean_acc, mean_bal, mean_f1, mean_edit) != (doc_acc, doc_bal, doc_f1, doc_edit):
            task_audit["metrics_recomputed_match"] = False
            task_audit["discrepancy"] = {
                "computed": (mean_acc, mean_bal, mean_f1, mean_edit),
                "documented": (doc_acc, doc_bal, doc_f1, doc_edit),
            }

        report[task] = task_audit

    return report


if __name__ == "__main__":
    rep = audit_splits_and_metrics()
    print(json.dumps(rep, indent=2))
