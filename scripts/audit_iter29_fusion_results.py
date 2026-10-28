"""scripts/audit_iter29_fusion_results.py

Comprehensive Audit of ITER-28 Multimodal Fusion Results across:
1. Late Fusion (Suturing, Knot Tying, Needle Passing)
2. Feature Concatenation Fusion (Suturing, Knot Tying, Needle Passing)
3. Adaptive Reliability Fusion (Suturing, Knot Tying, Needle Passing)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import get_louo_splits


def audit_iter29_artifacts() -> Dict[str, Any]:
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    fusion_types = ["late", "concat", "adaptive"]
    expected_folds = {
        "Suturing": 8,
        "Knot_Tying": 8,
        "Needle_Passing": 7,
    }

    audit_summary: Dict[str, Any] = {
        "files_audited": {},
        "integrity_checks": {
            "all_files_exist": True,
            "all_valid_json": True,
            "all_folds_correct": True,
            "all_numbers_finite": True,
            "metrics_math_verified": True,
        },
        "discrepancies": [],
    }

    for f_type in fusion_types:
        for task in tasks:
            key = f"{f_type}_{task}"
            fname = f"artifacts/surgmotion_fusion_{f_type}_{task}_results.json"
            fpath = PROJECT_ROOT / fname

            file_info: Dict[str, Any] = {
                "path": fname,
                "exists": fpath.is_file(),
                "valid_json": False,
                "active_folds": 0,
                "expected_folds": expected_folds[task],
                "has_robustness": False,
                "schema_keys": [],
                "metrics": {},
                "fold_users": [],
                "math_match": True,
            }

            if not fpath.is_file():
                audit_summary["integrity_checks"]["all_files_exist"] = False
                audit_summary["discrepancies"].append(f"Missing file: {fname}")
                audit_summary["files_audited"][key] = file_info
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_info["valid_json"] = True
            except Exception as e:
                audit_summary["integrity_checks"]["all_valid_json"] = False
                audit_summary["discrepancies"].append(f"Invalid JSON in {fname}: {e}")
                audit_summary["files_audited"][key] = file_info
                continue

            file_info["schema_keys"] = list(data.keys())
            file_info["active_folds"] = data.get("active_folds", 0)
            file_info["has_robustness"] = "robustness" in data
            file_info["metrics"] = data.get("metrics", {})

            # Check fold count
            if file_info["active_folds"] != expected_folds[task]:
                audit_summary["integrity_checks"]["all_folds_correct"] = False
                audit_summary["discrepancies"].append(
                    f"{fname}: active_folds={file_info['active_folds']} != expected {expected_folds[task]}"
                )

            # Check fold results
            fold_results = data.get("fold_results", [])
            file_info["fold_users"] = [f.get("test_user") for f in fold_results]

            # Verify math
            accs = [f["accuracy"] for f in fold_results]
            bal_accs = [f["balanced_accuracy"] for f in fold_results]
            f1s = [f["macro_f1"] for f in fold_results]
            edits = [f["edit_score"] for f in fold_results]

            for vals, name in [(accs, "accuracy"), (bal_accs, "balanced_accuracy"), (f1s, "macro_f1"), (edits, "edit_score")]:
                if not np.all(np.isfinite(vals)):
                    audit_summary["integrity_checks"]["all_numbers_finite"] = False
                    audit_summary["discrepancies"].append(f"{fname}: Non-finite values in {name}")

                mean_val = round(float(np.mean(vals)), 2)
                std_val = round(float(np.std(vals)), 2)

                doc_mean = data.get("metrics", {}).get(name, {}).get("mean")
                doc_std = data.get("metrics", {}).get(name, {}).get("std")

                if doc_mean != mean_val or doc_std != std_val:
                    file_info["math_match"] = False
                    audit_summary["integrity_checks"]["metrics_math_verified"] = False
                    audit_summary["discrepancies"].append(
                        f"{fname} {name} mismatch: calculated mean={mean_val}, std={std_val} vs documented mean={doc_mean}, std={doc_std}"
                    )

            audit_summary["files_audited"][key] = file_info

    return audit_summary


if __name__ == "__main__":
    report = audit_iter29_artifacts()
    print(json.dumps(report, indent=2))
