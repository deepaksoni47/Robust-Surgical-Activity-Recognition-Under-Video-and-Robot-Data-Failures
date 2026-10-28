"""scripts/build_iter29_unified_artifacts.py
Constructs machine-readable consolidated comparison and robustness comparison JSONs for ITER-29.
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]

# Models to include
models_cfg = [
    {
        "id": "majority_baseline",
        "name": "Majority Baseline",
        "type": "trivial",
        "modality": "none",
        "file_template": "artifacts/majority_baseline_results.json",
    },
    {
        "id": "resnet18_videogru",
        "name": "ResNet-18 VideoGRU",
        "type": "unimodal_video",
        "modality": "video",
        "file_template": "artifacts/video_videogru_{task}_results.json",
    },
    {
        "id": "surgmotion_videomlp",
        "name": "SurgMotion VideoMLP (W=1)",
        "type": "unimodal_video",
        "modality": "video",
        "file_template": "artifacts/surgmotion_VideoMLP_{task}_results.json",
    },
    {
        "id": "surgmotion_videogru",
        "name": "SurgMotion VideoGRU (W=8)",
        "type": "unimodal_video",
        "modality": "video",
        "file_template": "artifacts/surgmotion_VideoGRU_{task}_results.json",
    },
    {
        "id": "kinematics_gru",
        "name": "Kinematics GRU",
        "type": "unimodal_kinematics",
        "modality": "kinematics",
        "file_template": "artifacts/kinematics_gru_{task}_results.json",
    },
    {
        "id": "surgmotion_fusion_late",
        "name": "Fixed Late Fusion (SurgMotion + Kin)",
        "type": "multimodal_fusion",
        "modality": "multimodal",
        "file_template": "artifacts/surgmotion_fusion_late_{task}_results.json",
    },
    {
        "id": "surgmotion_fusion_concat",
        "name": "Feature Concatenation Fusion (SurgMotion + Kin)",
        "type": "multimodal_fusion",
        "modality": "multimodal",
        "file_template": "artifacts/surgmotion_fusion_concat_{task}_results.json",
    },
    {
        "id": "surgmotion_fusion_adaptive",
        "name": "Adaptive Reliability Fusion (SurgMotion + Kin)",
        "type": "multimodal_fusion",
        "modality": "multimodal",
        "file_template": "artifacts/surgmotion_fusion_adaptive_{task}_results.json",
    },
]


def extract_model_task_data(task: str, cfg: dict) -> dict:
    fname = cfg["file_template"].format(task=task)
    fpath = PROJECT_ROOT / fname
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine folds
    if "fold_results" in data:
        raw_folds = data["fold_results"]
    elif "folds" in data:
        raw_folds = data["folds"]
    elif task in data and "folds" in data[task]:
        raw_folds = data[task]["folds"]
    else:
        raise ValueError(f"Cannot parse folds from {fname}")

    # Standardize fold metrics to [0, 100]% scale based on model file format
    standardized_folds = []
    for f in raw_folds:
        if cfg["id"] == "majority_baseline":
            acc = float(f["accuracy"]) * 100.0
            bal_acc = float(f.get("balanced_accuracy", 0.0)) * 100.0
            f1 = float(f["macro_f1"]) * 100.0
            edit = float(f.get("edit_score", 0.0))
        elif cfg["id"] in ("resnet18_videogru", "kinematics_gru"):
            # Older PyTorch runners saved acc, bal, and f1 as fractions in [0, 1]
            acc = float(f["accuracy"]) * 100.0
            bal_acc = float(f.get("balanced_accuracy", 0.0)) * 100.0
            f1 = float(f["macro_f1"]) * 100.0
            edit = float(f.get("edit_score", 0.0))
        else:
            # SurgMotion baselines and fusion models are already formatted as percentages [0, 100]
            acc = float(f["accuracy"])
            bal_acc = float(f.get("balanced_accuracy", 0.0))
            f1 = float(f["macro_f1"])
            edit = float(f.get("edit_score", 0.0))

        user = f.get("test_user") or f.get("user")
        fold_num = f.get("fold", len(standardized_folds) + 1)

        standardized_folds.append({
            "fold": fold_num,
            "test_user": user,
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "macro_f1": round(f1, 2),
            "edit_score": round(edit, 2),
        })

    acc_vals = [f["accuracy"] for f in standardized_folds]
    bal_vals = [f["balanced_accuracy"] for f in standardized_folds]
    f1_vals = [f["macro_f1"] for f in standardized_folds]
    edit_vals = [f["edit_score"] for f in standardized_folds]

    summary_metrics = {
        "accuracy": {
            "mean": round(float(np.mean(acc_vals)), 2),
            "std": round(float(np.std(acc_vals)), 2),
            "min": round(float(np.min(acc_vals)), 2),
            "max": round(float(np.max(acc_vals)), 2),
        },
        "balanced_accuracy": {
            "mean": round(float(np.mean(bal_vals)), 2),
            "std": round(float(np.std(bal_vals)), 2),
            "min": round(float(np.min(bal_vals)), 2),
            "max": round(float(np.max(bal_vals)), 2),
        },
        "macro_f1": {
            "mean": round(float(np.mean(f1_vals)), 2),
            "std": round(float(np.std(f1_vals)), 2),
            "min": round(float(np.min(f1_vals)), 2),
            "max": round(float(np.max(f1_vals)), 2),
        },
        "edit_score": {
            "mean": round(float(np.mean(edit_vals)), 2),
            "std": round(float(np.std(edit_vals)), 2),
            "min": round(float(np.min(edit_vals)), 2),
            "max": round(float(np.max(edit_vals)), 2),
        },
    }

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "type": cfg["type"],
        "modality": cfg["modality"],
        "active_folds": len(standardized_folds),
        "source_file": fname,
        "metrics": summary_metrics,
        "fold_results": standardized_folds,
    }


def build_unified_comparison():
    unified = {
        "metadata": {
            "description": "Consolidated benchmark comparison across 8 surgical gesture recognition models on JIGSAWS official LOUO splits.",
            "iteration": "ITER-29",
            "benchmark": "JIGSAWS",
            "evaluation_protocol": "Leave-One-User-Out (LOUO) Cross-Validation",
            "units": "percentage (0.0 to 100.0) across all metrics",
            "tasks": tasks,
            "models_evaluated": [cfg["name"] for cfg in models_cfg],
        },
        "tasks": {},
    }

    for task in tasks:
        task_models = {}
        for cfg in models_cfg:
            task_models[cfg["id"]] = extract_model_task_data(task, cfg)

        # Identify strongest unimodal baseline for each metric
        unimodal_models = [
            m for m in task_models.values()
            if m["type"] in ("unimodal_video", "unimodal_kinematics")
        ]

        strongest_unimodal = {}
        for m_key in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]:
            best_m = max(unimodal_models, key=lambda m: m["metrics"][m_key]["mean"])
            strongest_unimodal[m_key] = {
                "model_id": best_m["id"],
                "model_name": best_m["name"],
                "mean": best_m["metrics"][m_key]["mean"],
                "std": best_m["metrics"][m_key]["std"],
            }

        # Calculate absolute difference for each fusion model against strongest unimodal
        comparisons = {}
        for f_id in ["surgmotion_fusion_late", "surgmotion_fusion_concat", "surgmotion_fusion_adaptive"]:
            f_model = task_models[f_id]
            comp_metrics = {}
            for m_key in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]:
                f_val = f_model["metrics"][m_key]["mean"]
                base_val = strongest_unimodal[m_key]["mean"]
                delta = round(float(f_val - base_val), 2)
                pct_delta = round(float((f_val - base_val) / base_val * 100.0), 2) if base_val > 0 else 0.0
                comp_metrics[m_key] = {
                    "fusion_mean": f_val,
                    "baseline_mean": base_val,
                    "strongest_baseline_name": strongest_unimodal[m_key]["model_name"],
                    "absolute_delta": delta,
                    "relative_percent_delta": pct_delta,
                    "beats_strongest_unimodal": bool(delta > 0),
                }
            comparisons[f_id] = comp_metrics

        unified["tasks"][task] = {
            "models": task_models,
            "strongest_unimodal_baselines": strongest_unimodal,
            "fusion_comparisons": comparisons,
        }

    out_path = PROJECT_ROOT / "artifacts/iter29_unified_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2)
    print(f"Saved unified comparison to {out_path}")


def build_robustness_comparison():
    robustness_dict = {
        "metadata": {
            "description": "Comprehensive robustness stress testing audit for SurgMotion Concat and Adaptive fusion across 5 conditions.",
            "iteration": "ITER-29",
            "tasks": tasks,
            "conditions": [
                "clean",
                "video_corrupted",
                "kinematics_corrupted",
                "video_missing",
                "kinematics_missing",
            ],
            "condition_descriptions": {
                "clean": "Both modalities fully available and unperturbed (availability a = [1, 1]).",
                "video_corrupted": "Optical telemetry perturbed with Gaussian noise (std=1.0) applied to 1024D SurgMotion features.",
                "kinematics_corrupted": "Robotic telemetry perturbed with Gaussian noise (std=1.0) applied to 76D kinematics.",
                "video_missing": "Optical telemetry completely lost (availability a = [0, 1], video features zeroed).",
                "kinematics_missing": "Robotic telemetry completely lost (availability a = [1, 0], kinematics zeroed).",
            },
        },
        "tasks": {},
    }

    for task in tasks:
        robustness_dict["tasks"][task] = {}
        for ftype in ["concat", "adaptive"]:
            src_file = f"artifacts/surgmotion_fusion_{ftype}_{task}_results.json"
            with open(PROJECT_ROOT / src_file, "r", encoding="utf-8") as f:
                d = json.load(f)

            raw_rob = d.get("robustness", {})
            clean_acc = raw_rob.get("clean", {}).get("accuracy", 0.0)
            clean_f1 = raw_rob.get("clean", {}).get("macro_f1", 0.0)
            clean_edit = raw_rob.get("clean", {}).get("edit_score", 0.0)

            cond_data = {}
            for cond, vals in raw_rob.items():
                acc = vals["accuracy"]
                f1 = vals["macro_f1"]
                edit = vals["edit_score"]

                delta_acc = round(float(acc - clean_acc), 2)
                delta_f1 = round(float(f1 - clean_f1), 2)
                delta_edit = round(float(edit - clean_edit), 2)

                pct_acc = round(float(delta_acc / clean_acc * 100.0), 2) if clean_acc > 0 else 0.0

                cond_data[cond] = {
                    "accuracy": acc,
                    "macro_f1": f1,
                    "edit_score": edit,
                    "degradation_vs_clean": {
                        "accuracy_delta": delta_acc,
                        "macro_f1_delta": delta_f1,
                        "edit_score_delta": delta_edit,
                        "accuracy_relative_drop_pct": pct_acc,
                    },
                }

            robustness_dict["tasks"][task][ftype] = {
                "model_name": f"SurgMotion + Kinematics {ftype.capitalize()} Fusion",
                "source_file": src_file,
                "conditions": cond_data,
            }

    out_path = PROJECT_ROOT / "artifacts/iter29_robustness_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(robustness_dict, f, indent=2)
    print(f"Saved robustness comparison to {out_path}")


if __name__ == "__main__":
    build_unified_comparison()
    build_robustness_comparison()
