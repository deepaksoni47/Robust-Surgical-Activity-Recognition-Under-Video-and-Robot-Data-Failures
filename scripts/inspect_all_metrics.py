"""scripts/inspect_all_metrics.py
Inspect and summarize all baseline and fusion results across all 3 tasks.
"""
import json
from pathlib import Path
import numpy as np

tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]

print("================================================================================")
print("1. ROBUSTNESS RESULTS IN AUDITED ARTIFACTS")
print("================================================================================")
for task in tasks:
    print(f"\n--- Task: {task} ---")
    for ftype in ["late", "concat", "adaptive"]:
        fname = f"artifacts/surgmotion_fusion_{ftype}_{task}_results.json"
        p = Path(fname)
        if not p.is_file():
            print(f"Missing: {fname}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        rob = d.get("robustness", {})
        if not rob:
            print(f"  [{ftype.upper()}] No robustness evaluated (fixed late fusion without dynamic gating)")
        else:
            print(f"  [{ftype.upper()}]:")
            for cond, m in rob.items():
                print(f"    {cond:22s} -> Acc: {m['accuracy']:6.2f}%, F1: {m['macro_f1']:6.2f}%, Edit: {m['edit_score']:6.2f}%")

print("\n================================================================================")
print("2. FOLD-LEVEL METRICS SUMMARY ACROSS ALL MODELS")
print("================================================================================")
models = [
    ("Majority Baseline", "majority_baseline_results.json"),
    ("ResNet-18 VideoGRU", "video_videogru_{task}_results.json"),
    ("SurgMotion VideoMLP", "surgmotion_VideoMLP_{task}_results.json"),
    ("SurgMotion VideoGRU", "surgmotion_VideoGRU_{task}_results.json"),
    ("Kinematics GRU", "kinematics_gru_{task}_results.json"),
    ("Late Fusion (SurgMotion+Kin)", "surgmotion_fusion_late_{task}_results.json"),
    ("Concat Fusion (SurgMotion+Kin)", "surgmotion_fusion_concat_{task}_results.json"),
    ("Adaptive Fusion (SurgMotion+Kin)", "surgmotion_fusion_adaptive_{task}_results.json"),
]

for task in tasks:
    print(f"\n==================== TASK: {task.upper()} ====================")
    print(f"{'Model':<35} | {'Accuracy (%)':<16} | {'Bal Acc (%)':<16} | {'Macro-F1 (%)':<16} | {'Edit Score (%)':<16}")
    print("-" * 105)
    for model_name, pattern in models:
        fname = "artifacts/" + pattern.format(task=task)
        p = Path(fname)
        if not p.is_file():
            print(f"{model_name:<35} | FILE NOT FOUND: {fname}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        if "fold_results" in d:
            folds = d["fold_results"]
            accs = [f["accuracy"] for f in folds]
            bal_accs = [f["balanced_accuracy"] for f in folds]
            f1s = [f["macro_f1"] for f in folds]
            edits = [f["edit_score"] for f in folds]
        elif "folds" in d:
            folds = d["folds"]
            accs = [f["accuracy"] * (100.0 if f["accuracy"] <= 1.0 else 1.0) for f in folds]
            bal_accs = [f.get("balanced_accuracy", 0.0) * (100.0 if f.get("balanced_accuracy", 0.0) <= 1.0 else 1.0) for f in folds]
            f1s = [f["macro_f1"] * (100.0 if f["macro_f1"] <= 1.0 else 1.0) for f in folds]
            edits = [f.get("edit_score", 0.0) for f in folds]
        elif task in d and "folds" in d[task]:
            folds = d[task]["folds"]
            accs = [f["accuracy"] * (100.0 if f["accuracy"] <= 1.0 else 1.0) for f in folds]
            bal_accs = [f.get("balanced_accuracy", 0.0) * (100.0 if f.get("balanced_accuracy", 0.0) <= 1.0 else 1.0) for f in folds]
            f1s = [f["macro_f1"] * (100.0 if f["macro_f1"] <= 1.0 else 1.0) for f in folds]
            edits = [f.get("edit_score", 0.0) for f in folds]
        else:
            print(f"Unknown format for {fname}")
            continue

        acc_str = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}"
        bal_str = f"{np.mean(bal_accs):.2f} ± {np.std(bal_accs):.2f}"
        f1_str = f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}"
        ed_str = f"{np.mean(edits):.2f} ± {np.std(edits):.2f}"
        print(f"{model_name:<35} | {acc_str:<16} | {bal_str:<16} | {f1_str:<16} | {ed_str:<16}")
