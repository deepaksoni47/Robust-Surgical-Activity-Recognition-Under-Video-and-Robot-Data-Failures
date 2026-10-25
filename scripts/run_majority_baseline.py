import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.dataset import get_fold_datasets
from src.data.splits import get_louo_splits
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score
from src.models.majority import MajorityClassClassifier
from src.utils.reproducibility import save_json


TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]


def run_majority_benchmark_for_task(
    task: str,
    window_size: int = 16,
    stride: int = 8,
    dataset_dir: str = "dataset",
) -> Dict[str, Any]:
    """Run 8-fold Leave-One-User-Out majority baseline evaluation for one task."""
    splits = get_louo_splits(task, create_val_split=False)
    fold_results: List[Dict[str, Any]] = []

    print(f"\n==========================================")
    print(f"Running Majority Baseline on Task: {task}")
    print(f"==========================================")

    for fold_num in sorted(splits.keys()):
        fold_data = get_fold_datasets(
            task=task,
            fold_num=fold_num,
            window_size=window_size,
            train_stride=stride,
            eval_stride=stride,
            dataset_dir=dataset_dir,
            create_val_split=False,
        )

        train_ds = fold_data["train"]
        test_ds = fold_data["test"]
        vocab = fold_data["vocab"]
        inv_vocab = fold_data["inv_vocab"]
        fold_info = fold_data["fold_info"]

        # 1. Collect training and test targets
        train_targets = [train_ds[i]["target"].item() for i in range(len(train_ds))]
        test_targets = [test_ds[i]["target"].item() for i in range(len(test_ds))]

        if len(test_targets) == 0:
            print(f"Fold {fold_num} (User {fold_info['test_user']}): Skipped (0 test trials in official benchmark split)")
            continue

        # 2. Fit majority model strictly on train targets
        clf = MajorityClassClassifier(num_classes=len(vocab))
        clf.fit(train_targets)

        # 3. Predict on test targets
        preds = clf.predict(len(test_targets))

        # 4. Compute framewise classification metrics
        metrics = compute_classification_metrics(
            y_true=test_targets,
            y_pred=preds,
            num_classes=len(vocab),
            class_names=inv_vocab,
        )

        # 5. Compute Segmental Edit Score per trial
        true_trial_seqs: List[List[int]] = []
        pred_trial_seqs: List[List[int]] = []

        for record in test_ds.trial_records:
            labels_seq = record["labels"].tolist()
            true_trial_seqs.append(labels_seq)
            pred_trial_seqs.append([clf.majority_class_] * len(labels_seq))

        edit_score = compute_edit_score(true_trial_seqs, pred_trial_seqs)

        fold_record = {
            "fold": fold_num,
            "test_user": fold_info["test_user"],
            "majority_class_id": clf.majority_class_,
            "majority_class_name": inv_vocab.get(clf.majority_class_, str(clf.majority_class_)),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "edit_score": edit_score,
            "test_samples": len(test_targets),
        }
        fold_results.append(fold_record)

        print(
            f"Fold {fold_num} (User {fold_info['test_user']}): "
            f"Maj={fold_record['majority_class_name']}, "
            f"Acc={metrics['accuracy']*100:.2f}%, "
            f"BalAcc={metrics['balanced_accuracy']*100:.2f}%, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"Edit={edit_score:.2f}%"
        )

    # Aggregate across 8 folds
    accs = [f["accuracy"] * 100 for f in fold_results]
    bal_accs = [f["balanced_accuracy"] * 100 for f in fold_results]
    macro_f1s = [f["macro_f1"] for f in fold_results]
    edits = [f["edit_score"] for f in fold_results]

    summary = {
        "task": task,
        "n_folds": len(fold_results),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "mean_edit_score": float(np.mean(edits)),
        "std_edit_score": float(np.std(edits)),
        "folds": fold_results,
    }

    print(f"\n---> {task} 8-Fold LOUO Summary:")
    print(f"Accuracy:          {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
    print(f"Balanced Accuracy: {summary['mean_balanced_accuracy']:.2f}% ± {summary['std_balanced_accuracy']:.2f}%")
    print(f"Macro-F1:          {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f}")
    print(f"Edit Score:        {summary['mean_edit_score']:.2f}% ± {summary['std_edit_score']:.2f}%")

    return summary


def run_all_tasks_majority_benchmark(
    window_size: int = 16,
    stride: int = 8,
    dataset_dir: str = "dataset",
    output_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Execute majority baseline across Suturing, Knot Tying, and Needle Passing."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}
    for task in TASKS:
        res = run_majority_benchmark_for_task(
            task=task,
            window_size=window_size,
            stride=stride,
            dataset_dir=dataset_dir,
        )
        all_results[task] = res

    # Save full JSON results
    json_path = out_dir / "majority_baseline_results.json"
    save_json(all_results, json_path)
    print(f"\nSaved full results to {json_path}")

    # Generate Markdown Summary
    md_lines = [
        "# JIGSAWS Majority-Class Trivial Baseline Benchmark",
        "",
        "Protocol: Official 8-fold Leave-One-User-Out (LOUO) cross-validation.",
        "Zero-leakage guarantee: Majority class fitted strictly on training trials of each fold.",
        "",
        "| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 | Edit Score (%) |",
        "|---|---:|---:|---:|---:|",
    ]

    for task in TASKS:
        r = all_results[task]
        md_lines.append(
            f"| **{task}** | {r['mean_accuracy']:.2f} ± {r['std_accuracy']:.2f} | "
            f"{r['mean_balanced_accuracy']:.2f} ± {r['std_balanced_accuracy']:.2f} | "
            f"{r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f} | "
            f"{r['mean_edit_score']:.2f} ± {r['std_edit_score']:.2f} |"
        )

    md_path = out_dir / "majority_baseline_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved summary table to {md_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8-fold LOUO Majority Baseline on JIGSAWS")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    run_all_tasks_majority_benchmark(
        window_size=args.window_size,
        stride=args.stride,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
