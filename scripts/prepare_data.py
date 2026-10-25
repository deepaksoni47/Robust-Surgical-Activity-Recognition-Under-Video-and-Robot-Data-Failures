import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.dataset import get_fold_dataloaders
from src.utils.reproducibility import load_config, save_json, set_seed


def prepare_sanity_subset(
    task: str = "Suturing",
    fold_num: int = 1,
    output_dir: str = "artifacts/sanity_subset",
    batch_size: int = 16,
    window_size: int = 16,
    stride: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full preprocessing pipeline on a task fold and cache diagnostic subset data.
    
    Verifies that raw files transform into clean model-ready batches with zero NaNs.
    """
    set_seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Preparing sanity subset for task='{task}', fold={fold_num}...")
    loader_dict = get_fold_dataloaders(
        task=task,
        fold_num=fold_num,
        batch_size=batch_size,
        window_size=window_size,
        train_stride=stride,
        eval_stride=stride,
        num_workers=0,
        dataset_dir="dataset",
    )

    train_loader = loader_dict["train_loader"]
    test_loader = loader_dict["test_loader"]
    normalizer = loader_dict["normalizer"]
    vocab = loader_dict["vocab"]
    fold_info = loader_dict["fold_info"]

    # Extract 1 sample batch from train and test
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    # Assert finite values
    assert torch.all(torch.isfinite(train_batch["kinematics"])), "Non-finite values in train batch!"
    assert torch.all(torch.isfinite(test_batch["kinematics"])), "Non-finite values in test batch!"

    # Save diagnostic summary
    summary = {
        "task": task,
        "fold_num": fold_num,
        "test_user": fold_info["test_user"],
        "num_classes": len(vocab),
        "vocabulary": vocab,
        "train_trials": fold_info["train_trials"],
        "test_trials": fold_info["test_trials"],
        "num_train_windows": len(loader_dict["datasets"]["train"]),
        "num_test_windows": len(loader_dict["datasets"]["test"]),
        "normalizer_mean_norm": float(np.linalg.norm(normalizer.mean)),
        "normalizer_std_norm": float(np.linalg.norm(normalizer.std)),
        "batch_shapes": {
            "train_kinematics": list(train_batch["kinematics"].shape),
            "train_labels": list(train_batch["labels"].shape),
            "train_target": list(train_batch["target"].shape),
            "test_kinematics": list(test_batch["kinematics"].shape),
            "test_labels": list(test_batch["labels"].shape),
            "test_target": list(test_batch["target"].shape),
        },
        "sample_trials": {
            "train_sample": train_batch["trial_id"][:4],
            "test_sample": test_batch["trial_id"][:4],
        },
    }

    summary_file = out_path / f"{task}_fold{fold_num}_summary.json"
    save_json(summary, summary_file)
    print(f"Saved diagnostic summary to {summary_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="JIGSAWS Data Pipeline Sanity Check")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to config file")
    parser.add_argument("--task", type=str, default="Suturing", help="Task name or 'all'")
    parser.add_argument("--fold", type=int, default=1, help="LOUO fold number (1..8)")
    parser.add_argument("--output-dir", type=str, default="artifacts/sanity_subset", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    window_cfg = cfg.get("windowing", {})
    window_size = window_cfg.get("window_size", 16)
    stride = window_cfg.get("stride", 8)
    seed = cfg.get("dataset", {}).get("seed", 42)

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]

    for t in tasks:
        prepare_sanity_subset(
            task=t,
            fold_num=args.fold,
            output_dir=args.output_dir,
            window_size=window_size,
            stride=stride,
            seed=seed,
        )


if __name__ == "__main__":
    main()
