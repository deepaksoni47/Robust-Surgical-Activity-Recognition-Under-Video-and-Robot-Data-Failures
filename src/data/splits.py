import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.utils.reproducibility import save_json

FOLD_USERS = {
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
}


def parse_split_file(filepath: str | Path) -> List[str]:
    """Parse JIGSAWS split file (Train.txt or Test.txt) and return unique trial IDs.
    
    Format per line:
    Suturing_B001_000080_000219.txt \t G1
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path.resolve()}")

    trial_ids: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            # Strip trailing .txt if present
            if token.endswith(".txt"):
                token = token[:-4]

            # Match Task_SubjectTrialNum pattern, e.g. Suturing_B001, Knot_Tying_C002, Needle_Passing_D001
            match = re.match(r"^(.+_[A-I]\d+)", token)
            if match:
                trial_ids.add(match.group(1))
            else:
                raise ValueError(f"Could not extract trial_id from token: '{token}' in {path.name}")

    return sorted(list(trial_ids))


def assert_no_leakage(
    train_trials: List[str],
    test_trials: List[str],
    val_trials: Optional[List[str]] = None,
) -> None:
    """Strictly assert zero trial and user leakage between train, val, and test splits.
    
    Raises:
        AssertionError: If any trial or user is shared across train/val and test.
    """
    train_set = set(train_trials)
    test_set = set(test_trials)
    val_set = set(val_trials) if val_trials is not None else set()

    # 1. Trial-level disjointness
    train_test_overlap = train_set & test_set
    if train_test_overlap:
        raise AssertionError(f"Leakage detected! Shared trials between train and test: {train_test_overlap}")

    if val_set:
        val_test_overlap = val_set & test_set
        if val_test_overlap:
            raise AssertionError(f"Leakage detected! Shared trials between val and test: {val_test_overlap}")

        train_val_overlap = train_set & val_set
        if train_val_overlap:
            raise AssertionError(f"Leakage detected! Shared trials between train and val: {train_val_overlap}")

    # 2. User-level disjointness (Leave-One-User-Out)
    def extract_user(trial: str) -> str:
        match = re.search(r"_([A-I])\d+", trial)
        return match.group(1) if match else ""

    test_users = {extract_user(t) for t in test_set}
    train_users = {extract_user(t) for t in train_set}

    user_overlap = train_users & test_users
    if user_overlap:
        raise AssertionError(f"User leakage detected! Test user {user_overlap} appears in training set.")

    if val_set:
        val_users = {extract_user(t) for t in val_set}
        val_user_overlap = val_users & test_users
        if val_user_overlap:
            raise AssertionError(f"User leakage detected! Test user {val_user_overlap} appears in validation set.")


def get_louo_splits(
    task: str,
    splits_root: str | Path = "dataset/Experimental_setup",
    create_val_split: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Load official 8-fold Leave-One-User-Out (LOUO) splits for a specific task.
    
    Args:
        task: Name of task ('Suturing', 'Knot_Tying', 'Needle_Passing').
        splits_root: Path to Experimental_setup directory.
        create_val_split: If True, carves out 1 user from the training pool as validation.
        
    Returns:
        Dict mapping fold_num (1..8) to split dict with train_trials, val_trials, test_trials.
    """
    root = Path(splits_root) / task / "unBalanced" / "GestureClassification" / "UserOut"
    if not root.is_dir():
        raise FileNotFoundError(f"LOUO split directory not found for task '{task}': {root.resolve()}")

    folds: Dict[int, Dict[str, Any]] = {}

    for fold_num in range(1, 9):
        fold_dir = root / f"{fold_num}_Out" / "itr_1"
        train_file = fold_dir / "Train.txt"
        test_file = fold_dir / "Test.txt"

        raw_train_trials = parse_split_file(train_file)
        test_trials = parse_split_file(test_file)
        test_user = FOLD_USERS[fold_num]

        if create_val_split:
            # Deterministically designate the next user in rotation as validation user
            # e.g., for test user B (fold 1), val user is C; for test user I (fold 8), val user is B.
            val_fold = (fold_num % 8) + 1
            val_user = FOLD_USERS[val_fold]

            val_trials = [t for t in raw_train_trials if f"_{val_user}" in t]
            train_trials = [t for t in raw_train_trials if f"_{val_user}" not in t]
        else:
            val_trials = []
            train_trials = raw_train_trials
            val_user = None

        # Assert zero leakage
        assert_no_leakage(train_trials, test_trials, val_trials)

        folds[fold_num] = {
            "fold": fold_num,
            "task": task,
            "test_user": test_user,
            "val_user": val_user,
            "train_trials": train_trials,
            "val_trials": val_trials,
            "test_trials": test_trials,
            "n_train": len(train_trials),
            "n_val": len(val_trials),
            "n_test": len(test_trials),
            "total_trials": len(train_trials) + len(val_trials) + len(test_trials),
        }

    return folds


def export_louo_splits(
    tasks: Optional[List[str]] = None,
    output_dir: str | Path = "configs/splits",
    splits_root: str | Path = "dataset/Experimental_setup",
) -> Dict[str, Path]:
    """Export all 8 LOUO folds for each task as tidy CSV and JSON files for reproducibility."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]

    exported_files = {}

    for task in tasks:
        folds = get_louo_splits(task, splits_root=splits_root, create_val_split=True)

        rows = []
        for fold_num, f_data in folds.items():
            for t in f_data["train_trials"]:
                rows.append({"fold": fold_num, "task": task, "trial_id": t, "split": "train", "user_id": t.split("_")[1][0]})
            for t in f_data["val_trials"]:
                rows.append({"fold": fold_num, "task": task, "trial_id": t, "split": "val", "user_id": t.split("_")[1][0]})
            for t in f_data["test_trials"]:
                rows.append({"fold": fold_num, "task": task, "trial_id": t, "split": "test", "user_id": t.split("_")[1][0]})

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"{task}_louo_splits.csv"
        df.to_csv(csv_path, index=False)
        exported_files[task] = csv_path

    # Export combined summary JSON
    summary_path = out_dir / "louo_splits_summary.json"
    save_json({task: get_louo_splits(task, splits_root=splits_root, create_val_split=True) for task in tasks}, summary_path)
    exported_files["summary"] = summary_path

    return exported_files
