from pathlib import Path
import pytest
import pandas as pd

from src.data.splits import (
    FOLD_USERS,
    assert_no_leakage,
    export_louo_splits,
    get_louo_splits,
    parse_split_file,
)


def test_parse_split_file():
    """Verify split file parsing extracts unique trial identifiers."""
    split_file = Path(
        "dataset/Experimental_setup/Suturing/unBalanced/GestureClassification/UserOut/1_Out/itr_1/Test.txt"
    )
    trials = parse_split_file(split_file)
    assert len(trials) == 5
    assert trials == [
        "Suturing_B001",
        "Suturing_B002",
        "Suturing_B003",
        "Suturing_B004",
        "Suturing_B005",
    ]


def test_louo_splits_suturing():
    """Verify all 8 folds for Suturing are disjoint and total 39 trials each."""
    folds = get_louo_splits("Suturing")
    assert len(folds) == 8

    for fold_num in range(1, 9):
        f = folds[fold_num]
        assert f["fold"] == fold_num
        assert f["test_user"] == FOLD_USERS[fold_num]
        assert f["total_trials"] == 39
        assert len(f["test_trials"]) > 0
        assert len(f["train_trials"]) > 0
        assert len(f["val_trials"]) > 0

        # Assert no test trial leaks into train or val
        assert_no_leakage(f["train_trials"], f["test_trials"], f["val_trials"])


def test_louo_splits_knot_tying():
    """Verify all 8 folds for Knot Tying are disjoint and total 36 trials each."""
    folds = get_louo_splits("Knot_Tying")
    assert len(folds) == 8

    for fold_num in range(1, 9):
        f = folds[fold_num]
        assert f["total_trials"] == 36
        assert_no_leakage(f["train_trials"], f["test_trials"], f["val_trials"])


def test_louo_splits_needle_passing():
    """Verify all 8 folds for Needle Passing are disjoint and total 28 trials each."""
    folds = get_louo_splits("Needle_Passing")
    assert len(folds) == 8

    for fold_num in range(1, 9):
        f = folds[fold_num]
        # In official JIGSAWS LOUO benchmark, Needle_Passing contains 27 trials (Needle_Passing_H002 is excluded)
        assert f["total_trials"] == 27
        assert_no_leakage(f["train_trials"], f["test_trials"], f["val_trials"])


def test_leakage_assertion_catches_trial_overlap():
    """Verify that an overlapping trial between train and test raises AssertionError."""
    train = ["Suturing_C001", "Suturing_C002", "Suturing_D001"]
    test = ["Suturing_B001", "Suturing_C001"]  # C001 leaks!
    with pytest.raises(AssertionError, match="Shared trials between train and test"):
        assert_no_leakage(train, test)


def test_leakage_assertion_catches_user_overlap():
    """Verify that a test user appearing in training set raises AssertionError."""
    train = ["Suturing_B001", "Suturing_C001"]  # User B is in test!
    test = ["Suturing_B002"]
    with pytest.raises(AssertionError, match="User leakage detected"):
        assert_no_leakage(train, test)


def test_export_louo_splits(tmp_path):
    """Verify export of split files generates CSVs for all 3 tasks and summary JSON."""
    exported = export_louo_splits(output_dir=tmp_path)
    assert "Suturing" in exported
    assert "Knot_Tying" in exported
    assert "Needle_Passing" in exported
    assert "summary" in exported

    suturing_csv = exported["Suturing"]
    df = pd.read_csv(suturing_csv)
    assert len(df) == 39 * 8  # 39 trials recorded across each of the 8 folds
    assert set(df["split"].unique()) == {"train", "val", "test"}
