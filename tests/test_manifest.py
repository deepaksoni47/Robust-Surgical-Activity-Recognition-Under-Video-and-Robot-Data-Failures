from pathlib import Path
import pytest
import pandas as pd

from src.data.manifest import TASKS, build_manifest, export_manifest, parse_meta_file


@pytest.fixture(scope="module")
def manifest_df():
    """Build and cache multi-task manifest for testing."""
    return build_manifest("dataset")


def test_manifest_total_trials(manifest_df):
    """Verify that exactly 103 trials are loaded across the 3 tasks."""
    assert len(manifest_df) == 103


def test_manifest_per_task_counts(manifest_df):
    """Verify trial distribution matches official JIGSAWS benchmark specification."""
    counts = manifest_df["task"].value_counts().to_dict()
    assert counts.get("Suturing") == 39
    assert counts.get("Knot_Tying") == 36
    assert counts.get("Needle_Passing") == 28


def test_manifest_unique_trial_ids(manifest_df):
    """Verify every trial has a unique identifier."""
    assert manifest_df["trial_id"].nunique() == 103


def test_manifest_user_ids(manifest_df):
    """Verify user IDs are non-null and belong to valid subjects B through I."""
    valid_subjects = {"B", "C", "D", "E", "F", "G", "H", "I"}
    assert manifest_df["user_id"].isna().sum() == 0
    assert set(manifest_df["user_id"].unique()).issubset(valid_subjects)


def test_manifest_skill_levels(manifest_df):
    """Verify skill levels are non-null and belong to Novice (N), Intermediate (I), Expert (E)."""
    valid_skills = {"N", "I", "E"}
    assert manifest_df["skill_level"].isna().sum() == 0
    assert set(manifest_df["skill_level"].unique()).issubset(valid_skills)


def test_manifest_kinematics_exist(manifest_df):
    """Verify that all 103 trials have verified kinematics files on disk."""
    assert manifest_df["kinematics_exists"].all()
    for path in manifest_df["kinematics_path"]:
        assert Path(path).is_file(), f"Missing kinematics file: {path}"


def test_manifest_transcriptions_exist(manifest_df):
    """Verify that all 103 trials have verified transcription files on disk."""
    assert manifest_df["transcription_exists"].all()
    for path in manifest_df["transcription_path"]:
        assert Path(path).is_file(), f"Missing transcription file: {path}"


def test_manifest_export(tmp_path, manifest_df):
    """Verify manifest export writes both CSV and JSON formats."""
    csv_path = export_manifest(manifest_df, tmp_path)
    assert csv_path.is_file()
    
    json_path = tmp_path / "manifest.json"
    assert json_path.is_file()

    # Read back and verify row count
    df_read = pd.read_csv(csv_path)
    assert len(df_read) == 103
