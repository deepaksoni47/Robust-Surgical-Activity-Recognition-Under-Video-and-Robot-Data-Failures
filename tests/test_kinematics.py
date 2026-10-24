from pathlib import Path
import numpy as np
import pytest

from src.data.manifest import build_manifest
from src.data.kinematics import (
    KINEMATICS_DIMENSION,
    audit_all_kinematics,
    get_kinematics_columns,
    load_kinematics,
    validate_kinematics_file,
)


@pytest.fixture(scope="module")
def manifest_df():
    """Build manifest once for testing."""
    return build_manifest("dataset")


def test_kinematics_columns_count_and_uniqueness():
    """Verify column names list has exactly 76 unique, well-structured names."""
    cols = get_kinematics_columns()
    assert len(cols) == KINEMATICS_DIMENSION
    assert len(set(cols)) == KINEMATICS_DIMENSION
    
    # Assert all 4 manipulators are present
    assert any(c.startswith("ML_") for c in cols)
    assert any(c.startswith("MR_") for c in cols)
    assert any(c.startswith("SL_") for c in cols)
    assert any(c.startswith("SR_") for c in cols)


def test_load_kinematics_suturing():
    """Verify loading real Suturing kinematics file."""
    path = Path("dataset/Suturing/kinematics/AllGestures/Suturing_B001.txt")
    data = load_kinematics(path)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.ndim == 2
    assert data.shape[1] == 76
    assert data.shape[0] > 1000
    assert np.all(np.isfinite(data))


def test_load_kinematics_knot_tying():
    """Verify loading real Knot Tying kinematics file."""
    path = Path("dataset/Knot_Tying/kinematics/AllGestures/Knot_Tying_B001.txt")
    data = load_kinematics(path)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 76
    assert np.all(np.isfinite(data))


def test_load_kinematics_needle_passing():
    """Verify loading real Needle Passing kinematics file."""
    path = Path("dataset/Needle_Passing/kinematics/AllGestures/Needle_Passing_B001.txt")
    data = load_kinematics(path)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 76
    assert np.all(np.isfinite(data))


def test_load_kinematics_file_not_found():
    """Verify FileNotFoundError on non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_kinematics("dataset/non_existent_kinematics.txt")


def test_load_kinematics_wrong_dimension(tmp_path):
    """Verify ValueError is raised if columns != 76."""
    bad_file = tmp_path / "bad_cols.txt"
    np.savetxt(bad_file, np.random.randn(10, 50))
    with pytest.raises(ValueError, match="Expected 76 kinematic columns"):
        load_kinematics(bad_file)


def test_load_kinematics_nan_detection(tmp_path):
    """Verify ValueError is raised if NaN or Inf values are present."""
    nan_file = tmp_path / "nan_kinematics.txt"
    data = np.random.randn(20, 76)
    data[5, 10] = np.nan
    np.savetxt(nan_file, data)
    with pytest.raises(ValueError, match="Non-finite values detected"):
        load_kinematics(nan_file)


def test_validate_kinematics_file():
    """Verify validate_kinematics_file returns valid summary dict."""
    path = Path("dataset/Suturing/kinematics/AllGestures/Suturing_B001.txt")
    report = validate_kinematics_file(path)
    assert report["n_channels"] == 76
    assert report["is_finite"] is True
    assert report["n_samples"] > 0
    assert report["min_val"] < report["max_val"]


def test_audit_all_103_kinematics(tmp_path, manifest_df):
    """Verify audit over all 103 trials across all 3 tasks passes without non-finite values."""
    audit_file = tmp_path / "kinematics_audit.json"
    summary = audit_all_kinematics(manifest_df, audit_file)
    assert summary["total_trials"] == 103
    assert summary["all_trials_finite"] is True
    assert audit_file.is_file()
    assert len(summary["per_task_summary"]) == 3
    assert summary["total_kinematic_samples"] > 200000
