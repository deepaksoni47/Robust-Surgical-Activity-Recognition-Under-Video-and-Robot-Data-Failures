"""
Unit and Schema Integrity Tests for SurgSync Dataset Parser (ITER-33).
"""

import pytest
import numpy as np
from src.data.surgsync_parser import (
    SurgSyncGestureMapper,
    SurgSyncKinematicsParser,
    SurgSyncSyncAuditor,
    SurgSyncSplitGenerator,
    SURGSYNC_SUTURING_GESTURES,
    AD_HOC_GESTURE_CODES,
)


def test_gesture_mapping_canonical():
    """Verify all 18 canonical suturing gestures map to discrete IDs 1-18."""
    mapper = SurgSyncGestureMapper()
    assert mapper.get_vocabulary_size() == 21  # 0 (bg) + 18 + 2 adhoc

    # Full verbalized strings
    for gid, desc in SURGSYNC_SUTURING_GESTURES.items():
        mapped_id = mapper.map_gesture(desc)
        assert mapped_id == gid, f"Failed mapping {desc} -> expected {gid}, got {mapped_id}"

    # Sample verbalized string with parenthetical notes
    long_desc = "Grasp needle (Closing the instrument jaws on the needle body to secure it.)"
    assert mapper.map_gesture(long_desc) == 2

    # Case-insensitivity test
    assert mapper.map_gesture("drive needle") == 6
    assert mapper.map_gesture("WRAP SUTURE AROUND INSTRUMENT") == 13


def test_gesture_mapping_adhoc_and_null():
    """Verify ad-hoc codes ('00', '01') and missing/null values map correctly."""
    mapper = SurgSyncGestureMapper()

    # Ad-hoc codes
    assert mapper.map_gesture("00") == 19
    assert mapper.map_gesture("01") == 20

    # Null / Background
    assert mapper.map_gesture(None) == 0
    assert mapper.map_gesture("") == 0
    assert mapper.map_gesture("None") == 0
    assert mapper.map_gesture("null") == 0
    assert mapper.map_gesture("NaN") == 0


def test_gesture_mapping_unknown_raises():
    """Verify that unknown/corrupted labels fail loudly rather than silently imputing."""
    mapper = SurgSyncGestureMapper()
    with pytest.raises(ValueError, match="Unknown gesture label"):
        mapper.map_gesture("Perform laparoscopic appendectomy")


def test_kinematics_14d_cartesian_extraction():
    """Verify 14D Cartesian kinematic feature vector extraction and values."""
    psm1_row = {
        "measured_cp.position": [0.01, -0.02, 0.03],
        "measured_cv.linear": [0.001, -0.002, 0.003],
        "jaw.measured_position": 0.45,
    }
    psm2_row = {
        "measured_cp.position": [-0.04, 0.05, -0.06],
        "measured_cv.linear": [-0.004, 0.005, -0.006],
        "jaw.measured_position": -0.12,
    }

    vec = SurgSyncKinematicsParser.extract_14d_cartesian(psm1_row, psm2_row)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (14,)
    assert vec.dtype == np.float32

    # Check PSM1 components (first 7)
    np.testing.assert_allclose(vec[:3], [0.01, -0.02, 0.03])
    np.testing.assert_allclose(vec[3:6], [0.001, -0.002, 0.003])
    assert vec[6] == pytest.approx(0.45, abs=1e-5)

    # Check PSM2 components (last 7)
    np.testing.assert_allclose(vec[7:10], [-0.04, 0.05, -0.06])
    np.testing.assert_allclose(vec[10:13], [-0.004, 0.005, -0.006])
    assert vec[13] == pytest.approx(-0.12, abs=1e-5)


def test_kinematics_14d_missing_or_nan_raises():
    """Verify that missing fields or NaN values in kinematics fail loudly."""
    # Missing linear velocity in PSM1
    bad_psm1 = {
        "measured_cp.position": [0.01, -0.02, 0.03],
        "jaw.measured_position": 0.45,
    }
    good_psm2 = {
        "measured_cp.position": [-0.04, 0.05, -0.06],
        "measured_cv.linear": [-0.004, 0.005, -0.006],
        "jaw.measured_position": -0.12,
    }
    with pytest.raises(KeyError, match="Missing required field"):
        SurgSyncKinematicsParser.extract_14d_cartesian(bad_psm1, good_psm2)

    # Non-finite (NaN) values
    nan_psm1 = {
        "measured_cp.position": [np.nan, -0.02, 0.03],
        "measured_cv.linear": [0.001, -0.002, 0.003],
        "jaw.measured_position": 0.45,
    }
    with pytest.raises(ValueError, match="Non-finite values detected"):
        SurgSyncKinematicsParser.extract_14d_cartesian(nan_psm1, good_psm2)


def test_kinematics_14d_joint_space():
    """Verify 14D joint-space feature vector extraction."""
    psm1_row = {
        "measured_js.position": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "jaw.measured_position": 0.5,
    }
    psm2_row = {
        "measured_js.position": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        "jaw.measured_position": -0.5,
    }

    vec = SurgSyncKinematicsParser.extract_14d_joint_space(psm1_row, psm2_row)
    assert vec.shape == (14,)
    assert vec[6] == pytest.approx(0.5, abs=1e-5)
    assert vec[13] == pytest.approx(-0.5, abs=1e-5)


def test_sync_auditor():
    """Verify that cross-modal time sync validation catches out-of-tolerance deltas."""
    auditor = SurgSyncSyncAuditor(max_kinematic_delta_ms=33.0)

    # Clean record within tolerance (< 33 ms)
    clean_row = {
        "frame_index": 100,
        "master_timestamp_ns": 10_000_000_000,
        "delta_to_master.PSM1.measured_cp_ns": 5_000_000,   # 5.0 ms
        "delta_to_master.PSM1.measured_cv_ns": -2_500_000,  # 2.5 ms
        "delta_to_master.PSM2.measured_cp_ns": 8_000_000,   # 8.0 ms
        "delta_to_master.PSM2.measured_cv_ns": 4_000_000,   # 4.0 ms
        "delta_to_master.image_right_ns": 6_000_000,        # 6.0 ms
    }
    is_valid, violations = auditor.validate_timestamp_row(clean_row)
    assert is_valid
    assert len(violations) == 0

    # Degraded record with massive jitter (55 ms delta)
    jitter_row = dict(clean_row)
    jitter_row["delta_to_master.PSM1.measured_cp_ns"] = 55_000_000  # 55.0 ms > 33.0 ms
    is_valid, violations = auditor.validate_timestamp_row(jitter_row)
    assert not is_valid
    assert len(violations) == 1
    assert "exceeds tolerance" in violations[0]


def test_stratified_split_generator():
    """Verify that stratified fold generation produces leak-free, balanced splits."""
    # Create synthetic dataset representing the 92 suturing episodes
    synthetic_episodes = []
    strata_counts = {
        ("Expert", "online"): 16,
        ("Expert", "offline"): 17,
        ("Intermediate", "online"): 22,
        ("Intermediate", "offline"): 24,
        ("Novice", "online"): 7,
        ("Novice", "offline"): 6,
    }

    ep_counter = 0
    for (skill, recorder), count in strata_counts.items():
        for i in range(count):
            synthetic_episodes.append({
                "episode_id": f"ep_{ep_counter:03d}",
                "operator_skill_level": skill,
                "recorder_variant": recorder,
            })
            ep_counter += 1

    assert len(synthetic_episodes) == 92

    folds = SurgSyncSplitGenerator.generate_stratified_folds(
        synthetic_episodes, n_splits=4, seed=42
    )

    assert len(folds) == 4
    all_val_ids = []

    for f_idx, fold in enumerate(folds):
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]

        # Strict leakage assertion: zero intersection between train and val
        assert len(set(train_ids).intersection(set(val_ids))) == 0

        # Exact total count
        assert len(train_ids) + len(val_ids) == 92

        # Val fold size should be balanced (23 episodes per fold)
        assert len(val_ids) == 23
        all_val_ids.extend(val_ids)

    # Exact coverage: all 92 episodes appear in validation exactly once
    assert len(set(all_val_ids)) == 92
