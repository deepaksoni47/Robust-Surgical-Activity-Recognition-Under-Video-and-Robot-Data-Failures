"""
SurgSync Dataset Parser and Schema Validator.

Provides schema validation, gesture vocabulary mapping, kinematics feature extraction,
synchronization verification, and skill-stratified fold generation for the SurgSync
multi-modal robotic surgery benchmark (Zhou et al., ICRA 2026).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


# Canonical 18-gesture taxonomy for single_interrupted_stitch (Zhou et al., ICRA 2026)
SURGSYNC_SUTURING_GESTURES = {
    1: "Reach for needle",
    2: "Grasp needle",
    3: "Release needle",
    4: "Adjust needle",
    5: "Approach tissue",
    6: "Drive needle",
    7: "Tissue manipulation",
    8: "Extract needle",
    9: "Pull suture",
    10: "Grasp suture tail",
    11: "Release suture",
    12: "Park needle",
    13: "Wrap suture around instrument",
    14: "Grasp suture through loop",
    15: "Tighten throw",
    16: "Cut suture",  # Unused in v1.0 (suture cutting omitted in protocol)
    17: "Hold / Idle",
    18: "Move to rest position",
}

# Ad-hoc codes observed in v1.0 data (332 frames across 8 clips)
AD_HOC_GESTURE_CODES = {
    "00": "bilateral pause (both hand not moving > 20 frames)",
    "01": "move camera",
}


class SurgSyncGestureMapper:
    """
    Maps verbalized strings, numeric IDs, ad-hoc codes, and NULL/background
    to a standardized discrete label space.
    """

    def __init__(self, include_idle_background: bool = True):
        self.include_idle_background = include_idle_background
        
        # Build canonical keyword matching table
        self.text_to_id: Dict[str, int] = {}
        for gid, desc in SURGSYNC_SUTURING_GESTURES.items():
            key = desc.lower().strip()
            self.text_to_id[key] = gid

        # Background class is 0
        self.background_id = 0
        # Ad-hoc codes mapped to 19 and 20 (or background if desired)
        self.adhoc_to_id = {
            "00": 19,
            "01": 20,
        }

    def map_gesture(self, raw_label: Optional[str]) -> int:
        """
        Map a raw annotation value to an integer gesture ID.
        
        Returns:
            int: 0 for background/None/unannotated, 1-18 for canonical gestures,
                 19-20 for recognized ad-hoc motion states.
        """
        if raw_label is None:
            return self.background_id
            
        raw_str = str(raw_label).strip()
        if not raw_str or raw_str.lower() in ("none", "null", "nan"):
            return self.background_id

        # Check ad-hoc literals first ('00', '01')
        if raw_str in self.adhoc_to_id:
            return self.adhoc_to_id[raw_str]

        # Check exact string match
        raw_lower = raw_str.lower()
        if raw_lower in self.text_to_id:
            return self.text_to_id[raw_lower]

        # Check substring match against canonical descriptions
        for gid, desc in SURGSYNC_SUTURING_GESTURES.items():
            if desc.lower() in raw_lower or raw_lower.startswith(desc.lower()):
                return gid

        # Unrecognized non-empty label
        raise ValueError(f"Unknown gesture label in SurgSync: {raw_label!r}")

    def get_vocabulary_size(self, include_adhoc: bool = True) -> int:
        """Total vocabulary size including background (0)."""
        base = 19  # 0 (bg) + 18 gestures
        return base + 2 if include_adhoc else base


class SurgSyncKinematicsParser:
    """
    Extracts standardized kinematic vectors from PSM1 and PSM2 records.
    Standardized default is 14D Cartesian/Velocity vector:
      PSM1: [x, y, z, vx, vy, vz, jaw_angle] (7D)
      PSM2: [x, y, z, vx, vy, vz, jaw_angle] (7D)
    """

    @staticmethod
    def extract_14d_cartesian(
        psm1_row: Dict[str, Any],
        psm2_row: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct a 14-dimensional bimanual kinematic feature vector.
        
        Args:
            psm1_row: Dictionary containing PSM1 kinematic fields.
            psm2_row: Dictionary containing PSM2 kinematic fields.
            
        Returns:
            np.ndarray of shape (14,) and dtype float32.
        """
        # Validate required fields
        for arm_name, row in [("PSM1", psm1_row), ("PSM2", psm2_row)]:
            for field in ("measured_cp.position", "measured_cv.linear", "jaw.measured_position"):
                if field not in row:
                    raise KeyError(f"Missing required field {field!r} in {arm_name} record")

        p1_pos = np.asarray(psm1_row["measured_cp.position"], dtype=np.float32)
        p1_vel = np.asarray(psm1_row["measured_cv.linear"], dtype=np.float32)
        p1_jaw = float(psm1_row["jaw.measured_position"])

        p2_pos = np.asarray(psm2_row["measured_cp.position"], dtype=np.float32)
        p2_vel = np.asarray(psm2_row["measured_cv.linear"], dtype=np.float32)
        p2_jaw = float(psm2_row["jaw.measured_position"])

        if len(p1_pos) != 3 or len(p1_vel) != 3:
            raise ValueError(f"PSM1 position/velocity must be 3D, got {len(p1_pos)}, {len(p1_vel)}")
        if len(p2_pos) != 3 or len(p2_vel) != 3:
            raise ValueError(f"PSM2 position/velocity must be 3D, got {len(p2_pos)}, {len(p2_vel)}")

        vec = np.concatenate([
            p1_pos, p1_vel, [p1_jaw],
            p2_pos, p2_vel, [p2_jaw]
        ]).astype(np.float32)

        assert vec.shape == (14,), f"Expected shape (14,), got {vec.shape}"
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"Non-finite values detected in 14D kinematic vector: {vec}")

        return vec

    @staticmethod
    def extract_14d_joint_space(
        psm1_row: Dict[str, Any],
        psm2_row: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct a 14-dimensional bimanual joint feature vector:
          PSM1: 6 joint angles + 1 jaw angle (7D)
          PSM2: 6 joint angles + 1 jaw angle (7D)
        """
        p1_js = np.asarray(psm1_row["measured_js.position"], dtype=np.float32)
        p1_jaw = float(psm1_row["jaw.measured_position"])
        p2_js = np.asarray(psm2_row["measured_js.position"], dtype=np.float32)
        p2_jaw = float(psm2_row["jaw.measured_position"])

        if len(p1_js) != 6 or len(p2_js) != 6:
            raise ValueError(f"Expected 6 joint angles per PSM, got {len(p1_js)} and {len(p2_js)}")

        vec = np.concatenate([p1_js, [p1_jaw], p2_js, [p2_jaw]]).astype(np.float32)
        if not np.all(np.isfinite(vec)):
            raise ValueError("Non-finite values detected in 14D joint vector")
        return vec


class SurgSyncSyncAuditor:
    """
    Validates cross-modal time alignment between video master clock and
    kinematics/annotations.
    """

    def __init__(self, max_kinematic_delta_ms: float = 33.0):
        self.max_kinematic_delta_ms = max_kinematic_delta_ms

    def validate_timestamp_row(self, row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if a timestamp record satisfies temporal synchronization bounds.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check frame index and master timestamp
        if "frame_index" not in row or "master_timestamp_ns" not in row:
            violations.append("Missing frame_index or master_timestamp_ns")
            return False, violations

        # Check key topic deltas
        check_topics = [
            ("delta_to_master.PSM1.measured_cp_ns", "PSM1 Cartesian position"),
            ("delta_to_master.PSM1.measured_cv_ns", "PSM1 Cartesian velocity"),
            ("delta_to_master.PSM2.measured_cp_ns", "PSM2 Cartesian position"),
            ("delta_to_master.PSM2.measured_cv_ns", "PSM2 Cartesian velocity"),
            ("delta_to_master.image_right_ns", "Stereo-right frame"),
        ]

        for col, desc in check_topics:
            if col in row and row[col] is not None:
                delta_ms = abs(float(row[col])) / 1e6
                if delta_ms > self.max_kinematic_delta_ms:
                    violations.append(f"{desc} delta ({delta_ms:.2f} ms) exceeds tolerance ({self.max_kinematic_delta_ms} ms)")

        return len(violations) == 0, violations


class SurgSyncSplitGenerator:
    """
    Generates balanced, leak-free cross-validation splits.
    Because operator IDs are withheld in SurgSync v1.0, splits are stratified
    across operator skill levels (Expert, Intermediate, Novice) and recorder variants
    (Online, Offline).
    """

    @staticmethod
    def generate_stratified_folds(
        episodes: List[Dict[str, Any]],
        n_splits: int = 4,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generate n_splits stratified folds for single_interrupted_stitch.
        
        Args:
            episodes: List of episode metadata dictionaries.
            n_splits: Number of cross-validation folds (default: 4).
            seed: Random seed for reproducible assignment.
            
        Returns:
            List of fold definitions containing train_ids and val_ids.
        """
        rng = np.random.RandomState(seed)

        # Group episodes by (operator_skill_level, recorder_variant) stratum
        strata: Dict[Tuple[str, str], List[str]] = {}
        for ep in episodes:
            skill = ep.get("operator_skill_level", "Unknown")
            recorder = ep.get("recorder_variant", "Unknown")
            key = (skill, recorder)
            strata.setdefault(key, []).append(ep["episode_id"])

        folds: List[Dict[str, Any]] = [
            {"fold": f, "train_ids": [], "val_ids": [], "strata_counts": {}}
            for f in range(n_splits)
        ]

        # Allocate each stratum uniformly across folds with a rolling offset
        fold_offset = 0
        for stratum_key, ep_ids in sorted(strata.items()):
            shuffled = list(ep_ids)
            rng.shuffle(shuffled)
            
            for idx, ep_id in enumerate(shuffled):
                val_fold = (fold_offset + idx) % n_splits
                for f in range(n_splits):
                    if f == val_fold:
                        folds[f]["val_ids"].append(ep_id)
                    else:
                        folds[f]["train_ids"].append(ep_id)
            fold_offset = (fold_offset + len(shuffled)) % n_splits

        # Integrity checks: no overlap between train and val within any fold
        for fold in folds:
            train_set = set(fold["train_ids"])
            val_set = set(fold["val_ids"])
            assert len(train_set.intersection(val_set)) == 0, f"Leakage detected in fold {fold['fold']}"
            assert len(train_set) + len(val_set) == len(episodes), "Mismatch in total episode count"

        return folds
