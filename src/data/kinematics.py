from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.reproducibility import save_json

KINEMATICS_DIMENSION = 76

# Manipulator prefixes in JIGSAWS kinematics order:
# 1-19: Master Left (ML)
# 20-38: Master Right (MR)
# 39-57: Slave Left (SL)
# 58-76: Slave Right (SR)
MANIPULATOR_PREFIXES = ["ML", "MR", "SL", "SR"]


def get_kinematics_columns() -> List[str]:
    """Return explicit semantic column names for all 76 kinematic features."""
    feature_names = []
    for prefix in MANIPULATOR_PREFIXES:
        # Tooltip Cartesian Position (3)
        feature_names.extend([f"{prefix}_pos_x", f"{prefix}_pos_y", f"{prefix}_pos_z"])
        # Tooltip Orientation Rotation Matrix R (9)
        feature_names.extend([f"{prefix}_rot_r{i}{j}" for i in range(1, 4) for j in range(1, 4)])
        # Tooltip Translational Linear Velocity (3)
        feature_names.extend([f"{prefix}_vel_trans_x", f"{prefix}_vel_trans_y", f"{prefix}_vel_trans_z"])
        # Tooltip Rotational Angular Velocity (3)
        feature_names.extend([f"{prefix}_vel_rot_x", f"{prefix}_vel_rot_y", f"{prefix}_vel_rot_z"])
        # Gripper Jaw Angle (1)
        feature_names.append(f"{prefix}_gripper_angle")

    if len(feature_names) != KINEMATICS_DIMENSION:
        raise ValueError(
            f"Expected {KINEMATICS_DIMENSION} column names, but generated {len(feature_names)}"
        )
    return feature_names


def load_kinematics(filepath: str | Path) -> np.ndarray:
    """Load a JIGSAWS 76D kinematics file into a float32 numpy array.
    
    Args:
        filepath: Path to the kinematics .txt file.
        
    Returns:
        np.ndarray of shape (N, 76) with dtype float32.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If column count != 76 or values are non-finite (NaN/Inf).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Kinematics file not found: {path.resolve()}")

    try:
        data = np.loadtxt(path, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to parse kinematics file {path.name}: {e}") from e

    # Handle 1D array if a file has only 1 row
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2:
        raise ValueError(
            f"Kinematics data must be 2D array, got {data.ndim}D with shape {data.shape} in {path.name}"
        )

    if data.shape[1] != KINEMATICS_DIMENSION:
        raise ValueError(
            f"Expected {KINEMATICS_DIMENSION} kinematic columns, got {data.shape[1]} in {path.name}"
        )

    if not np.all(np.isfinite(data)):
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        raise ValueError(
            f"Non-finite values detected in {path.name}: {n_nan} NaNs, {n_inf} Infs"
        )

    return data


def validate_kinematics_file(filepath: str | Path) -> Dict[str, Any]:
    """Validate a single kinematics file and return descriptive summary statistics."""
    data = load_kinematics(filepath)
    return {
        "filepath": str(filepath).replace("\\", "/"),
        "n_samples": int(data.shape[0]),
        "n_channels": int(data.shape[1]),
        "is_finite": bool(np.all(np.isfinite(data))),
        "mean_magnitude": float(np.mean(np.abs(data))),
        "min_val": float(np.min(data)),
        "max_val": float(np.max(data)),
    }


def audit_all_kinematics(
    manifest_df: pd.DataFrame,
    output_path: Optional[str | Path] = "artifacts/kinematics_audit.json"
) -> Dict[str, Any]:
    """Audit kinematics files across all trials recorded in the manifest.
    
    Args:
        manifest_df: Manifest DataFrame from build_manifest().
        output_path: Optional destination to save audit summary JSON.
        
    Returns:
        Dictionary summarizing total samples, min/max lengths, and per-task stats.
    """
    results_by_task: Dict[str, List[Dict[str, Any]]] = {}
    total_samples = 0
    all_lengths = []

    for _, row in manifest_df.iterrows():
        task = row["task"]
        trial_id = row["trial_id"]
        kin_path = row["kinematics_path"]

        stats = validate_kinematics_file(kin_path)
        stats["trial_id"] = trial_id
        stats["user_id"] = row["user_id"]

        results_by_task.setdefault(task, []).append(stats)
        total_samples += stats["n_samples"]
        all_lengths.append(stats["n_samples"])

    summary = {
        "total_trials": len(manifest_df),
        "total_kinematic_samples": int(total_samples),
        "min_samples_per_trial": int(min(all_lengths)),
        "max_samples_per_trial": int(max(all_lengths)),
        "mean_samples_per_trial": float(np.mean(all_lengths)),
        "median_samples_per_trial": float(np.median(all_lengths)),
        "all_trials_finite": True,
        "per_task_summary": {
            task: {
                "n_trials": len(trials),
                "total_samples": sum(t["n_samples"] for t in trials),
                "min_samples": min(t["n_samples"] for t in trials),
                "max_samples": max(t["n_samples"] for t in trials),
                "mean_samples": float(np.mean([t["n_samples"] for t in trials])),
            }
            for task, trials in results_by_task.items()
        },
    }

    if output_path is not None:
        save_json(summary, output_path)

    return summary
