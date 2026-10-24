import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.reproducibility import save_json


GESTURE_DESCRIPTIONS: Dict[str, str] = {
    "G0": "Idle / Background",
    "G1": "Reaching for needle with right hand",
    "G2": "Positioning tip of needle",
    "G3": "Pushing needle through tissue",
    "G4": "Transferring needle from left to right hand",
    "G5": "Moving to point with needle in robot hand",
    "G6": "Pulling suture with left hand",
    "G7": "Pulling suture with right hand",
    "G8": "Orienting needle",
    "G9": "Using right hand to help tighten suture",
    "G10": "Loosening more suture",
    "G11": "Dropping suture / needle at end",
    "G12": "Reaching for needle with left hand",
    "G13": "Making C loop around right hand",
    "G14": "Reaching for suture with right hand",
    "G15": "Knot tying",
}

TASK_GESTURES: Dict[str, List[str]] = {
    "Suturing": ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"],
    "Knot_Tying": ["G1", "G11", "G12", "G13", "G14", "G15"],
    # In Needle_Passing, G9 and G10 are present in actual annotations in addition to G1-G6, G8, G11
    "Needle_Passing": ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"],
}


def get_gesture_vocabulary(
    task: Optional[str] = None, include_background: bool = True
) -> Dict[str, int]:
    """Return dictionary mapping gesture string IDs to integer class indices.
    
    Args:
        task: Optional task name ('Suturing', 'Knot_Tying', 'Needle_Passing').
              If None, returns vocabulary across all 15 gestures in JIGSAWS.
        include_background: Whether to include 'G0' at index 0.
        
    Returns:
        Dict[str, int] mapping gesture IDs to unique contiguous class indices.
    """
    if task is not None:
        if task not in TASK_GESTURES:
            raise ValueError(f"Unknown task: {task}. Expected one of {list(TASK_GESTURES.keys())}")
        gesture_list = list(TASK_GESTURES[task])
    else:
        # Sort all unique gestures numerically (G1..G15)
        gesture_list = [f"G{i}" for i in range(1, 16)]

    vocab: Dict[str, int] = {}
    if include_background:
        vocab["G0"] = 0
        for idx, g in enumerate(gesture_list, start=1):
            vocab[g] = idx
    else:
        for idx, g in enumerate(gesture_list, start=0):
            vocab[g] = idx

    return vocab


def parse_transcription(filepath: str | Path) -> List[Dict[str, Any]]:
    """Parse a JIGSAWS gesture transcription file.
    
    Args:
        filepath: Path to the transcription .txt file.
        
    Returns:
        List of segment dicts with keys: start_frame, end_frame, gesture_id, duration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If line format is invalid or start_frame > end_frame.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Transcription file not found: {path.resolve()}")

    segments: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed line {line_num} in {path.name}: expected at least 3 tokens, got '{line}'"
                )

            try:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid frame numbers on line {line_num} in {path.name}: '{line}'"
                ) from e

            gesture_id = parts[2]

            if start_frame > end_frame:
                raise ValueError(
                    f"Invalid interval on line {line_num} in {path.name}: start ({start_frame}) > end ({end_frame})"
                )

            segments.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "gesture_id": gesture_id,
                "duration": end_frame - start_frame + 1,
            })

    return segments


def transcription_to_framewise_labels(
    segments: List[Dict[str, Any]],
    total_frames: int,
    background_label: str = "G0",
) -> List[str]:
    """Convert segment intervals into continuous 0-indexed per-frame label array.
    
    JIGSAWS frame indices in transcription files are 1-indexed.
    Frames not covered by any annotation segment are assigned background_label.
    
    Args:
        segments: List of segment dictionaries from parse_transcription().
        total_frames: Total number of frames in the video/kinematics stream.
        background_label: Label for unannotated frames (default: 'G0').
        
    Returns:
        List[str] of length total_frames containing gesture IDs.
    """
    labels = [background_label] * total_frames

    for seg in segments:
        # Convert 1-indexed inclusive interval [start_frame, end_frame] to 0-indexed slice
        start_idx = max(0, seg["start_frame"] - 1)
        end_idx = min(total_frames, seg["end_frame"])
        gesture = seg["gesture_id"]

        for i in range(start_idx, end_idx):
            labels[i] = gesture

    return labels


def audit_all_transcriptions(
    manifest_df: pd.DataFrame,
    output_path: Optional[str | Path] = "artifacts/class_counts.json",
) -> Dict[str, Any]:
    """Audit gesture transcriptions across all 103 trials in the manifest.
    
    Args:
        manifest_df: Manifest DataFrame from build_manifest().
        output_path: Optional destination to save class counts summary.
        
    Returns:
        Dictionary summarizing segment counts, frame counts, and class distributions.
    """
    gesture_counts_by_task: Dict[str, Dict[str, int]] = {}
    total_segments = 0
    all_gestures_seen: Set[str] = set()

    for _, row in manifest_df.iterrows():
        task = row["task"]
        trans_path = row["transcription_path"]

        segments = parse_transcription(trans_path)
        total_segments += len(segments)

        task_counts = gesture_counts_by_task.setdefault(task, {})
        for seg in segments:
            gid = seg["gesture_id"]
            all_gestures_seen.add(gid)
            task_counts[gid] = task_counts.get(gid, 0) + 1

    summary = {
        "total_trials": len(manifest_df),
        "total_annotated_segments": total_segments,
        "unique_gestures_found": sorted(list(all_gestures_seen)),
        "segment_counts_by_task": gesture_counts_by_task,
        "per_task_vocabulary_coverage": {
            task: {
                "expected_gestures": TASK_GESTURES[task],
                "observed_gestures": sorted(list(gesture_counts_by_task.get(task, {}).keys())),
                "all_expected_observed": set(TASK_GESTURES[task]) == set(gesture_counts_by_task.get(task, {}).keys()),
            }
            for task in TASK_GESTURES
        },
    }

    if output_path is not None:
        save_json(summary, output_path)

    return summary
