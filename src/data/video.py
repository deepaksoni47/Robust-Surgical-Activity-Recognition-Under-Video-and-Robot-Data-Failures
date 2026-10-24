from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.data.kinematics import load_kinematics
from src.data.transcriptions import parse_transcription
from src.utils.reproducibility import save_json


def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
    """Inspect an endoscopic AVI video file and extract metadata via OpenCV.
    
    Args:
        video_path: Path to the AVI video file.
        
    Returns:
        Dict containing frame count, fps, resolution, duration_sec, and readability flag.
        
    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV fails to open the video file.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path.resolve()}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video file: {path.resolve()}")

    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        ret, frame = cap.read()
        can_read_first = bool(ret and frame is not None and frame.size > 0)
        duration_sec = float(n_frames / fps) if fps > 0 else 0.0

    finally:
        cap.release()

    return {
        "video_path": str(path).replace("\\", "/"),
        "filename": path.name,
        "n_frames": n_frames,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "resolution": [width, height],
        "fourcc": fourcc_str,
        "duration_sec": round(duration_sec, 2),
        "can_read_first_frame": can_read_first,
    }


def read_frame_at_index(
    video_path: str | Path,
    frame_idx: int,
    to_rgb: bool = True
) -> np.ndarray:
    """Read a specific 0-indexed frame from a video file.
    
    Args:
        video_path: Path to the video file.
        frame_idx: 0-indexed frame index to read.
        to_rgb: If True, convert BGR OpenCV image to RGB.
        
    Returns:
        np.ndarray with shape (H, W, 3), dtype uint8.
        
    Raises:
        IndexError: If frame_idx cannot be retrieved or read.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path.name}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise IndexError(f"Could not read frame at index {frame_idx} from {path.name}")
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

    return frame


def audit_video_kinematics_alignment(
    manifest_df: pd.DataFrame,
    sample_size: Optional[int] = None,
    output_json: Optional[str | Path] = "artifacts/alignment_audit.json",
    output_md: Optional[str | Path] = "artifacts/alignment_audit.md",
) -> Dict[str, Any]:
    """Audit video frame count against kinematics sample count and transcription intervals.
    
    Args:
        manifest_df: Manifest DataFrame from build_manifest().
        sample_size: Optional number of trials to sample per task (None audits all trials).
        output_json: Path to save JSON report.
        output_md: Path to save Markdown summary report.
        
    Returns:
        Summary dictionary with alignment offsets and tolerances.
    """
    if sample_size is not None and sample_size < len(manifest_df):
        # Sample evenly across tasks
        audit_df = manifest_df.groupby("task", group_keys=False).apply(
            lambda x: x.head(sample_size), include_groups=False
        ).reset_index(drop=True)
        # Re-merge task column if needed
        if "task" not in audit_df.columns:
            audit_df["task"] = manifest_df.loc[audit_df.index, "task"]
    else:
        audit_df = manifest_df

    alignment_records = []

    for _, row in audit_df.iterrows():
        trial_id = row["trial_id"]
        task = row["task"]
        vid_path = row["video_capture1_path"]
        kin_path = row["kinematics_path"]
        trans_path = row["transcription_path"]

        # Video metadata
        vid_meta = get_video_metadata(vid_path)
        n_vid_frames = vid_meta["n_frames"]
        fps = vid_meta["fps"]

        # Kinematics samples
        kin_data = load_kinematics(kin_path)
        n_kin_samples = kin_data.shape[0]

        # Transcription last frame
        segments = parse_transcription(trans_path)
        last_trans_frame = segments[-1]["end_frame"] if segments else 0

        # Offsets
        delta_vid_kin = abs(n_vid_frames - n_kin_samples)
        delta_kin_trans = abs(n_kin_samples - last_trans_frame)

        alignment_records.append({
            "trial_id": trial_id,
            "task": task,
            "fps": fps,
            "n_vid_frames": n_vid_frames,
            "n_kin_samples": n_kin_samples,
            "last_trans_frame": last_trans_frame,
            "delta_vid_kin": delta_vid_kin,
            "delta_kin_trans": delta_kin_trans,
            "vid_duration_sec": vid_meta["duration_sec"],
            "kin_duration_sec": round(n_kin_samples / 30.0, 2),
        })

    deltas_vid_kin = [r["delta_vid_kin"] for r in alignment_records]
    deltas_kin_trans = [r["delta_kin_trans"] for r in alignment_records]

    summary = {
        "trials_audited": len(alignment_records),
        "mean_delta_vid_kin_frames": float(np.mean(deltas_vid_kin)),
        "max_delta_vid_kin_frames": int(max(deltas_vid_kin)),
        "median_delta_vid_kin_frames": float(np.median(deltas_vid_kin)),
        "mean_delta_kin_trans_frames": float(np.mean(deltas_kin_trans)),
        "max_delta_kin_trans_frames": int(max(deltas_kin_trans)),
        "alignment_rule": (
            "Kinematics samples and video frames are 1-to-1 synchronized at nominal 30 Hz. "
            "Slight recording tail offsets (< 0.5s) are clipped to min(N_vid, N_kin) during windowing."
        ),
        "records": alignment_records,
    }

    if output_json is not None:
        save_json(summary, output_json)

    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# JIGSAWS Multi-Modal Video-Kinematics Alignment Audit\n\n")
            f.write(f"- **Trials Audited:** {summary['trials_audited']}\n")
            f.write(f"- **Mean Video vs Kinematics Difference:** {summary['mean_delta_vid_kin_frames']:.2f} frames\n")
            f.write(f"- **Median Video vs Kinematics Difference:** {summary['median_delta_vid_kin_frames']:.1f} frames\n")
            f.write(f"- **Max Video vs Kinematics Difference:** {summary['max_delta_vid_kin_frames']} frames\n")
            f.write(f"- **Mean Kinematics vs Transcription Difference:** {summary['mean_delta_kin_trans_frames']:.2f} frames\n\n")
            f.write("### Alignment Convention\n")
            f.write(summary["alignment_rule"] + "\n\n")
            f.write("### Sample Trials Alignment Table\n\n")
            f.write("| Trial ID | Task | FPS | Video Frames | Kinematics Samples | Delta (Frames) | Vid Duration | Kin Duration |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in alignment_records[:15]:
                f.write(f"| {r['trial_id']} | {r['task']} | {r['fps']} | {r['n_vid_frames']} | {r['n_kin_samples']} | {r['delta_vid_kin']} | {r['vid_duration_sec']}s | {r['kin_duration_sec']}s |\n")

    return summary
