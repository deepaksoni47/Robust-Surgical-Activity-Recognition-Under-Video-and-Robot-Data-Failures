from pathlib import Path
import numpy as np
import pytest

from src.data.manifest import build_manifest
from src.data.video import (
    audit_video_kinematics_alignment,
    get_video_metadata,
    read_frame_at_index,
)


@pytest.fixture(scope="module")
def manifest_df():
    """Build manifest once for testing."""
    return build_manifest("dataset")


def test_get_video_metadata():
    """Verify video metadata extraction on Suturing video."""
    vid_path = Path("dataset/Suturing/video/Suturing_B001_capture1.avi")
    meta = get_video_metadata(vid_path)
    assert meta["n_frames"] == 5640
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert abs(meta["fps"] - 30.0) < 0.1
    assert meta["duration_sec"] > 180.0
    assert meta["can_read_first_frame"] is True


def test_read_frame_at_index():
    """Verify reading a specific frame returns a valid RGB image."""
    vid_path = Path("dataset/Suturing/video/Suturing_B001_capture1.avi")
    frame = read_frame_at_index(vid_path, frame_idx=50, to_rgb=True)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.min() >= 0
    assert frame.max() <= 255


def test_read_frame_invalid_index():
    """Verify IndexError on out-of-range frame index."""
    vid_path = Path("dataset/Suturing/video/Suturing_B001_capture1.avi")
    with pytest.raises(IndexError):
        read_frame_at_index(vid_path, frame_idx=999999)


def test_video_file_not_found():
    """Verify FileNotFoundError on non-existent video."""
    with pytest.raises(FileNotFoundError):
        get_video_metadata("dataset/non_existent_video.avi")


def test_audit_video_kinematics_alignment(tmp_path, manifest_df):
    """Verify alignment audit between video frames and kinematics samples."""
    json_out = tmp_path / "alignment_audit.json"
    md_out = tmp_path / "alignment_audit.md"

    # Audit 3 trials per task (9 total) for fast, thorough integration verification
    summary = audit_video_kinematics_alignment(
        manifest_df,
        sample_size=3,
        output_json=json_out,
        output_md=md_out,
    )

    assert summary["trials_audited"] == 9
    assert json_out.is_file()
    assert md_out.is_file()
    # Mean difference between video and kinematics should be small (recording head/tail tolerance < 1.0s = 30 frames)
    assert summary["mean_delta_vid_kin_frames"] < 30.0
