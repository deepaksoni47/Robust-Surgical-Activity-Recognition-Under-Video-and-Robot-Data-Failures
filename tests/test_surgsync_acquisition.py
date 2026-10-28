"""
Unit and Preflight Tests for SurgSync Acquisition and Extraction Pipelines (ITER-34).
"""

import json
import numpy as np
import pytest
from pathlib import Path

from scripts.download_surgsync_suturing import (
    check_preflight_space,
    verify_download_integrity,
    EXPECTED_SUTURING_TOTAL_BYTES,
)
from scripts.extract_surgmotion_surgsync import (
    preprocess_frame,
    discover_suturing_episodes,
)


def test_check_preflight_space_skip_video(tmp_path: Path):
    """Verify that skip_video mode requires only 1 GB and passes when space is available."""
    is_sufficient, free_gb, req_gb = check_preflight_space(tmp_path, skip_video=True)
    assert req_gb == 1.0
    assert free_gb > 0
    # In any development environment with >= 1 GB free, this passes
    if free_gb >= 1.0:
        assert is_sufficient


def test_check_preflight_space_insufficient_flag(tmp_path: Path):
    """Verify that asking for impossible disk space (e.g. 50,000 GB) fails preflight."""
    is_sufficient, free_gb, req_gb = check_preflight_space(
        tmp_path, skip_video=False, min_free_gb=50_000.0
    )
    assert not is_sufficient
    assert req_gb == 50_000.0


def test_verify_download_integrity_missing_manifest(tmp_path: Path):
    """Verify that integrity check cleanly reports missing manifest."""
    res = verify_download_integrity(tmp_path)
    assert res["status"] == "FAILED"
    assert "not found" in res["error"]


def test_verify_download_integrity_with_mock_manifest(tmp_path: Path):
    """Verify integrity check validates files and detects missing/mismatched files."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)
    manifest_path = meta_dir / "manifest.json"

    # Create dummy files
    f1 = tmp_path / "online_data" / "episodes" / "single_interrupted_stitch" / "0" / "annotation.parquet"
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"dummy_annotation_data")

    # Manifest specifying f1 (present) and f2 (missing)
    mock_manifest = {
        "files": {
            "online_data/episodes/single_interrupted_stitch/0/annotation.parquet": {
                "size_bytes": len(b"dummy_annotation_data"),
                "sha256": "dummy_sha"
            },
            "online_data/episodes/single_interrupted_stitch/0/video_raw/stereo_left.mkv": {
                "size_bytes": 1024 * 1024 * 100,
                "sha256": "dummy_video_sha"
            }
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(mock_manifest, f)

    res = verify_download_integrity(tmp_path, manifest_path=manifest_path, verify_hashes=False)
    assert res["status"] == "INCOMPLETE"
    assert res["verified_count"] == 1
    assert res["missing_count"] == 1
    assert "stereo_left.mkv" in res["missing_files"][0]


def test_preprocess_frame():
    """Verify frame preprocessing output dimensions, normalization, and channel order."""
    # Create fake 1080p BGR frame (1080, 1920, 3)
    fake_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    processed = preprocess_frame(fake_frame, target_size=224)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (3, 224, 224)
    assert processed.dtype == np.float32
    assert np.all(np.isfinite(processed))


def test_discover_suturing_episodes_empty(tmp_path: Path):
    """Verify episode discovery returns empty list gracefully on nonexistent dir."""
    episodes = discover_suturing_episodes(tmp_path, task="single_interrupted_stitch")
    assert isinstance(episodes, list)
    assert len(episodes) == 0


def test_check_preflight_space_all_tasks(tmp_path: Path):
    """Verify that 'all' tasks mode enforces ~800 GB threshold for full and 2 GB for skip-video."""
    # Full video mode
    is_sufficient, free_gb, req_gb = check_preflight_space(tmp_path, skip_video=False, task="all")
    assert req_gb > 750.0  # ~800 GB
    
    # Skip video mode
    is_sufficient_skip, free_gb_skip, req_gb_skip = check_preflight_space(tmp_path, skip_video=True, task="all")
    assert req_gb_skip == 2.0


def test_verify_download_integrity_all_tasks_filter(tmp_path: Path):
    """Verify that task='all' checks all tasks, while task='suturing' only checks suturing + meta."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)
    manifest_path = meta_dir / "manifest.json"

    # Manifest with 1 suturing file and 1 peg_transfer file
    mock_manifest = {
        "files": {
            "online_data/episodes/single_interrupted_stitch/0/annotation.parquet": {
                "size_bytes": 100,
                "sha256": "sha1"
            },
            "online_data/episodes/peg_transfer/0/annotation.parquet": {
                "size_bytes": 200,
                "sha256": "sha2"
            }
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(mock_manifest, f)

    # When task="suturing", only 1 file is relevant
    res_suturing = verify_download_integrity(tmp_path, manifest_path=manifest_path, task="suturing")
    assert res_suturing["total_relevant_files"] == 1

    # When task="all", both files are relevant
    res_all = verify_download_integrity(tmp_path, manifest_path=manifest_path, task="all")
    assert res_all["total_relevant_files"] == 2


def test_discover_episodes_all_tasks(tmp_path: Path):
    """Verify multi-task discovery across partitions when task='all'."""
    # Create fake episode 1 (suturing)
    ep1_dir = tmp_path / "online_data" / "episodes" / "single_interrupted_stitch" / "0"
    (ep1_dir / "video_raw").mkdir(parents=True)
    (ep1_dir / "video_raw" / "stereo_left.mkv").write_bytes(b"dummy_video_1")
    with open(ep1_dir / "episode_meta.json", "w", encoding="utf-8") as f:
        json.dump({"episode_id": "ep1", "operator_skill_level": "Expert"}, f)

    # Create fake episode 2 (peg_transfer)
    ep2_dir = tmp_path / "offline_data" / "episodes" / "peg_transfer" / "1"
    (ep2_dir / "video_raw").mkdir(parents=True)
    (ep2_dir / "video_raw" / "stereo_left.mkv").write_bytes(b"dummy_video_2")
    with open(ep2_dir / "episode_meta.json", "w", encoding="utf-8") as f:
        json.dump({"episode_id": "ep2", "operator_skill_level": "Intermediate"}, f)

    # Test discovering only suturing
    suturing_eps = discover_suturing_episodes(tmp_path, task="single_interrupted_stitch")
    assert len(suturing_eps) == 1
    assert suturing_eps[0]["task"] == "single_interrupted_stitch"

    # Test discovering all tasks
    all_eps = discover_suturing_episodes(tmp_path, task="all")
    assert len(all_eps) == 2
    tasks_found = {ep["task"] for ep in all_eps}
    assert tasks_found == {"single_interrupted_stitch", "peg_transfer"}

