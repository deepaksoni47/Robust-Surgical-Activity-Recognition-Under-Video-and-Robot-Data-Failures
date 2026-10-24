from pathlib import Path
import pytest

from src.data.manifest import build_manifest
from src.data.transcriptions import (
    GESTURE_DESCRIPTIONS,
    TASK_GESTURES,
    audit_all_transcriptions,
    get_gesture_vocabulary,
    parse_transcription,
    transcription_to_framewise_labels,
)


@pytest.fixture(scope="module")
def manifest_df():
    """Build manifest once for testing."""
    return build_manifest("dataset")


def test_gesture_vocabulary_global():
    """Verify global vocabulary contains all 15 gestures plus background G0."""
    vocab = get_gesture_vocabulary(task=None, include_background=True)
    assert len(vocab) == 16
    assert vocab["G0"] == 0
    assert vocab["G15"] == 15
    for i in range(1, 16):
        assert f"G{i}" in vocab


def test_gesture_vocabulary_per_task():
    """Verify task-specific vocabularies match JIGSAWS definitions."""
    suturing_vocab = get_gesture_vocabulary(task="Suturing", include_background=True)
    assert len(suturing_vocab) == 11  # 10 gestures + G0
    assert set(TASK_GESTURES["Suturing"]).issubset(set(suturing_vocab.keys()))

    knot_vocab = get_gesture_vocabulary(task="Knot_Tying", include_background=True)
    assert len(knot_vocab) == 7  # 6 gestures + G0
    assert set(TASK_GESTURES["Knot_Tying"]).issubset(set(knot_vocab.keys()))

    needle_vocab = get_gesture_vocabulary(task="Needle_Passing", include_background=True)
    assert len(needle_vocab) == 11  # 10 observed gestures + G0
    assert set(TASK_GESTURES["Needle_Passing"]).issubset(set(needle_vocab.keys()))


def test_parse_transcription_suturing():
    """Verify parsing real Suturing transcription file."""
    path = Path("dataset/Suturing/transcriptions/Suturing_B001.txt")
    segments = parse_transcription(path)
    assert len(segments) == 29
    first = segments[0]
    assert first["start_frame"] == 80
    assert first["end_frame"] == 219
    assert first["gesture_id"] == "G1"
    assert first["duration"] == 140

    last = segments[-1]
    assert last["gesture_id"] == "G11"
    assert last["end_frame"] == 5635


def test_parse_transcription_knot_tying():
    """Verify parsing real Knot Tying transcription file."""
    path = Path("dataset/Knot_Tying/transcriptions/Knot_Tying_B001.txt")
    segments = parse_transcription(path)
    assert len(segments) > 5
    assert segments[0]["start_frame"] == 45
    assert segments[0]["gesture_id"] == "G12"


def test_parse_transcription_needle_passing():
    """Verify parsing real Needle Passing transcription file."""
    path = Path("dataset/Needle_Passing/transcriptions/Needle_Passing_B001.txt")
    segments = parse_transcription(path)
    assert len(segments) > 5
    assert segments[0]["gesture_id"] == "G1"


def test_transcription_to_framewise_labels():
    """Verify converting intervals to contiguous frame-wise labels with background."""
    segments = [
        {"start_frame": 3, "end_frame": 5, "gesture_id": "G1", "duration": 3},
        {"start_frame": 6, "end_frame": 8, "gesture_id": "G2", "duration": 3},
    ]
    total_frames = 10
    labels = transcription_to_framewise_labels(segments, total_frames, background_label="G0")
    assert len(labels) == 10
    # JIGSAWS 1-indexed frames:
    # 1 -> index 0: G0
    # 2 -> index 1: G0
    # 3, 4, 5 -> indices 2, 3, 4: G1
    # 6, 7, 8 -> indices 5, 6, 7: G2
    # 9, 10 -> indices 8, 9: G0
    assert labels == ["G0", "G0", "G1", "G1", "G1", "G2", "G2", "G2", "G0", "G0"]


def test_parse_transcription_file_not_found():
    """Verify FileNotFoundError on non-existent file."""
    with pytest.raises(FileNotFoundError):
        parse_transcription("dataset/non_existent_transcription.txt")


def test_parse_transcription_malformed_intervals(tmp_path):
    """Verify ValueError is raised if start_frame > end_frame."""
    bad_file = tmp_path / "inverted_interval.txt"
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write("500 200 G1\n")
    with pytest.raises(ValueError, match="Invalid interval"):
        parse_transcription(bad_file)


def test_audit_all_103_transcriptions(tmp_path, manifest_df):
    """Verify audit over all 103 transcriptions across the 3 tasks."""
    audit_file = tmp_path / "class_counts.json"
    summary = audit_all_transcriptions(manifest_df, audit_file)
    assert summary["total_trials"] == 103
    assert summary["total_annotated_segments"] > 1000
    assert audit_file.is_file()

    # Verify that all expected gestures are observed in each task
    for task, coverage in summary["per_task_vocabulary_coverage"].items():
        assert coverage["all_expected_observed"] is True, f"Missing gesture classes in {task}"
