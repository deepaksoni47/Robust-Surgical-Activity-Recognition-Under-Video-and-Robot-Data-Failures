import random
from pathlib import Path
import numpy as np
import pytest
import torch

from src.utils.reproducibility import get_device, load_config, load_json, save_json, set_seed


def test_set_seed_deterministic_python():
    """Verify Python random produces identical values with set_seed."""
    set_seed(42)
    val1 = [random.random() for _ in range(5)]
    set_seed(42)
    val2 = [random.random() for _ in range(5)]
    assert val1 == val2


def test_set_seed_deterministic_numpy():
    """Verify NumPy produces identical arrays with set_seed."""
    set_seed(123)
    arr1 = np.random.randn(4, 4)
    set_seed(123)
    arr2 = np.random.randn(4, 4)
    np.testing.assert_array_equal(arr1, arr2)


def test_set_seed_deterministic_torch():
    """Verify PyTorch produces identical tensors with set_seed."""
    set_seed(999)
    t1 = torch.randn(3, 5)
    set_seed(999)
    t2 = torch.randn(3, 5)
    assert torch.equal(t1, t2)


def test_load_config_valid():
    """Verify load_config correctly parses existing YAML configuration."""
    config_path = Path("configs/data.yaml")
    config = load_config(config_path)
    assert "dataset" in config
    assert config["dataset"]["total_trials"] == 103
    assert config["dataset"]["split_protocol"] == "UserOut"
    assert len(config["dataset"]["tasks"]) == 3


def test_load_config_not_found():
    """Verify load_config raises FileNotFoundError for nonexistent paths."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_config.yaml")


def test_save_load_json(tmp_path):
    """Verify save_json and load_json roundtrip."""
    sample_data = {
        "project": "Multimodal Surgical Gesture",
        "tasks": ["Suturing", "Knot_Tying", "Needle_Passing"],
        "iterations": 26,
    }
    json_file = tmp_path / "artifacts" / "test_manifest.json"
    saved_path = save_json(sample_data, json_file)
    assert saved_path.exists()
    
    loaded = load_json(json_file)
    assert loaded == sample_data


def test_get_device():
    """Verify get_device returns a valid torch.device instance."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
