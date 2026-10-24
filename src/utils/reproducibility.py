import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> int:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is empty or invalid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if config is None:
        raise ValueError(f"Configuration file is empty: {path.resolve()}")
    if not isinstance(config, dict):
        raise ValueError(f"YAML configuration must parse to a dictionary, got {type(config).__name__}")
        
    return config


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> Path:
    """Save serializable data to a JSON file, creating parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def load_json(file_path: str | Path) -> Any:
    """Load data from a JSON file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Return torch.device('cuda') if available, otherwise torch.device('cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
