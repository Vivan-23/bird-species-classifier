"""Utility functions for reproducibility, configuration, device management, and checkpoints.
"""

import os
import json
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_setting: str = "auto") -> torch.device:
    """Detect and return appropriate torch.device based on availability and config setting."""
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU acceleration: {device_name}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU execution.")
    return device


def save_checkpoint(checkpoint: dict, file_path: str) -> None:
    """Save PyTorch checkpoint safely."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(checkpoint, file_path)
    print(f"[Checkpoint] Successfully saved checkpoint to {file_path}")


def load_checkpoint(file_path: str, device: torch.device = None) -> dict:
    """Load PyTorch checkpoint safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {file_path}")
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    return checkpoint


def save_json(data: dict, file_path: str) -> None:
    """Save dictionary to formatted JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[JSON] Saved data to {file_path}")


def load_json(file_path: str) -> dict:
    """Load dictionary from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
