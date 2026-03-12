"""
Configuration module for voicebox backend.

Handles data directory configuration for production bundling.
"""

import json
import os
from pathlib import Path

# Allow users to override the HuggingFace model download directory.
# Set VOICEBOX_MODELS_DIR to an absolute path before starting the server.
# This sets HF_HUB_CACHE so all huggingface_hub downloads go to that path.
_custom_models_dir = os.environ.get("VOICEBOX_MODELS_DIR")
if _custom_models_dir:
    os.environ["HF_HUB_CACHE"] = _custom_models_dir
    print(f"[config] Model download path set to: {_custom_models_dir}")

# Default data directory (used in development)
_data_dir = Path("data")

def set_data_dir(path: str | Path):
    """
    Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _data_dir
    _data_dir = Path(path)
    _data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory set to: {_data_dir.absolute()}")

def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path to the data directory
    """
    return _data_dir

def get_db_path() -> Path:
    """Get database file path."""
    return _data_dir / "voicebox.db"

def get_profiles_dir() -> Path:
    """Get profiles directory path."""
    path = _data_dir / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_generations_dir() -> Path:
    """Get generations directory path."""
    path = _data_dir / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_cache_dir() -> Path:
    """Get cache directory path."""
    path = _data_dir / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir() -> Path:
    """Get models directory path."""
    path = _data_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_settings_path() -> Path:
    """Get app settings file path."""
    return _data_dir / "settings.json"

def load_app_settings() -> dict:
    """Load app settings from JSON file. Returns empty dict if file not found."""
    path = get_settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_app_settings(data: dict) -> None:
    """Save app settings to JSON file."""
    path = get_settings_path()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
