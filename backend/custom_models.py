"""
Custom voice model management module.

Handles adding, removing, and listing user-defined HuggingFace TTS models.
Models are persisted in a JSON config file in the data directory.

@author AJ - Kamyab (Ankit Jain)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import config


def _get_config_path() -> Path:
    """Get path to the custom models JSON config file."""
    return config.get_data_dir() / "custom_models.json"


def _load_config() -> dict:
    """Load custom models config from disk."""
    path = _get_config_path()
    if not path.exists():
        return {"models": []}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"models": []}


def _save_config(data: dict) -> None:
    """Save custom models config to disk."""
    path = _get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _generate_id(hf_repo_id: str) -> str:
    """Generate a slug ID from a HuggingFace repo ID.
    
    Example: 'AryanNsc/IND-QWENTTS-V1' -> 'aryansc-ind-qwentts-v1'
    """
    slug = hf_repo_id.lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9-]", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def list_custom_models() -> List[dict]:
    """List all custom models.
    
    Returns:
        List of custom model dicts
    """
    data = _load_config()
    return data.get("models", [])


def get_custom_model(model_id: str) -> Optional[dict]:
    """Get a single custom model by ID.
    
    Args:
        model_id: Custom model ID (slug)
        
    Returns:
        Model dict or None if not found
    """
    models = list_custom_models()
    for model in models:
        if model["id"] == model_id:
            return model
    return None


def add_custom_model(hf_repo_id: str, display_name: str) -> dict:
    """Add a new custom model.
    
    Args:
        hf_repo_id: HuggingFace repo ID (e.g. 'AryanNsc/IND-QWENTTS-V1')
        display_name: User-friendly display name
        
    Returns:
        Created model dict
        
    Raises:
        ValueError: If model already exists or inputs are invalid
    """
    hf_repo_id = hf_repo_id.strip()
    display_name = display_name.strip()
    
    if not hf_repo_id:
        raise ValueError("HuggingFace repo ID is required")
    if not display_name:
        raise ValueError("Display name is required")
    if "/" not in hf_repo_id:
        raise ValueError("HuggingFace repo ID must be in format 'owner/model-name'")
    
    model_id = _generate_id(hf_repo_id)
    
    data = _load_config()
    models = data.get("models", [])
    
    # Check for duplicates
    for existing in models:
        if existing["id"] == model_id:
            raise ValueError(f"Model '{hf_repo_id}' already exists")
        if existing["hf_repo_id"] == hf_repo_id:
            raise ValueError(f"Model with repo ID '{hf_repo_id}' already exists")
    
    model = {
        "id": model_id,
        "display_name": display_name,
        "hf_repo_id": hf_repo_id,
        "added_at": datetime.utcnow().isoformat() + "Z",
    }
    
    models.append(model)
    data["models"] = models
    _save_config(data)
    
    return model


def remove_custom_model(model_id: str) -> bool:
    """Remove a custom model by ID.
    
    Args:
        model_id: Custom model ID (slug)
        
    Returns:
        True if removed, False if not found
    """
    data = _load_config()
    models = data.get("models", [])
    
    original_count = len(models)
    models = [m for m in models if m["id"] != model_id]
    
    if len(models) == original_count:
        return False
    
    data["models"] = models
    _save_config(data)
    return True


def get_hf_repo_id_for_custom_model(model_id: str) -> Optional[str]:
    """Get the HuggingFace repo ID for a custom model.
    
    Args:
        model_id: Custom model ID (slug, without 'custom:' prefix)
        
    Returns:
        HuggingFace repo ID or None if not found
    """
    model = get_custom_model(model_id)
    if model:
        return model["hf_repo_id"]
    return None
