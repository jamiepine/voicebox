"""
Custom voice model management module.

Handles adding, removing, and listing user-defined HuggingFace TTS models.
Models are persisted in a JSON config file in the data directory.

@author AJ - Kamyab (Ankit Jain)
"""

import fcntl
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import config

logger = logging.getLogger(__name__)

# Module-level lock to serialise in-process reads/writes to the config file.
_config_lock = threading.Lock()


def _get_config_path() -> Path:
    """Get path to the custom models JSON config file."""
    return config.get_data_dir() / "custom_models.json"


def _load_config() -> dict:
    """Load custom models config from disk.

    On IOError the file is simply missing — return an empty config.
    On JSONDecodeError the file is corrupt — back it up, log the error,
    and re-raise so callers do not accidentally overwrite it.
    """
    path = _get_config_path()
    if not path.exists():
        return {"models": []}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        # Back up the corrupt file so we don't lose data
        backup = path.with_suffix(
            f".json.corrupt.{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            path.rename(backup)
            logger.error(
                "Corrupt custom_models.json backed up to %s: %s", backup, exc
            )
        except OSError as rename_err:
            logger.error(
                "Failed to back up corrupt config %s: %s (original error: %s)",
                path, rename_err, exc,
            )
        raise
    except IOError:
        return {"models": []}


def _save_config(data: dict) -> None:
    """Save custom models config to disk atomically.

    Writes to a temp file in the same directory, fsyncs, then atomically
    replaces the original via os.replace.  The caller MUST hold _config_lock.
    """
    path = _get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd_int, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".custom_models_"
        )
        fd = os.fdopen(fd_int, "w")
        json.dump(data, fd, indent=2, default=str)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        fd = None  # prevent double-close
        os.replace(tmp_path, str(path))
        tmp_path = None  # prevent cleanup
    finally:
        if fd is not None:
            fd.close()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# Regex for valid HuggingFace repo IDs: owner/repo where each segment is
# non-empty and contains only alphanumeric characters, dots, underscores,
# and hyphens.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


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
    with _config_lock:
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
    if not _HF_REPO_RE.match(hf_repo_id):
        raise ValueError(
            "HuggingFace repo ID must be in format 'owner/model-name' "
            "(alphanumeric, dots, underscores, and hyphens only, no leading/trailing slashes)"
        )
    
    model_id = _generate_id(hf_repo_id)

    with _config_lock:
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
    with _config_lock:
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
