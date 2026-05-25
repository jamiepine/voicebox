"""
Thread-safe CRUD for user-defined HuggingFace TTS models.

Persists custom model entries as a JSON file inside the configured data
directory.  All reads and writes are serialised through a module-level
``threading.Lock`` and writes use atomic rename (``os.replace``) so a
crash mid-write can never leave a half-written config file.
"""

import json
import logging
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# HuggingFace repo ID pattern: owner/model
_HF_REPO_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def _config_path() -> Path:
    """Return the path to the custom models JSON file."""
    return config.get_data_dir() / "custom_models.json"


def _read() -> list[dict]:
    """Read the custom models list from disk.

    On ``json.JSONDecodeError`` the corrupt file is backed up with a
    timestamped suffix, the error is logged, and the exception is
    re-raised so callers can decide how to proceed.
    """
    path = _config_path()
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_suffix(f".json.corrupt.{ts}")
        path.rename(backup)
        logger.error(
            "Corrupt custom_models.json — backed up to %s",
            backup,
        )
        raise

    if not isinstance(data, list):
        return []
    return data


def _write(entries: list[dict]) -> None:
    """Atomically write the custom models list to disk."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix="custom_models_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _slug(hf_repo_id: str) -> str:
    """Convert a HuggingFace repo ID to a filesystem-safe slug."""
    return hf_repo_id.replace("/", "--")


# ── Public API ────────────────────────────────────────────────────────


def list_custom_models() -> list[dict]:
    """Return all custom model entries."""
    with _lock:
        return _read()


def add_custom_model(
    hf_repo_id: str,
    display_name: str | None = None,
    engine: str | None = None,
) -> dict:
    """Add a new custom HuggingFace model definition.

    Args:
        hf_repo_id: Full HuggingFace repository ID (e.g. ``owner/model``).
        display_name: Human-friendly label; defaults to *hf_repo_id*.
        engine: Optional engine hint.

    Returns:
        The newly created entry dict.

    Raises:
        ValueError: If the repo ID is malformed or already registered.
    """
    if not _HF_REPO_RE.match(hf_repo_id):
        raise ValueError(
            f"Invalid HuggingFace repo ID: '{hf_repo_id}'. "
            "Expected format: owner/model"
        )

    model_id = _slug(hf_repo_id)

    with _lock:
        entries = _read()

        if any(e.get("id") == model_id for e in entries):
            raise ValueError(
                f"Custom model '{hf_repo_id}' is already registered"
            )

        entry = {
            "id": model_id,
            "hf_repo_id": hf_repo_id,
            "display_name": display_name or hf_repo_id,
            "engine": engine,
            "created_at": datetime.now(UTC).isoformat(),
        }
        entries.append(entry)
        _write(entries)

    return entry


def get_custom_model(model_id: str) -> dict | None:
    """Look up a custom model by its slug id.

    Returns:
        The entry dict, or ``None`` if not found.
    """
    with _lock:
        entries = _read()
    return next((e for e in entries if e.get("id") == model_id), None)


def delete_custom_model(model_id: str) -> bool:
    """Delete a custom model entry by id.

    Returns:
        ``True`` if the entry was found and removed, ``False`` otherwise.
    """
    with _lock:
        entries = _read()
        before = len(entries)
        entries = [e for e in entries if e.get("id") != model_id]
        if len(entries) == before:
            return False
        _write(entries)
        return True
