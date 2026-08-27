"""Persisted model-download-source setting (HuggingFace / ModelScope).

Stored as a flat JSON file under the data directory (simpler than a
database row for a single string, with no schema/migration to maintain).
Read fresh on every ``resolve_model_source()`` call — a change applies to
the very next download, no restart or process-wide env-var mutation needed.
See specs/001-modelscope-download-source/research.md §4.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

VALID_SOURCES = ("huggingface", "modelscope")
DEFAULT_SOURCE = "huggingface"

_SETTINGS_FILENAME = "model_source.json"


def _settings_path():
    # Deferred import: avoids importing backend.config (and everything it
    # pulls in) unless a caller actually needs the data dir.
    from ..config import get_data_dir

    return get_data_dir() / _SETTINGS_FILENAME


def get_model_source() -> str:
    """Return the persisted download source, defaulting to HuggingFace."""
    path = _settings_path()
    if not path.exists():
        return DEFAULT_SOURCE

    try:
        data = json.loads(path.read_text())
        source = data.get("source")
    except (json.JSONDecodeError, OSError):
        logger.warning("model_source.json is unreadable — defaulting to %s", DEFAULT_SOURCE)
        return DEFAULT_SOURCE

    if source not in VALID_SOURCES:
        logger.warning("model_source.json has an unknown source %r — defaulting to %s", source, DEFAULT_SOURCE)
        return DEFAULT_SOURCE

    return source


def set_model_source(source: str) -> None:
    """Persist a new download source. Raises ValueError for an unknown value.

    Writes to a temp file and renames it into place (atomic on POSIX and
    Windows) instead of truncating the target in place — a concurrent
    reader, or a write that fails partway, must never see invalid JSON and
    silently fall back to the default, masking the user's actual setting.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown model source: {source!r}. Must be one of {VALID_SOURCES}.")

    path = _settings_path()
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps({"source": source}))
    os.replace(tmp_path, path)
