"""
Server-side user settings — singleton rows persisted in SQLite so every
client window, API consumer, and headless flow reads the same preferences.

Two domains live here: capture/refine defaults and long-form generation
defaults. Each has a ``get_*`` that lazily creates the row with defaults and
an ``update_*`` that accepts a partial payload.
"""

from typing import Any

from sqlalchemy.orm import Session

from ..database import CaptureSettings as DBCaptureSettings
from ..database import GenerationSettings as DBGenerationSettings
from ..utils.capture_chords import (
    default_push_to_talk_chord,
    default_toggle_to_talk_chord,
)


SINGLETON_ID = 1


def _get_or_create_capture_row(db: Session) -> DBCaptureSettings:
    """Fetch the singleton capture-settings row, lazily creating it on first read.

    Every setter path is a partial update against this row, so callers
    can assume the returned object exists and holds the schema defaults
    for any field the user hasn't customised yet.
    """
    row = db.query(DBCaptureSettings).filter(DBCaptureSettings.id == SINGLETON_ID).first()
    if row is None:
        row = DBCaptureSettings(
            id=SINGLETON_ID,
            chord_push_to_talk_keys=default_push_to_talk_chord(),
            chord_toggle_to_talk_keys=default_toggle_to_talk_chord(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def _get_or_create_generation_row(db: Session) -> DBGenerationSettings:
    """Fetch the singleton generation-settings row, lazily creating it on first read.

    Symmetric counterpart to :func:`_get_or_create_capture_row` for the
    long-form generation defaults (chunk sizer, crossfade, normalize).
    """
    row = db.query(DBGenerationSettings).filter(DBGenerationSettings.id == SINGLETON_ID).first()
    if row is None:
        row = DBGenerationSettings(id=SINGLETON_ID)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def _apply_patch(row: Any, patch: dict[str, Any]) -> None:
    """Apply a partial update to a settings row.

    Values explicitly set to ``None`` are honored only for columns where the
    schema allows it — clearing ``default_playback_voice_id`` works, but a
    ``None`` for a non-nullable field is dropped rather than crashing the
    request. Unknown keys are ignored.
    """
    columns = type(row).__table__.columns
    for key, value in patch.items():
        col = columns.get(key)
        if col is None:
            continue
        if value is None and not col.nullable:
            continue
        setattr(row, key, value)


def get_capture_settings(db: Session) -> DBCaptureSettings:
    """Return the capture settings row, creating it with defaults if missing."""
    return _get_or_create_capture_row(db)


def update_capture_settings(db: Session, patch: dict[str, Any]) -> DBCaptureSettings:
    """Apply a partial update to the capture-settings singleton and persist it.

    After the commit, propagates the (possibly changed) custom-LLM
    endpoint into the backend module state via
    :func:`_sync_llm_backend_config` so subsequent refinement /
    personality calls route to the new URL without needing a restart.
    """
    row = _get_or_create_capture_row(db)
    _apply_patch(row, patch)
    db.commit()
    db.refresh(row)
    _sync_llm_backend_config(row)
    return row


def _sync_llm_backend_config(row: DBCaptureSettings) -> None:
    """Push the persisted custom-LLM endpoint into the backend module state.

    Called after every capture-settings write and once on startup from
    ``bootstrap_llm_backend_config``. Keeps the runtime dispatch in
    ``backends.get_llm_backend()`` in sync with what the DB row holds so
    the user's next refinement / personality call routes to the new URL
    without a restart.
    """
    # Imported inline to avoid a circular import at module load — the
    # backends package pulls in HF patches that in turn import services.
    from .. import backends

    backends.set_llm_config(
        endpoint=row.custom_llm_endpoint,
        model=row.custom_llm_model,
        api_key=row.custom_llm_api_key,
    )


def bootstrap_llm_backend_config(db: Session) -> None:
    """Seed the backend LLM config from the persisted row on server startup."""
    row = _get_or_create_capture_row(db)
    _sync_llm_backend_config(row)


def get_generation_settings(db: Session) -> DBGenerationSettings:
    """Return the generation settings row, creating it with defaults if missing."""
    return _get_or_create_generation_row(db)


def update_generation_settings(db: Session, patch: dict[str, Any]) -> DBGenerationSettings:
    """Apply a partial update to the generation-settings singleton and persist it.

    Unlike :func:`update_capture_settings`, no runtime state depends on
    these values — the fields flow directly into the next
    ``run_generation`` invocation via ``GenerationSettings`` — so the
    commit is enough.
    """
    row = _get_or_create_generation_row(db)
    _apply_patch(row, patch)
    db.commit()
    db.refresh(row)
    return row
