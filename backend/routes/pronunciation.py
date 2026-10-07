"""Pronunciation dictionary endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models
from ..database import PronunciationEntry, VoiceProfile as DBVoiceProfile, get_db
from ..services import pronunciation

logger = logging.getLogger(__name__)

router = APIRouter()


_SCOPE_TAKEN = (
    "That scope already has an entry for this term. Update the existing entry instead."
)


def _commit_or_conflict(db: Session) -> None:
    """Commit, turning a scope collision into a 409.

    ``find_duplicate`` runs first and gives a friendlier message naming the
    existing row, but it is a check-then-act pair: two concurrent creates can
    both pass it. ``uq_pronunciation_scope`` is what actually holds, so its
    violation has to surface as a conflict rather than a 500.
    """
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if "uq_pronunciation_scope" in str(exc.orig):
            raise HTTPException(status_code=409, detail=_SCOPE_TAKEN) from exc
        raise


def _validate_profile(profile_id: str | None, db: Session) -> None:
    if profile_id is None:
        return
    if db.query(DBVoiceProfile).filter_by(id=profile_id).first() is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")


@router.get("/pronunciations", response_model=list[models.PronunciationEntryResponse])
async def list_pronunciations(
    language: str | None = Query(None, description="Filter to entries that apply to this language"),
    profile_id: str | None = Query(None, description="Filter to entries that apply to this voice"),
    include_disabled: bool = Query(True),
    db: Session = Depends(get_db),
):
    """List dictionary entries.

    With no filters this returns everything, which is what a management screen
    wants. Passing ``language`` or ``profile_id`` narrows it to what would
    actually apply to a generation with those settings.
    """
    if language is None and profile_id is None:
        q = db.query(PronunciationEntry)
        if not include_disabled:
            q = q.filter(PronunciationEntry.enabled.is_(True))
        return q.order_by(PronunciationEntry.term).all()

    return pronunciation.get_entries(
        db, language=language, profile_id=profile_id, include_disabled=include_disabled
    )


@router.post("/pronunciations", response_model=models.PronunciationEntryResponse)
async def create_pronunciation(
    data: models.PronunciationEntryCreate,
    db: Session = Depends(get_db),
):
    """Add a term and how to say it."""
    _validate_profile(data.profile_id, db)

    existing = pronunciation.find_duplicate(db, data.term, data.language, data.profile_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An entry for '{data.term}' already exists in this scope "
                f"(id {existing.id}). Update it instead."
            ),
        )

    entry = PronunciationEntry(
        term=data.term,
        replacement=data.replacement,
        language=data.language,
        profile_id=data.profile_id,
        enabled=data.enabled,
        notes=data.notes,
    )
    db.add(entry)
    _commit_or_conflict(db)
    db.refresh(entry)
    return entry


@router.put("/pronunciations/{entry_id}", response_model=models.PronunciationEntryResponse)
async def update_pronunciation(
    entry_id: str,
    data: models.PronunciationEntryUpdate,
    db: Session = Depends(get_db),
):
    """Update an entry. Omitted fields are left as they are."""
    entry = db.query(PronunciationEntry).filter_by(id=entry_id).first()
    if entry is None:
        raise HTTPException(status_code=404, detail="Pronunciation entry not found")

    fields = data.model_dump(exclude_unset=True)
    if "profile_id" in fields:
        _validate_profile(fields["profile_id"], db)

    # Re-check the scope only when something that defines it moved.
    if {"term", "language", "profile_id"} & fields.keys():
        clash = pronunciation.find_duplicate(
            db,
            fields.get("term", entry.term),
            fields.get("language", entry.language),
            fields.get("profile_id", entry.profile_id),
            exclude_id=entry_id,
        )
        if clash is not None:
            raise HTTPException(
                status_code=409,
                detail=f"That scope already has an entry for this term (id {clash.id}).",
            )

    for key, value in fields.items():
        setattr(entry, key, value)

    _commit_or_conflict(db)
    db.refresh(entry)
    return entry


@router.delete("/pronunciations/{entry_id}")
async def delete_pronunciation(entry_id: str, db: Session = Depends(get_db)):
    """Delete an entry."""
    entry = db.query(PronunciationEntry).filter_by(id=entry_id).first()
    if entry is None:
        raise HTTPException(status_code=404, detail="Pronunciation entry not found")
    db.delete(entry)
    db.commit()
    return {"message": "Pronunciation entry deleted"}


@router.post("/pronunciations/preview", response_model=models.PronunciationPreviewResponse)
async def preview_pronunciations(
    data: models.PronunciationPreviewRequest,
    db: Session = Depends(get_db),
):
    """Show what the engine would be given for this text.

    The dictionary runs at generation time and the rewritten text is never
    stored, so without this there is no way to see what a rule actually does
    short of listening to the output.
    """
    _validate_profile(data.profile_id, db)
    result, applied = pronunciation.apply_pronunciations(
        data.text, data.language, db, profile_id=data.profile_id
    )
    return {"original": data.text, "result": result, "applied": applied}
