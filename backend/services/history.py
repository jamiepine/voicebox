"""
Generation history management module.
"""

from typing import List, Optional, Tuple
from datetime import datetime
import uuid
import shutil
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import or_

from ..models import GenerationRequest, GenerationResponse, HistoryQuery, HistoryResponse, HistoryListResponse, GenerationVersionResponse, EffectConfig
from ..database import Generation as DBGeneration, GenerationVersion as DBGenerationVersion, VoiceProfile as DBVoiceProfile
from .. import config


def _get_versions_for_generation(generation_id: str, db: Session) -> tuple:
    """Get versions list and active version ID for a generation."""
    import json
    versions_rows = (
        db.query(DBGenerationVersion)
        .filter_by(generation_id=generation_id)
        .order_by(DBGenerationVersion.created_at)
        .all()
    )
    if not versions_rows:
        return None, None

    versions = []
    active_version_id = None
    for v in versions_rows:
        effects_chain = None
        if v.effects_chain:
            try:
                raw = json.loads(v.effects_chain)
                effects_chain = [EffectConfig(**e) for e in raw]
            except Exception:
                pass
        versions.append(GenerationVersionResponse(
            id=v.id,
            generation_id=v.generation_id,
            label=v.label,
            audio_path=v.audio_path,
            effects_chain=effects_chain,
            is_default=v.is_default,
            created_at=v.created_at,
        ))
        if v.is_default:
            active_version_id = v.id

    return versions, active_version_id


async def create_generation(
    profile_id: str,
    text: str,
    language: str,
    audio_path: str,
    duration: float,
    seed: Optional[int],
    db: Session,
    instruct: Optional[str] = None,
    generation_id: Optional[str] = None,
    status: str = "completed",
    engine: Optional[str] = "qwen",
    model_size: Optional[str] = None,
    source: str = "manual",
) -> GenerationResponse:
    """
    Create a new generation history entry.

    Args:
        profile_id: Profile ID used for generation
        text: Generated text
        language: Language code
        audio_path: Path where audio was saved
        duration: Audio duration in seconds
        seed: Random seed used (if any)
        db: Database session
        instruct: Natural language instruction used (if any)
        generation_id: Pre-assigned ID (for async generation flow)
        status: Generation status (generating, completed, failed)
        engine: TTS engine used (qwen, luxtts, chatterbox, chatterbox_turbo)
        model_size: Model size variant (1.7B, 0.6B) — only relevant for qwen
        source: Origin marker stored on the row. ``"manual"`` for regular
            /generate calls; ``"personality_speak"`` for rows created
            by the /profiles/{id}/speak endpoint. Enables filtering the
            history view for personality-driven output.

    Returns:
        Created generation entry
    """
    db_generation = DBGeneration(
        id=generation_id or str(uuid.uuid4()),
        profile_id=profile_id,
        text=text,
        language=language,
        audio_path=audio_path,
        duration=duration,
        seed=seed,
        instruct=instruct,
        engine=engine,
        model_size=model_size,
        status=status,
        source=source,
        created_at=datetime.utcnow(),
    )

    db.add(db_generation)
    db.commit()
    db.refresh(db_generation)

    return GenerationResponse.model_validate(db_generation)


async def update_generation_status(
    generation_id: str,
    status: str,
    db: Session,
    audio_path: Optional[str] = None,
    duration: Optional[float] = None,
    error: Optional[str] = None,
) -> Optional[GenerationResponse]:
    """Update the status of a generation (used by async generation flow)."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return None

    generation.status = status
    if audio_path is not None:
        generation.audio_path = audio_path
    if duration is not None:
        generation.duration = duration
    if error is not None:
        generation.error = error

    db.commit()
    db.refresh(generation)
    return GenerationResponse.model_validate(generation)


async def get_generation(
    generation_id: str,
    db: Session,
) -> Optional[GenerationResponse]:
    """
    Get a generation by ID.
    
    Args:
        generation_id: Generation ID
        db: Database session
        
    Returns:
        Generation or None if not found
    """
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return None
    
    return GenerationResponse.model_validate(generation)


async def list_generations(
    query: HistoryQuery,
    db: Session,
) -> HistoryListResponse:
    """
    List generations with optional filters.
    
    Args:
        query: Query parameters (filters, pagination)
        db: Database session
        
    Returns:
        HistoryListResponse with items and total count
    """
    # Build base query with join to get profile name
    q = db.query(
        DBGeneration,
        DBVoiceProfile.name.label('profile_name')
    ).join(
        DBVoiceProfile,
        DBGeneration.profile_id == DBVoiceProfile.id
    )
    
    # Apply profile filter
    if query.profile_id:
        q = q.filter(DBGeneration.profile_id == query.profile_id)
    
    # Apply search filter (searches in text content)
    if query.search:
        search_pattern = f"%{query.search}%"
        q = q.filter(DBGeneration.text.like(search_pattern))
    
    # Get total count before pagination
    total_count = q.count()
    
    # Apply ordering (newest first)
    q = q.order_by(DBGeneration.created_at.desc())
    
    # Apply pagination
    q = q.offset(query.offset).limit(query.limit)
    
    # Execute query
    results = q.all()
    
    # Convert to HistoryResponse with profile_name
    items = []
    for generation, profile_name in results:
        versions, active_version_id = _get_versions_for_generation(generation.id, db)
        items.append(HistoryResponse(
            id=generation.id,
            profile_id=generation.profile_id,
            profile_name=profile_name,
            text=generation.text,
            language=generation.language,
            audio_path=generation.audio_path,
            duration=generation.duration,
            seed=generation.seed,
            instruct=generation.instruct,
            engine=generation.engine or "qwen",
            model_size=generation.model_size,
            status=generation.status or "completed",
            error=generation.error,
            is_favorited=bool(generation.is_favorited),
            created_at=generation.created_at,
            versions=versions,
            active_version_id=active_version_id,
        ))
    
    return HistoryListResponse(
        items=items,
        total=total_count,
    )


async def delete_generation(
    generation_id: str,
    db: Session,
) -> bool:
    """
    Delete a generation.
    
    Args:
        generation_id: Generation ID
        db: Database session
        
    Returns:
        True if deleted, False if not found
    """
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return False

    # Delete all version files and records
    from . import versions as versions_mod
    versions_mod.delete_versions_for_generation(generation_id, db)

    # Delete main audio file (if not already removed by version cleanup)
    if generation.audio_path:
        audio_path = config.resolve_storage_path(generation.audio_path)
        if audio_path is not None and audio_path.exists():
            audio_path.unlink()

    # Delete from database
    db.delete(generation)
    db.commit()
    
    return True


async def delete_failed_generations(db: Session) -> int:
    """
    Delete every generation whose status is 'failed'.

    Used by the "Clear failed" action in the UI so users can tidy up
    history after the model wasn't loaded, the app was closed mid-run,
    or a generation otherwise errored out (see issue #410).

    Returns:
        Number of generations deleted.
    """
    from . import versions as versions_mod

    failed = db.query(DBGeneration).filter(DBGeneration.status == "failed").all()
    count = 0
    for generation in failed:
        # Clean up version files/rows first.
        versions_mod.delete_versions_for_generation(generation.id, db)

        # Remove the main audio file if it somehow made it to disk.
        if generation.audio_path:
            audio_path = config.resolve_storage_path(generation.audio_path)
            if audio_path is not None and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    # Best-effort cleanup — don't abort the whole sweep
                    # if a single file can't be removed.
                    pass

        db.delete(generation)
        count += 1

    db.commit()
    return count


async def delete_generations_by_profile(
    profile_id: str,
    db: Session,
) -> int:
    """
    Delete all generations for a profile.
    
    Args:
        profile_id: Profile ID
        db: Database session
        
    Returns:
        Number of generations deleted
    """
    generations = db.query(DBGeneration).filter_by(profile_id=profile_id).all()
    
    count = 0
    for generation in generations:
        # Delete associated version files and rows first
        from . import versions as versions_mod
        versions_mod.delete_versions_for_generation(generation.id, db)

        # Delete audio file
        audio_path = config.resolve_storage_path(generation.audio_path)
        if audio_path is not None and audio_path.exists():
            audio_path.unlink()
        
        # Delete from database
        db.delete(generation)
        count += 1
    
    db.commit()
    
    return count


async def get_generation_stats(db: Session) -> dict:
    """
    Get generation statistics.
    
    Args:
        db: Database session
        
    Returns:
        Statistics dictionary
    """
    from sqlalchemy import func
    
    total = db.query(func.count(DBGeneration.id)).scalar()
    
    total_duration = db.query(func.sum(DBGeneration.duration)).scalar() or 0
    
    # Get generations by profile
    by_profile = db.query(
        DBGeneration.profile_id,
        func.count(DBGeneration.id).label('count')
    ).group_by(DBGeneration.profile_id).all()
    
    return {
        "total_generations": total,
        "total_duration_seconds": total_duration,
        "generations_by_profile": {
            profile_id: count for profile_id, count in by_profile
        },
    }
