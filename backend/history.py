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

from .models import GenerationRequest, GenerationResponse, HistoryQuery, HistoryResponse, HistoryListResponse
from .database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from . import config


def _get_generations_dir() -> Path:
    """Get generations directory from config."""
    return config.get_generations_dir()


async def create_generation(
    profile_id: str,
    text: str,
    language: str,
    audio_path: str,
    duration: float,
    seed: Optional[int],
    db: Session,
    instruct: Optional[str] = None,
    model_size: Optional[str] = None,
    backend_type: Optional[str] = None,
    request_user_id: Optional[str] = None,
    request_user_first_name: Optional[str] = None,
    request_ip: Optional[str] = None,
    generation_time_seconds: Optional[float] = None,
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

    Returns:
        Created generation entry
    """
    db_generation = DBGeneration(
        id=str(uuid.uuid4()),
        profile_id=profile_id,
        text=text,
        language=language,
        audio_path=audio_path,
        duration=duration,
        generation_time_seconds=generation_time_seconds,
        seed=seed,
        instruct=instruct,
        model_size=model_size,
        backend_type=backend_type,
        request_user_id=request_user_id,
        request_user_first_name=request_user_first_name,
        request_ip=request_ip,
        created_at=datetime.utcnow(),
    )

    db.add(db_generation)
    db.commit()
    db.refresh(db_generation)

    return GenerationResponse.model_validate(db_generation)


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
    if generation.deleted_at is not None:
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
    ).filter(
        DBGeneration.deleted_at.is_(None)
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
        items.append(HistoryResponse(
            id=generation.id,
            profile_id=generation.profile_id,
            profile_name=profile_name,
            text=generation.text,
            language=generation.language,
            audio_path=generation.audio_path,
            duration=generation.duration,
            generation_time_seconds=generation.generation_time_seconds,
            seed=generation.seed,
            instruct=generation.instruct,
            model_size=generation.model_size,
            backend_type=generation.backend_type,
            request_user_id=generation.request_user_id,
            request_user_first_name=generation.request_user_first_name,
            request_ip=generation.request_ip,
            created_at=generation.created_at,
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
    if generation.deleted_at is not None:
        return False
    
    # Delete audio file
    audio_path = Path(generation.audio_path)
    if audio_path.exists():
        audio_path.unlink()
    
    # Soft-delete in database (keep record for auditing/restore workflows).
    generation.deleted_at = datetime.utcnow()
    db.commit()
    
    return True


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
        # Delete audio file
        audio_path = Path(generation.audio_path)
        if audio_path.exists():
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
