"""
Voice profile management module.
"""

from typing import List, Optional
from datetime import datetime
import uuid
import shutil
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import (
    VoiceProfileCreate,
    VoiceProfileResponse,
    ProfileSampleCreate,
    ProfileSampleResponse,
)
from .database import (
    VoiceProfile as DBVoiceProfile,
    ProfileSample as DBProfileSample,
)
from .utils.audio import validate_reference_audio, load_audio, save_audio
from .tts import get_tts_model
from . import config


def _get_profiles_dir() -> Path:
    """Get profiles directory from config."""
    return config.get_profiles_dir()


async def create_profile(
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse:
    """
    Create a new voice profile.
    
    Args:
        data: Profile creation data
        db: Database session
        
    Returns:
        Created profile
    """
    # Create profile in database
    db_profile = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        language=data.language,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    # Create profile directory
    profile_dir = _get_profiles_dir() / db_profile.id
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    return VoiceProfileResponse.model_validate(db_profile)


async def add_profile_sample(
    profile_id: str,
    audio_path: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse:
    """
    Add a sample to a voice profile.
    
    Args:
        profile_id: Profile ID
        audio_path: Path to temporary audio file
        reference_text: Transcript of audio
        db: Database session
        
    Returns:
        Created sample
    """
    # Validate profile exists
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")
    
    # Validate audio
    is_valid, error_msg = validate_reference_audio(audio_path)
    if not is_valid:
        raise ValueError(f"Invalid reference audio: {error_msg}")
    
    # Create sample ID and directory
    sample_id = str(uuid.uuid4())
    profile_dir = _get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy audio file to profile directory
    dest_path = profile_dir / f"{sample_id}.wav"
    audio, sr = load_audio(audio_path)
    save_audio(audio, str(dest_path), sr)
    
    # Create database entry
    db_sample = DBProfileSample(
        id=sample_id,
        profile_id=profile_id,
        audio_path=str(dest_path),
        reference_text=reference_text,
    )
    
    db.add(db_sample)
    
    # Update profile timestamp
    profile.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_sample)
    
    return ProfileSampleResponse.model_validate(db_sample)


async def get_profile(
    profile_id: str,
    db: Session,
) -> Optional[VoiceProfileResponse]:
    """
    Get a voice profile by ID.
    
    Args:
        profile_id: Profile ID
        db: Database session
        
    Returns:
        Profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None
    
    return VoiceProfileResponse.model_validate(profile)


async def get_profile_samples(
    profile_id: str,
    db: Session,
) -> List[ProfileSampleResponse]:
    """
    Get all samples for a profile.
    
    Args:
        profile_id: Profile ID
        db: Database session
        
    Returns:
        List of samples
    """
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    return [ProfileSampleResponse.model_validate(s) for s in samples]


async def list_profiles(db: Session) -> List[VoiceProfileResponse]:
    """
    List all voice profiles.
    
    Args:
        db: Database session
        
    Returns:
        List of profiles
    """
    profiles = db.query(DBVoiceProfile).order_by(
        DBVoiceProfile.created_at.desc()
    ).all()
    
    return [VoiceProfileResponse.model_validate(p) for p in profiles]


async def update_profile(
    profile_id: str,
    data: VoiceProfileCreate,
    db: Session,
) -> Optional[VoiceProfileResponse]:
    """
    Update a voice profile.
    
    Args:
        profile_id: Profile ID
        data: Updated profile data
        db: Database session
        
    Returns:
        Updated profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None
    
    # Update fields
    profile.name = data.name
    profile.description = data.description
    profile.language = data.language
    profile.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(profile)
    
    return VoiceProfileResponse.model_validate(profile)


async def delete_profile(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete a voice profile and all associated data.
    
    Args:
        profile_id: Profile ID
        db: Database session
        
    Returns:
        True if deleted, False if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return False
    
    # Delete samples from database
    db.query(DBProfileSample).filter_by(profile_id=profile_id).delete()
    
    # Delete profile from database
    db.delete(profile)
    db.commit()
    
    # Delete profile directory
    profile_dir = _get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    
    return True


async def delete_profile_sample(
    sample_id: str,
    db: Session,
) -> bool:
    """
    Delete a profile sample.
    
    Args:
        sample_id: Sample ID
        db: Database session
        
    Returns:
        True if deleted, False if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return False
    
    # Delete audio file
    audio_path = Path(sample.audio_path)
    if audio_path.exists():
        audio_path.unlink()
    
    # Delete from database
    db.delete(sample)
    db.commit()
    
    return True


async def update_profile_sample(
    sample_id: str,
    reference_text: str,
    db: Session,
) -> Optional[ProfileSampleResponse]:
    """
    Update a profile sample's reference text.
    
    Args:
        sample_id: Sample ID
        reference_text: Updated reference text
        db: Database session
        
    Returns:
        Updated sample or None if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return None
    
    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)
    
    return ProfileSampleResponse.model_validate(sample)


async def create_voice_prompt_for_profile(
    profile_id: str,
    db: Session,
    use_cache: bool = True,
) -> dict:
    """
    Create a combined voice prompt from all samples in a profile.
    
    Args:
        profile_id: Profile ID
        db: Database session
        use_cache: Whether to use cached prompts
        
    Returns:
        Voice prompt dictionary
    """
    # Get all samples for profile
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    
    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")
    
    tts_model = get_tts_model()
    
    if len(samples) == 1:
        # Single sample - use directly
        sample = samples[0]
        voice_prompt, _ = await tts_model.create_voice_prompt(
            sample.audio_path,
            sample.reference_text,
            use_cache=use_cache,
        )
        return voice_prompt
    else:
        # Multiple samples - combine them
        audio_paths = [s.audio_path for s in samples]
        reference_texts = [s.reference_text for s in samples]
        
        # Combine audio
        combined_audio, combined_text = await tts_model.combine_voice_prompts(
            audio_paths,
            reference_texts,
        )
        
        # Save combined audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_audio(combined_audio, tmp.name, 24000)
            tmp_path = tmp.name
        
        try:
            # Create prompt from combined audio
            voice_prompt, _ = await tts_model.create_voice_prompt(
                tmp_path,
                combined_text,
                use_cache=use_cache,
            )
            return voice_prompt
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
