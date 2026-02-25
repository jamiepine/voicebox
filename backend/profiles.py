"""
Voice profile management module.
"""

from typing import List, Optional
from datetime import datetime, timezone
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
from .utils.images import validate_image, process_avatar
from .utils.cache import _get_cache_dir, clear_profile_cache
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
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
    profile.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(db_sample)
    
    # Invalidate combined audio cache for this profile
    # Since a new sample was added, any cached combined audio is now stale
    clear_profile_cache(profile_id)
    
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
    profile.updated_at = datetime.now(timezone.utc)
    
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
    
    # Clean up combined audio cache files for this profile
    clear_profile_cache(profile_id)
    
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
    
    # Store profile_id before deleting
    profile_id = sample.profile_id
    
    # Delete audio file
    audio_path = Path(sample.audio_path)
    if audio_path.exists():
        audio_path.unlink()
    
    # Delete from database
    db.delete(sample)
    db.commit()
    
    # Invalidate combined audio cache for this profile
    # Since the sample set changed, any cached combined audio is now stale
    clear_profile_cache(profile_id)
    
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
    
    # Store profile_id before updating
    profile_id = sample.profile_id
    
    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)
    
    # Invalidate combined audio cache for this profile
    # Since the reference text changed, cache keys and combined text are now stale
    clear_profile_cache(profile_id)
    
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

        # Save combined audio to cache directory (persistent)
        # Create a hash of sample IDs to identify this specific combination
        import hashlib
        sample_ids_str = "-".join(sorted([s.id for s in samples]))
        combination_hash = hashlib.md5(sample_ids_str.encode()).hexdigest()[:12]
        
        # Store in cache directory
        cache_dir = _get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined_path = cache_dir / f"combined_{profile_id}_{combination_hash}.wav"
        
        # Save combined audio
        save_audio(combined_audio, str(combined_path), 24000)

        # Create prompt from combined audio
        voice_prompt, _ = await tts_model.create_voice_prompt(
            str(combined_path),
            combined_text,
            use_cache=use_cache,
        )
        return voice_prompt


async def upload_avatar(
    profile_id: str,
    image_path: str,
    db: Session,
) -> VoiceProfileResponse:
    """
    Upload and process avatar image for a profile.

    Args:
        profile_id: Profile ID
        image_path: Path to uploaded image file
        db: Database session

    Returns:
        Updated profile
    """
    # Validate profile exists
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    # Validate image
    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    # Delete existing avatar if present
    if profile.avatar_path:
        old_avatar = Path(profile.avatar_path)
        if old_avatar.exists():
            old_avatar.unlink()

    # Determine file extension from uploaded file
    from PIL import Image
    with Image.open(image_path) as img:
        # Normalize JPEG variants (MPO is multi-picture format from some cameras)
        img_format = img.format
        if img_format in ('MPO', 'JPG'):
            img_format = 'JPEG'
        
        ext_map = {
            'PNG': '.png',
            'JPEG': '.jpg',
            'WEBP': '.webp'
        }
        ext = ext_map.get(img_format, '.png')

    # Save processed image to profile directory
    profile_dir = _get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"avatar{ext}"

    process_avatar(image_path, str(output_path))

    # Update database
    profile.avatar_path = str(output_path)
    profile.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(profile)

    return VoiceProfileResponse.model_validate(profile)


async def delete_avatar(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete avatar image for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found or no avatar
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile or not profile.avatar_path:
        return False

    # Delete avatar file
    avatar_path = Path(profile.avatar_path)
    if avatar_path.exists():
        avatar_path.unlink()

    # Update database
    profile.avatar_path = None
    profile.updated_at = datetime.now(timezone.utc)

    db.commit()

    return True
