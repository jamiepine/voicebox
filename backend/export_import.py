"""
Voice profile export/import module.

Handles exporting profiles to ZIP archives and importing them back.
Also handles exporting individual generations.
"""

import json
import zipfile
import io
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from .models import VoiceProfileResponse
from .database import VoiceProfile as DBVoiceProfile, ProfileSample as DBProfileSample, Generation as DBGeneration
from .profiles import create_profile, add_profile_sample
from .models import VoiceProfileCreate
from . import config


def _get_profiles_dir() -> Path:
    """Get profiles directory from config."""
    return config.get_profiles_dir()


def _get_unique_profile_name(name: str, db: Session) -> str:
    """
    Get a unique profile name by appending a number if needed.
    
    Args:
        name: Original profile name
        db: Database session
        
    Returns:
        Unique profile name
    """
    base_name = name
    counter = 1
    
    while True:
        existing = db.query(DBVoiceProfile).filter_by(name=name).first()
        if not existing:
            return name
        
        name = f"{base_name} ({counter})"
        counter += 1


def export_profile_to_zip(profile_id: str, db: Session) -> bytes:
    """
    Export a voice profile to a ZIP archive.
    
    Args:
        profile_id: Profile ID to export
        db: Database session
        
    Returns:
        ZIP file contents as bytes
        
    Raises:
        ValueError: If profile not found or has no samples
    """
    # Get profile
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")
    
    # Get all samples
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    if not samples:
        raise ValueError(f"Profile {profile_id} has no samples")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Check if profile has avatar
        has_avatar = False
        if profile.avatar_path:
            avatar_path = Path(profile.avatar_path)
            if avatar_path.exists():
                has_avatar = True
                # Add avatar to ZIP root with original extension
                avatar_ext = avatar_path.suffix
                zip_file.write(avatar_path, f"avatar{avatar_ext}")

        # Create manifest.json
        manifest = {
            "version": "1.0",
            "profile": {
                "name": profile.name,
                "description": profile.description,
                "language": profile.language,
            },
            "has_avatar": has_avatar,
        }
        zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Create samples.json mapping
        samples_data = {}
        profile_dir = _get_profiles_dir() / profile_id

        for sample in samples:
            # Get filename from audio_path (should be {sample_id}.wav)
            audio_path = Path(sample.audio_path)
            filename = audio_path.name

            # Read audio file
            if not audio_path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")

            # Add to samples directory in ZIP
            zip_path = f"samples/{filename}"
            zip_file.write(audio_path, zip_path)

            # Map filename to reference text
            samples_data[filename] = sample.reference_text

        zip_file.writestr("samples.json", json.dumps(samples_data, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.read()


async def import_profile_from_zip(file_bytes: bytes, db: Session) -> VoiceProfileResponse:
    """
    Import a voice profile from a ZIP archive.
    
    Args:
        file_bytes: ZIP file contents
        db: Database session
        
    Returns:
        Created profile
        
    Raises:
        ValueError: If ZIP is invalid or missing required files
    """
    zip_buffer = io.BytesIO(file_bytes)
    
    try:
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            # Validate ZIP structure
            namelist = zip_file.namelist()
            
            if "manifest.json" not in namelist:
                raise ValueError("ZIP archive missing manifest.json")
            
            if "samples.json" not in namelist:
                raise ValueError("ZIP archive missing samples.json")
            
            # Read manifest
            manifest_data = json.loads(zip_file.read("manifest.json"))
            
            if "version" not in manifest_data:
                raise ValueError("Invalid manifest.json: missing version")
            
            if "profile" not in manifest_data:
                raise ValueError("Invalid manifest.json: missing profile")
            
            profile_data = manifest_data["profile"]
            
            # Read samples mapping
            samples_data = json.loads(zip_file.read("samples.json"))
            
            if not isinstance(samples_data, dict):
                raise ValueError("Invalid samples.json: must be a dictionary")
            
            # Get unique profile name
            original_name = profile_data.get("name", "Imported Profile")
            unique_name = _get_unique_profile_name(original_name, db)
            
            # Create profile
            profile_create = VoiceProfileCreate(
                name=unique_name,
                description=profile_data.get("description"),
                language=profile_data.get("language", "en"),
            )
            
            profile = await create_profile(profile_create, db)

            # Extract and add samples
            profile_dir = _get_profiles_dir() / profile.id
            profile_dir.mkdir(parents=True, exist_ok=True)

            # Handle avatar if present
            avatar_files = [f for f in namelist if f.startswith("avatar.")]
            if avatar_files:
                try:
                    avatar_file = avatar_files[0]
                    # Extract to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=Path(avatar_file).suffix, delete=False) as tmp:
                        tmp.write(zip_file.read(avatar_file))
                        tmp_path = tmp.name

                    try:
                        from .profiles import upload_avatar
                        await upload_avatar(profile.id, tmp_path, db)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    # Avatar import is optional - continue even if it fails
                    pass

            for filename, reference_text in samples_data.items():
                # Validate filename
                if not filename.endswith('.wav'):
                    raise ValueError(f"Invalid sample filename: {filename} (must be .wav)")
                
                # Extract audio file to temp location
                zip_path = f"samples/{filename}"
                
                if zip_path not in namelist:
                    raise ValueError(f"Sample file not found in ZIP: {zip_path}")
                
                # Extract to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(zip_file.read(zip_path))
                    tmp_path = tmp.name
                
                try:
                    # Add sample to profile
                    await add_profile_sample(
                        profile.id,
                        tmp_path,
                        reference_text,
                        db,
                    )
                finally:
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)
            
            return profile
            
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in archive: {e}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error importing profile: {str(e)}")


def export_generation_to_zip(generation_id: str, db: Session) -> bytes:
    """
    Export a generation to a ZIP archive.
    
    Args:
        generation_id: Generation ID to export
        db: Database session
        
    Returns:
        ZIP file contents as bytes
        
    Raises:
        ValueError: If generation not found
    """
    # Get generation
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        raise ValueError(f"Generation {generation_id} not found")
    
    # Get profile info
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    if not profile:
        raise ValueError(f"Profile {generation.profile_id} not found")
    
    # Get audio file
    audio_path = Path(generation.audio_path)
    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create manifest.json
        manifest = {
            "version": "1.0",
            "generation": {
                "id": generation.id,
                "text": generation.text,
                "language": generation.language,
                "duration": generation.duration,
                "seed": generation.seed,
                "instruct": generation.instruct,
                "created_at": generation.created_at.isoformat(),
            },
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "language": profile.language,
            }
        }
        zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))
        
        # Add audio file
        filename = audio_path.name
        zip_file.write(audio_path, f"audio/{filename}")
    
    zip_buffer.seek(0)
    return zip_buffer.read()


async def import_generation_from_zip(file_bytes: bytes, db: Session) -> dict:
    """
    Import a generation from a ZIP archive.
    
    Args:
        file_bytes: ZIP file contents
        db: Database session
        
    Returns:
        Dictionary with generation ID and profile info
        
    Raises:
        ValueError: If ZIP is invalid or missing required files
    """
    from pathlib import Path
    import tempfile
    import shutil
    from datetime import datetime
    from . import config
    
    zip_buffer = io.BytesIO(file_bytes)
    
    try:
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            # Validate ZIP structure
            namelist = zip_file.namelist()
            
            if "manifest.json" not in namelist:
                raise ValueError("ZIP archive missing manifest.json")
            
            # Read manifest
            manifest_data = json.loads(zip_file.read("manifest.json"))
            
            if "version" not in manifest_data:
                raise ValueError("Invalid manifest.json: missing version")
            
            if "generation" not in manifest_data:
                raise ValueError("Invalid manifest.json: missing generation data")
            
            generation_data = manifest_data["generation"]
            profile_data = manifest_data.get("profile", {})
            
            # Validate required fields
            required_fields = ["text", "language", "duration"]
            for field in required_fields:
                if field not in generation_data:
                    raise ValueError(f"Invalid manifest.json: missing generation.{field}")
            
            # Find audio file in archive
            audio_files = [f for f in namelist if f.startswith("audio/") and f.endswith(".wav")]
            if not audio_files:
                raise ValueError("No audio file found in ZIP archive")
            
            audio_file_path = audio_files[0]
            
            # Check if we should match an existing profile or create metadata
            profile_id = None
            profile_name = profile_data.get("name", "Unknown Profile")
            
            # Try to find matching profile by name
            if profile_name and profile_name != "Unknown Profile":
                existing_profile = db.query(DBVoiceProfile).filter_by(name=profile_name).first()
                if existing_profile:
                    profile_id = existing_profile.id
            
            # If no matching profile, use a placeholder or the first available profile
            if not profile_id:
                # Get any profile, or None if no profiles exist
                any_profile = db.query(DBVoiceProfile).first()
                if any_profile:
                    profile_id = any_profile.id
                    profile_name = any_profile.name
                else:
                    raise ValueError("No voice profiles found. Please create a profile before importing generations.")
            
            # Extract audio file to temporary location
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(zip_file.read(audio_file_path))
                tmp_path = tmp.name
            
            try:
                # Create generations directory
                generations_dir = config.get_generations_dir()
                generations_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate new ID for this generation
                new_generation_id = str(__import__('uuid').uuid4())
                
                # Copy audio to generations directory
                audio_dest = generations_dir / f"{new_generation_id}.wav"
                shutil.copy(tmp_path, audio_dest)
                
                # Create generation record
                db_generation = DBGeneration(
                    id=new_generation_id,
                    profile_id=profile_id,
                    text=generation_data["text"],
                    language=generation_data["language"],
                    audio_path=str(audio_dest),
                    duration=generation_data["duration"],
                    seed=generation_data.get("seed"),
                    instruct=generation_data.get("instruct"),
                    created_at=datetime.utcnow(),
                )
                
                db.add(db_generation)
                db.commit()
                db.refresh(db_generation)
                
                return {
                    "id": db_generation.id,
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "text": db_generation.text,
                    "message": f"Generation imported successfully (assigned to profile: {profile_name})"
                }
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
            
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in archive: {e}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error importing generation: {str(e)}")
