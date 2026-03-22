"""Voice profile endpoints."""

import io
import json as _json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..app import safe_content_disposition
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..services import channels, export_import, profiles
from ..services.profiles import _profile_to_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/profiles", response_model=models.VoiceProfileResponse)
async def create_profile(
    data: models.VoiceProfileCreate,
    db: Session = Depends(get_db),
):
    """Create a new voice profile."""
    try:
        return await profiles.create_profile(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profiles", response_model=list[models.VoiceProfileResponse])
async def list_profiles(db: Session = Depends(get_db)):
    """List all voice profiles."""
    return await profiles.list_profiles(db)


@router.post("/profiles/import", response_model=models.VoiceProfileResponse)
async def import_profile(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a voice profile from a ZIP archive."""
    MAX_FILE_SIZE = 100 * 1024 * 1024

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB"
        )

    try:
        profile = await export_import.import_profile_from_zip(content, db)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Preset Voice Endpoints ───────────────────────────────────────────
# These MUST be declared before /profiles/{profile_id} to avoid the
# wildcard swallowing "presets" as a profile_id.


@router.get("/profiles/presets/{engine}")
async def list_preset_voices(engine: str):
    """List available preset voices for an engine."""
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return {
            "engine": engine,
            "voices": [
                {
                    "voice_id": vid,
                    "name": name,
                    "gender": gender,
                    "language": lang,
                }
                for vid, name, gender, lang in KOKORO_VOICES
            ],
        }
    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return {
            "engine": engine,
            "voices": [
                {
                    "voice_id": speaker_id,
                    "name": display_name,
                    "gender": gender,
                    "language": lang,
                }
                for speaker_id, display_name, gender, lang, _desc in QWEN_CUSTOM_VOICES
            ],
        }
    return {"engine": engine, "voices": []}

@router.get("/profiles/{profile_id}", response_model=models.VoiceProfileResponse)
async def get_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get a voice profile by ID."""
    profile = await profiles.get_profile(profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.put("/profiles/{profile_id}", response_model=models.VoiceProfileResponse)
async def update_profile(
    profile_id: str,
    data: models.VoiceProfileCreate,
    db: Session = Depends(get_db),
):
    """Update a voice profile."""
    try:
        profile = await profiles.update_profile(profile_id, data, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Delete a voice profile."""
    success = await profiles.delete_profile(profile_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"message": "Profile deleted successfully"}


SAMPLE_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
SAMPLE_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.post("/profiles/{profile_id}/samples", response_model=models.ProfileSampleResponse)
async def add_profile_sample(
    profile_id: str,
    file: UploadFile = File(...),
    reference_text: str = Form(...),
    db: Session = Depends(get_db),
):
    """Add a sample to a voice profile."""
    _allowed_audio_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}
    _uploaded_ext = Path(file.filename or "").suffix.lower()
    file_suffix = _uploaded_ext if _uploaded_ext in _allowed_audio_exts else ".wav"

    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
        total_size = 0
        while chunk := await file.read(SAMPLE_UPLOAD_CHUNK_SIZE):
            total_size += len(chunk)
            if total_size > SAMPLE_MAX_FILE_SIZE:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {SAMPLE_MAX_FILE_SIZE // (1024 * 1024)} MB)",
                )
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        sample = await profiles.add_profile_sample(
            profile_id,
            tmp_path,
            reference_text,
            db,
        )
        return sample
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/profiles/{profile_id}/samples", response_model=list[models.ProfileSampleResponse])
async def get_profile_samples(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get all samples for a profile."""
    return await profiles.get_profile_samples(profile_id, db)


@router.delete("/profiles/samples/{sample_id}")
async def delete_profile_sample(
    sample_id: str,
    db: Session = Depends(get_db),
):
    """Delete a profile sample."""
    success = await profiles.delete_profile_sample(sample_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"message": "Sample deleted successfully"}


@router.put("/profiles/samples/{sample_id}", response_model=models.ProfileSampleResponse)
async def update_profile_sample(
    sample_id: str,
    data: models.ProfileSampleUpdate,
    db: Session = Depends(get_db),
):
    """Update a profile sample's reference text."""
    sample = await profiles.update_profile_sample(sample_id, data.reference_text, db)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample


@router.post("/profiles/{profile_id}/avatar", response_model=models.VoiceProfileResponse)
async def upload_profile_avatar(
    profile_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload or update avatar image for a profile."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        profile = await profiles.upload_avatar(profile_id, tmp_path, db)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/profiles/{profile_id}/avatar")
async def get_profile_avatar(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get avatar image for a profile."""
    profile = await profiles.get_profile(profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if not profile.avatar_path:
        raise HTTPException(status_code=404, detail="No avatar found for this profile")

    avatar_path = config.resolve_storage_path(profile.avatar_path)
    if avatar_path is None or not avatar_path.exists():
        raise HTTPException(status_code=404, detail="Avatar file not found")

    return FileResponse(avatar_path)


@router.delete("/profiles/{profile_id}/avatar")
async def delete_profile_avatar(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Delete avatar image for a profile."""
    success = await profiles.delete_avatar(profile_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found or no avatar to delete")
    return {"message": "Avatar deleted successfully"}


@router.get("/profiles/{profile_id}/export")
async def export_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Export a voice profile as a ZIP archive."""
    try:
        profile = await profiles.get_profile(profile_id, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        zip_bytes = export_import.export_profile_to_zip(profile_id, db)

        safe_name = "".join(c for c in profile.name if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_name:
            safe_name = "profile"
        filename = f"profile-{safe_name}.voicebox.zip"

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{profile_id}/channels")
async def get_profile_channels(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get list of channel IDs assigned to a profile."""
    try:
        channel_ids = await channels.get_profile_channels(profile_id, db)
        return {"channel_ids": channel_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/profiles/{profile_id}/channels")
async def set_profile_channels(
    profile_id: str,
    data: models.ProfileChannelAssignment,
    db: Session = Depends(get_db),
):
    """Set which channels a profile is assigned to."""
    try:
        await channels.set_profile_channels(profile_id, data, db)
        return {"message": "Profile channels updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/profiles/{profile_id}/effects", response_model=models.VoiceProfileResponse)
async def update_profile_effects(
    profile_id: str,
    data: models.ProfileEffectsUpdate,
    db: Session = Depends(get_db),
):
    """Set or clear the default effects chain for a voice profile."""
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if data.effects_chain is not None:
        from ..utils.effects import validate_effects_chain

        chain_dicts = [e.model_dump() for e in data.effects_chain]
        error = validate_effects_chain(chain_dicts)
        if error:
            raise HTTPException(status_code=400, detail=error)
        profile.effects_chain = _json.dumps(chain_dicts)
    else:
        profile.effects_chain = None

    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)
