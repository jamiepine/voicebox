"""Audio file serving endpoints."""

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history
from ..database import get_db

router = APIRouter()


def _audio_media_type(path: Path) -> str:
    """Derive the Content-Type from the file extension.

    Imported audio retains its source format (.mp3, .m4a, .ogg, …) so a
    blanket ``audio/wav`` would mislead strict clients trying to decode
    via the response header instead of sniffing the bytes."""
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "audio/wav"


@router.get("/audio/version/{version_id}")
async def get_version_audio(version_id: str, db: Session = Depends(get_db)):
    """Serve audio for a specific version."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    audio_path = config.resolve_storage_path(version.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type=_audio_media_type(audio_path),
        filename=f"generation_{version.generation_id}_{version.label}{audio_path.suffix}",
    )


@router.get("/audio/{generation_id}")
async def get_audio(generation_id: str, db: Session = Depends(get_db)):
    """Serve generated audio file (serves the default version)."""
    generation = await history.get_generation(generation_id, db)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    audio_path = config.resolve_storage_path(generation.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type=_audio_media_type(audio_path),
        filename=f"generation_{generation_id}{audio_path.suffix}",
    )


@router.get("/samples/{sample_id}")
async def get_sample_audio(sample_id: str, db: Session = Depends(get_db)):
    """Serve profile sample audio file."""
    from ..database import ProfileSample as DBProfileSample

    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    audio_path = config.resolve_storage_path(sample.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"sample_{sample_id}.wav",
    )
