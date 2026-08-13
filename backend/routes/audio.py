"""Audio file serving endpoints."""

import logging
import mimetypes
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history
from ..database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

# MP3 conversion settings
_MP3_BITRATE_K = 192


def _audio_media_type(path: Path) -> str:
    """Derive the Content-Type from the file extension.

    Imported audio retains its source format (.mp3, .m4a, .ogg, …) so a
    blanket ``audio/wav`` would mislead strict clients trying to decode
    via the response header instead of sniffing the bytes."""
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "audio/wav"


def _ensure_mp3(wav_path: Path) -> Path | None:
    """Convert *wav_path* to MP3 (cached on disk next to the WAV).

    Returns the path of the MP3 file.  The first request for a given WAV
    transcodes it with ffmpeg and caches ``<name>.mp3`` beside it, so
    subsequent requests are cheap.  Returns ``None`` when ffmpeg is
    missing or the conversion fails."""
    mp3_path = wav_path.with_suffix(".mp3")
    if mp3_path.exists() and mp3_path.stat().st_mtime >= wav_path.stat().st_mtime:
        return mp3_path
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", f"{_MP3_BITRATE_K}k",
                str(mp3_path),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning("MP3 conversion failed for %s: %s", wav_path, e)
        return None
    return mp3_path


@router.get("/audio/version/{version_id}")
async def get_version_audio(version_id: str, db: Session = Depends(get_db), format: str = "mp3"):
    """Serve audio for a specific version."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    audio_path = config.resolve_storage_path(version.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    if format == "mp3" and audio_path.suffix.lower() in (".wav", ".flac", ".ogg"):
        mp3_path = _ensure_mp3(audio_path)
        if mp3_path is not None:
            return FileResponse(
                mp3_path,
                media_type="audio/mpeg",
                filename=f"generation_{version.generation_id}_{version.label}.mp3",
            )

    return FileResponse(
        audio_path,
        media_type=_audio_media_type(audio_path),
        filename=f"generation_{version.generation_id}_{version.label}{audio_path.suffix}",
    )


@router.get("/audio/{generation_id}")
async def get_audio(generation_id: str, db: Session = Depends(get_db), format: str = "mp3"):
    """Serve generated audio file (serves the default version).

    Defaults to MP3 (converted on demand and cached beside the WAV); pass
    ``?format=wav`` to get the original file."""
    generation = await history.get_generation(generation_id, db)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    audio_path = config.resolve_storage_path(generation.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    if format == "mp3" and audio_path.suffix.lower() in (".wav", ".flac", ".ogg"):
        mp3_path = _ensure_mp3(audio_path)
        if mp3_path is not None:
            return FileResponse(
                mp3_path,
                media_type="audio/mpeg",
                filename=f"generation_{generation_id}.mp3",
            )

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
