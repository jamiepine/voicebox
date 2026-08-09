"""Audio file serving endpoints."""

import asyncio
import io
import mimetypes
from pathlib import Path

import librosa
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history
from ..database import get_db
from ..utils.audio import EXPORT_FORMATS, encode_audio

router = APIRouter()


def _audio_media_type(path: Path) -> str:
    """Derive the Content-Type from the file extension.

    Imported audio retains its source format (.mp3, .m4a, .ogg, …) so a
    blanket ``audio/wav`` would mislead strict clients trying to decode
    via the response header instead of sniffing the bytes."""
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "audio/wav"


def _transcode(path: Path, fmt: str) -> bytes:
    """Decode a stored file and re-encode it into ``fmt``.

    Decoded at the file's own rate and channel count, matching the story
    mixdown, so a transcode is a container change rather than a resample."""
    audio, sr = librosa.load(str(path), sr=None, mono=False)
    return encode_audio(audio, int(sr), fmt=fmt)


async def _serve_audio(path: Path, fmt: str | None, stem: str):
    """Serve a stored audio file, optionally transcoded to ``fmt``.

    With no ``fmt`` the file is streamed untouched by ``FileResponse``, which
    keeps range requests working for the player. A transcode has to buffer the
    whole encode, so it is only paid for when a caller explicitly asks."""
    if fmt is None:
        return FileResponse(
            path,
            media_type=_audio_media_type(path),
            filename=f"{stem}{path.suffix}",
        )

    spec = EXPORT_FORMATS.get(fmt.lower())
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Supported: {sorted(EXPORT_FORMATS)}",
        )

    # Already in the requested container: hand back the bytes on disk rather
    # than decoding and re-encoding, which would only lose quality.
    if path.suffix.lower() == spec["ext"]:
        return FileResponse(
            path,
            media_type=spec["mime"],
            filename=f"{stem}{spec['ext']}",
        )

    try:
        audio_bytes = await asyncio.to_thread(_transcode, path, fmt.lower())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcode failed: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=spec["mime"],
        headers={"Content-Disposition": f'attachment; filename="{stem}{spec["ext"]}"'},
    )


@router.get("/audio/version/{version_id}")
async def get_version_audio(
    version_id: str,
    format: str | None = None,
    db: Session = Depends(get_db),
):
    """Serve audio for a specific version.

    ``format`` is one of :data:`EXPORT_FORMATS`; omitted, the stored file is
    served as-is."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    audio_path = config.resolve_storage_path(version.audio_path)
    if audio_path is None or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return await _serve_audio(
        audio_path,
        format,
        f"generation_{version.generation_id}_{version.label}",
    )


@router.get("/audio/{generation_id}")
async def get_audio(
    generation_id: str,
    format: str | None = None,
    db: Session = Depends(get_db),
):
    """Serve generated audio file (serves the default version).

    ``format`` is one of :data:`EXPORT_FORMATS` — ``mp3``, ``ogg``, ``opus``,
    ``flac`` or ``wav``. Omitted, the stored file is served as-is, so existing
    callers and range requests are unaffected."""
    generation = await history.get_generation(generation_id, db)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    audio_path = config.resolve_storage_path(generation.audio_path)
    if audio_path is None or not audio_path.is_file():
        detail = (
            "Generation failed; no audio available"
            if generation.status == "failed"
            else "Audio file not found"
        )
        raise HTTPException(status_code=404, detail=detail)

    return await _serve_audio(audio_path, format, f"generation_{generation_id}")


@router.get("/samples/{sample_id}")
async def get_sample_audio(sample_id: str, db: Session = Depends(get_db)):
    """Serve profile sample audio file."""
    from ..database import ProfileSample as DBProfileSample

    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    audio_path = config.resolve_storage_path(sample.audio_path)
    if audio_path is None or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"sample_{sample_id}.wav",
    )
