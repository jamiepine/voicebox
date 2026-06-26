"""TTS generation endpoints."""

import asyncio
import logging
import struct
import uuid
from pathlib import Path

import numpy as np

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history, personality, profiles
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services.generation import run_generation
from ..services.task_queue import cancel_generation as cancel_generation_job, enqueue_generation
from ..utils.audio import load_audio
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()

IMPORTED_AUDIO_PROFILE_NAME = "Imported Audio"
IMPORT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
IMPORT_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


def _get_or_create_import_profile(db: Session) -> DBVoiceProfile:
    """Singleton profile every imported audio clip points at — keeps the
    Generation FK happy without making profile_id nullable across the schema."""
    row = (
        db.query(DBVoiceProfile)
        .filter(DBVoiceProfile.name == IMPORTED_AUDIO_PROFILE_NAME)
        .first()
    )
    if row is not None:
        return row
    row = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=IMPORTED_AUDIO_PROFILE_NAME,
        description="External audio imported into a story timeline.",
        language="en",
        voice_type="import",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _resolve_generation_engine(data: models.GenerationRequest, profile) -> str:
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


def _wav_stream_header(
    sample_rate: int,
    *,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Build a stream-friendly WAV header with unknown data length."""
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,
    )


def _audio_to_pcm16le(audio: np.ndarray) -> bytes:
    """Convert float audio to little-endian 16-bit PCM bytes."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767.0).astype("<i2").tobytes()


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from ..backends import engine_has_model_sizes

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None

    text = data.text
    source = "manual"
    if data.personality and getattr(profile, "personality", None):
        try:
            llm_result = await personality.rewrite_as_profile(profile.personality, data.text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        text = llm_result.text.strip()
        if not text:
            raise HTTPException(status_code=500, detail="LLM produced empty output; nothing to speak.")
        source = "personality_speak"

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine_has_model_sizes(engine) else None,
        source=source,
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=text,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    else:
        import json as _json

        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            try:
                effects_chain_config = _json.loads(profile_obj.effects_chain)
            except Exception:
                pass

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=data.profile_id,
            text=text,
            language=data.language,
            engine=engine,
            model_size=model_size,
            seed=data.seed,
            normalize=data.normalize,
            effects_chain=effects_chain_config,
            instruct=data.instruct,
            mode="generate",
            max_chunk_chars=data.max_chunk_chars,
            crossfade_ms=data.crossfade_ms,
        )
    )

    return generation


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or "1.7B",
            seed=gen.seed,
            instruct=gen.instruct,
            mode="retry",
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    gen.status = "generating"
    gen.error = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or "1.7B",
            seed=gen.seed,
            instruct=gen.instruct,
            mode="regenerate",
            version_id=version_id,
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post("/generate/{generation_id}/cancel")
async def cancel_generation(generation_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") not in ("loading_model", "generating"):
        raise HTTPException(status_code=400, detail="Only active generations can be cancelled")

    cancellation_state = cancel_generation_job(generation_id)
    if cancellation_state is None:
        # Row says active but the worker is no longer tracking it — the gen
        # coroutine exited without writing a terminal status (most often a
        # SQLite lock racing with the failed-status write inside the worker's
        # exception handler). Fail the row here so the user can move on.
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation orphaned by worker",
        )
        return {"message": "Orphaned generation cleared"}

    if cancellation_state == "queued":
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation cancelled",
        )
        return {"message": "Queued generation cancelled"}

    return {"message": "Generation cancellation requested"}


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                if not gen:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                payload = {
                    "id": gen.id,
                    "status": gen.status or "completed",
                    "duration": gen.duration,
                    "error": gen.error,
                    # Agent-originated sources ("mcp", "rest") skip main-window
                    # autoplay — the floating pill plays those directly.
                    "source": gen.source,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (gen.status or "completed") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech and stream WAV audio as chunks without saving to disk."""
    from ..backends import (
        engine_needs_trim,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import split_text_into_chunks

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    tts_model = get_tts_backend_for_engine(engine)
    model_size = data.model_size or "1.7B"

    text = data.text
    if data.personality and getattr(profile, "personality", None):
        try:
            llm_result = await personality.rewrite_as_profile(
                profile.personality,
                data.text,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        text = llm_result.text.strip()
        if not text:
            raise HTTPException(
                status_code=500,
                detail="LLM produced empty output; nothing to speak.",
            )

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        data.profile_id,
        db,
        engine=engine,
    )

    trim_fn = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    elif profile.effects_chain:
        import json as _json

        try:
            effects_chain_config = _json.loads(profile.effects_chain)
        except Exception:
            effects_chain_config = None

    async def _wav_stream():
        try:
            chunks = split_text_into_chunks(text, data.max_chunk_chars)

            apply_effects_fn = None
            if effects_chain_config:
                from ..utils.effects import apply_effects as _apply_effects

                apply_effects_fn = _apply_effects

            normalize_fn = None
            if data.normalize:
                from ..utils.audio import normalize_audio as _normalize_audio

                normalize_fn = _normalize_audio

            sample_rate: int | None = None
            crossfade_samples = 0
            fade_out: np.ndarray | None = None
            fade_in: np.ndarray | None = None
            tail: np.ndarray | None = None

            for i, chunk_text in enumerate(chunks):
                chunk_seed = (data.seed + i) if data.seed is not None else None
                chunk_audio, chunk_sr = await tts_model.generate(
                    chunk_text,
                    voice_prompt,
                    data.language,
                    chunk_seed,
                    data.instruct,
                )

                chunk_audio = np.asarray(chunk_audio, dtype=np.float32)
                if trim_fn is not None:
                    chunk_audio = trim_fn(chunk_audio, chunk_sr)
                if apply_effects_fn is not None:
                    chunk_audio = apply_effects_fn(
                        chunk_audio, chunk_sr, effects_chain_config
                    )
                if normalize_fn is not None:
                    chunk_audio = normalize_fn(chunk_audio)

                if sample_rate is None:
                    sample_rate = chunk_sr
                    crossfade_samples = int(
                        max(0, data.crossfade_ms) * sample_rate / 1000
                    )
                    if crossfade_samples > 0:
                        fade_out = np.linspace(
                            1.0, 0.0, crossfade_samples, dtype=np.float32
                        )
                        fade_in = np.linspace(
                            0.0, 1.0, crossfade_samples, dtype=np.float32
                        )
                    yield _wav_stream_header(sample_rate)
                elif chunk_sr != sample_rate:
                    logger.warning(
                        "Stream chunk sample-rate mismatch (%s vs %s); using first value",
                        chunk_sr,
                        sample_rate,
                    )

                if tail is None:
                    tail = chunk_audio
                else:
                    overlap = min(
                        len(tail),
                        len(chunk_audio),
                        crossfade_samples,
                    )
                    if overlap > 0 and fade_out is not None and fade_in is not None:
                        if len(tail) > overlap:
                            yield _audio_to_pcm16le(tail[:-overlap])
                        blended = (
                            tail[-overlap:] * fade_out[:overlap]
                            + chunk_audio[:overlap] * fade_in[:overlap]
                        )
                        tail = np.concatenate([blended, chunk_audio[overlap:]])
                    else:
                        tail = np.concatenate([tail, chunk_audio])

                if crossfade_samples == 0 and tail is not None and tail.size > 0:
                    yield _audio_to_pcm16le(tail)
                    tail = np.array([], dtype=np.float32)
                elif (
                    crossfade_samples > 0
                    and tail is not None
                    and len(tail) > crossfade_samples
                ):
                    emit = tail[:-crossfade_samples]
                    tail = tail[-crossfade_samples:]
                    if emit.size > 0:
                        yield _audio_to_pcm16le(emit)

            if tail is not None and tail.size > 0:
                yield _audio_to_pcm16le(tail)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("Client disconnected during audio stream")

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate/import", response_model=models.GenerationResponse)
async def import_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an external audio file as a generation row.

    Designed for the story timeline so users can drop in music or other
    non-TTS audio. The row points at a singleton "Imported Audio" profile
    so the existing generation/story plumbing keeps working unchanged."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMPORT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(IMPORT_AUDIO_EXTENSIONS)}",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > IMPORT_AUDIO_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {IMPORT_AUDIO_MAX_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    generation_id = str(uuid.uuid4())
    target = config.get_generations_dir() / f"{generation_id}{suffix}"
    target.write_bytes(audio_bytes)

    try:
        audio, sr = load_audio(str(target))
        duration = float(len(audio) / sr) if sr else 0.0
    except Exception as decode_err:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {decode_err}",
        ) from decode_err

    profile = _get_or_create_import_profile(db)
    display_name = Path(file.filename or "Imported audio").stem or "Imported audio"

    return await history.create_generation(
        profile_id=profile.id,
        text=display_name,
        language="en",
        audio_path=config.to_storage_path(target),
        duration=duration,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="completed",
        engine="import",
        model_size=None,
        source="import",
    )
