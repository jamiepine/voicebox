"""
OpenAI-compatible TTS API endpoints.

Implements a subset of the OpenAI Audio API so that clients using the
official openai Python SDK (or any OpenAI-compatible caller) can point at
voicebox without code changes.

Supported endpoints
-------------------
POST /v1/audio/speech
    Accept an OpenAI ``speech`` request body, map the model and voice to an
    internal engine / voice-prompt, and return raw WAV bytes.

GET /v1/models
    Stub that returns the three model IDs understood by this server.

Model mapping
-------------
tts-1           → Kokoro (fast, CPU-friendly)
tts-1-hd        → Qwen3-TTS 1.7B
gpt-4o-mini-tts → Qwen3-TTS 0.6B

Voice mapping (when no matching profile is found by name)
---------------------------------------------------------
alloy   → af_alloy
echo    → am_echo
fable   → bm_fable
onyx    → am_onyx
nova    → af_nova
shimmer → af_sky

Limitations
-----------
- ``response_format`` is always treated as WAV; other formats (mp3, opus,
  aac, flac, pcm) are not yet supported.  Pass ``response_format="wav"``
  or omit it to avoid surprises.
- ``speed`` is accepted in the schema for API compatibility but is not yet
  forwarded to the TTS backends.  Non-default speed values are silently
  ignored.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Model / voice mappings
# ---------------------------------------------------------------------------

_MODEL_MAP: dict[str, tuple[str, str]] = {
    # openai_model_id -> (engine, model_size)
    "tts-1": ("kokoro", "default"),
    "tts-1-hd": ("qwen", "1.7B"),
    "gpt-4o-mini-tts": ("qwen", "0.6B"),
}

_OPENAI_VOICE_TO_KOKORO: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}

_AVAILABLE_MODELS = list(_MODEL_MAP.keys())


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    # instructions / system prompt (OpenAI "instruct" equivalent)
    instructions: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest,
    db: Session = Depends(get_db),
) -> Response:
    """Generate speech from text, returning WAV audio bytes.

    Always returns ``audio/wav`` regardless of ``response_format`` — this
    keeps the implementation simple while remaining correct for most callers.
    """
    from ..backends import load_engine_model, get_tts_backend_for_engine, engine_needs_trim
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import normalize_audio, trim_tts_output
    from ..services.tts import audio_to_wav_bytes

    # --- Resolve model -------------------------------------------------
    mapping = _MODEL_MAP.get(request.model)
    if mapping is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Supported: {_AVAILABLE_MODELS}",
        )
    engine, model_size = mapping

    # --- Resolve voice prompt ------------------------------------------
    voice_prompt = await _resolve_voice_prompt(request.voice, engine, db)

    # --- Load model and generate ---------------------------------------
    tts_model = get_tts_backend_for_engine(engine)
    await load_engine_model(engine, model_size)

    trim_fn = trim_tts_output if engine_needs_trim(engine) else None

    audio, sample_rate = await generate_chunked(
        tts_model,
        request.input,
        voice_prompt,
        language="en",
        seed=None,
        instruct=request.instructions,
        trim_fn=trim_fn,
    )

    audio = normalize_audio(audio)
    wav_bytes = audio_to_wav_bytes(audio, sample_rate)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@router.get("/v1/models")
async def list_models():
    """Return model IDs understood by this server."""
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "voicebox"}
            for model_id in _AVAILABLE_MODELS
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve_voice_prompt(voice: str, engine: str, db: Session) -> dict:
    """Return a voice_prompt dict for the requested voice name.

    Strategy:
    1. Look up a VoiceProfile by name (case-insensitive).  If found, delegate
       to ``profiles.create_voice_prompt_for_profile`` which handles reference
       audio encoding and caching.
    2. If no profile matches (or the engine is kokoro), fall back to a
       static voice-id dict using the ``_OPENAI_VOICE_TO_KOKORO`` mapping.
    """
    from ..database import VoiceProfile as DBVoiceProfile
    from ..services import profiles as profiles_svc

    # Try profile lookup first
    profile = (
        db.query(DBVoiceProfile)
        .filter(func.lower(DBVoiceProfile.name) == voice.lower())
        .first()
    )

    if profile is not None:
        try:
            return await profiles_svc.create_voice_prompt_for_profile(
                str(profile.id),
                db,
                use_cache=True,
                engine=engine,
            )
        except Exception as e:
            # Profile found but voice-prompt creation failed; fall through to
            # built-in voice.  Log so the failure is not silently swallowed.
            logger.warning(
                "Failed to create voice prompt for profile '%s': %s. "
                "Falling back to built-in Kokoro voice.",
                voice,
                e,
            )

    # Fall back to built-in Kokoro voice id using engine-specific key names.
    kokoro_voice = _OPENAI_VOICE_TO_KOKORO.get(voice.lower(), "af_alloy")
    if engine == "kokoro":
        return {"kokoro_voice": kokoro_voice}
    # qwen_custom_voice and all other engines expect preset_voice_id
    return {"preset_voice_id": kokoro_voice}
