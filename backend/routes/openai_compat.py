"""OpenAI-compatible audio API: ``POST /v1/audio/speech`` and ``GET /v1/models``.

Any client built against OpenAI's text-to-speech API (the official ``openai``
SDKs, Open WebUI, Home Assistant, SillyTavern, shell scripts...) can point at
a local Voicebox by changing only the base URL. Voicebox has no API keys, so
the ``Authorization`` header is accepted and ignored.

Request field mapping
---------------------
``voice``
    A Voicebox voice profile, by name (case-insensitive) or id. When omitted
    or empty, the per-client MCP binding (``X-Voicebox-Client-Id``) and then
    the global default playback voice are tried, exactly like ``POST /speak``.
``model``
    A Voicebox TTS engine (``qwen``, ``chatterbox``, ``kokoro``...) or a
    model variant from ``/v1/models`` (``qwen-tts-0.6B``). OpenAI's own ids
    (``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``) are aliases for "the
    profile's configured engine", so unchanged client code works.
``input``
    The text to speak, in the profile's language.
``response_format``
    ``wav`` (default), ``mp3``, ``flac``, ``opus``, ``pcm`` (raw 16-bit
    little-endian mono at 24 kHz, as OpenAI defines it), ``aac`` (needs ffmpeg).
``speed``
    0.25-4.0, applied as a pitch-preserving time stretch after synthesis.
``instructions``
    Forwarded as the ``instruct`` prompt to engines that support it (Qwen).

Errors use OpenAI's ``{"error": {"message", "type", "param", "code"}}`` body
so SDK exceptions carry a readable message.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from ..database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compatible"])

# OpenAI's own model ids. They carry no engine information for us, so they
# resolve to whatever engine the voice profile is configured for.
OPENAI_MODEL_ALIASES = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")

# response_format -> (soundfile format, soundfile subtype, media type)
_SOUNDFILE_FORMATS: dict[str, tuple[str, str | None, str]] = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "flac": ("FLAC", None, "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
}
_FFMPEG_FORMATS: dict[str, tuple[list[str], str]] = {
    "aac": (["-c:a", "aac", "-b:a", "128k", "-f", "adts"], "audio/aac"),
}
RESPONSE_FORMATS = (*_SOUNDFILE_FORMATS, "pcm", *_FFMPEG_FORMATS)
PCM_SAMPLE_RATE = 24000

# Voicebox launched long after the OpenAI `created` epoch convention; a fixed
# value keeps the field deterministic for clients that display it.
_MODELS_CREATED = 1737763200  # 2026-01-25, the Voicebox repo's creation date


class OpenAIError(Exception):
    """An error rendered in OpenAI's JSON error shape."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={
                "error": {
                    "message": self.message,
                    "type": self.error_type,
                    "param": self.param,
                    "code": self.code,
                }
            },
        )


class SpeechRequest(BaseModel):
    model: str = Field(..., min_length=1)
    input: str = Field(..., min_length=1, max_length=10000)
    voice: str | None = None
    response_format: str = "wav"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = None


def _available_formats() -> list[str]:
    """Formats this server can actually produce, given its libsndfile build and PATH."""
    formats = ["pcm"]
    formats.extend(f for f, (fmt, _, _) in _SOUNDFILE_FORMATS.items() if fmt in sf.available_formats())
    if shutil.which("ffmpeg"):
        formats.extend(_FFMPEG_FORMATS)
    return formats


def _model_entries() -> list[dict[str, Any]]:
    """Every id ``model`` accepts, in OpenAI's model-object shape."""
    from ..backends import TTS_ENGINES, get_tts_model_configs

    entries = [
        {
            "id": alias,
            "object": "model",
            "created": _MODELS_CREATED,
            "owned_by": "voicebox",
            "description": "Alias: uses the engine configured on the requested voice profile.",
        }
        for alias in OPENAI_MODEL_ALIASES
    ]
    for engine, display_name in TTS_ENGINES.items():
        entries.append(
            {
                "id": engine,
                "object": "model",
                "created": _MODELS_CREATED,
                "owned_by": "voicebox",
                "description": display_name,
            }
        )
    for cfg in get_tts_model_configs():
        if cfg.model_name in TTS_ENGINES:
            continue
        entries.append(
            {
                "id": cfg.model_name,
                "object": "model",
                "created": _MODELS_CREATED,
                "owned_by": "voicebox",
                "description": cfg.display_name,
            }
        )
    return entries


def resolve_model(model: str, profile) -> tuple[str, str | None]:
    """Map the request's ``model`` onto ``(engine, model_size)``.

    ``model_size`` is ``None`` for engines with a single variant.
    """
    from ..backends import TTS_ENGINES, engine_has_model_sizes, get_tts_model_configs

    if model in OPENAI_MODEL_ALIASES:
        engine = getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"
    elif model in TTS_ENGINES:
        engine = model
    else:
        cfg = next((c for c in get_tts_model_configs() if c.model_name == model), None)
        if cfg is None:
            known = [e["id"] for e in _model_entries()]
            raise OpenAIError(
                404,
                f"The model '{model}' does not exist. Available models: {', '.join(known)}.",
                param="model",
                code="model_not_found",
            )
        return cfg.engine, (cfg.model_size if engine_has_model_sizes(cfg.engine) else None)

    if not engine_has_model_sizes(engine):
        return engine, None
    # Bare engine name: the first registered variant is that engine's default size.
    default_cfg = next(c for c in get_tts_model_configs() if c.engine == engine)
    return engine, default_cfg.model_size


def encode_audio(audio: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Encode float audio into the requested container; returns (bytes, media type)."""
    import io

    if response_format == "pcm":
        # OpenAI defines pcm as headerless 16-bit mono at 24 kHz; the body
        # carries no rate, so engines at another rate (LuxTTS: 48 kHz) must be resampled.
        if sample_rate != PCM_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=PCM_SAMPLE_RATE)
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767).astype("<i2").tobytes(), "audio/pcm"

    if response_format in _SOUNDFILE_FORMATS:
        fmt, subtype, media_type = _SOUNDFILE_FORMATS[response_format]
        if fmt not in sf.available_formats():
            raise OpenAIError(
                400,
                f"response_format '{response_format}' is not supported by this server's libsndfile build.",
                param="response_format",
                code="unsupported_format",
            )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=fmt, subtype=subtype)
        return buffer.getvalue(), media_type

    ffmpeg_args, media_type = _FFMPEG_FORMATS[response_format]
    if shutil.which("ffmpeg") is None:
        raise OpenAIError(
            400,
            f"response_format '{response_format}' requires ffmpeg, which is not installed on this server. "
            f"Supported formats here: {', '.join(_available_formats())}.",
            param="response_format",
            code="unsupported_format",
        )
    wav = io.BytesIO()
    sf.write(wav, audio, sample_rate, format="WAV", subtype="PCM_16")
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"],
            input=wav.getvalue(),
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenAIError(500, "ffmpeg timed out while encoding.", error_type="server_error") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise OpenAIError(500, f"ffmpeg failed to encode audio: {stderr}", error_type="server_error")
    return result.stdout, media_type


def apply_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Time-stretch without changing pitch. ``speed`` > 1 shortens the clip."""
    if abs(speed - 1.0) < 1e-3 or audio.size == 0:
        return audio
    import librosa

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
    return stretched.astype(audio.dtype, copy=False)


@router.get("/v1/models")
async def list_models():
    """List every id ``POST /v1/audio/speech`` accepts as ``model``."""
    return {"object": "list", "data": _model_entries()}


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    for entry in _model_entries():
        if entry["id"] == model_id:
            return entry
    return OpenAIError(404, f"The model '{model_id}' does not exist.", param="model", code="model_not_found").response()


@router.post("/v1/audio/speech")
async def create_speech(request: Request, db: Session = Depends(get_db)):
    """Synthesize ``input`` in a Voicebox voice and return the encoded audio."""
    try:
        return await _create_speech(request, db)
    except OpenAIError as e:
        return e.response()
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        return OpenAIError(e.status_code, detail).response()


async def _create_speech(request: Request, db: Session) -> Response:
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..mcp_server.resolve import resolve_profile
    from ..services import profiles
    from ..services.generation import release_generation_memory
    from ..utils.chunked_tts import generate_chunked

    try:
        body = await request.json()
    except ValueError as exc:
        raise OpenAIError(400, "Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise OpenAIError(400, "Request body must be a JSON object.")
    try:
        data = SpeechRequest.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first.get("loc", ())) or None
        raise OpenAIError(400, f"Invalid value for '{param}': {first['msg']}", param=param) from exc

    if data.response_format not in _available_formats():
        raise OpenAIError(
            400,
            f"Invalid response_format '{data.response_format}'. Supported on this server: "
            f"{', '.join(_available_formats())}.",
            param="response_format",
            code="unsupported_format",
        )

    client_id = request.headers.get("X-Voicebox-Client-Id")
    profile = resolve_profile(data.voice or None, client_id, db)
    if profile is None:
        if data.voice:
            raise OpenAIError(
                400,
                f"Voice '{data.voice}' is not a Voicebox voice profile. "
                "Pass a profile name or id; GET /profiles lists them.",
                param="voice",
                code="voice_not_found",
            )
        raise OpenAIError(
            400,
            "No voice given and no default voice configured. Pass `voice` (a profile name or id), "
            "or set a default in Voicebox → Settings → MCP.",
            param="voice",
            code="voice_required",
        )

    engine, model_size = resolve_model(data.model, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as exc:
        raise OpenAIError(400, str(exc), param="model", code="engine_not_compatible") from exc

    try:
        await ensure_model_cached_or_raise(engine, model_size or "default")
    except HTTPException as exc:
        raise OpenAIError(
            400,
            f"The model for '{data.model}' (engine '{engine}') is not downloaded yet. "
            "Download it in Voicebox → Models, or POST /models/load.",
            param="model",
            code="model_not_downloaded",
        ) from exc
    await load_engine_model(engine, model_size or "default")
    tts_model = get_tts_backend_for_engine(engine)

    voice_prompt = await profiles.create_voice_prompt_for_profile(str(profile.id), db, engine=engine)

    trim_fn = None
    runaway_detector = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output
    if engine_retries_runaway(engine):
        from ..utils.audio import has_tts_runaway

        runaway_detector = has_tts_runaway

    started = time.monotonic()
    try:
        audio, sample_rate = await generate_chunked(
            tts_model,
            data.input,
            voice_prompt,
            language=getattr(profile, "language", None) or "en",
            instruct=data.instructions,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )
    finally:
        release_generation_memory(tts_model)

    if profile.effects_chain:
        import json

        from ..utils.effects import apply_effects

        try:
            effects_chain = json.loads(profile.effects_chain)
        except ValueError:
            effects_chain = None
        if effects_chain:
            audio = apply_effects(audio, sample_rate, effects_chain)

    from ..utils.audio import normalize_audio

    audio = normalize_audio(audio)
    audio = apply_speed(audio, sample_rate, data.speed)
    content, media_type = encode_audio(audio, sample_rate, data.response_format)

    logger.info(
        "OpenAI-compat speech: profile=%s engine=%s format=%s %.1fs audio in %.1fs",
        profile.name,
        engine,
        data.response_format,
        len(audio) / sample_rate,
        time.monotonic() - started,
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{data.response_format}"'},
    )
