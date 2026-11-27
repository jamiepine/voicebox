"""Short audio previews of voices, rendered from a fixed sample phrase.

Lets the UI play a voice before the user commits to generating with it.
Built-in voices are deterministic, so their previews are cached on disk and
rendered at most once.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import HTTPException

from .. import config

logger = logging.getLogger(__name__)

# A greeting that is short enough to render in a couple of seconds yet long
# enough to show a voice's pacing. Keep one per language the built-in voices
# cover; anything else falls back to English.
SAMPLE_PHRASES: dict[str, str] = {
    "en": "Hi there! This is a quick preview of how I sound.",
    "es": "¡Hola! Esta es una breve muestra de cómo sueno.",
    "fr": "Bonjour ! Voici un court aperçu de ma voix.",
    "de": "Hallo! Das ist eine kurze Hörprobe meiner Stimme.",
    "it": "Ciao! Questa è una breve anteprima della mia voce.",
    "pt": "Olá! Esta é uma breve amostra da minha voz.",
    "ru": "Привет! Это короткий пример моего голоса.",
    "hi": "नमस्ते! यह मेरी आवाज़ का एक छोटा सा नमूना है।",
    "ja": "こんにちは！これは私の声の短いサンプルです。",
    "ko": "안녕하세요! 제 목소리를 짧게 들려드릴게요.",
    "zh": "你好！这是我声音的简短试听。",
}
DEFAULT_PHRASE_LANGUAGE = "en"

# Fixed so a given voice always previews the same way.
PREVIEW_SEED = 1234

# Previews share the model with real generations, and someone skimming a voice
# list can click faster than the model can answer. Run them one at a time.
_preview_lock = asyncio.Lock()


def sample_phrase_for(language: str) -> str:
    return SAMPLE_PHRASES.get(language, SAMPLE_PHRASES[DEFAULT_PHRASE_LANGUAGE])


def preview_cache_path(engine: str, voice_id: str, language: str) -> Path:
    """Where the preview for a built-in voice lives.

    Keyed by a digest rather than the voice id so an id can never shape a path,
    and so editing the phrase invalidates old files.
    """
    key = f"{engine}\0{voice_id}\0{language}\0{sample_phrase_for(language)}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return config.get_cache_dir() / "voice_previews" / f"{engine}-{digest}.wav"


def _model_size_for(engine: str) -> str:
    from ..backends import get_tts_model_configs

    configs = [c for c in get_tts_model_configs() if c.engine == engine]
    sizes = [c.model_size for c in configs]
    if "1.7B" in sizes:
        return "1.7B"
    return sizes[0] if sizes else "default"


def _display_name_for(engine: str, model_size: str) -> str:
    from ..backends import get_tts_model_configs

    for cfg in get_tts_model_configs():
        if cfg.engine == engine and cfg.model_size == model_size:
            return cfg.display_name
    return engine


async def render_preview(
    engine: str,
    language: str,
    make_voice_prompt: Callable[[], Awaitable[dict]],
) -> bytes:
    """Synthesise the sample phrase for ``language`` and return WAV bytes.

    ``make_voice_prompt`` runs after the model is loaded and under the preview
    lock, because building a prompt from a cloned voice's samples needs the model.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..services import tts
    from ..services.generation import release_generation_memory
    from .audio import has_tts_runaway, normalize_audio, trim_tts_output
    from .chunked_tts import generate_chunked

    model_size = _model_size_for(engine)

    async with _preview_lock:
        try:
            await ensure_model_cached_or_raise(engine, model_size)
        except HTTPException as exc:
            # The stock message points at /generate, which means nothing in a UI.
            name = _display_name_for(engine, model_size)
            raise HTTPException(
                status_code=409,
                detail=f"{name} isn't downloaded yet. Download it from the Models tab to preview this voice.",
            ) from exc

        await load_engine_model(engine, model_size)
        tts_model = get_tts_backend_for_engine(engine)
        voice_prompt = await make_voice_prompt()

        try:
            audio, sample_rate = await generate_chunked(
                tts_model,
                sample_phrase_for(language),
                voice_prompt,
                language=language,
                seed=PREVIEW_SEED,
                trim_fn=trim_tts_output if engine_needs_trim(engine) else None,
                runaway_detector=has_tts_runaway if engine_retries_runaway(engine) else None,
            )
        finally:
            release_generation_memory(tts_model)

    return tts.audio_to_wav_bytes(normalize_audio(audio), sample_rate)


async def cached_preset_preview(engine: str, voice_id: str, language: str) -> Path:
    """Path to the preview of a built-in voice, rendering it on first use."""
    path = preview_cache_path(engine, voice_id, language)
    if path.exists():
        return path

    async def preset_prompt() -> dict:
        return {"voice_type": "preset", "preset_engine": engine, "preset_voice_id": voice_id}

    wav = await render_preview(engine, language, preset_prompt)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a request racing this one never reads a half-written file.
    tmp = path.with_suffix(".wav.tmp")
    tmp.write_bytes(wav)
    tmp.replace(path)
    return path
