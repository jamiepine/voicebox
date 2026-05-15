"""
Supertonic-3 TTS backend implementation.

Wraps the Supertonic-3 model (Supertone/supertonic-3 on HuggingFace) — a fast
ONNX-based multilingual TTS that runs entirely on CPU. ~66-99M params, 31
languages, ~167x realtime on consumer hardware.

Supertonic uses pre-built voice style files (M1-M5, F1-F5), not zero-shot
cloning from arbitrary audio. Voice prompts are preset references.

Cache layout differs from other backends: Supertonic downloads to
~/.cache/supertonic3/ rather than the HuggingFace Hub cache, so we detect
cached state by inspecting that directory directly.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import TTSBackend
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

SUPERTONIC_HF_REPO = "Supertone/supertonic-3"
SUPERTONIC_DEFAULT_VOICE = "M1"
SUPERTONIC_FALLBACK_SAMPLE_RATE = 44100  # Resolved from tts.sample_rate at runtime when possible

SUPERTONIC_VOICES = [
    ("M1", "Male 1", "male", "multi"),
    ("M2", "Male 2", "male", "multi"),
    ("M3", "Male 3", "male", "multi"),
    ("M4", "Male 4", "male", "multi"),
    ("M5", "Male 5", "male", "multi"),
    ("F1", "Female 1", "female", "multi"),
    ("F2", "Female 2", "female", "multi"),
    ("F3", "Female 3", "female", "multi"),
    ("F4", "Female 4", "female", "multi"),
    ("F5", "Female 5", "female", "multi"),
]

SUPERTONIC_LANGUAGES = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]


def _supertonic_cache_dir() -> Path:
    return Path(os.path.expanduser("~/.cache/supertonic3"))


class SupertonicTTSBackend:
    """Supertonic-3 TTS backend — ONNX, CPU, 31 langs, preset voices."""

    def __init__(self):
        self._tts = None
        self.model_size = "default"

    def is_loaded(self) -> bool:
        return self._tts is not None

    def _get_model_path(self, model_size: str) -> str:
        return SUPERTONIC_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Supertonic ships ONNX bundles to ~/.cache/supertonic3/."""
        cache = _supertonic_cache_dir()
        if not cache.exists():
            return False
        # Any of the four ONNX files indicates the bundle was unpacked.
        for name in ("vocoder.onnx", "text_encoder.onnx", "duration_predictor.onnx", "vector_estimator.onnx"):
            if any(cache.rglob(name)):
                return True
        return False

    async def load_model(self, model_size: str = "default") -> None:
        if self._tts is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        model_name = "supertonic"
        is_cached = self._is_model_cached()

        # supertonic downloads via huggingface_hub, so HF tqdm progress fires
        # through model_load_progress even though the destination dir is custom.
        with model_load_progress(model_name, is_cached):
            from supertonic import TTS

            logger.info("Loading Supertonic-3 (ONNX, CPU)...")
            self._tts = TTS(auto_download=True)

        logger.info("Supertonic-3 loaded successfully")

    def unload_model(self) -> None:
        if self._tts is not None:
            del self._tts
            self._tts = None
            logger.info("Supertonic-3 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Supertonic does not support zero-shot cloning from arbitrary audio.
        Voice prompts are always preset references; the cloned-profile path
        falls back to the default voice.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "supertonic",
            "preset_voice_id": SUPERTONIC_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        sr = self._sample_rate()
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=sr)

    def _sample_rate(self) -> int:
        if self._tts is not None:
            sr = getattr(self._tts, "sample_rate", None)
            if isinstance(sr, (int, float)) and sr > 0:
                return int(sr)
        return SUPERTONIC_FALLBACK_SAMPLE_RATE

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        await self.load_model()

        voice_name = (
            voice_prompt.get("preset_voice_id")
            or voice_prompt.get("supertonic_voice")
            or SUPERTONIC_DEFAULT_VOICE
        )
        lang = language if language in SUPERTONIC_LANGUAGES else "en"

        def _generate_sync():
            if seed is not None:
                # Supertonic's vector estimator samples noisy latents via np.random.randn.
                np.random.seed(seed)

            style = self._tts.get_voice_style(voice_name=voice_name)
            wav, _duration = self._tts.synthesize(text, voice_style=style, lang=lang)

            audio = np.asarray(wav, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            if audio.ndim == 0:
                audio = audio.reshape(1)
            return audio, self._sample_rate()

        return await asyncio.to_thread(_generate_sync)
