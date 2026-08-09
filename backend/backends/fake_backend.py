"""Fake TTS backend for UI and E2E testing.

Activated by ``VOICEBOX_FAKE_TTS=1``. Every engine resolves to this backend,
which synthesizes a quiet sine tone sized to the input text — so the full
generation pipeline (task queue, SSE progress, database rows, audio serving)
runs exactly as in production, minus model weights and GPU time.
"""

import asyncio
import logging
from typing import ClassVar, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
SECONDS_PER_CHAR = 0.02
MIN_DURATION_S = 0.25
TONE_HZ = 440.0
AMPLITUDE = 0.1


class FakeTTSBackend:
    """Implements the TTSBackend protocol without any model."""

    MODEL_CONFIGS: ClassVar[list] = []

    def __init__(self) -> None:
        self._loaded = False

    async def load_model(self, model_size: str = "default") -> None:
        if self._loaded:
            return
        # Brief pause so the UI's loading_model state is observable.
        await asyncio.sleep(0.1)
        self._loaded = True
        logger.info("Fake TTS backend loaded (VOICEBOX_FAKE_TTS)")

    async def load_model_async(self, model_size: str = "default") -> None:
        # Qwen engines are loaded through this variant (see load_engine_model).
        await self.load_model(model_size)

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        return ({"fake": True, "audio_path": audio_path, "reference_text": reference_text}, False)

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        combined_text = " ".join(reference_texts)
        return np.zeros(SAMPLE_RATE, dtype=np.float32), combined_text

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        duration_s = max(MIN_DURATION_S, len(text) * SECONDS_PER_CHAR)
        # Yield once so cancellation has a window, mirroring real inference.
        await asyncio.sleep(0.05)
        t = np.linspace(0.0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
        audio = (AMPLITUDE * np.sin(2.0 * np.pi * TONE_HZ * t)).astype(np.float32)
        return audio, SAMPLE_RATE

    def unload_model(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def _get_model_path(self, model_size: str) -> str:
        return "fake"


_fake_backend: Optional[FakeTTSBackend] = None


def get_fake_backend() -> FakeTTSBackend:
    global _fake_backend
    if _fake_backend is None:
        _fake_backend = FakeTTSBackend()
    return _fake_backend
