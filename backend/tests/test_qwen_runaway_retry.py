"""Regression coverage for runaway MLX Qwen TTS output."""

from unittest.mock import patch

import numpy as np
import pytest

from backend.backends import engine_needs_trim, engine_retries_runaway
from backend.utils.audio import has_tts_runaway
from backend.utils.chunked_tts import generate_chunked

SAMPLE_RATE = 1000


def test_mlx_qwen_enables_runaway_retry_without_aggressive_trim():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        assert engine_needs_trim("qwen") is False
        assert engine_retries_runaway("qwen") is True


def test_pytorch_qwen_keeps_runaway_retry_disabled():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        assert engine_needs_trim("qwen") is False
        assert engine_retries_runaway("qwen") is False


def test_detector_flags_long_internal_silence():
    speech = np.full(2 * SAMPLE_RATE, 0.2, dtype=np.float32)
    runaway_gap = np.zeros(2500, dtype=np.float32)
    hallucinated_noise = np.full(2 * SAMPLE_RATE, 0.8, dtype=np.float32)
    audio = np.concatenate([speech, runaway_gap, hallucinated_noise])

    assert has_tts_runaway(audio, SAMPLE_RATE) is True


def test_detector_ignores_normal_internal_pause():
    speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    normal_pause = np.zeros(1200, dtype=np.float32)
    audio = np.concatenate([speech, normal_pause, speech])

    assert has_tts_runaway(audio, SAMPLE_RATE) is False


def test_trailing_silence_is_not_a_runaway():
    speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    trailing_silence = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)

    assert (
        has_tts_runaway(
            np.concatenate([speech, trailing_silence]),
            SAMPLE_RATE,
        )
        is False
    )


@pytest.mark.asyncio
async def test_runaway_chunk_is_retried_as_smaller_chunks():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            if len(text) > 200:
                speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
                silence = np.zeros(2500, dtype=np.float32)
                noise = np.full(SAMPLE_RATE, 0.8, dtype=np.float32)
                return np.concatenate([speech, silence, noise]), SAMPLE_RATE
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()
    text = f"{'A' * 119}. {'B' * 119}."

    audio, sample_rate = await generate_chunked(
        backend,
        text,
        {},
        max_chunk_chars=800,
        crossfade_ms=50,
        runaway_detector=has_tts_runaway,
    )

    assert sample_rate == SAMPLE_RATE
    assert backend.calls == [text, f"{'A' * 119}.", f"{'B' * 119}."]
    assert len(audio) == 1950


@pytest.mark.asyncio
async def test_persistent_runaway_fails_instead_of_returning_corrupt_audio():
    class AlwaysRunawayBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
            silence = np.zeros(2500, dtype=np.float32)
            noise = np.full(SAMPLE_RATE, 0.8, dtype=np.float32)
            return np.concatenate([speech, silence, noise]), SAMPLE_RATE

    backend = AlwaysRunawayBackend()
    text = f"{'A' * 119}. {'B' * 119}."

    with pytest.raises(
        RuntimeError,
        match="remained unstable after retrying smaller text chunks",
    ):
        await generate_chunked(
            backend,
            text,
            {},
            max_chunk_chars=800,
            runaway_detector=has_tts_runaway,
        )

    assert [len(call) for call in backend.calls] == [241, 120, 100]
