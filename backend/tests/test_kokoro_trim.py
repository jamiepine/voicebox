"""Test Kokoro short prompt audio trimming and engine config."""

import numpy as np
import pytest

from backend.backends import engine_needs_trim, get_model_config
from backend.utils.audio import trim_tts_output


def test_kokoro_engine_does_not_use_generic_trim():
    """Kokoro trims inside the backend; the generic needs_trim path would cut
    multi-segment output at the first >1s inter-segment gap."""
    assert engine_needs_trim("kokoro") is False
    config = get_model_config("kokoro")
    assert config is not None
    assert config.needs_trim is False


def test_trim_tts_output_edge_only_keeps_internal_gaps():
    """With max_internal_silence_ms=None only the edges are trimmed."""
    sr = 24000
    speech = np.full(int(sr * 1.0), 0.2, dtype=np.float32)
    gap = np.zeros(int(sr * 1.5), dtype=np.float32)  # longer than the 1s default cut
    pad = np.zeros(int(sr * 0.5), dtype=np.float32)
    raw_audio = np.concatenate([pad, speech, gap, speech, pad])

    default_trim = trim_tts_output(raw_audio, sample_rate=sr)
    edge_trim = trim_tts_output(raw_audio, sample_rate=sr, max_internal_silence_ms=None)

    # Default behaviour cuts at the internal gap and drops the second utterance.
    assert len(default_trim) / sr == pytest.approx(1.0, abs=0.25)
    # Edge-only trim keeps both utterances and the gap between them.
    assert len(edge_trim) / sr == pytest.approx(1.0 + 1.5 + 1.0 + 0.2, abs=0.05)


def test_kokoro_trim_tts_output_removes_trailing_dead_space():
    """Verify trim_tts_output removes trailing silence past speech."""
    sr = 24000
    speech = np.full(int(sr * 1.5), 0.2, dtype=np.float32)  # 1.5s speech
    trailing_silence = np.zeros(int(sr * 1.0), dtype=np.float32)  # 1.0s trailing dead space
    raw_audio = np.concatenate([speech, trailing_silence])

    trimmed = trim_tts_output(raw_audio, sample_rate=sr)

    # Trimming cuts trailing silence from 2.5s down to speech duration boundary (1.5s)
    expected_dur_samples = int(sr * 1.5)
    assert len(trimmed) == expected_dur_samples
    assert len(trimmed) < len(raw_audio)


@pytest.mark.asyncio
async def test_kokoro_backend_generate_applies_trimming(monkeypatch):
    """Verify KokoroTTSBackend.generate applies trimming on synthesized output."""
    from backend.backends.kokoro_backend import KokoroTTSBackend, KOKORO_SAMPLE_RATE

    backend = KokoroTTSBackend()

    # Mock _load_model_sync to avoid requiring real model load in pure unit test
    monkeypatch.setattr(backend, "_load_model_sync", lambda: None)
    monkeypatch.setattr(backend, "_model", object())

    # Mock KPipeline output to yield audio with 1s trailing silence
    sr = KOKORO_SAMPLE_RATE
    speech = np.full(int(sr * 1.0), 0.2, dtype=np.float32)
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    fake_audio = np.concatenate([speech, silence])

    class FakeResult:
        def __init__(self, audio):
            self.audio = audio

    class FakePipeline:
        def __call__(self, text, voice, speed=1.0):
            yield FakeResult(fake_audio)
            yield FakeResult(fake_audio)

    monkeypatch.setattr(backend, "_get_pipeline", lambda lang: FakePipeline())

    audio, sample_rate = await backend.generate("Read it back to me.", voice_prompt={})

    assert sample_rate == sr
    # Two segments of (1s speech + 1s silence) = 4.0s raw. Only the trailing
    # pad is trimmed (down to a 0.2s tail); the 1s gap between the segments
    # must survive, otherwise the second segment would be dropped.
    assert len(audio) / sr == pytest.approx(1.0 + 1.0 + 1.0 + 0.2, abs=0.05)
    assert len(audio) < 2 * len(fake_audio)
