"""Tests for preserving real speech after long internal pauses."""

import numpy as np
import pytest

from backend.utils.audio import trim_tts_output

SAMPLE_RATE = 24_000


def _speech(seconds: float) -> np.ndarray:
    return np.full(round(SAMPLE_RATE * seconds), 0.2, dtype=np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(SAMPLE_RATE * seconds), dtype=np.float32)


def _noise(seconds: float) -> np.ndarray:
    return np.full(round(SAMPLE_RATE * seconds), 0.05, dtype=np.float32)


def test_keeps_long_speech_continuation_after_internal_pause():
    audio = np.concatenate([_speech(3), _silence(1.5), _speech(4)])

    trimmed = trim_tts_output(audio, sample_rate=SAMPLE_RATE)

    assert len(trimmed) / SAMPLE_RATE == pytest.approx(8.5, abs=0.02)


def test_still_cuts_short_noise_after_internal_pause():
    audio = np.concatenate([_speech(3), _silence(1.5), _noise(0.4)])

    trimmed = trim_tts_output(audio, sample_rate=SAMPLE_RATE)

    assert len(trimmed) / SAMPLE_RATE == pytest.approx(3.0, abs=0.02)


def test_keeps_long_continuation_then_cuts_short_noise_tail():
    audio = np.concatenate([_speech(3), _silence(1.5), _speech(4), _silence(1.5), _noise(0.3)])

    trimmed = trim_tts_output(audio, sample_rate=SAMPLE_RATE)

    assert len(trimmed) / SAMPLE_RATE == pytest.approx(8.5, abs=0.02)


def test_output_without_internal_pause_is_unchanged():
    audio = _speech(5)

    trimmed = trim_tts_output(audio, sample_rate=SAMPLE_RATE)

    assert len(trimmed) == len(audio)
    fade_samples = int(SAMPLE_RATE * 30 / 1000)
    np.testing.assert_array_equal(trimmed[:-fade_samples], audio[:-fade_samples])
