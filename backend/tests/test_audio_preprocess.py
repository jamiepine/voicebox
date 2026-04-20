"""
Unit tests for reference-audio preprocessing.

Covers :func:`backend.utils.audio.preprocess_reference_audio` and
:func:`backend.utils.audio.validate_and_load_reference_audio`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio import (  # noqa: E402
    preprocess_reference_audio,
    validate_and_load_reference_audio,
)


SR = 24000


def _tone(duration_s: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_peak_cap_scales_hot_input():
    audio = _tone(3.0, amp=0.99)
    out = preprocess_reference_audio(audio, SR)
    assert np.abs(out).max() <= 0.951


def test_peak_cap_leaves_moderate_input_untouched():
    audio = _tone(3.0, amp=0.5)
    out = preprocess_reference_audio(audio, SR)
    assert np.isclose(np.abs(out).max(), 0.5, atol=1e-3)


def test_dc_offset_removed():
    audio = _tone(3.0, amp=0.3) + 0.1
    out = preprocess_reference_audio(audio, SR)
    assert abs(float(np.mean(out))) < 1e-3


def test_silence_is_trimmed_with_padding_kept():
    silence = np.zeros(int(SR * 1.0), dtype=np.float32)
    speech = _tone(3.0, amp=0.3)
    audio = np.concatenate([silence, speech, silence])
    out = preprocess_reference_audio(audio, SR)
    # Most of the 2s of leading/trailing silence should be gone, but the
    # 3s of speech plus ~200ms of padding should remain.
    assert len(audio) - len(out) >= SR, "expected >=1s of silence trimmed"
    assert len(out) >= int(3.0 * SR), "speech body should be preserved"


def test_clean_audio_is_not_padded_past_original_length():
    # Well-recorded audio with no edge silence shouldn't get longer after
    # preprocessing — otherwise a 29.9 s upload could be pushed past the
    # 30 s max_duration ceiling downstream.
    audio = _tone(3.0, amp=0.3)
    out = preprocess_reference_audio(audio, SR)
    assert len(out) <= len(audio)


def test_empty_input_returns_empty():
    out = preprocess_reference_audio(np.zeros(0, dtype=np.float32), SR)
    assert out.size == 0


def test_validate_accepts_previously_rejected_hot_file(tmp_path):
    audio = _tone(3.0, amp=0.995)
    path = tmp_path / "hot.wav"
    sf.write(str(path), audio, SR)

    ok, err, out_audio, out_sr = validate_and_load_reference_audio(str(path))

    assert ok, f"expected pass, got error: {err}"
    assert out_audio is not None
    assert out_sr == SR
    assert np.abs(out_audio).max() <= 0.951


def test_validate_still_rejects_silent_input(tmp_path):
    audio = np.zeros(int(SR * 3.0), dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, SR)

    ok, err, _, _ = validate_and_load_reference_audio(str(path))

    assert not ok
    assert err is not None
    assert "too short" in err.lower() or "quiet" in err.lower()


def test_validate_rejects_too_short(tmp_path):
    audio = _tone(0.5, amp=0.3)
    path = tmp_path / "short.wav"
    sf.write(str(path), audio, SR)

    ok, err, _, _ = validate_and_load_reference_audio(str(path))

    assert not ok
    assert "too short" in (err or "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
