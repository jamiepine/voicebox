"""Coverage for the Qwen3-TTS EOS-miss guards.

Qwen3-TTS occasionally misses its codec EOS. The decode then runs to
``max_new_tokens`` (the checkpoint ships 8192 ≈ 11.4 min of audio) and
appends minutes of near-silent gibberish to the clip. These tests cover
the three layers that bound the damage:

1. ``estimate_max_new_tokens`` — per-call decode budget.
2. ``exceeds_plausible_speech_duration`` / ``detect_tts_runaway`` —
   catching runaways the internal-silence detector misses (measured in
   the wild: 175 chars of text produced 613s of low-level gibberish
   with no internal silence gap).
3. Engine wiring — qwen engines retry flagged chunks on every backend
   and default to smaller chunks unless the request overrides.
"""

import numpy as np

from backend.backends import (
    get_engine_default_chunk_chars,
    get_tts_model_configs,
)
from backend.backends.base import (
    QWEN_MAX_NEW_TOKENS_CAP,
    estimate_max_new_tokens,
)
from backend.utils.audio import (
    detect_tts_runaway,
    exceeds_plausible_speech_duration,
)

SAMPLE_RATE = 24000


# ── decode budget ────────────────────────────────────────────────────


def test_short_text_gets_minimum_budget():
    assert estimate_max_new_tokens("hi") == 1500


def test_budget_scales_with_text_length():
    # 300 chars ≈ a default qwen chunk; the budget must exceed the
    # slowest sane delivery (~6 tokens/char at 12 Hz) with headroom.
    budget = estimate_max_new_tokens("a" * 300)
    assert budget == 300 * 12 + 750
    assert budget / 300 >= 12


def test_budget_is_capped_at_model_limit():
    assert estimate_max_new_tokens("a" * 100_000) == QWEN_MAX_NEW_TOKENS_CAP


def test_budget_never_truncates_slow_normal_delivery():
    # Worst sane pace: ~2 chars/sec incl. pauses = 6 tokens/char.
    # The budget must exceed that for any length below the cap.
    for chars in (50, 175, 300, 500, 800):
        budget = estimate_max_new_tokens("a" * chars)
        slow_normal = chars * 6
        expected_cap = QWEN_MAX_NEW_TOKENS_CAP
        if slow_normal < expected_cap:
            assert budget >= slow_normal, chars


# ── duration-based runaway detection ─────────────────────────────────


def test_duration_detector_flags_eos_miss_runaway():
    # Real-world failure: 175 chars of text → 613s of low-level noise.
    audio = np.full(613 * SAMPLE_RATE, 0.01, dtype=np.float32)
    assert exceeds_plausible_speech_duration(audio, SAMPLE_RATE, "a" * 175)
    assert detect_tts_runaway(audio, SAMPLE_RATE, "a" * 175)


def test_duration_detector_passes_normal_delivery():
    # 800 chars at a very deliberate 2 chars/sec ≈ 400s — well within
    # the plausible envelope (800 / 1.2 + 15 ≈ 682s).
    audio = np.full(400 * SAMPLE_RATE, 0.2, dtype=np.float32)
    assert not exceeds_plausible_speech_duration(audio, SAMPLE_RATE, "a" * 800)
    assert not detect_tts_runaway(audio, SAMPLE_RATE, "a" * 800)


def test_duration_detector_tolerates_trailing_silence():
    # Normal clip plus a couple seconds of trailing silence.
    speech = np.full(30 * SAMPLE_RATE, 0.2, dtype=np.float32)
    silence = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    audio = np.concatenate([speech, silence])
    assert not detect_tts_runaway(audio, SAMPLE_RATE, "a" * 100)


def test_duration_detector_ignores_whitespace_padding():
    # Blank lines / indent runs must not raise the threshold: 100 spoken
    # chars + 1500 padding chars still caps the plausible duration at
    # 100/1.2 + 15 ≈ 98s, so a 110s runaway is flagged.
    padded_text = ("a" * 100) + ("\n\n  \t" * 250)
    audio = np.full(110 * SAMPLE_RATE, 0.01, dtype=np.float32)
    assert exceeds_plausible_speech_duration(audio, SAMPLE_RATE, padded_text)


def test_duration_detector_handles_empty_input():
    assert not exceeds_plausible_speech_duration(np.array([], dtype=np.float32), SAMPLE_RATE, "text")
    assert not exceeds_plausible_speech_duration(np.full(1000, 0.2, dtype=np.float32), SAMPLE_RATE, "")


# ── engine wiring ────────────────────────────────────────────────────


def test_qwen_engines_default_to_smaller_chunks():
    assert get_engine_default_chunk_chars("qwen") == 300
    assert get_engine_default_chunk_chars("qwen_custom_voice") == 300


def test_other_engines_have_no_chunk_override():
    assert get_engine_default_chunk_chars("kokoro") is None
    assert get_engine_default_chunk_chars("luxtts") is None


def test_every_qwen_config_carries_the_guards():
    qwen_configs = [
        c
        for c in get_tts_model_configs()
        if c.engine in ("qwen", "qwen_custom_voice")
    ]
    assert len(qwen_configs) == 4
    for cfg in qwen_configs:
        assert cfg.retries_runaway, cfg.model_name
        assert cfg.default_chunk_chars == 300, cfg.model_name
