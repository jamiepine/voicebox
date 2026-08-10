"""
Tests for the prosody renderer — plan to audio.

Everything here is assembly, so the engine is a stub that returns tones. That
keeps the whole file model-free while still exercising the arithmetic that
actually matters: whether a pause lasts as long as it says, whether cutting
costs duration, and whether joins behave differently around silence.

The two rules worth pinning are both measured facts rather than preferences:
trimming run edges is what makes segmentation duration-neutral, and crossfading
into a pause would eat it from both ends.

Usage:
    python -m pytest backend/tests/test_prosody_renderer.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.prosody import Silence, Speech, compile_plan
from backend.services.prosody.ir import RenderPlan
from backend.services.prosody.renderer import (
    apply_rate,
    assemble,
    edge_silence_ms,
    render,
    trim_edges,
)

SR = 24000


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def padded(speech_s: float, lead_ms: int = 340, trail_ms: int = 100) -> np.ndarray:
    """A run shaped like a real generation: speech wrapped in the model's own
    silence. The measured figures were 340ms lead and 100ms trail."""
    lead = np.zeros(int(SR * lead_ms / 1000), dtype=np.float32)
    trail = np.zeros(int(SR * trail_ms / 1000), dtype=np.float32)
    return np.concatenate([lead, tone(speech_s), trail])


def fake_engine(seconds=1.0, sr=SR, lead_ms=340, trail_ms=100):
    """A stub engine returning realistically padded audio, fixed length."""

    async def generate_run(node: Speech):
        return padded(seconds, lead_ms, trail_ms), sr

    return generate_run


def proportional_engine(secs_per_char=0.1, sr=SR, lead_ms=340, trail_ms=100):
    """A stub whose speech length tracks the text.

    Needed for any comparison across different numbers of runs: splitting a
    sentence distributes the same words, so total speech is constant and only
    the padding multiplies. A fixed-length stub would add a whole extra run of
    speech per cut and measure nothing.
    """

    async def generate_run(node: Speech):
        return padded(len(node.text) * secs_per_char, lead_ms, trail_ms), sr

    return generate_run


def secs(audio: np.ndarray, sr: int = SR) -> float:
    return len(audio) / sr


# ── Edge trimming: the finding the design rests on ───────────────────


def test_edge_silence_is_measured():
    lead, trail = edge_silence_ms(padded(1.0, 340, 100), SR)
    assert lead == pytest.approx(340, abs=20)
    assert trail == pytest.approx(100, abs=20)


def test_trimming_removes_the_padding_but_keeps_a_cushion():
    trimmed = trim_edges(padded(1.0, 340, 100), SR)
    assert secs(trimmed) < secs(padded(1.0, 340, 100))
    lead, _ = edge_silence_ms(trimmed, SR)
    assert lead < 100, "most of the leading silence should be gone"
    assert secs(trimmed) > 1.0, "the speech itself must survive"


def test_silence_only_audio_is_not_destroyed():
    quiet = np.zeros(SR, dtype=np.float32)
    assert trim_edges(quiet, SR).size > 0


def test_empty_audio_is_handled():
    assert trim_edges(np.array([], dtype=np.float32), SR).size == 0
    assert edge_silence_ms(np.array([], dtype=np.float32), SR) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_cutting_is_duration_neutral_when_trimmed():
    """The measured result: three runs untrimmed ran ~0.7s longer than one;
    trimmed, the difference collapses. That is what makes segmentation viable."""
    # Same nine characters either way, so only the number of cuts differs.
    one_run = RenderPlan(nodes=[Speech("aaabbbccc", "en")])
    three_runs = RenderPlan(
        nodes=[Speech("aaa", "en"), Speech("bbb", "es"), Speech("ccc", "en")]
    )

    engine = proportional_engine()
    untrimmed_1, _ = await render(one_run, engine, trim_runs=False)
    untrimmed_3, _ = await render(three_runs, engine, trim_runs=False)
    trimmed_1, _ = await render(one_run, engine, trim_runs=True)
    trimmed_3, _ = await render(three_runs, engine, trim_runs=True)

    added_untrimmed = secs(untrimmed_3) - secs(untrimmed_1)
    added_trimmed = secs(trimmed_3) - secs(trimmed_1)

    assert added_untrimmed > 0.6, "two extra cuts should add real dead air"
    assert added_trimmed < added_untrimmed / 2, (
        f"trimming should recover most of it: {added_untrimmed:.2f}s -> {added_trimmed:.2f}s"
    )


# ── Pauses ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_pause_lasts_as_long_as_it_says():
    plan = RenderPlan(nodes=[Speech("a", "en"), Silence(700), Speech("b", "en")])
    with_pause, sr = await render(plan, fake_engine(), crossfade_ms=0)
    without, _ = await render(
        RenderPlan(nodes=[Speech("a", "en"), Speech("b", "en")]),
        fake_engine(),
        crossfade_ms=0,
    )
    assert secs(with_pause, sr) - secs(without, sr) == pytest.approx(0.7, abs=0.02)


@pytest.mark.asyncio
async def test_a_crossfade_does_not_eat_the_pause():
    """Overlapping a silence with its neighbours would shorten it from both
    ends, so a 700ms break would not last 700ms."""
    plan = RenderPlan(nodes=[Speech("a", "en"), Silence(700), Speech("b", "en")])
    faded, sr = await render(plan, fake_engine(), crossfade_ms=50)
    butted, _ = await render(plan, fake_engine(), crossfade_ms=0)
    assert secs(faded, sr) == pytest.approx(secs(butted, sr), abs=0.005)


@pytest.mark.asyncio
async def test_adjacent_runs_do_overlap():
    """The crossfade must still apply where it is wanted, or joins click."""
    plan = RenderPlan(nodes=[Speech("a", "en"), Speech("b", "en")])
    faded, sr = await render(plan, fake_engine(), crossfade_ms=100)
    butted, _ = await render(plan, fake_engine(), crossfade_ms=0)
    assert secs(butted, sr) - secs(faded, sr) == pytest.approx(0.1, abs=0.01)


@pytest.mark.asyncio
async def test_a_leading_pause_survives():
    """A plan can open with a break, before any run has established the rate."""
    plan = RenderPlan(nodes=[Silence(500), Speech("a", "en")])
    audio, sr = await render(plan, fake_engine(), crossfade_ms=0)
    lead, _ = edge_silence_ms(audio, sr)
    assert lead >= 450


@pytest.mark.asyncio
async def test_a_trailing_pause_survives():
    plan = RenderPlan(nodes=[Speech("a", "en"), Silence(500)])
    audio, sr = await render(plan, fake_engine(), crossfade_ms=0)
    _, trail = edge_silence_ms(audio, sr)
    assert trail >= 450


@pytest.mark.asyncio
async def test_a_plan_of_only_silence_renders_nothing():
    """Nothing established a sample rate, so there is no meaningful output."""
    audio, sr = await render(RenderPlan(nodes=[Silence(500)]), fake_engine())
    assert audio.size == 0
    assert sr == 0


# ── Rate ─────────────────────────────────────────────────────────────


def test_a_slower_rate_lengthens_without_resampling():
    original = tone(1.0)
    slower = apply_rate(original, 0.5)
    assert secs(slower) == pytest.approx(2.0, rel=0.05)


def test_rate_one_is_a_no_op():
    original = tone(1.0)
    assert apply_rate(original, 1.0) is original


@pytest.mark.asyncio
async def test_rate_applies_per_run():
    plan = RenderPlan(nodes=[Speech("a", "en"), Speech("b", "en", rate=0.5)])
    audio, sr = await render(plan, fake_engine(seconds=1.0), crossfade_ms=0)
    plain, _ = await render(
        RenderPlan(nodes=[Speech("a", "en"), Speech("b", "en")]),
        fake_engine(seconds=1.0),
        crossfade_ms=0,
    )
    assert secs(audio, sr) > secs(plain, sr)


# ── Mixed engine output ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runs_at_different_rates_are_resampled():
    """Concatenating mismatched rates would come out as a pitch shift."""
    calls = {"n": 0}

    async def varying(node: Speech):
        calls["n"] += 1
        return (padded(1.0), SR) if calls["n"] == 1 else (padded(1.0), 48000)

    plan = RenderPlan(nodes=[Speech("a", "en"), Speech("b", "en")])
    audio, sr = await render(plan, varying, crossfade_ms=0)
    assert sr == SR
    # Both runs are ~1s of speech; a mishandled rate would halve or double one.
    assert secs(audio, sr) == pytest.approx(2.0, abs=0.4)


@pytest.mark.asyncio
async def test_multichannel_output_is_folded_to_mono():
    async def stereo(node: Speech):
        mono = padded(1.0)
        return np.stack([mono, mono]), SR

    audio, _ = await render(RenderPlan(nodes=[Speech("a", "en")]), stereo)
    assert audio.ndim == 1


# ── Assembly primitives ──────────────────────────────────────────────


def test_assemble_of_nothing_is_empty():
    assert assemble([], SR).size == 0


def test_assemble_of_one_piece_is_that_piece():
    piece = tone(0.5)
    assert np.array_equal(assemble([(piece, False)], SR), piece)


# ── End to end from markup ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_compiled_script_renders():
    """The join the whole feature exists for: English, a Spanish span, a pause."""
    plan = compile_plan(
        'The shot is a <lang xml:lang="es">bandeja, no un smash,</lang> here.'
        '<break time="700ms"/>Not a smash.',
        engine="qwen",
        default_language="en",
        engine_languages=["en", "es"],
    )
    languages = [n.language for n in plan.nodes if isinstance(n, Speech)]
    assert "es" in languages

    audio, sr = await render(plan, fake_engine(seconds=0.5))
    assert sr == SR
    assert secs(audio, sr) > 0.7, "should contain every run plus the pause"
