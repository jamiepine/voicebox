"""Turn a RenderPlan into audio.

Everything here is assembly. The engine is called once per speech run with that
run's own settings; silences, rate and joins are produced by arithmetic on the
resulting arrays, which is why pauses and per-span language work on all eight
engines including the ones that accept no directives at all.

Two things this must get right, both measured rather than assumed:

*trim the interior joins only*
    Every generation carries its own leading and trailing silence (250-360ms
    lead, 100-400ms trail on the voices measured). Where two runs meet, that is
    two lots of dead air the author never asked for, and it compounds: a
    three-run sentence runs ~0.7s longer than the same words in one shot.
    Trimmed at the joins, the difference is ~0.03s.

    But the *outer* edges are the utterance's natural lead-in and release.
    Trimming those makes every clip end abruptly after its last sound, which
    reads as a forced delivery even on text carrying no markup at all.

*do not crossfade into a pause*
    A crossfade across a Silence eats the pause from both ends. Runs joined to
    each other overlap; runs adjacent to silence are butted.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .ir import RenderPlan, Silence, Speech

logger = logging.getLogger(__name__)

DEFAULT_CROSSFADE_MS = 50

# Below this level a frame counts as silence for edge trimming.
_SILENCE_DB = -45.0
_FRAME_MS = 10
# Left on after trimming so a crossfade has something to work with and the
# speech does not start hard against the join.
_EDGE_CUSHION_MS = 30


def edge_silence_ms(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Leading and trailing near-silence, in milliseconds."""
    if audio.size == 0:
        return 0.0, 0.0
    frame = max(1, int(sr * _FRAME_MS / 1000))
    usable = len(audio) - (len(audio) % frame)
    if usable < frame:
        return 0.0, 0.0
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
    loud = np.where(20 * np.log10(rms) > _SILENCE_DB)[0]
    if len(loud) == 0:
        return float(len(audio) / sr * 1000), 0.0
    return float(loud[0] * _FRAME_MS), float((len(frames) - 1 - loud[-1]) * _FRAME_MS)


def trim_edges(
    audio: np.ndarray,
    sr: int,
    cushion_ms: int = _EDGE_CUSHION_MS,
    *,
    lead: bool = True,
    trail: bool = True,
) -> np.ndarray:
    """Strip the model's own silence from the chosen edges.

    Only *interior* edges should be trimmed. A generation's trailing silence is
    the utterance's natural release -- roughly 290ms on the voices measured --
    and cutting it back to the cushion makes every clip end abruptly, which
    reads as a forced delivery even on text with no markup at all. The leading
    silence at the very start is the same story.

    What genuinely accumulates is the *join*: run N's trail butted against run
    N+1's lead is two lots of dead air the author never asked for, and it
    compounds with the number of runs. So the renderer trims where runs meet
    and leaves the outer boundary of the whole utterance alone.
    """
    if audio.size == 0:
        return audio
    lead_ms, trail_ms = edge_silence_ms(audio, sr)
    cushion = int(sr * cushion_ms / 1000)
    start = max(0, int(sr * lead_ms / 1000) - cushion) if lead else 0
    end = len(audio) - (max(0, int(sr * trail_ms / 1000) - cushion) if trail else 0)
    return audio[start:end] if end > start else audio


def apply_rate(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change tempo without changing pitch.

    Phase vocoder, matching the story mixer -- resampling would transpose the
    voice, which is not what a rate directive means.
    """
    if rate == 1.0 or audio.size == 0:
        return audio
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)


def _crossfade(a: np.ndarray, b: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate([a, b])
    overlap = min(samples, len(a), len(b))
    out = np.array(a, dtype=np.float32, copy=True)
    fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    out[-overlap:] = out[-overlap:] * fade_out + b[:overlap] * fade_in
    return np.concatenate([out, b[overlap:]])


def assemble(
    pieces: list[tuple[np.ndarray, bool]],
    sr: int,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
) -> np.ndarray:
    """Join rendered pieces.

    ``pieces`` is ``(audio, is_silence)``. A crossfade is applied only between
    two speech runs: overlapping a pause with its neighbours would shorten it
    from both ends, so a 700ms break would not last 700ms.
    """
    if not pieces:
        return np.array([], dtype=np.float32)

    samples = int(sr * crossfade_ms / 1000)
    out = np.asarray(pieces[0][0], dtype=np.float32)
    prev_is_silence = pieces[0][1]

    for audio, is_silence in pieces[1:]:
        audio = np.asarray(audio, dtype=np.float32)
        if is_silence or prev_is_silence:
            out = np.concatenate([out, audio])
        else:
            out = _crossfade(out, audio, samples)
        prev_is_silence = is_silence

    return out


async def render(
    plan: RenderPlan,
    generate_run,
    *,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    trim_runs: bool = True,
) -> tuple[np.ndarray, int]:
    """Render *plan* to audio.

    Args:
        plan: A compiled plan. ``plan.is_trivial`` should be handled by the
            caller on the existing single-shot path; this still renders it
            correctly, just with no benefit.
        generate_run: ``async (Speech) -> (audio, sample_rate)``. Injected
            rather than imported so the renderer can be tested without a model
            and so long runs can still go through ``generate_chunked``.
        crossfade_ms: Overlap between adjacent speech runs. 0 for a butt join.
        trim_runs: Trim silence at the joins between runs. The first run's
            lead-in and the last run's release are always kept -- they are the
            utterance's own boundary, not an artefact of cutting.

    Returns:
        ``(audio, sample_rate)``.
    """
    pieces: list[tuple[np.ndarray, bool]] = []
    sample_rate: int | None = None
    pending_silence_ms = 0

    # Which speech runs sit at the very start and very end of the utterance.
    # Those outer edges keep the model's own lead-in and release; everything
    # between them is an interior join and gets trimmed.
    speech_positions = [i for i, n in enumerate(plan.nodes) if isinstance(n, Speech)]
    first_speech = speech_positions[0] if speech_positions else None
    last_speech = speech_positions[-1] if speech_positions else None

    for index, node in enumerate(plan.nodes):
        if isinstance(node, Silence):
            # Held until a sample rate is known -- a plan can legitimately open
            # with a pause, before any run has told us the rate.
            pending_silence_ms += node.ms
            if sample_rate is not None:
                pieces.append((_silence(pending_silence_ms, sample_rate), True))
                pending_silence_ms = 0
            continue

        if not isinstance(node, Speech):
            continue

        audio, run_sr = await generate_run(node)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        if sample_rate is None:
            sample_rate = int(run_sr)
            if pending_silence_ms:
                pieces.append((_silence(pending_silence_ms, sample_rate), True))
                pending_silence_ms = 0
        elif int(run_sr) != sample_rate:
            # Mixed rates would otherwise concatenate into a pitch shift.
            logger.info(
                "Resampling a prosody run from %dHz to %dHz", int(run_sr), sample_rate
            )
            audio = librosa.resample(audio, orig_sr=int(run_sr), target_sr=sample_rate)

        if trim_runs:
            audio = trim_edges(
                audio,
                sample_rate,
                lead=index != first_speech,
                trail=index != last_speech,
            )
        audio = apply_rate(audio, node.rate)

        if audio.size:
            pieces.append((audio, False))

    if sample_rate is None:
        # Silence-only plan; nothing established a rate.
        return np.array([], dtype=np.float32), 0

    if pending_silence_ms:
        pieces.append((_silence(pending_silence_ms, sample_rate), True))

    return assemble(pieces, sample_rate, crossfade_ms=crossfade_ms), sample_rate


def _silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(sr * ms / 1000)), dtype=np.float32)
