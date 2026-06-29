"""Services for SRT-driven dubbing projects."""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import io
import json
import logging
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np
import soundfile as sf
from sqlalchemy.orm import Session

from .. import config, models
from ..database import DubbingProject, DubbingSegment, Generation as DBGeneration, get_db
from ..services import history, profiles, transcribe
from ..services.generation import run_generation
from ..services.task_queue import create_background_task, enqueue_generation
from ..utils.audio import load_audio, time_stretch_audio_file_with_ffmpeg
from ..utils.tasks import get_task_manager
from .srt_parser import parse_srt_text

logger = logging.getLogger(__name__)
PACE_MIN = 0.8
PACE_MAX = 1.2
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.2
DUBBING_CUT_LEAD_IN_MS = 50
DUBBING_CUT_TAIL_OUT_MS = 180
WORD_ALIGNMENT_MIN_SCORE = 0.72
WORD_ALIGNMENT_SEARCH_SLACK = 6
AUTO_CUT_RMS_FRAME_MS = 8
AUTO_CUT_RMS_SEARCH_MS = 160
AUTO_CUT_TAIL_SEARCH_AFTER_MS = 420
AUTO_CUT_ATTACK_SEARCH_BEFORE_MS = 260
AUTO_CUT_ATTACK_SEARCH_AFTER_MS = 240
AUTO_CUT_WORD_ATTACK_PRE_MS = 120
AUTO_CUT_WORD_ATTACK_POST_MS = 180
AUTO_CUT_WORD_ATTACK_MIN_RANGE = 1e-6
AUTO_CUT_MISSING_SILENCE_THRESHOLD_MS = 50
AUTO_CUT_SOFT_ACOUSTIC_GAP_MIN_MS = 55
AUTO_CUT_SOFT_ACOUSTIC_MAX_DRIFT_MS = 140
AUTO_CUT_DEBUG_SCHEMA_VERSION = 3
AUTO_CUT_ZCR_MIN_RMS_FACTOR = 0.04
AUTO_CUT_OVERLAP_GUARD_MS = 8
MATCH_APOSTROPHE_RE = re.compile(r"['’`´]")
MATCH_PUNCTUATION_RE = re.compile(r"[^\w\sÀ-ÖØ-öø-ÿ]", re.UNICODE)
TERMINAL_PUNCTUATION_RE = re.compile(r'[.!?…]["”»\')\]]*\s*$')
SOFT_PUNCTUATION_RE = re.compile(r'[,;:]["â€Â»\')\]]*\s*$')
DUBBING_TIMING_RETRY_RE = re.compile(
    r"\s*Timing fit retry\s+\d+\s*:\s*.*?(?=(?:\s+Timing fit retry\s+\d+\s*:)|$)",
    re.IGNORECASE | re.DOTALL,
)
DUBBING_FORCED_TIMING_SENTENCES_RE = re.compile(
    r"\s*(?:target the subtitle window precisely|speak noticeably faster|minimize pauses|"
    r"keep the sentence very compact)[^.?!]*(?:[.?!]|$)",
    re.IGNORECASE,
)
FULL_NARRATION_GENERATION_PREFIX = "dubbing-full-narration"
DUBBING_CUT_GENERATION_PREFIX = "dubbing-cut"
QWEN_DUBBING_ENGINES = {"qwen", "qwen_custom_voice", "qwen_voice_design"}
AUTO_CUT_LANGUAGE_NAMES = {"en": "English", "fr": "French"}
AUTO_CUT_LANGUAGE_HINTS = {
    "en": {
        "the",
        "and",
        "you",
        "your",
        "this",
        "that",
        "with",
        "for",
        "from",
        "are",
        "is",
        "in",
        "of",
        "to",
        "we",
        "will",
        "today",
    },
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "de",
        "du",
        "un",
        "une",
        "et",
        "vous",
        "nous",
        "dans",
        "pour",
        "sur",
        "avec",
        "qui",
        "que",
        "est",
        "ce",
        "cette",
    },
}
FRENCH_ACCENT_RE = re.compile(r"[àâçéèêëîïôùûüÿœæ]", re.IGNORECASE)


def clamp_pace(value: float | None) -> float | None:
    if value is None:
        return None
    return max(PACE_MIN, min(PACE_MAX, float(value)))


def clamp_temperature(value: float | None) -> float | None:
    if value is None:
        return None
    return max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, float(value)))


def sanitize_dubbing_instructions(value: str | None) -> str | None:
    """Keep dubbing delivery natural by stripping old retry/timing coercion hints."""
    text = (value or "").strip()
    if not text:
        return None
    text = DUBBING_TIMING_RETRY_RE.sub(" ", text)
    text = DUBBING_FORCED_TIMING_SENTENCES_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text or None


def is_qwen_dubbing_engine(engine: str | None) -> bool:
    return (engine or "qwen") in QWEN_DUBBING_ENGINES


def detect_srt_text_language(segments: list[DubbingSegment]) -> str | None:
    """Small en/fr text-language guard for Auto Cut alignment safety."""
    text = " ".join((segment.text or "") for segment in segments).lower()
    if not text.strip():
        return None

    tokens = re.findall(r"[a-zàâçéèêëîïôùûüÿœæ]+", text, flags=re.IGNORECASE)
    if len(tokens) < 4:
        return None

    scores = {
        language: sum(1 for token in tokens if token in hints)
        for language, hints in AUTO_CUT_LANGUAGE_HINTS.items()
    }
    if FRENCH_ACCENT_RE.search(text):
        scores["fr"] += 2

    best_language = max(scores, key=scores.get)
    other_language = "fr" if best_language == "en" else "en"
    best_score = scores[best_language]
    other_score = scores[other_language]
    if best_score < 2:
        return None
    if best_score < other_score + 2 and best_score < other_score * 1.5:
        return None
    return best_language


def validate_auto_cut_language(project: DubbingProject, segments: list[DubbingSegment]) -> None:
    """Prevent Whisper word alignment when project language obviously mismatches SRT text."""
    project_language = (project.language or "").strip().lower()
    if project_language not in AUTO_CUT_LANGUAGE_NAMES:
        return
    detected_language = detect_srt_text_language(segments)
    if detected_language is None or detected_language == project_language:
        return

    expected = AUTO_CUT_LANGUAGE_NAMES.get(project_language, project_language)
    detected = AUTO_CUT_LANGUAGE_NAMES.get(detected_language, detected_language)
    raise ValueError(
        "Auto Cut language mismatch: "
        f"project language is {expected}, but SRT text appears to be {detected}. "
        f"Set the project language to {detected} before running Auto Cut."
    )


def release_dubbing_stt_memory(reason: str) -> None:
    """Release SRT2Voice-only STT/GPU memory after alignment-heavy tasks."""
    try:
        transcribe.unload_whisper_model()
    except Exception:
        logger.debug("SRT2Voice STT unload skipped after %s", reason, exc_info=True)
    try:
        import gc

        gc.collect()
    except Exception:
        logger.debug("SRT2Voice GC cleanup skipped after %s", reason, exc_info=True)
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logger.debug("SRT2Voice CUDA cache cleanup skipped after %s", reason, exc_info=True)


def release_dubbing_tts_memory(reason: str) -> int:
    """Release TTS engines before/after SRT2Voice work so VRAM is not pinned."""
    unloaded = 0
    try:
        from ..backends import unload_all_tts_backends

        unloaded = unload_all_tts_backends()
    except Exception:
        logger.debug("SRT2Voice TTS unload skipped after %s", reason, exc_info=True)
    try:
        import gc

        gc.collect()
    except Exception:
        logger.debug("SRT2Voice TTS GC cleanup skipped after %s", reason, exc_info=True)
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logger.debug("SRT2Voice TTS CUDA cache cleanup skipped after %s", reason, exc_info=True)
    return unloaded


def full_narration_generation_id(project_id: str) -> str:
    """Stable generation id used for one-piece SRT narration beta output."""
    return f"{FULL_NARRATION_GENERATION_PREFIX}-{project_id}"


def _full_narration_timing_path(generation_id: str) -> Path:
    return config.get_generations_dir() / "dubbing_full_narration_timing" / f"{generation_id}.json"


def _clean_srt_narration_text_path(project_id: str) -> Path:
    return config.get_generations_dir() / "srt2voice_clean_text" / f"{project_id}.txt"


def _safe_debug_filename(value: str | None, fallback: str) -> str:
    """Return a Windows-safe debug filename without changing user-facing names."""
    name = (value or fallback).strip() or fallback
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    safe = re.sub(r"\s+", " ", safe).strip(" .")
    return safe or fallback


def _clean_srt_narration_text_alias_path(project: DubbingProject, generation_id: str | None = None) -> Path:
    """Human-readable clean text copy near full narration timing debug files."""
    project_name = _safe_debug_filename(project.name, project.id)
    if generation_id:
        safe_generation_id = _safe_debug_filename(generation_id, project.id)
        filename = f"{project_name}__{safe_generation_id}"
    else:
        filename = project_name
    return config.get_generations_dir() / "dubbing_full_narration_timing" / f"{filename}.txt"


def reset_full_narration_timing(generation_id: str) -> None:
    """Remove stale timing metadata before starting a new full narration run."""
    timing_path = _full_narration_timing_path(generation_id)
    try:
        timing_path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not reset full narration timing metadata for %s", generation_id, exc_info=True)


def write_full_narration_timing(generation_id: str, elapsed_ms: int) -> None:
    """Persist the real runtime of a full narration generation."""
    timing_path = _full_narration_timing_path(generation_id)
    try:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(
                {
                    "generation_id": generation_id,
                    "elapsed_ms": max(0, int(elapsed_ms)),
                    "recorded_at_ms": int(round(time.time() * 1000)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Could not write full narration timing metadata for %s", generation_id, exc_info=True)


def read_full_narration_elapsed_ms(generation_id: str) -> int | None:
    """Read the persisted real runtime of a full narration generation."""
    timing_path = _full_narration_timing_path(generation_id)
    if not timing_path.exists():
        return None
    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
        elapsed_ms = int(payload.get("elapsed_ms"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return max(0, elapsed_ms)


def get_full_narration_generation(project_id: str, db: Session) -> DBGeneration | None:
    return db.query(DBGeneration).filter_by(id=full_narration_generation_id(project_id)).first()


async def invalidate_project_cut_artifacts(project_id: str, db: Session) -> None:
    """Drop derived Auto Cut/manual cut state when the source full WAV changes."""
    segments = (
        db.query(DubbingSegment)
        .filter_by(project_id=project_id)
        .order_by(DubbingSegment.segment_order.asc())
        .all()
    )
    for segment in segments:
        cut_generation = get_cut_generation(segment, db)
        if cut_generation is not None:
            await history.delete_generation(cut_generation.id, db)
        segment.actual_duration_ms = None
        segment.delta_ms = None
        segment.fit_status = "unknown"
        if segment.status not in {"failed", "generating"}:
            segment.status = "pending"

    cut_dir = config.get_generations_dir() / "dubbing_cuts" / project_id
    if cut_dir.exists():
        shutil.rmtree(cut_dir, ignore_errors=True)
    db.commit()


def cut_generation_id(segment: DubbingSegment) -> str:
    """Stable generation id for a segment cut derived from the full narration WAV."""
    return f"{DUBBING_CUT_GENERATION_PREFIX}-{segment.id}"


def get_cut_generation(segment: DubbingSegment, db: Session) -> DBGeneration | None:
    return db.query(DBGeneration).filter_by(id=cut_generation_id(segment)).first()


def list_cut_generations(project_id: str, db: Session) -> dict[str, DBGeneration]:
    segments = list_project_segments(project_id, db)
    ids_by_segment_id = {cut_generation_id(segment): segment.id for segment in segments}
    if not ids_by_segment_id:
        return {}
    rows = db.query(DBGeneration).filter(DBGeneration.id.in_(ids_by_segment_id.keys())).all()
    return {ids_by_segment_id[row.id]: row for row in rows if row.audio_path}


def _latest_manual_cut_bounds(project_id: str) -> dict[str, dict[str, int]]:
    """Return the latest persisted manual cut bounds by segment id."""
    debug_path = config.get_generations_dir() / "dubbing_cuts" / project_id / "manual_cuts.jsonl"
    if not debug_path.exists():
        return {}

    bounds: dict[str, dict[str, int]] = {}
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        segment_id = item.get("segment_id")
        if not isinstance(segment_id, str):
            continue
        try:
            bounds[segment_id] = {
                "cut_start_ms": int(item.get("cut_start_ms", 0)),
                "cut_end_ms": int(item.get("cut_end_ms", 0)),
            }
        except (TypeError, ValueError):
            continue
    return bounds


def _latest_auto_cut_bounds(project_id: str) -> dict[str, dict[str, int]]:
    """Return source-space cut bounds created by the automatic post-processor."""
    debug_path = config.get_generations_dir() / "dubbing_cuts" / project_id / "alignment_debug.json"
    if not debug_path.exists():
        return {}

    try:
        payload = json.loads(debug_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    bounds: dict[str, dict[str, int]] = {}
    cuts = payload.get("cuts")
    if not isinstance(cuts, list):
        return bounds

    for item in cuts:
        if not isinstance(item, dict):
            continue
        segment_id = item.get("segment_id")
        if not isinstance(segment_id, str):
            continue
        try:
            bounds[segment_id] = {
                "cut_start_ms": int(item.get("cut_start_ms", 0)),
                "cut_end_ms": int(item.get("cut_end_ms", 0)),
            }
        except (TypeError, ValueError):
            continue
    return bounds


def get_cut_source_bounds(project_id: str, segment_id: str) -> dict[str, int | str] | None:
    """Return the source-space bounds of a cut inside the full narration WAV."""
    manual_bounds = _latest_manual_cut_bounds(project_id).get(segment_id)
    if manual_bounds is not None:
        return {
            "cut_start_ms": manual_bounds["cut_start_ms"],
            "cut_end_ms": manual_bounds["cut_end_ms"],
            "source_type": "manual",
        }

    auto_bounds = _latest_auto_cut_bounds(project_id).get(segment_id)
    if auto_bounds is not None:
        return {
            "cut_start_ms": auto_bounds["cut_start_ms"],
            "cut_end_ms": auto_bounds["cut_end_ms"],
            "source_type": "auto",
        }
    return None


def _previous_manual_cut_end(project: DubbingProject, segment: DubbingSegment, db: Session) -> int:
    """Find the end of the previous manual cut for sequential full-WAV cutting."""
    bounds_by_segment = _latest_manual_cut_bounds(project.id)
    segments = list_project_segments(project.id, db)
    previous_segments = [
        item
        for item in segments
        if (item.start_ms, item.segment_order, item.srt_index) < (segment.start_ms, segment.segment_order, segment.srt_index)
    ]
    # Manual cuts are source-space cuts in the full narration WAV. Prefer the
    # persisted source bounds; if the debug ledger is missing, fall back to the
    # cumulative duration of already-created cut files so the next cut still
    # starts after the previous one instead of restarting from the SRT timecode.
    fallback_end_ms = 0
    for previous in previous_segments:
        bounds = bounds_by_segment.get(previous.id)
        if bounds and bounds.get("cut_end_ms", 0) > 0:
            fallback_end_ms = bounds["cut_end_ms"]
            continue
        generation = get_cut_generation(previous, db)
        if generation is not None and generation.duration:
            fallback_end_ms += int(round(float(generation.duration) * 1000))
    for previous in reversed(previous_segments):
        bounds = bounds_by_segment.get(previous.id)
        if bounds and bounds.get("cut_end_ms", 0) > 0:
            return bounds["cut_end_ms"]
    return fallback_end_ms


def normalize_srt2voice_tts_text(text: str, language: str | None = None) -> str:
    """Flatten SRT text into one TTS-friendly line with light typography normalization."""
    normalized = re.sub(r"[\r\n\t]+", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    language_key = (language or "").strip().lower()
    if language_key in {"fr", "french", "fr-fr"}:
        normalized = re.sub(r"\s*([:;!?])", r" \1", normalized)
        normalized = re.sub(r"\s+([,.])", r"\1", normalized)
    else:
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_clean_srt_narration_text(
    segments: list[DubbingSegment],
    language: str | None = None,
) -> str:
    """Build one continuous TTS input from SRT segments without indexes or timecodes."""
    text = " ".join((segment.text or "").strip() for segment in segments if (segment.text or "").strip())
    return normalize_srt2voice_tts_text(text, language)


def write_clean_srt_narration_text(project: DubbingProject, text: str, generation_id: str | None = None) -> Path:
    """Persist the cleaned SRT text used as full narration TTS input."""
    path = _clean_srt_narration_text_path(project.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    alias_path = _clean_srt_narration_text_alias_path(project, generation_id)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(text, encoding="utf-8")
    return path


def format_srt_timecode(ms: int) -> str:
    """Format milliseconds as SRT timecode."""
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ends_phrase(text: str) -> bool:
    return bool(TERMINAL_PUNCTUATION_RE.search((text or "").strip()))


def _boundary_punctuation_kind(text: str) -> str:
    stripped = (text or "").strip()
    if TERMINAL_PUNCTUATION_RE.search(stripped):
        return "hard"
    if SOFT_PUNCTUATION_RE.search(stripped):
        return "soft"
    return "none"


def normalize_alignment_tokens(text: str) -> list[str]:
    """Normalize French text for ASR/SRT matching while preserving accents."""
    normalized = MATCH_APOSTROPHE_RE.sub(" ", (text or "").lower())
    normalized = MATCH_PUNCTUATION_RE.sub(" ", normalized)
    return [token for token in re.split(r"\s+", normalized.strip()) if token]


def _timestamp_word_tokens(word: dict) -> list[dict]:
    tokens = normalize_alignment_tokens(str(word.get("word", "")))
    if not tokens:
        return []
    start_ms = int(round(float(word["start"]) * 1000))
    end_ms = int(round(float(word["end"]) * 1000))
    return [{"token": token, "start_ms": start_ms, "end_ms": end_ms} for token in tokens]


def _alignment_tokens_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if len(expected) > 3 and expected.endswith("s") and expected[:-1] == actual:
        return True
    if len(actual) > 3 and actual.endswith("s") and actual[:-1] == expected:
        return True
    return SequenceMatcher(None, expected, actual).ratio() >= 0.82


def _alignment_lcs_score(expected_tokens: list[str], actual_tokens: list[str]) -> float:
    if not expected_tokens:
        return 0.0
    previous = [0] * (len(actual_tokens) + 1)
    for expected in expected_tokens:
        current = [0]
        for index, actual in enumerate(actual_tokens, start=1):
            if _alignment_tokens_match(expected, actual):
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(expected_tokens)


def _find_alignment_span(
    segment_tokens: list[str],
    transcript_tokens: list[dict],
    search_start: int,
) -> tuple[int, int, float] | None:
    if not segment_tokens or not transcript_tokens:
        return None

    max_start = max(0, len(transcript_tokens) - 1)
    exact_limit = max(0, len(transcript_tokens) - len(segment_tokens))
    for start in range(min(search_start, exact_limit), exact_limit + 1):
        window = transcript_tokens[start:start + len(segment_tokens)]
        if [item["token"] for item in window] == segment_tokens:
            return start, start + len(segment_tokens) - 1, 1.0

    best: tuple[int, int, float] | None = None
    max_window = min(len(transcript_tokens), len(segment_tokens) + WORD_ALIGNMENT_SEARCH_SLACK)
    min_window = max(1, len(segment_tokens) - WORD_ALIGNMENT_SEARCH_SLACK)
    for start in range(max(0, search_start - WORD_ALIGNMENT_SEARCH_SLACK), max_start + 1):
        for window_size in range(min_window, max_window + 1):
            end = start + window_size
            if end > len(transcript_tokens):
                continue
            score = _alignment_lcs_score(
                segment_tokens,
                [item["token"] for item in transcript_tokens[start:end]],
            )
            if (
                best is None
                or score > best[2]
                or (
                    abs(score - best[2]) < 0.0001
                    and (
                        abs(start - search_start) < abs(best[0] - search_start)
                        or (
                            abs(start - search_start) == abs(best[0] - search_start)
                            and window_size < (best[1] - best[0] + 1)
                        )
                    )
                )
            ):
                best = (start, end - 1, score)

    if best is not None and best[2] >= WORD_ALIGNMENT_MIN_SCORE:
        return best
    return None


def _rms_frames(audio: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> list[dict]:
    start_sample = max(0, min(len(audio), int(round(start_ms * sample_rate / 1000))))
    end_sample = max(start_sample, min(len(audio), int(round(end_ms * sample_rate / 1000))))
    frame_samples = max(1, int(round(AUTO_CUT_RMS_FRAME_MS * sample_rate / 1000)))
    frames: list[dict] = []
    for frame_start in range(start_sample, end_sample, frame_samples):
        frame_end = min(end_sample, frame_start + frame_samples)
        chunk = audio[frame_start:frame_end].astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        if chunk.size > 1:
            zcr = float(np.mean(np.signbit(chunk[1:]) != np.signbit(chunk[:-1])))
        else:
            zcr = 0.0
        frames.append(
            {
                "start_ms": int(round(frame_start * 1000 / sample_rate)),
                "end_ms": int(round(frame_end * 1000 / sample_rate)),
                "rms": rms,
                "zcr": zcr,
            }
        )
    return frames


def _estimate_acoustic_gap_boundary(
    *,
    audio: np.ndarray,
    sample_rate: int,
    previous_word_end_ms: int,
    next_word_start_ms: int,
) -> tuple[int, dict]:
    """Find the safest dynamic cut point between two matched words."""
    audio_end_ms = int(round(len(audio) * 1000 / sample_rate))
    search_start_ms = max(0, previous_word_end_ms - AUTO_CUT_RMS_SEARCH_MS)
    search_end_ms = min(audio_end_ms, next_word_start_ms + AUTO_CUT_ATTACK_SEARCH_AFTER_MS)
    semantic_mid_ms = (previous_word_end_ms + next_word_start_ms) / 2
    frames = _rms_frames(audio, sample_rate, search_start_ms, search_end_ms)
    if not frames:
        fallback = int(round(semantic_mid_ms))
        return fallback, {
            "confidence": "fallback",
            "reason": "no_rms_frames",
            "previous_word_end_ms": previous_word_end_ms,
            "next_word_start_ms": next_word_start_ms,
            "cut_ms": fallback,
        }

    min_rms = min(frame["rms"] for frame in frames)
    max_rms = max(frame["rms"] for frame in frames)
    min_zcr = min(frame["zcr"] for frame in frames)
    max_zcr = max(frame["zcr"] for frame in frames)
    threshold = min_rms + (max_rms - min_rms) * 0.20
    energy_threshold = min_rms + (max_rms - min_rms) * 0.26
    zcr_threshold = max(0.06, min_zcr + (max_zcr - min_zcr) * 0.35)
    zcr_rms_floor = min_rms + (max_rms - min_rms) * AUTO_CUT_ZCR_MIN_RMS_FACTOR

    def has_voice_energy(frame: dict) -> bool:
        if frame["rms"] >= energy_threshold:
            return True
        return frame["rms"] >= zcr_rms_floor and frame["zcr"] >= zcr_threshold

    energy_regions: list[dict] = []
    current_energy: dict | None = None
    for frame in frames:
        if has_voice_energy(frame):
            if current_energy is None:
                current_energy = {
                    "start_ms": int(frame["start_ms"]),
                    "end_ms": int(frame["end_ms"]),
                    "peak_rms": float(frame["rms"]),
                    "peak_zcr": float(frame["zcr"]),
                }
            else:
                current_energy["end_ms"] = int(frame["end_ms"])
                current_energy["peak_rms"] = max(float(current_energy["peak_rms"]), float(frame["rms"]))
                current_energy["peak_zcr"] = max(float(current_energy["peak_zcr"]), float(frame["zcr"]))
        elif current_energy is not None:
            energy_regions.append(current_energy)
            current_energy = None
    if current_energy is not None:
        energy_regions.append(current_energy)

    previous_tail_limit_ms = min(audio_end_ms, previous_word_end_ms + AUTO_CUT_TAIL_SEARCH_AFTER_MS)
    next_attack_min_ms = max(0, next_word_start_ms - AUTO_CUT_ATTACK_SEARCH_BEFORE_MS)
    next_attack_limit_ms = min(audio_end_ms, next_word_start_ms + AUTO_CUT_ATTACK_SEARCH_AFTER_MS)

    previous_candidates = [
        region
        for region in energy_regions
        if int(region["start_ms"]) <= previous_tail_limit_ms
        and int(region["end_ms"]) >= previous_word_end_ms - AUTO_CUT_RMS_SEARCH_MS
        and int(region["start_ms"]) < next_word_start_ms
        and int(region["start_ms"]) <= semantic_mid_ms
    ]
    previous_region = (
        max(
            previous_candidates,
            key=lambda region: (
                int(region["end_ms"]),
                -abs(((int(region["start_ms"]) + int(region["end_ms"])) / 2) - previous_word_end_ms),
            ),
        )
        if previous_candidates
        else None
    )

    previous_energy_end_ms = int(previous_region["end_ms"]) if previous_region is not None else previous_word_end_ms
    next_candidates = [
        region
        for region in energy_regions
        if int(region["end_ms"]) >= next_attack_min_ms
        and int(region["start_ms"]) <= next_attack_limit_ms
        and int(region["end_ms"]) > previous_energy_end_ms
    ]
    next_region = (
        min(
            next_candidates,
            key=lambda region: (
                max(0, int(region["start_ms"]) - previous_energy_end_ms),
                abs(int(region["start_ms"]) - next_word_start_ms),
            ),
        )
        if next_candidates
        else None
    )

    if previous_region is not None and next_region is not None:
        next_energy_start_ms = int(next_region["start_ms"])
        if next_energy_start_ms > previous_energy_end_ms:
            cut_ms = int(round((previous_energy_end_ms + next_energy_start_ms) / 2))
            return cut_ms, {
                "confidence": "high" if next_energy_start_ms - previous_energy_end_ms >= 40 else "medium",
                "previous_word_end_ms": previous_word_end_ms,
                "next_word_start_ms": next_word_start_ms,
                "search_start_ms": search_start_ms,
                "search_end_ms": search_end_ms,
                "previous_energy_end_ms": previous_energy_end_ms,
                "next_energy_start_ms": next_energy_start_ms,
                "energy_gap_duration_ms": next_energy_start_ms - previous_energy_end_ms,
                "cut_ms": cut_ms,
                "min_rms": min_rms,
                "max_rms": max_rms,
                "min_zcr": min_zcr,
                "max_zcr": max_zcr,
                "quiet_threshold_rms": threshold,
                "energy_threshold_rms": energy_threshold,
                "zcr_threshold": zcr_threshold,
                "zcr_rms_floor": zcr_rms_floor,
                "cut_method": "targeted_tail_attack_gap_midpoint",
                "previous_region": previous_region,
                "next_region": next_region,
            }

    quiet_runs: list[list[dict]] = []
    current: list[dict] = []
    for frame in frames:
        if frame["rms"] <= threshold:
            current.append(frame)
        elif current:
            quiet_runs.append(current)
            current = []
    if current:
        quiet_runs.append(current)

    if quiet_runs:
        best_run = min(
            quiet_runs,
            key=lambda run: (
                0
                if (
                    int(run[0]["start_ms"]) <= next_word_start_ms
                    and int(run[-1]["end_ms"]) >= previous_word_end_ms
                )
                else 1,
                abs(((run[0]["start_ms"] + run[-1]["end_ms"]) / 2) - semantic_mid_ms),
                -len(run),
            ),
        )
        gap_start_ms = int(best_run[0]["start_ms"])
        gap_end_ms = int(best_run[-1]["end_ms"])
        cut_ms = int(round((gap_start_ms + gap_end_ms) / 2))
        confidence = "high" if gap_end_ms - gap_start_ms >= 40 else "medium"
        return cut_ms, {
            "confidence": confidence,
            "previous_word_end_ms": previous_word_end_ms,
            "next_word_start_ms": next_word_start_ms,
            "search_start_ms": search_start_ms,
            "search_end_ms": search_end_ms,
            "gap_start_ms": gap_start_ms,
            "gap_end_ms": gap_end_ms,
            "gap_duration_ms": gap_end_ms - gap_start_ms,
            "cut_ms": cut_ms,
            "min_rms": min_rms,
            "max_rms": max_rms,
            "threshold_rms": threshold,
            "cut_method": "nearest_quiet_run_midpoint",
        }

    fallback = int(round(semantic_mid_ms))
    return fallback, {
            "confidence": "fallback",
            "reason": "no_quiet_run",
            "previous_word_end_ms": previous_word_end_ms,
            "next_word_start_ms": next_word_start_ms,
            "cut_ms": fallback,
            "min_rms": min_rms,
            "max_rms": max_rms,
            "min_zcr": min_zcr,
            "max_zcr": max_zcr,
            "threshold_rms": threshold,
            "zcr_threshold": zcr_threshold,
            "zcr_rms_floor": zcr_rms_floor,
        }


def _select_auto_cut_boundary(
    *,
    audio: np.ndarray,
    sample_rate: int,
    current_segment: DubbingSegment,
    current_span: tuple[int, int],
    next_span: tuple[int, int],
) -> tuple[int, str, dict]:
    """Pick a punctuation-aware cut point without inventing artificial silence."""
    previous_word_end_ms = int(current_span[1])
    next_word_start_ms = int(next_span[0])
    semantic_mid_ms = int(round((previous_word_end_ms + next_word_start_ms) / 2))
    semantic_gap_ms = next_word_start_ms - previous_word_end_ms
    punctuation_kind = _boundary_punctuation_kind(current_segment.text)

    acoustic_cut_ms, acoustic_debug = _estimate_acoustic_gap_boundary(
        audio=audio,
        sample_rate=sample_rate,
        previous_word_end_ms=previous_word_end_ms,
        next_word_start_ms=next_word_start_ms,
    )
    acoustic_gap_ms = acoustic_debug.get("energy_gap_duration_ms", acoustic_debug.get("gap_duration_ms"))
    acoustic_gap_ms = int(acoustic_gap_ms) if acoustic_gap_ms is not None else None
    acoustic_drift_ms = abs(int(acoustic_cut_ms) - semantic_mid_ms)

    base_debug = {
        "punctuation_kind": punctuation_kind,
        "previous_word_end_ms": previous_word_end_ms,
        "next_word_start_ms": next_word_start_ms,
        "semantic_mid_ms": semantic_mid_ms,
        "semantic_gap_ms": semantic_gap_ms,
        "acoustic_cut_ms": int(acoustic_cut_ms),
        "acoustic_gap_ms": acoustic_gap_ms,
        "acoustic_drift_ms": acoustic_drift_ms,
        "soft_acoustic_gap_min_ms": AUTO_CUT_SOFT_ACOUSTIC_GAP_MIN_MS,
        "soft_acoustic_max_drift_ms": AUTO_CUT_SOFT_ACOUSTIC_MAX_DRIFT_MS,
        "acoustic": acoustic_debug,
    }

    if punctuation_kind == "hard":
        return int(acoustic_cut_ms), "hard_punctuation_rms_zcr", {
            **base_debug,
            "confidence": acoustic_debug.get("confidence", "medium"),
            "cut_ms": int(acoustic_cut_ms),
            "cut_method": "hard_punctuation_rms_zcr",
        }

    should_trust_acoustic_gap = (
        acoustic_gap_ms is not None
        and acoustic_gap_ms >= AUTO_CUT_SOFT_ACOUSTIC_GAP_MIN_MS
        and acoustic_drift_ms <= AUTO_CUT_SOFT_ACOUSTIC_MAX_DRIFT_MS
    )
    if should_trust_acoustic_gap:
        method = "soft_punctuation_rms_zcr_gap_midpoint" if punctuation_kind == "soft" else "continuous_rms_zcr_gap_midpoint"
        return int(acoustic_cut_ms), method, {
            **base_debug,
            "confidence": acoustic_debug.get("confidence", "medium"),
            "cut_ms": int(acoustic_cut_ms),
            "cut_method": method,
        }

    method = "soft_punctuation_semantic_midpoint" if punctuation_kind == "soft" else "continuous_semantic_midpoint"
    confidence = "medium" if semantic_gap_ms >= AUTO_CUT_MISSING_SILENCE_THRESHOLD_MS else "low"
    return semantic_mid_ms, method, {
        **base_debug,
        "confidence": confidence,
        "reason": "acoustic_gap_untrusted_or_too_short",
        "cut_ms": semantic_mid_ms,
        "cut_method": method,
    }


def _estimate_word_attack(
    *,
    audio: np.ndarray,
    sample_rate: int,
    word_start_ms: int,
) -> tuple[int, dict]:
    """Refine a Whisper word start to the first sustained acoustic energy."""
    audio_end_ms = int(round(len(audio) * 1000 / sample_rate))
    search_start_ms = max(0, int(word_start_ms) - AUTO_CUT_WORD_ATTACK_PRE_MS)
    search_end_ms = min(audio_end_ms, int(word_start_ms) + AUTO_CUT_WORD_ATTACK_POST_MS)
    frames = _rms_frames(audio, sample_rate, search_start_ms, search_end_ms)
    if not frames:
        fallback = max(0, min(audio_end_ms, int(word_start_ms)))
        return fallback, {
            "confidence": "fallback",
            "reason": "no_rms_frames",
            "word_start_ms": int(word_start_ms),
            "attack_ms": fallback,
        }

    rms_values = [frame["rms"] for frame in frames]
    zcr_values = [frame["zcr"] for frame in frames]
    min_rms = min(rms_values)
    max_rms = max(rms_values)
    min_zcr = min(zcr_values)
    max_zcr = max(zcr_values)
    rms_range = max_rms - min_rms
    if rms_range <= AUTO_CUT_WORD_ATTACK_MIN_RANGE:
        fallback = max(0, min(audio_end_ms, int(word_start_ms)))
        return fallback, {
            "confidence": "fallback",
            "reason": "flat_energy",
            "word_start_ms": int(word_start_ms),
            "attack_ms": fallback,
            "min_rms": min_rms,
            "max_rms": max_rms,
        }

    speech_threshold = min_rms + rms_range * 0.28
    sustain_threshold = min_rms + rms_range * 0.16
    zcr_threshold = max(0.06, min_zcr + (max_zcr - min_zcr) * 0.35)
    zcr_rms_floor = min_rms + rms_range * AUTO_CUT_ZCR_MIN_RMS_FACTOR

    def is_voiced_attack(frame: dict) -> bool:
        if frame["rms"] >= speech_threshold:
            return True
        return frame["rms"] >= zcr_rms_floor and frame["zcr"] >= zcr_threshold

    def is_sustained(frame: dict) -> bool:
        if frame["rms"] >= sustain_threshold:
            return True
        return frame["rms"] >= zcr_rms_floor and frame["zcr"] >= zcr_threshold

    earliest_candidate_ms = int(word_start_ms) - 80
    candidate_index: int | None = None
    for index, frame in enumerate(frames):
        if frame["end_ms"] < earliest_candidate_ms:
            continue
        next_frames = frames[index : index + 3]
        sustained = sum(1 for item in next_frames if is_sustained(item)) >= min(2, len(next_frames))
        if is_voiced_attack(frame) and sustained:
            candidate_index = index
            break

    if candidate_index is None:
        fallback = max(0, min(audio_end_ms, int(word_start_ms)))
        return fallback, {
            "confidence": "fallback",
            "reason": "no_sustained_attack",
            "word_start_ms": int(word_start_ms),
            "attack_ms": fallback,
            "search_start_ms": search_start_ms,
            "search_end_ms": search_end_ms,
            "min_rms": min_rms,
            "max_rms": max_rms,
            "min_zcr": min_zcr,
            "max_zcr": max_zcr,
            "speech_threshold_rms": speech_threshold,
            "sustain_threshold_rms": sustain_threshold,
            "zcr_threshold": zcr_threshold,
            "zcr_rms_floor": zcr_rms_floor,
        }

    attack_index = candidate_index
    while attack_index > 0 and is_sustained(frames[attack_index - 1]):
        attack_index -= 1
    attack_ms = int(frames[attack_index]["start_ms"])
    return attack_ms, {
        "confidence": "high" if abs(attack_ms - int(word_start_ms)) <= 80 else "medium",
        "word_start_ms": int(word_start_ms),
        "attack_ms": attack_ms,
        "candidate_ms": int(frames[candidate_index]["start_ms"]),
        "search_start_ms": search_start_ms,
        "search_end_ms": search_end_ms,
        "min_rms": min_rms,
        "max_rms": max_rms,
        "min_zcr": min_zcr,
        "max_zcr": max_zcr,
        "speech_threshold_rms": speech_threshold,
        "sustain_threshold_rms": sustain_threshold,
        "zcr_threshold": zcr_threshold,
        "zcr_rms_floor": zcr_rms_floor,
    }


def _select_cached_whisper_model() -> str:
    stt = transcribe.get_whisper_model()
    is_cached = getattr(stt, "_is_model_cached", None)
    for model_size in ("turbo", "large", "medium", "small", "base"):
        if callable(is_cached):
            try:
                if is_cached(model_size):
                    return model_size
            except Exception:
                continue
    return getattr(stt, "model_size", "base") or "base"


async def _align_segments_to_full_narration(
    *,
    audio_path: Path,
    project: DubbingProject,
    segments: list[DubbingSegment],
) -> tuple[dict[str, tuple[int, int, float]], dict]:
    def _empty_debug() -> dict:
        return {
            "project_id": project.id,
            "audio_path": str(audio_path),
            "model_size": None,
            "language": None,
            "word_count": 0,
            "matched_segment_count": 0,
            "transcript_words": [],
            "transcript_tokens": [],
            "segments": [],
            "error": None,
        }

    debug: dict = {
        "project_id": project.id,
        "audio_path": str(audio_path),
        "model_size": None,
        "language": None,
        "word_count": 0,
        "matched_segment_count": 0,
        "attempts": [],
        "transcript_words": [],
        "transcript_tokens": [],
        "segments": [],
        "error": None,
    }
    stt = transcribe.get_whisper_model()
    if not hasattr(stt, "transcribe_word_timestamps"):
        debug["error"] = "Selected STT backend does not expose word timestamps."
        return {}, debug

    model_size = _select_cached_whisper_model()
    debug["model_size"] = model_size

    async def run_attempt(language: str | None) -> tuple[dict[str, tuple[int, int, float]], dict]:
        attempt_debug = _empty_debug()
        attempt_debug["model_size"] = model_size
        attempt_debug["language"] = language or "auto"
        try:
            words = await stt.transcribe_word_timestamps(
                str(audio_path),
                language=language,
                model_size=model_size,
            )
        except Exception:
            logger.exception(
                "Dubbing word alignment failed with Whisper %s language=%s",
                model_size,
                language or "auto",
            )
            attempt_debug["error"] = (
                f"Dubbing word alignment failed with Whisper {model_size} "
                f"language={language or 'auto'}."
            )
            return {}, attempt_debug

        attempt_debug["word_count"] = len(words)
        attempt_debug["transcript_words"] = [
            {
                "word": str(word.get("word", "")),
                "start_ms": int(round(float(word["start"]) * 1000)),
                "end_ms": int(round(float(word["end"]) * 1000)),
            }
            for word in words
            if "start" in word and "end" in word
        ]
        transcript_tokens: list[dict] = []
        for word in words:
            transcript_tokens.extend(_timestamp_word_tokens(word))
        if not transcript_tokens:
            attempt_debug["error"] = "Whisper returned no usable timestamp tokens."
            return {}, attempt_debug
        attempt_debug["transcript_tokens"] = transcript_tokens

        spans: dict[str, tuple[int, int, float]] = {}
        search_start = 0
        for segment in segments:
            segment_tokens = normalize_alignment_tokens(segment.text)
            span = _find_alignment_span(segment_tokens, transcript_tokens, search_start)
            if span is None:
                attempt_debug["segments"].append(
                    {
                        "segment_id": segment.id,
                        "srt_index": segment.srt_index,
                        "text": segment.text,
                        "normalized_tokens": segment_tokens,
                        "match": None,
                        "search_start_token_index": search_start,
                        "used_fallback": True,
                    }
                )
                continue
            start_index, end_index, score = span
            start_ms = transcript_tokens[start_index]["start_ms"]
            end_ms = transcript_tokens[end_index]["end_ms"]
            spans[segment.id] = (start_ms, end_ms, score)
            attempt_debug["segments"].append(
                {
                    "segment_id": segment.id,
                    "srt_index": segment.srt_index,
                    "text": segment.text,
                    "normalized_tokens": segment_tokens,
                    "match": {
                        "score": score,
                        "start_token_index": start_index,
                        "end_token_index": end_index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "matched_tokens": [
                            item["token"] for item in transcript_tokens[start_index : end_index + 1]
                        ],
                    },
                    "search_start_token_index": search_start,
                    "used_fallback": False,
                }
            )
            search_start = end_index + 1
        attempt_debug["matched_segment_count"] = len(spans)
        return spans, attempt_debug

    language_attempts: list[str | None] = []
    project_language = (project.language or "").strip()
    if project_language:
        language_attempts.append(project_language)
    language_attempts.append(None)

    best_spans: dict[str, tuple[int, int, float]] = {}
    best_debug: dict | None = None
    seen_languages: set[str] = set()
    minimum_good_matches = max(1, int(round(len(segments) * 0.6)))
    for language in language_attempts:
        language_key = language or "auto"
        if language_key in seen_languages:
            continue
        seen_languages.add(language_key)
        spans, attempt_debug = await run_attempt(language)
        debug["attempts"].append(
            {
                "language": language_key,
                "matched_segment_count": len(spans),
                "segment_count": len(segments),
                "word_count": attempt_debug.get("word_count", 0),
                "error": attempt_debug.get("error"),
            }
        )
        if best_debug is None or len(spans) > len(best_spans):
            best_spans = spans
            best_debug = attempt_debug
        if len(spans) >= minimum_good_matches:
            break

    if best_debug is None:
        debug["error"] = "Whisper alignment did not run."
        return {}, debug

    if len(best_spans) == 0:
        for segment in segments:
            logger.warning("No ASR word alignment for dubbing segment %s", segment.srt_index)

    best_debug["attempts"] = debug["attempts"]
    return best_spans, best_debug


async def build_auto_cut_timeline_clips(project: DubbingProject, db: Session) -> dict:
    """Build timeline-only clips from full narration using word matching + RMS gaps."""
    segments = list_project_segments(project.id, db)
    if not segments:
        raise ValueError("Dubbing project has no SRT segments.")
    validate_auto_cut_language(project, segments)

    full_generation = get_full_narration_generation(project.id, db)
    if (
        full_generation is None
        or (full_generation.status or "completed") != "completed"
        or not full_generation.audio_path
    ):
        raise ValueError("Generate the full SRT narration before running Auto Cut.")

    full_audio_path = config.resolve_storage_path(full_generation.audio_path)
    if full_audio_path is None or not full_audio_path.exists():
        raise ValueError("Full narration audio file was not found.")

    sample_rate = 24000
    audio, sample_rate = load_audio(str(full_audio_path), sample_rate=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError("Full narration audio file is empty.")

    full_duration_ms = int(round(len(audio) * 1000 / sample_rate))
    alignment_spans, alignment_debug = await _align_segments_to_full_narration(
        audio_path=full_audio_path,
        project=project,
        segments=segments,
    )
    total_target_duration_ms = max(1, sum(max(1, segment.target_duration_ms) for segment in segments))
    proportional_boundaries: list[int] = [0]
    cursor_ms = 0
    for index, segment in enumerate(segments[:-1]):
        cursor_ms += int(round(full_duration_ms * max(1, segment.target_duration_ms) / total_target_duration_ms))
        proportional_boundaries.append(max(0, min(full_duration_ms, cursor_ms)))
    proportional_boundaries.append(full_duration_ms)

    source_boundaries: list[int] = [0]
    boundary_debug: list[dict] = []
    for index in range(len(segments) - 1):
        current_segment = segments[index]
        next_segment = segments[index + 1]
        current_span = alignment_spans.get(current_segment.id)
        next_span = alignment_spans.get(next_segment.id)
        if current_span is not None and next_span is not None:
            cut_ms, boundary_source, cut_debug = _select_auto_cut_boundary(
                audio=audio,
                sample_rate=sample_rate,
                current_segment=current_segment,
                current_span=current_span,
                next_span=next_span,
            )
            cut_ms = max(source_boundaries[-1], min(full_duration_ms, cut_ms))
        else:
            cut_ms = proportional_boundaries[index + 1]
            cut_debug = {
                "confidence": "fallback",
                "reason": "missing_word_alignment",
                "cut_ms": cut_ms,
            }
            boundary_source = "proportional_fallback"
        source_boundaries.append(cut_ms)
        boundary_debug.append(
            {
                "after_segment_id": current_segment.id,
                "after_srt_index": current_segment.srt_index,
                "before_segment_id": next_segment.id,
                "before_srt_index": next_segment.srt_index,
                "source": boundary_source,
                **cut_debug,
            }
        )
    source_boundaries.append(full_duration_ms)

    clips: list[dict] = []
    placement_debug: list[dict] = []
    for index, segment in enumerate(segments):
        source_start_ms = source_boundaries[index]
        source_end_ms = max(source_start_ms + 1, source_boundaries[index + 1])
        timeline_start_ms = segment.start_ms
        segment_span = alignment_spans.get(segment.id)
        placement: dict = {
            "segment_id": segment.id,
            "srt_index": segment.srt_index,
            "srt_start_ms": segment.start_ms,
            "cut_source_start_ms": source_start_ms,
            "timeline_start_ms": timeline_start_ms,
            "placement_source": "srt_start_fallback",
        }
        if segment_span is not None:
            first_word_start_ms = int(segment_span[0])
            attack_ms, attack_debug = _estimate_word_attack(
                audio=audio,
                sample_rate=sample_rate,
                word_start_ms=first_word_start_ms,
            )
            leading_offset_ms = max(0, attack_ms - source_start_ms)
            timeline_start_ms = max(0, segment.start_ms - leading_offset_ms)
            placement = {
                **placement,
                "first_word_start_ms": first_word_start_ms,
                "refined_first_word_attack_ms": attack_ms,
                "leading_offset_ms": leading_offset_ms,
                "timeline_start_ms": timeline_start_ms,
                "placement_source": "first_word_energy_attack",
                "attack": attack_debug,
            }
        if index > 0 and clips:
            previous_clip = clips[-1]
            previous_effective_duration_ms = max(
                0,
                int(previous_clip["duration_ms"])
                - int(previous_clip["trim_start_ms"])
                - int(previous_clip["trim_end_ms"]),
            )
            previous_clip_end_ms = int(previous_clip["start_ms"]) + max(
                0,
                previous_effective_duration_ms,
            )
            previous_boundary = boundary_debug[index - 1] if index - 1 < len(boundary_debug) else {}
            punctuation_kind = str(previous_boundary.get("punctuation_kind") or "")
            anchored_timeline_start_ms = timeline_start_ms
            if punctuation_kind in {"none", "soft"}:
                desired_previous_end_ms = anchored_timeline_start_ms
                previous_clip["start_ms"] = max(0, desired_previous_end_ms - previous_effective_duration_ms)
                placement = {
                    **placement,
                    "timeline_start_ms": anchored_timeline_start_ms,
                    "previous_clip_end_ms": previous_clip_end_ms,
                    "previous_adjusted_start_ms": previous_clip["start_ms"],
                    "desired_previous_end_ms": desired_previous_end_ms,
                    "punctuation_kind": punctuation_kind,
                    "placement_source": (
                        "soft_punctuation_adjacent_anchor_next_adjust_previous"
                        if punctuation_kind == "soft"
                        else "continuous_no_punctuation_anchor_next_adjust_previous"
                    ),
                }
        placement_debug.append(placement)
        clips.append(
            {
                "id": f"full-narration-clip-auto-{segment.id}",
                "generation_id": full_generation.id,
                "segment_id": segment.id,
                "srt_index": segment.srt_index,
                "start_ms": timeline_start_ms,
                "duration_ms": full_duration_ms,
                "trim_start_ms": source_start_ms,
                "trim_end_ms": max(0, full_duration_ms - source_end_ms),
                "track": index % 2,
                "volume": 1.0,
                "confidence": (
                    boundary_debug[index - 1].get("confidence", "fallback")
                    if index > 0 and index - 1 < len(boundary_debug)
                    else "start"
                ),
                "cut_source": (
                    boundary_debug[index - 1].get("source", "start")
                    if index > 0 and index - 1 < len(boundary_debug)
                    else "start"
                ),
            }
        )

    debug = {
        "schema_version": AUTO_CUT_DEBUG_SCHEMA_VERSION,
        "project_id": project.id,
        "audio_path": str(full_audio_path),
        "audio_mtime_ms": int(round(full_audio_path.stat().st_mtime * 1000)),
        "full_duration_ms": full_duration_ms,
        "alignment": alignment_debug,
        "boundaries": boundary_debug,
        "placements": placement_debug,
        "clips": clips,
    }
    debug_dir = config.get_generations_dir() / "dubbing_cuts" / project.id
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "word_matching_debug.json"
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"clips": clips, "debug_path": str(debug_path)}


def _auto_cut_debug_path(project_id: str) -> Path:
    return config.get_generations_dir() / "dubbing_cuts" / project_id / "word_matching_debug.json"


def _tempo_range(multiplier: float) -> tuple[str, str]:
    if 0.9 <= multiplier <= 1.1:
        return "safe", "Minimal tempo change expected."
    if 0.8 <= multiplier <= 1.2:
        return "warning", "Noticeable tempo change. Listen before export."
    return "critical", "Quality degradation likely. Consider editing text/CPS before applying tempo."


def _load_cached_auto_cut_debug(
    *,
    project: DubbingProject,
    segments: list[DubbingSegment],
    full_audio_path: Path,
    full_duration_ms: int,
) -> tuple[dict | None, Path | None]:
    """Reuse the last Auto Cut debug data only when it still matches this project/audio."""
    debug_path = _auto_cut_debug_path(project.id)
    if not debug_path.exists():
        return None, None
    try:
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Ignoring unreadable SRT2Voice tempo debug cache %s", debug_path, exc_info=True)
        return None, None

    if debug.get("project_id") != project.id:
        return None, None
    if int(debug.get("schema_version") or 0) != AUTO_CUT_DEBUG_SCHEMA_VERSION:
        return None, None
    if abs(int(debug.get("full_duration_ms") or 0) - full_duration_ms) > 25:
        return None, None
    current_mtime_ms = int(round(full_audio_path.stat().st_mtime * 1000))
    cached_mtime_ms = debug.get("audio_mtime_ms")
    if isinstance(cached_mtime_ms, int) and abs(cached_mtime_ms - current_mtime_ms) > 25:
        return None, None
    cached_audio_path = str(debug.get("audio_path") or "")
    if cached_audio_path and Path(cached_audio_path) != full_audio_path:
        return None, None

    segment_ids = {segment.id for segment in segments}
    clip_segment_ids = {str(clip.get("segment_id")) for clip in debug.get("clips", [])}
    placement_segment_ids = {str(item.get("segment_id")) for item in debug.get("placements", [])}
    if not segment_ids.issubset(clip_segment_ids) or not segment_ids.issubset(placement_segment_ids):
        return None, None
    return debug, debug_path


def _build_tempo_suggestion_from_debug(
    *,
    project: DubbingProject,
    segments: list[DubbingSegment],
    debug: dict,
    debug_path: Path | None,
    from_cached_alignment: bool,
) -> models.DubbingTempoSuggestionResponse:
    """Estimate the global atempo factor from the same mounted clips Auto Cut will export."""
    if not segments:
        raise ValueError("Dubbing project has no SRT segments.")
    ordered_segments = sorted(segments, key=lambda item: item.segment_order)
    target_duration_ms = max(1, ordered_segments[-1].end_ms - ordered_segments[0].start_ms)
    clips_by_segment = {
        str(item.get("segment_id")): item
        for item in debug.get("clips", [])
        if item.get("segment_id")
    }

    projected_start_ms = ordered_segments[0].start_ms
    projected_end_ms = projected_start_ms
    has_projected_clip = False
    for segment in ordered_segments:
        clip = clips_by_segment.get(segment.id)
        if not clip:
            continue

        source_start_ms = int(clip.get("trim_start_ms") or 0)
        source_end_ms = max(
            source_start_ms + 1,
            int(clip.get("duration_ms") or 0) - int(clip.get("trim_end_ms") or 0),
        )
        clip_start_ms = int(clip.get("start_ms") or segment.start_ms)
        effective_duration_ms = max(1, source_end_ms - source_start_ms)
        projected_start_ms = min(projected_start_ms, clip_start_ms)
        projected_end_ms = max(projected_end_ms, clip_start_ms + effective_duration_ms)
        has_projected_clip = True

    if not has_projected_clip:
        raise ValueError("Auto Cut word alignment did not produce tempo clips.")

    projected_duration_ms = max(1, int(round(projected_end_ms - ordered_segments[0].start_ms)))
    multiplier = projected_duration_ms / target_duration_ms
    range_name, message = _tempo_range(multiplier)
    return models.DubbingTempoSuggestionResponse(
        multiplier=round(multiplier, 4),
        target_duration_ms=target_duration_ms,
        projected_duration_ms=projected_duration_ms,
        delta_ms=projected_duration_ms - target_duration_ms,
        range=range_name,
        message=message,
        from_cached_alignment=from_cached_alignment,
        debug_path=str(debug_path) if debug_path is not None else None,
    )


async def suggest_project_tempo(project: DubbingProject, db: Session) -> models.DubbingTempoSuggestionResponse:
    """Suggest one global pitch-preserving tempo factor for the current full narration."""
    segments = list_project_segments(project.id, db)
    if not segments:
        raise ValueError("Dubbing project has no SRT segments.")

    full_generation = get_full_narration_generation(project.id, db)
    if (
        full_generation is None
        or (full_generation.status or "completed") != "completed"
        or not full_generation.audio_path
    ):
        raise ValueError("Generate the full SRT narration before suggesting tempo.")

    full_audio_path = config.resolve_storage_path(full_generation.audio_path)
    if full_audio_path is None or not full_audio_path.exists():
        raise ValueError("Full narration audio file was not found.")

    audio, sample_rate = load_audio(str(full_audio_path), sample_rate=24000, mono=True)
    full_duration_ms = int(round(len(audio) * 1000 / sample_rate))
    cached_debug, cached_debug_path = _load_cached_auto_cut_debug(
        project=project,
        segments=segments,
        full_audio_path=full_audio_path,
        full_duration_ms=full_duration_ms,
    )
    if cached_debug is not None:
        return _build_tempo_suggestion_from_debug(
            project=project,
            segments=segments,
            debug=cached_debug,
            debug_path=cached_debug_path,
            from_cached_alignment=True,
        )

    auto_cut = await build_auto_cut_timeline_clips(project, db)
    debug_path = Path(str(auto_cut.get("debug_path"))) if auto_cut.get("debug_path") else _auto_cut_debug_path(project.id)
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    return _build_tempo_suggestion_from_debug(
        project=project,
        segments=segments,
        debug=debug,
        debug_path=debug_path,
        from_cached_alignment=False,
    )


async def apply_project_suggested_tempo(
    project: DubbingProject,
    db: Session,
    *,
    multiplier: float | None = None,
) -> dict:
    """Apply global atempo, invalidate previous cuts, then re-run Auto Cut on the processed WAV."""
    suggestion = await suggest_project_tempo(project, db)
    tempo = float(multiplier if multiplier is not None else suggestion.multiplier)
    if tempo < PACE_MIN or tempo > PACE_MAX:
        raise ValueError(
            "Suggested tempo is outside the supported 0.80x-1.20x range. "
            "Edit the SRT text/timing first, then regenerate."
        )

    full_generation = get_full_narration_generation(project.id, db)
    if full_generation is None or not full_generation.audio_path:
        raise ValueError("Generate the full SRT narration before applying tempo.")

    if abs(tempo - 1.0) >= 0.005:
        applied = apply_generation_pace(full_generation, tempo, db)
        if not applied:
            raise ValueError("Tempo processing was skipped because ffmpeg atempo is unavailable or the audio is missing.")

    project.pace_override = tempo
    await invalidate_project_cut_artifacts(project.id, db)
    db.commit()

    auto_cut = await build_auto_cut_timeline_clips(project, db)
    return {
        "suggestion": suggestion,
        "clips": auto_cut["clips"],
        "debug_path": auto_cut.get("debug_path"),
    }


def assign_pace_groups(segments: list[DubbingSegment]) -> list[dict]:
    """Compute phrase-like groups and mirror their id onto the segment rows."""
    groups: list[dict] = []
    current_segments: list[DubbingSegment] = []
    current_start_ms: int | None = None
    current_end_ms: int | None = None
    group_index = 1

    def flush_group() -> None:
        nonlocal current_segments, current_start_ms, current_end_ms, group_index
        if not current_segments or current_start_ms is None or current_end_ms is None:
            return
        group_id = f"group-{group_index}"
        for segment in current_segments:
            segment.pace_group_id = group_id
        groups.append(
            {
                "id": group_id,
                "label": f"Phrase {group_index}",
                "segment_ids": [segment.id for segment in current_segments],
                "segment_orders": [segment.segment_order for segment in current_segments],
                "start_ms": current_start_ms,
                "end_ms": current_end_ms,
                "target_duration_ms": current_end_ms - current_start_ms,
            }
        )
        current_segments = []
        current_start_ms = None
        current_end_ms = None
        group_index += 1

    for segment in segments:
        if not current_segments:
            current_start_ms = segment.start_ms
        current_segments.append(segment)
        current_end_ms = segment.end_ms
        if _ends_phrase(segment.text):
            flush_group()

    flush_group()
    return groups


def get_group_override_map(project: DubbingProject) -> dict[str, float]:
    raw = project.group_pace_overrides or {}
    if isinstance(raw, dict):
        return {
            str(key): clamped
            for key, value in raw.items()
            if (clamped := clamp_pace(value)) is not None
        }
    return {}


def set_group_override(project: DubbingProject, group_id: str, pace_override: float | None) -> None:
    overrides = get_group_override_map(project)
    if pace_override is None:
        overrides.pop(group_id, None)
    else:
        overrides[group_id] = clamp_pace(pace_override)
    project.group_pace_overrides = overrides


def compute_group_effective_pace(
    *,
    project: DubbingProject,
    group: dict,
    segments_by_id: dict[str, DubbingSegment],
) -> float:
    overrides = get_group_override_map(project)
    if group["id"] in overrides:
        return overrides[group["id"]]
    if project.pace_override is not None:
        return clamp_pace(project.pace_override) or 1.0

    actual_duration_ms = sum(
        max(0, segments_by_id[segment_id].actual_duration_ms or 0) for segment_id in group["segment_ids"]
    )
    target_duration_ms = max(1, group["target_duration_ms"])
    if actual_duration_ms <= 0:
        return 1.0
    return clamp_pace(actual_duration_ms / target_duration_ms) or 1.0


def get_persisted_segment_pace(
    *,
    project: DubbingProject,
    segment: DubbingSegment,
    segments: list[DubbingSegment],
) -> float:
    """Return only user-saved pace overrides for generation post-processing."""
    groups = assign_pace_groups(segments)
    overrides = get_group_override_map(project)
    for group in groups:
        if segment.id in group["segment_ids"] and group["id"] in overrides:
            return overrides[group["id"]]
    if project.pace_override is not None:
        return clamp_pace(project.pace_override) or 1.0
    return 1.0


def apply_generation_pace(generation: DBGeneration, pace: float, db: Session) -> bool:
    """Apply pitch-preserving tempo change to a completed dubbing WAV."""
    pace = clamp_pace(pace) or 1.0
    if abs(pace - 1.0) < 0.005:
        return False
    if (generation.status or "completed") != "completed" or not generation.audio_path:
        return False

    audio_path = config.resolve_storage_path(generation.audio_path)
    if audio_path is None or not audio_path.exists():
        return False

    if not time_stretch_audio_file_with_ffmpeg(str(audio_path), pace, sample_rate=24000):
        logger.warning(
            "Dubbing pace %.2fx was skipped because ffmpeg atempo is not available",
            pace,
        )
        return False

    audio, sample_rate = load_audio(str(audio_path), sample_rate=24000, mono=True)
    generation.duration = len(audio) / sample_rate
    db.commit()
    return True


async def run_dubbing_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: int | None,
    instruct: str | None,
    pace: float,
    temperature: float | None = None,
    project_id: str | None = None,
    segment_id: str | None = None,
    max_chunk_chars: int | None = None,
    crossfade_ms: int | None = None,
    use_voice_prompt_cache: bool = True,
    unload_after: bool = False,
) -> None:
    """Run TTS, then apply Dubbing-only pitch-preserving pace override."""
    is_full_narration = generation_id.startswith(f"{FULL_NARRATION_GENERATION_PREFIX}-")
    started_at = time.perf_counter()
    db = None
    try:
        await run_generation(
            generation_id=generation_id,
            profile_id=profile_id,
            text=text,
            language=language,
            engine=engine,
            model_size=model_size,
            seed=seed,
            normalize=False,
            effects_chain=None,
            instruct=instruct,
            temperature=temperature if is_qwen_dubbing_engine(engine) else None,
            mode="generate",
            max_chunk_chars=max_chunk_chars,
            crossfade_ms=crossfade_ms,
            use_voice_prompt_cache=use_voice_prompt_cache,
            unload_after=unload_after,
        )

        db = next(get_db())
        try:
            generation = db.query(DBGeneration).filter_by(id=generation_id).first()
            if generation is not None:
                try:
                    pace_applied = apply_generation_pace(generation, pace, db) if is_qwen_dubbing_engine(engine) else False
                except Exception as exc:
                    logger.exception("Dubbing pace post-processing failed for %s", generation_id)
                    await history.update_generation_status(
                        generation_id=generation_id,
                        status="failed",
                        db=db,
                        error=f"Dubbing pace post-processing failed: {exc}",
                    )
                    pace_applied = False

                if pace_applied and project_id is not None and segment_id is not None:
                    segment = get_segment_or_none(project_id, segment_id, db)
                    if segment is not None and sync_segment_generation_state(segment, db):
                        db.commit()

            if project_id is not None:
                project = get_project_or_none(project_id, db)
                if project is not None and update_project_status(project, db):
                    db.commit()
        finally:
            if db is not None:
                db.close()
    finally:
        if is_full_narration:
            write_full_narration_timing(
                generation_id,
                int(round((time.perf_counter() - started_at) * 1000)),
            )


def build_pace_group_responses(
    project: DubbingProject,
    segments: list[DubbingSegment],
) -> list[dict]:
    groups = assign_pace_groups(segments)
    segments_by_id = {segment.id: segment for segment in segments}
    overrides = get_group_override_map(project)
    return [
        {
            **group,
            "pace_override": overrides.get(group["id"]),
            "effective_pace": compute_group_effective_pace(
                project=project,
                group=group,
                segments_by_id=segments_by_id,
            ),
        }
        for group in groups
    ]


def resolve_dubbing_engine_for_profile(profile, requested_engine: str | None = None) -> str:
    """Resolve the engine to use for a dubbing profile."""
    voice_type = getattr(profile, "voice_type", None) or "cloned"

    if voice_type == "designed":
        if requested_engine and requested_engine != "qwen_voice_design":
            raise ValueError(
                f"Designed profile {profile.id} only supports engine 'qwen_voice_design', not '{requested_engine}'"
            )
        return "qwen_voice_design"

    if voice_type == "preset":
        preset_engine = getattr(profile, "preset_engine", None)
        if preset_engine == "qwen_custom_voice":
            if requested_engine and requested_engine != "qwen_custom_voice":
                raise ValueError(
                    f"Preset profile {profile.id} only supports engine 'qwen_custom_voice', not '{requested_engine}'"
                )
            return "qwen_custom_voice"
        if preset_engine:
            if requested_engine and requested_engine != preset_engine:
                raise ValueError(
                    f"Preset profile {profile.id} only supports engine '{preset_engine}', not '{requested_engine}'"
                )
            return preset_engine
        raise ValueError(
            f"Preset profile {profile.id} only supports engine '{preset_engine}', not 'qwen'"
        )

    if requested_engine:
        return requested_engine

    default_engine = getattr(profile, "default_engine", None)
    if default_engine == "qwen_custom_voice":
        return "qwen_custom_voice"
    if default_engine == "qwen_voice_design":
        return "qwen_voice_design"
    if default_engine == "chatterbox":
        return "chatterbox"
    if default_engine == "chatterbox_turbo":
        return "chatterbox_turbo"
    if default_engine == "luxtts":
        return "luxtts"
    if default_engine == "tada":
        return "tada"

    return "qwen"


def create_project_from_srt(
    *,
    filename: str,
    content: str,
    db: Session,
    source_path: str | None = None,
    engine: str = "qwen",
    language: str = "fr",
) -> DubbingProject:
    """Create a persisted dubbing project from an SRT file."""
    segments = parse_srt_text(content)
    project_name = Path(filename).stem or "Imported SRT"

    project = DubbingProject(
        name=project_name,
        source_type="srt",
        source_path=source_path,
        engine=engine,
        language=language,
        pace_override=None,
        group_pace_overrides={},
        status="draft",
    )
    db.add(project)
    db.flush()

    for order, segment in enumerate(segments, start=1):
        db.add(
            DubbingSegment(
                project_id=project.id,
                segment_order=order,
                srt_index=segment.srt_index,
                start_tc=segment.start_tc,
                end_tc=segment.end_tc,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                target_duration_ms=segment.target_duration_ms,
                text_lines=segment.text_lines,
                text=segment.text,
                status="pending",
                fit_status="unknown",
            )
        )

    db.commit()
    db.refresh(project)
    return project


def get_project_or_none(project_id: str, db: Session) -> DubbingProject | None:
    return db.query(DubbingProject).filter_by(id=project_id).first()


def list_project_segments(project_id: str, db: Session) -> list[DubbingSegment]:
    segments = (
        db.query(DubbingSegment)
        .filter_by(project_id=project_id)
        .order_by(DubbingSegment.segment_order.asc())
        .all()
    )
    dirty = False
    previous_group_ids = {segment.id: segment.pace_group_id for segment in segments}
    for segment in segments:
        if sync_segment_generation_state(segment, db):
            dirty = True
    assign_pace_groups(segments)
    if any(segment.pace_group_id != previous_group_ids.get(segment.id) for segment in segments):
        dirty = True
    project = get_project_or_none(project_id, db)
    if project is not None and update_project_status(project, db):
        dirty = True
    if dirty:
        db.commit()
        for segment in segments:
            db.refresh(segment)
    return segments


def get_segment_or_none(project_id: str, segment_id: str, db: Session) -> DubbingSegment | None:
    segment = db.query(DubbingSegment).filter_by(id=segment_id, project_id=project_id).first()
    if segment is not None and sync_segment_generation_state(segment, db):
        db.commit()
        db.refresh(segment)
    return segment


def classify_timing_fit(delta_ms: int | None) -> str:
    """Treat overflows as warnings and reserve failed for real generation errors."""
    if delta_ms is None:
        return "unknown"
    if delta_ms <= 0:
        return "exact"
    return "warning"


def sync_segment_generation_state(segment: DubbingSegment, db: Session) -> bool:
    """Mirror linked generation status back to the dubbing segment."""
    if not segment.generation_id:
        return False

    generation = db.query(DBGeneration).filter_by(id=segment.generation_id).first()
    if generation is None:
        segment.generation_id = None
        segment.actual_duration_ms = None
        segment.delta_ms = None
        segment.fit_status = "unknown"
        if segment.status == "generating":
            segment.status = "pending"
        return True

    new_status = segment.status
    new_fit_status = segment.fit_status
    actual_duration_ms = segment.actual_duration_ms
    delta_ms = segment.delta_ms

    generation_status = generation.status or "completed"
    if generation_status in {"loading_model", "generating"}:
        new_status = "generating"
    elif generation_status == "failed":
        new_status = "failed"
        new_fit_status = "failed"
    elif generation.duration is not None:
        actual_duration_ms = int(round(generation.duration * 1000))
        delta_ms = actual_duration_ms - segment.target_duration_ms
        new_fit_status = classify_timing_fit(delta_ms)
        new_status = "generated" if new_fit_status == "exact" else "warning"

    changed = (
        actual_duration_ms != segment.actual_duration_ms
        or delta_ms != segment.delta_ms
        or new_fit_status != segment.fit_status
        or new_status != segment.status
    )
    if changed:
        segment.actual_duration_ms = actual_duration_ms
        segment.delta_ms = delta_ms
        segment.fit_status = new_fit_status
        segment.status = new_status
    return changed


def update_project_status(project: DubbingProject, db: Session) -> bool:
    """Recompute project status from the active SRT2Voice workflow."""
    full_generation = get_full_narration_generation(project.id, db)
    if full_generation is not None:
        full_status = full_generation.status or "completed"
        if full_status in {"loading_model", "generating"}:
            changed = project.status != "processing"
            if changed:
                project.status = "processing"
            return changed
        if full_status == "failed":
            changed = project.status != "failed"
            if changed:
                project.status = "failed"
            return changed
        if full_status == "completed" and full_generation.audio_path:
            changed = project.status != "completed"
            if changed:
                project.status = "completed"
            return changed

    segments = (
        db.query(DubbingSegment)
        .filter_by(project_id=project.id)
        .order_by(DubbingSegment.segment_order.asc())
        .all()
    )
    if not segments:
        next_status = "draft"
    elif any(segment.status == "failed" for segment in segments):
        next_status = "failed"
    elif any(segment.status in {"pending", "generating"} for segment in segments):
        next_status = "processing"
    else:
        next_status = "completed"

    changed = project.status != next_status
    if changed:
        project.status = next_status
    return changed


async def queue_full_narration_generation(
    *,
    project: DubbingProject,
    request: models.DubbingFullNarrationRequest,
    db: Session,
    engine: str,
) -> DBGeneration:
    """Queue one TTS generation for the complete cleaned SRT narration."""
    segments = list_project_segments(project.id, db)
    clean_text = build_clean_srt_narration_text(segments, request.language or project.language)
    if not clean_text:
        raise ValueError("Dubbing project has no text to narrate.")
    generation_id = full_narration_generation_id(project.id)
    write_clean_srt_narration_text(project, clean_text, generation_id)

    task_manager = get_task_manager()
    existing = db.query(DBGeneration).filter_by(id=generation_id).first()
    if existing is not None:
        if (existing.status or "completed") in {"loading_model", "generating"}:
            if task_manager.is_generation_active(existing.id):
                raise ValueError("Full narration is already generating for this project.")
            await history.update_generation_status(
                generation_id=existing.id,
                status="failed",
                db=db,
                error="Previous full narration task was interrupted.",
            )
        await history.delete_generation(existing.id, db)
    await invalidate_project_cut_artifacts(project.id, db)
    reset_full_narration_timing(generation_id)

    delivery_instructions = (
        sanitize_dubbing_instructions(request.instruct or request.style_prompt)
        if is_qwen_dubbing_engine(engine)
        else None
    )
    effective_pace = (project.pace_override or 1.0) if is_qwen_dubbing_engine(engine) else 1.0
    effective_temperature = (
        clamp_temperature(request.temperature)
        if request.temperature is not None
        else clamp_temperature(project.temperature)
    )
    if not is_qwen_dubbing_engine(engine):
        effective_temperature = None
    generation = await history.create_generation(
        profile_id=request.profile_id,
        text=clean_text,
        language=request.language,
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        instruct=delivery_instructions,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=request.model_size,
        source="dubbing_full_narration",
    )

    task_manager.start_generation(
        task_id=generation.id,
        profile_id=request.profile_id,
        text=clean_text,
    )

    project.profile_id = request.profile_id
    project.style_prompt = delivery_instructions
    project.language = request.language
    project.engine = engine
    project.temperature = effective_temperature
    project.status = "processing"
    db.commit()

    enqueue_generation(
        generation.id,
        run_dubbing_generation(
            generation_id=generation.id,
            profile_id=request.profile_id,
            text=clean_text,
            language=request.language,
            engine=engine,
            model_size=request.model_size or "1.7B",
            seed=None,
            instruct=delivery_instructions,
            pace=effective_pace,
            temperature=effective_temperature,
            project_id=project.id,
            use_voice_prompt_cache=True,
            unload_after=True,
        ),
    )
    db_generation = db.query(DBGeneration).filter_by(id=generation.id).first()
    if db_generation is None:
        raise ValueError("Full narration generation could not be created.")
    return db_generation


async def delete_segment_generation(segment: DubbingSegment, db: Session) -> bool:
    """Delete the linked segment audio, preferring derived cuts when present."""
    cut_generation = get_cut_generation(segment, db)
    if cut_generation is not None:
        deleted = await history.delete_generation(cut_generation.id, db)
        if deleted and not segment.generation_id:
            segment.actual_duration_ms = None
            segment.delta_ms = None
            segment.fit_status = "unknown"
            segment.status = "pending"
        db.commit()
        return deleted

    if segment.generation_id:
        deleted = await history.delete_generation(segment.generation_id, db)
        segment.generation_id = None
        if not deleted:
            return False
        segment.actual_duration_ms = None
        segment.delta_ms = None
        segment.fit_status = "unknown"
        segment.status = "pending"
        db.commit()
        return deleted

    return False


async def invalidate_project_derived_audio(project_id: str, db: Session) -> None:
    """Delete full narration and cuts after editable SRT changes."""
    full_generation = get_full_narration_generation(project_id, db)
    if full_generation is not None:
        await history.delete_generation(full_generation.id, db)

    for cut_generation in list_cut_generations(project_id, db).values():
        await history.delete_generation(cut_generation.id, db)


async def delete_segment(segment: DubbingSegment, db: Session) -> None:
    """Delete a Dubbing SRT segment and invalidate derived project audio."""
    project = get_project_or_none(segment.project_id, db)
    if project is None:
        raise ValueError("Dubbing project not found.")

    remaining_count = db.query(DubbingSegment).filter_by(project_id=segment.project_id).count()
    if remaining_count <= 1:
        raise ValueError("Cannot delete the last Dubbing segment.")

    if segment.generation_id:
        await history.delete_generation(segment.generation_id, db)

    # Full narration and cuts are derived from the whole editable SRT. Once a
    # segment is removed, those outputs are stale and must be regenerated.
    await invalidate_project_derived_audio(segment.project_id, db)

    db.delete(segment)
    db.flush()

    remaining_segments = (
        db.query(DubbingSegment)
        .filter_by(project_id=project.id)
        .order_by(DubbingSegment.start_ms.asc(), DubbingSegment.segment_order.asc())
        .all()
    )
    for order, remaining in enumerate(remaining_segments, start=1):
        remaining.segment_order = order
        remaining.srt_index = order
        remaining.pace_group_id = None

    project.status = "draft"
    assign_pace_groups(remaining_segments)
    update_project_status(project, db)
    db.commit()


async def delete_project(project: DubbingProject, db: Session) -> None:
    """Delete a dubbing project and every linked segment generation."""
    full_generation = get_full_narration_generation(project.id, db)
    if full_generation is not None:
        await history.delete_generation(full_generation.id, db)

    segments = (
        db.query(DubbingSegment)
        .filter_by(project_id=project.id)
        .order_by(DubbingSegment.segment_order.asc())
        .all()
    )
    for segment in segments:
        if segment.generation_id:
            await history.delete_generation(segment.generation_id, db)
        cut_generation = get_cut_generation(segment, db)
        if cut_generation is not None:
            await history.delete_generation(cut_generation.id, db)
        db.delete(segment)

    db.delete(project)
    db.commit()


async def update_segment_text(segment: DubbingSegment, db: Session, *, text: str) -> None:
    value = text.strip()
    if not value:
        raise ValueError("Segment text cannot be empty.")
    if segment.generation_id:
        await delete_segment_generation(segment, db)
    await invalidate_project_derived_audio(segment.project_id, db)
    segment.text = value
    segment.text_lines = [value]
    segment.pace_group_id = None
    segment.status = "pending"
    segment.fit_status = "unknown"
    segment.actual_duration_ms = None
    segment.delta_ms = None
    db.commit()


async def update_segment_timing(
    segment: DubbingSegment,
    db: Session,
    *,
    start_ms: int,
    end_ms: int,
    preserve_audio: bool = False,
) -> None:
    """Update editable timeline timing for one segment."""
    if end_ms <= start_ms:
        raise ValueError("Segment end must be after segment start.")

    if not preserve_audio and segment.generation_id:
        await delete_segment_generation(segment, db)
    if not preserve_audio:
        await invalidate_project_derived_audio(segment.project_id, db)
    segment.start_ms = int(start_ms)
    segment.end_ms = int(end_ms)
    segment.target_duration_ms = int(end_ms - start_ms)
    segment.start_tc = format_srt_timecode(segment.start_ms)
    segment.end_tc = format_srt_timecode(segment.end_ms)
    if preserve_audio and segment.actual_duration_ms is not None:
        segment.delta_ms = segment.actual_duration_ms - segment.target_duration_ms
        segment.fit_status = classify_timing_fit(segment.delta_ms)
        segment.status = "generated" if segment.fit_status == "exact" else "warning"
    else:
        segment.fit_status = "unknown"
        segment.delta_ms = None
    segment.pace_group_id = None
    db.commit()


async def update_project_settings(
    project: DubbingProject,
    db: Session,
    *,
    pace_override: float | None,
    temperature: float | None,
    name: str | None = None,
) -> None:
    if name is not None:
        value = name.strip()
        if not value:
            raise ValueError("Project name cannot be empty.")
        project.name = value
    project.pace_override = clamp_pace(pace_override)
    project.temperature = clamp_temperature(temperature)
    db.commit()


async def update_group_pace_override(
    project: DubbingProject,
    db: Session,
    *,
    group_id: str,
    pace_override: float | None,
) -> None:
    segments = list_project_segments(project.id, db)
    groups = {group["id"]: group for group in assign_pace_groups(segments)}
    if group_id not in groups:
        raise ValueError("Dubbing pace group not found.")
    set_group_override(project, group_id, pace_override)
    db.commit()


async def queue_segment_generation(
    *,
    project: DubbingProject,
    segment: DubbingSegment,
    request: models.DubbingSegmentGenerateRequest,
    db: Session,
    engine: str,
) -> None:
    """Create one queued generation for the segment."""
    if segment.generation_id:
        await delete_segment_generation(segment, db)

    delivery_instructions = (
        sanitize_dubbing_instructions(request.instruct or request.style_prompt)
        if is_qwen_dubbing_engine(engine)
        else None
    )
    effective_temperature = (
        clamp_temperature(request.temperature)
        if request.temperature is not None
        else clamp_temperature(project.temperature)
    )
    if not is_qwen_dubbing_engine(engine):
        effective_temperature = None
    generation_id = str(segment.id)
    generation = await history.create_generation(
        profile_id=request.profile_id,
        text=segment.text,
        language=request.language,
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        instruct=delivery_instructions,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=request.model_size,
        source="dubbing_segment",
    )

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation.id,
        profile_id=request.profile_id,
        text=segment.text,
    )

    segment.generation_id = generation.id
    segment.status = "generating"
    segment.fit_status = "unknown"
    segment.actual_duration_ms = None
    segment.delta_ms = None
    project.profile_id = request.profile_id
    project.style_prompt = delivery_instructions
    project.language = request.language
    project.engine = engine
    project.temperature = effective_temperature
    project.status = "processing"
    db.commit()

    segments = list_project_segments(project.id, db)
    persisted_pace = (
        get_persisted_segment_pace(
            project=project,
            segment=segment,
            segments=segments,
        )
        if is_qwen_dubbing_engine(engine)
        else 1.0
    )

    enqueue_generation(
        generation.id,
        run_dubbing_generation(
            generation_id=generation.id,
            profile_id=request.profile_id,
            text=segment.text,
            language=request.language,
            engine=engine,
            model_size=request.model_size or "1.7B",
            seed=None,
            instruct=delivery_instructions,
            pace=persisted_pace,
            temperature=effective_temperature,
            project_id=project.id,
            segment_id=segment.id,
        ),
    )


async def _wait_for_segment_completion(project_id: str, segment_id: str) -> None:
    """Poll the generation until the segment leaves the generating state."""
    while True:
        await asyncio.sleep(0.5)
        db = next(get_db())
        try:
            segment = get_segment_or_none(project_id, segment_id, db)
            if segment is None:
                return
            if segment.status != "generating":
                return
        finally:
            db.close()


async def _auto_fit_segment_worker(
    *,
    project_id: str,
    segment_id: str,
    request: models.DubbingAutoFitRequest,
    engine: str,
) -> None:
    """Sequentially retry a segment until it stops overflowing or exhausts attempts."""
    for _attempt in range(max(1, request.max_attempts)):
        db = next(get_db())
        try:
            project = get_project_or_none(project_id, db)
            segment = get_segment_or_none(project_id, segment_id, db)
            if project is None or segment is None:
                return
            await queue_segment_generation(
                project=project,
                segment=segment,
                request=request,
                db=db,
                engine=engine,
            )
        finally:
            db.close()

        await _wait_for_segment_completion(project_id, segment_id)

        db = next(get_db())
        try:
            project = get_project_or_none(project_id, db)
            segment = get_segment_or_none(project_id, segment_id, db)
            if project is None or segment is None:
                return
            if segment.status == "generated":
                update_project_status(project, db)
                db.commit()
                return
            if segment.status == "warning" and (segment.delta_ms or 0) <= 0:
                update_project_status(project, db)
                db.commit()
                return
        finally:
            db.close()

    db = next(get_db())
    try:
        project = get_project_or_none(project_id, db)
        if project is not None and update_project_status(project, db):
            db.commit()
    finally:
        db.close()


def start_auto_fit_segment(
    *,
    project_id: str,
    segment_id: str,
    request: models.DubbingAutoFitRequest,
    engine: str,
) -> None:
    create_background_task(
        _auto_fit_segment_worker(
            project_id=project_id,
            segment_id=segment_id,
            request=request,
            engine=engine,
        )
    )


async def _auto_fit_project_worker(
    *,
    project_id: str,
    request: models.DubbingAutoFitRequest,
    engine: str,
) -> None:
    db = next(get_db())
    try:
        segments = (
            db.query(DubbingSegment)
            .filter_by(project_id=project_id)
            .order_by(DubbingSegment.segment_order.asc())
            .all()
        )
        segment_ids = [
            segment.id
            for segment in segments
            if segment.status != "generating"
        ]
    finally:
        db.close()

    for segment_id in segment_ids:
        await _auto_fit_segment_worker(
            project_id=project_id,
            segment_id=segment_id,
            request=request,
            engine=engine,
        )

    db = next(get_db())
    try:
        project = get_project_or_none(project_id, db)
        if project is not None and update_project_status(project, db):
            db.commit()
    finally:
        db.close()


def start_auto_fit_project(
    *,
    project_id: str,
    request: models.DubbingAutoFitRequest,
    engine: str,
) -> None:
    create_background_task(
        _auto_fit_project_worker(project_id=project_id, request=request, engine=engine)
    )


def _audio_bytes_from_timeline(placed_audio: list[tuple[int, np.ndarray]], sample_rate: int) -> bytes:
    if not placed_audio:
        return b""
    total_end_ms = 0
    for start_ms, audio in placed_audio:
        duration_ms = int(round((len(audio) / sample_rate) * 1000))
        total_end_ms = max(total_end_ms, start_ms + duration_ms)
    total_samples = int(np.ceil(total_end_ms * sample_rate / 1000))
    timeline = np.zeros(total_samples, dtype=np.float32)
    for start_ms, audio in placed_audio:
        start_sample = int(round(start_ms * sample_rate / 1000))
        end_sample = start_sample + len(audio)
        if end_sample > len(timeline):
            timeline = np.pad(timeline, (0, end_sample - len(timeline)))
        timeline[start_sample:end_sample] += audio.astype(np.float32, copy=False)
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(timeline, -1.0, 1.0), sample_rate, format="WAV")
    return buffer.getvalue()


def _apply_micro_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 6) -> np.ndarray:
    """Apply a tiny anti-click fade without changing clip duration."""
    if audio.size == 0 or fade_ms <= 0:
        return audio
    fade_samples = int(round(fade_ms * sample_rate / 1000))
    if fade_samples <= 1:
        return audio
    fade_samples = min(fade_samples, max(1, audio.size // 4))
    if fade_samples <= 1:
        return audio
    faded = audio.astype(np.float32, copy=True)
    faded[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return faded


async def post_process_full_narration_cuts(project: DubbingProject, db: Session) -> int:
    """Cut the full narration WAV into SRT segment audio files.

    This pre-alignment stage is intentionally deterministic: it consumes the
    complete narration in order and distributes it across SRT blocks by their
    relative time budget. Each cut keeps a small lead-in and tail so playback is
    not abrupt. If a cut runs longer than the subtitle window, it is preserved
    and marked as a timing warning rather than truncated.
    """
    segments = list_project_segments(project.id, db)
    if not segments:
        raise ValueError("Dubbing project has no SRT segments.")

    full_generation = get_full_narration_generation(project.id, db)
    if (
        full_generation is None
        or (full_generation.status or "completed") != "completed"
        or not full_generation.audio_path
    ):
        raise ValueError("Generate the full SRT narration before post-processing cuts.")

    full_audio_path = config.resolve_storage_path(full_generation.audio_path)
    if full_audio_path is None or not full_audio_path.exists():
        raise ValueError("Full narration audio file was not found.")

    sample_rate = 24000
    audio, sample_rate = load_audio(str(full_audio_path), sample_rate=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError("Full narration audio file is empty.")

    cut_dir = config.get_generations_dir() / "dubbing_cuts" / project.id
    cut_dir.mkdir(parents=True, exist_ok=True)
    alignment_spans, alignment_debug = await _align_segments_to_full_narration(
        audio_path=full_audio_path,
        project=project,
        segments=segments,
    )
    alignment_debug["cuts"] = []
    total_segment_duration_ms = max(1, sum(max(1, segment.target_duration_ms) for segment in segments))
    cursor_sample = 0

    created = 0
    for segment_index, segment in enumerate(segments):
        alignment_span = alignment_spans.get(segment.id)
        next_alignment_span = (
            alignment_spans.get(segments[segment_index + 1].id)
            if segment_index + 1 < len(segments)
            else None
        )
        if alignment_span is not None:
            aligned_start_ms, aligned_end_ms, score = alignment_span
            raw_start_sample = max(0, int(round(aligned_start_ms * sample_rate / 1000)))
            raw_end_sample = min(len(audio), int(round(aligned_end_ms * sample_rate / 1000)))
            cut_source = "whisper_word_alignment"
            end_boundary_source = "current_segment_last_word"
            if next_alignment_span is not None:
                next_start_ms, _, _ = next_alignment_span
                next_start_sample = max(0, int(round(next_start_ms * sample_rate / 1000)))
                next_boundary_sample = max(
                    raw_start_sample + int(round(0.01 * sample_rate)),
                    next_start_sample - int(round(DUBBING_CUT_LEAD_IN_MS * sample_rate / 1000)),
                )
                if next_boundary_sample > raw_end_sample:
                    raw_end_sample = min(len(audio), next_boundary_sample)
                    end_boundary_source = "next_segment_start_minus_lead"
            cursor_sample = max(cursor_sample, raw_end_sample)
        else:
            if segment_index == len(segments) - 1:
                raw_end_sample = len(audio)
            else:
                segment_ratio = max(1, segment.target_duration_ms) / total_segment_duration_ms
                raw_end_sample = min(len(audio), cursor_sample + int(round(len(audio) * segment_ratio)))
            raw_start_sample = cursor_sample
            cursor_sample = raw_end_sample
            score = None
            cut_source = "srt_ratio_fallback"
            end_boundary_source = "srt_ratio_fallback"

        cut_start_sample = max(
            0,
            raw_start_sample - int(round(DUBBING_CUT_LEAD_IN_MS * sample_rate / 1000)),
        )
        cut_tail_ms = (
            0
            if end_boundary_source == "next_segment_start_minus_lead"
            else DUBBING_CUT_TAIL_OUT_MS
        )
        cut_end_sample = min(len(audio), raw_end_sample + int(round(cut_tail_ms * sample_rate / 1000)))

        segment_audio = audio[cut_start_sample:cut_end_sample].astype(np.float32, copy=False)

        generation_id = cut_generation_id(segment)
        existing = db.query(DBGeneration).filter_by(id=generation_id).first()
        if existing is not None:
            await history.delete_generation(existing.id, db)

        if segment_audio.size == 0:
            segment_audio = np.zeros(max(1, int(sample_rate * 0.01)), dtype=np.float32)

        cut_path = cut_dir / f"segment_{segment.srt_index:04d}.wav"
        sf.write(cut_path, np.clip(segment_audio, -1.0, 1.0), sample_rate, format="WAV")
        duration = len(segment_audio) / sample_rate
        actual_duration_ms = int(round(duration * 1000))
        alignment_debug["cuts"].append(
            {
                "segment_id": segment.id,
                "srt_index": segment.srt_index,
                "source": cut_source,
                "end_boundary_source": end_boundary_source,
                "score": score,
                "raw_start_ms": int(round(raw_start_sample * 1000 / sample_rate)),
                "raw_end_ms": int(round(raw_end_sample * 1000 / sample_rate)),
                "cut_start_ms": int(round(cut_start_sample * 1000 / sample_rate)),
                "cut_end_ms": int(round(cut_end_sample * 1000 / sample_rate)),
                "target_start_ms": segment.start_ms,
                "target_end_ms": segment.end_ms,
                "target_duration_ms": segment.target_duration_ms,
                "actual_duration_ms": actual_duration_ms,
                "audio_path": str(cut_path),
            }
        )
        segment.actual_duration_ms = actual_duration_ms
        segment.delta_ms = actual_duration_ms - segment.target_duration_ms
        segment.fit_status = classify_timing_fit(segment.delta_ms)
        segment.status = "generated" if segment.fit_status == "exact" else "warning"
        if alignment_span is None:
            segment.fit_status = "warning"
            segment.status = "warning"
        await history.create_generation(
            profile_id=full_generation.profile_id,
            text=(segment.text or "").strip(),
            language=project.language,
            audio_path=config.to_storage_path(cut_path),
            duration=duration,
            seed=None,
            db=db,
            instruct=full_generation.instruct,
            generation_id=generation_id,
            status="completed",
            engine=project.engine,
            model_size=full_generation.model_size,
            source="dubbing_segment_cut",
        )
        created += 1

    debug_path = cut_dir / "alignment_debug.json"
    debug_path.write_text(
        json.dumps(alignment_debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    project.status = "completed"
    db.commit()
    return created


async def create_manual_cut_from_full_narration(
    project: DubbingProject,
    segment: DubbingSegment,
    db: Session,
    *,
    cut_start_ms: int,
    cut_end_ms: int,
    use_previous_cut_end: bool = False,
) -> None:
    """Create or replace one segment cut using explicit full-narration bounds."""
    if use_previous_cut_end:
        cut_start_ms = _previous_manual_cut_end(project, segment, db)

    if cut_end_ms <= cut_start_ms:
        raise ValueError("Manual cut end must be after cut start.")

    full_generation = get_full_narration_generation(project.id, db)
    if (
        full_generation is None
        or (full_generation.status or "completed") != "completed"
        or not full_generation.audio_path
    ):
        raise ValueError("Generate the full SRT narration before creating manual cuts.")

    full_audio_path = config.resolve_storage_path(full_generation.audio_path)
    if full_audio_path is None or not full_audio_path.exists():
        raise ValueError("Full narration audio file was not found.")

    sample_rate = 24000
    audio, sample_rate = load_audio(str(full_audio_path), sample_rate=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError("Full narration audio file is empty.")

    start_sample = max(0, min(len(audio), int(round(cut_start_ms * sample_rate / 1000))))
    end_sample = max(start_sample, min(len(audio), int(round(cut_end_ms * sample_rate / 1000))))
    if end_sample <= start_sample:
        raise ValueError("Manual cut is outside the full narration audio.")

    segment_audio = audio[start_sample:end_sample].astype(np.float32, copy=False)
    if segment_audio.size == 0:
        segment_audio = np.zeros(max(1, int(sample_rate * 0.01)), dtype=np.float32)

    generation_id = cut_generation_id(segment)
    existing = db.query(DBGeneration).filter_by(id=generation_id).first()
    if existing is not None:
        await history.delete_generation(existing.id, db)

    cut_dir = config.get_generations_dir() / "dubbing_cuts" / project.id
    cut_dir.mkdir(parents=True, exist_ok=True)
    cut_path = cut_dir / f"segment_{segment.srt_index:04d}.wav"
    sf.write(cut_path, np.clip(segment_audio, -1.0, 1.0), sample_rate, format="WAV")

    duration = len(segment_audio) / sample_rate
    actual_duration_ms = int(round(duration * 1000))
    segment.actual_duration_ms = actual_duration_ms
    segment.delta_ms = actual_duration_ms - segment.target_duration_ms
    segment.fit_status = classify_timing_fit(segment.delta_ms)
    segment.status = "generated" if segment.fit_status == "exact" else "warning"

    await history.create_generation(
        profile_id=full_generation.profile_id,
        text=(segment.text or "").strip(),
        language=project.language,
        audio_path=config.to_storage_path(cut_path),
        duration=duration,
        seed=None,
        db=db,
        instruct=full_generation.instruct,
        generation_id=generation_id,
        status="completed",
        engine=project.engine,
        model_size=full_generation.model_size,
        source="dubbing_segment_cut_manual",
    )

    debug_path = cut_dir / "manual_cuts.jsonl"
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "segment_id": segment.id,
                    "srt_index": segment.srt_index,
                    "cut_start_ms": cut_start_ms,
                    "cut_end_ms": cut_end_ms,
                    "actual_duration_ms": actual_duration_ms,
                    "audio_path": str(cut_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    project.status = "completed"
    db.add(segment)
    db.add(project)
    db.commit()


def build_edited_srt(project_id: str, db: Session) -> str:
    segments = list_project_segments(project_id, db)
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = (segment.text or "").strip()
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{segment.start_tc} --> {segment.end_tc}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


async def build_project_export_package(
    project_id: str,
    db: Session,
    *,
    timeline_wav: bytes | None = None,
) -> bytes:
    project = get_project_or_none(project_id, db)
    if project is None:
        return b""

    segments = list_project_segments(project_id, db)
    full_generation = get_full_narration_generation(project_id, db)
    cut_generations = list_cut_generations(project_id, db)
    if timeline_wav is None:
        timeline_wav = await build_project_timeline_wav(project_id, db)

    manifest = {
        "project": {
            "id": project.id,
            "name": project.name,
            "engine": project.engine,
            "language": project.language,
            "profile_id": project.profile_id,
            "style_prompt": project.style_prompt,
        },
        "full_narration": {
            "generation_id": full_generation.id if full_generation else None,
            "status": full_generation.status if full_generation else None,
            "duration_ms": (
                int(round(full_generation.duration * 1000))
                if full_generation is not None and full_generation.duration is not None
                else None
            ),
        },
        "segments": [],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if full_generation is not None and full_generation.audio_path:
            full_path = config.resolve_storage_path(full_generation.audio_path)
            if full_path is not None and full_path.exists():
                archive.write(full_path, "audio/full_narration.wav")

        if timeline_wav:
            archive.writestr("audio/resequenced_timeline.wav", timeline_wav)

        for segment in segments:
            cut_generation = cut_generations.get(segment.id)
            cut_filename = None
            if cut_generation is not None and cut_generation.audio_path:
                cut_path = config.resolve_storage_path(cut_generation.audio_path)
                if cut_path is not None and cut_path.exists():
                    cut_filename = f"segments/segment_{segment.srt_index:04d}.wav"
                    archive.write(cut_path, cut_filename)
            manifest["segments"].append(
                {
                    "segment_id": segment.id,
                    "srt_index": segment.srt_index,
                    "start_tc": segment.start_tc,
                    "end_tc": segment.end_tc,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "text": segment.text,
                    "segment_audio": cut_filename,
                    "cut_generation_id": cut_generation.id if cut_generation is not None else None,
                    "cut_duration_ms": (
                        int(round(cut_generation.duration * 1000))
                        if cut_generation is not None and cut_generation.duration is not None
                        else None
                    ),
                }
            )

        edited_srt = build_edited_srt(project_id, db)
        debug_path = config.get_generations_dir() / "dubbing_cuts" / project_id / "alignment_debug.json"
        if debug_path.exists():
            archive.write(debug_path, "debug/alignment_debug.json")
        word_matching_path = config.get_generations_dir() / "dubbing_cuts" / project_id / "word_matching_debug.json"
        if word_matching_path.exists():
            archive.write(word_matching_path, "debug/word_matching_debug.json")
        clean_text_path = _clean_srt_narration_text_path(project_id)
        if clean_text_path.exists():
            archive.write(clean_text_path, "debug/clean_srt_narration.txt")
        archive.writestr("srt/edited.srt", edited_srt)
        archive.writestr("srt/original.srt", edited_srt)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return buffer.getvalue()


async def build_project_timeline_wav(project_id: str, db: Session) -> bytes:
    """Assemble generated segment audio into a single WAV following the SRT timeline."""
    project = get_project_or_none(project_id, db)
    if project is None:
        return b""
    segments = list_project_segments(project_id, db)
    if not segments:
        return b""

    sample_rate = 24000
    cut_audio_by_segment = list_cut_generations(project_id, db)
    if cut_audio_by_segment:
        placed_audio: list[tuple[int, np.ndarray]] = []
        previous_end_ms = 0
        for segment in segments:
            generation = cut_audio_by_segment.get(segment.id)
            if generation is None or not generation.audio_path:
                continue
            audio_path = config.resolve_storage_path(generation.audio_path)
            if audio_path is None or not audio_path.exists():
                continue
            audio, sample_rate = load_audio(str(audio_path), sample_rate=sample_rate, mono=True)
            if audio.size:
                start_ms = max(segment.start_ms, previous_end_ms)
                placed_audio.append((start_ms, audio.astype(np.float32, copy=False)))
                previous_end_ms = start_ms + int(round((len(audio) / sample_rate) * 1000))
        wav_bytes = _audio_bytes_from_timeline(placed_audio, sample_rate)
        if wav_bytes:
            return wav_bytes

    full_generation = get_full_narration_generation(project_id, db)
    if (
        full_generation is not None
        and (full_generation.status or "completed") == "completed"
        and full_generation.audio_path
    ):
        audio_path = config.resolve_storage_path(full_generation.audio_path)
        if audio_path is not None and audio_path.exists():
            audio, sample_rate = load_audio(str(audio_path), sample_rate=sample_rate, mono=True)
            if audio.size:
                first_start_ms = min(segment.start_ms for segment in segments)
                total_end_ms = first_start_ms + int(round((len(audio) / sample_rate) * 1000))
                total_samples = int(np.ceil(total_end_ms * sample_rate / 1000))
                timeline = np.zeros(total_samples, dtype=np.float32)
                start_sample = int(round(first_start_ms * sample_rate / 1000))
                timeline[start_sample:start_sample + len(audio)] = audio.astype(np.float32, copy=False)
                buffer = io.BytesIO()
                sf.write(buffer, np.clip(timeline, -1.0, 1.0), sample_rate, format="WAV")
                return buffer.getvalue()

    placed_audio: list[tuple[int, np.ndarray]] = []
    previous_end_ms = 0
    total_end_ms = 0
    generated_audio: dict[str, np.ndarray] = {}

    for segment in segments:
        if not segment.generation_id:
            continue
        generation = db.query(DBGeneration).filter_by(id=segment.generation_id).first()
        if generation is None or not generation.audio_path:
            continue

        audio_path = config.resolve_storage_path(generation.audio_path)
        if audio_path is None or not audio_path.exists():
            continue

        audio, sample_rate = load_audio(str(audio_path), sample_rate=sample_rate, mono=True)
        if audio.size == 0:
            continue
        generated_audio[segment.id] = audio.astype(np.float32, copy=False)

    for segment in segments:
        audio = generated_audio.get(segment.id)
        if audio is None:
            continue
        start_ms = max(segment.start_ms, previous_end_ms)
        duration_ms = int(round((len(audio) / sample_rate) * 1000))
        end_ms = start_ms + duration_ms
        placed_audio.append((start_ms, audio))
        previous_end_ms = end_ms
        total_end_ms = max(total_end_ms, end_ms, segment.end_ms)

    if not placed_audio:
        return b""

    total_samples = int(np.ceil(total_end_ms * sample_rate / 1000))
    timeline = np.zeros(total_samples, dtype=np.float32)

    for start_ms, audio in placed_audio:
        start_sample = int(round(start_ms * sample_rate / 1000))
        end_sample = start_sample + len(audio)
        if end_sample > len(timeline):
            timeline = np.pad(timeline, (0, end_sample - len(timeline)))
        timeline[start_sample:end_sample] += audio.astype(np.float32)

    timeline = np.clip(timeline, -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, timeline, sample_rate, format="WAV")
    return buffer.getvalue()


async def build_project_visible_timeline_wav(
    project_id: str,
    db: Session,
    *,
    clips: list,
) -> bytes:
    """Render the visible desktop timeline clips instead of the raw full narration.

    The clip list is intentionally provided by the UI because split/trim/move
    edits are currently stored as desktop timeline state, like the Stories editor.
    """
    project = get_project_or_none(project_id, db)
    segments = list_project_segments(project_id, db) if project is not None else []
    if project is None or not segments or not clips:
        return b""

    sample_rate = 24000
    placed_audio: list[tuple[int, np.ndarray]] = []
    total_end_ms = max(segment.end_ms for segment in segments)

    ordered_clips = sorted(clips, key=lambda item: int(getattr(item, "start_ms", 0) or 0))
    previous_audible_end_ms = 0
    for clip in ordered_clips:
        generation_id = getattr(clip, "generation_id", None)
        generation = db.query(DBGeneration).filter_by(id=generation_id).first()
        if generation is None or not generation.audio_path:
            continue

        audio_path = config.resolve_storage_path(generation.audio_path)
        if audio_path is None or not audio_path.exists():
            continue

        audio, sample_rate = load_audio(str(audio_path), sample_rate=sample_rate, mono=True)
        if audio.size == 0:
            continue

        trim_start_ms = max(0, int(getattr(clip, "trim_start_ms", 0) or 0))
        trim_end_ms = max(0, int(getattr(clip, "trim_end_ms", 0) or 0))
        start_sample = min(len(audio), int(round(trim_start_ms * sample_rate / 1000)))
        end_sample = max(start_sample, len(audio) - int(round(trim_end_ms * sample_rate / 1000)))
        trimmed = audio[start_sample:end_sample].astype(np.float32, copy=False)
        if trimmed.size == 0:
            continue
        trimmed = _apply_micro_fade(trimmed, sample_rate)

        volume = float(getattr(clip, "volume", 1.0) or 1.0)
        if volume <= 0.001:
            continue
        if volume != 1.0:
            trimmed = trimmed * volume

        start_ms = max(0, int(getattr(clip, "start_ms", 0) or 0))
        duration_ms = int(round((len(trimmed) / sample_rate) * 1000))
        if start_ms < previous_audible_end_ms:
            start_ms = previous_audible_end_ms
        previous_audible_end_ms = start_ms + duration_ms
        total_end_ms = max(total_end_ms, start_ms + duration_ms)
        placed_audio.append((start_ms, trimmed))

    if not placed_audio:
        return b""

    total_samples = int(np.ceil(total_end_ms * sample_rate / 1000))
    timeline = np.zeros(total_samples, dtype=np.float32)
    for start_ms, audio in placed_audio:
        start_sample = int(round(start_ms * sample_rate / 1000))
        end_sample = start_sample + len(audio)
        if end_sample > len(timeline):
            timeline = np.pad(timeline, (0, end_sample - len(timeline)))
        timeline[start_sample:end_sample] = audio.astype(np.float32, copy=False)

    buffer = io.BytesIO()
    sf.write(buffer, np.clip(timeline, -1.0, 1.0), sample_rate, format="WAV")
    return buffer.getvalue()
