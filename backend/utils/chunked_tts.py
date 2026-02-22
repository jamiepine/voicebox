"""
Chunked TTS generation with quality selection.

Splits long text into sentence-boundary chunks, generates audio per-chunk,
and concatenates with crossfade. Optionally upsamples to 44.1kHz for
higher quality output.

Environment variables:
    TTS_QUALITY: "standard" (24kHz native) or "high" (44.1kHz upsampled)
    TTS_MAX_CHUNK_CHARS: Max characters per chunk (default 800)
    TTS_UPSAMPLE_RATE: Target sample rate for high quality (default 44100)
"""

import logging
import os
import re
from typing import List

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# ---------------------------------------------------------------------------
# Runtime-mutable settings
# ---------------------------------------------------------------------------

_tts_settings = {
    "quality": os.getenv("TTS_QUALITY", "standard"),
    "max_chunk_chars": int(os.getenv("TTS_MAX_CHUNK_CHARS", "800")),
    "upsample_rate": int(os.getenv("TTS_UPSAMPLE_RATE", "44100")),
}

QUALITY_RATES = {
    "standard": 24000,   # Qwen3-TTS native sample rate
    "high": 44100,       # CD-quality upsampled via soxr
}


def get_tts_settings() -> dict:
    """Return current TTS chunking/quality settings."""
    quality = _tts_settings["quality"]
    return {
        "quality": quality,
        "sample_rate": QUALITY_RATES.get(quality, 24000),
        "max_chunk_chars": _tts_settings["max_chunk_chars"],
        "available_qualities": list(QUALITY_RATES.keys()),
    }


def update_tts_settings(updates: dict) -> dict:
    """Update TTS settings at runtime. Returns new settings."""
    if "quality" in updates:
        q = updates["quality"]
        if q not in QUALITY_RATES:
            raise ValueError(
                f"Invalid quality '{q}'. Must be one of {list(QUALITY_RATES.keys())}"
            )
        _tts_settings["quality"] = q
    if "max_chunk_chars" in updates:
        val = int(updates["max_chunk_chars"])
        if val < 100 or val > 5000:
            raise ValueError("max_chunk_chars must be between 100 and 5000")
        _tts_settings["max_chunk_chars"] = val
    return get_tts_settings()


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """Split text at sentence boundaries, with clause and word fallbacks.

    Priority: sentence-end (.!?) > clause boundary (;:,) > whitespace > hard cut.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        remaining = remaining.strip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to find last sentence end
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            split_pos = max_chars - 1

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    best = -1
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        best = m.start()
    return best


# ---------------------------------------------------------------------------
# Audio concatenation
# ---------------------------------------------------------------------------

def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sr: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to avoid clicks."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sr * crossfade_ms / 1000)
    result = chunks[0].copy()

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio using soxr (VHQ), with linear-interp fallback."""
    if src_rate == dst_rate:
        return audio
    try:
        import soxr

        return soxr.resample(audio, src_rate, dst_rate, quality="VHQ")
    except ImportError:
        logger.warning("soxr not installed; falling back to linear interpolation")
        ratio = dst_rate / src_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
