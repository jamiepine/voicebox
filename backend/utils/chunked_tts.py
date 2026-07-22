"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, List, Tuple

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_first_sentence_end(text: str) -> int:
    """Return the index of the *first* sentence-ending punctuation in ``text``.

    Mirror of :func:`_find_last_sentence_end` but scans forward instead of
    backward — used by the streaming pipeline, which needs to flush the
    earliest complete sentence as soon as it lands so the TTS backend can
    start on it while the LLM keeps generating.
    """
    ascii_positions: List[int] = []
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        if char == ".":
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            if word_start >= 0 and text[word_start].isdigit():
                continue
        if _inside_bracket_tag(text, pos):
            continue
        ascii_positions.append(pos)

    cjk_positions = [m.start() for m in re.finditer(r"[。！？]", text)]

    all_positions = ascii_positions + cjk_positions
    return min(all_positions) if all_positions else -1


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.)
    and periods inside bracket tags (``[laugh]``).  Also handles CJK
    sentence-ending punctuation (``。！？``).
    """
    best = -1
    # ASCII sentence ends
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        # Skip periods after abbreviations
        if char == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        # Skip if we're inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

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


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
) -> Tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
    """
    chunks = split_text_into_chunks(text, max_chunk_chars)

    if len(chunks) <= 1:
        # Short text — single-shot fast path
        audio, sample_rate = await backend.generate(
            text,
            voice_prompt,
            language,
            seed,
            instruct,
        )
        if trim_fn is not None:
            audio = trim_fn(audio, sample_rate)
        return audio, sample_rate

    # Long text — chunked generation
    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    audio_chunks: List[np.ndarray] = []
    sample_rate: int | None = None

    for i, chunk_text in enumerate(chunks):
        logger.info(
            "Generating chunk %d/%d (%d chars)",
            i + 1,
            len(chunks),
            len(chunk_text),
        )
        # Vary the seed per chunk to avoid correlated RNG artefacts,
        # but keep it deterministic so the same (text, seed) pair
        # always produces the same output.
        chunk_seed = (seed + i) if seed is not None else None

        chunk_audio, chunk_sr = await backend.generate(
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
        )
        if trim_fn is not None:
            chunk_audio = trim_fn(chunk_audio, chunk_sr)

        audio_chunks.append(np.asarray(chunk_audio, dtype=np.float32))
        if sample_rate is None:
            sample_rate = chunk_sr

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate


async def stream_sentences(
    llm_stream: AsyncIterator[str],
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> AsyncIterator[str]:
    """Buffer an LLM text stream and emit one complete sentence at a time.

    Consumes ``llm_stream`` (typically ``LLMBackend.generate_stream``), keeps
    a rolling text buffer, and yields as soon as a sentence-ending
    punctuation lands inside it. If the buffer ever exceeds
    ``max_chunk_chars`` without a sentence end, it flushes at a clause
    boundary or whitespace so a runaway generation can't stall the TTS
    pipeline forever. Trailing text (no punctuation on the last chunk)
    is flushed at end-of-stream.
    """
    buffer = ""
    async for delta in llm_stream:
        if not delta:
            continue
        buffer += delta

        while True:
            # Preferred: a real sentence boundary.
            end = _find_first_sentence_end(buffer)
            if end != -1:
                sentence = buffer[: end + 1].strip()
                buffer = buffer[end + 1 :]
                if sentence:
                    yield sentence
                continue

            # Fallback: buffer got too long without punctuation. Cut at
            # a clause boundary (or whitespace) so the TTS backend keeps
            # moving even when the LLM produces one long run-on line.
            if len(buffer) < max_chunk_chars:
                break

            segment = buffer[:max_chunk_chars]
            cut = _find_last_clause_boundary(segment)
            if cut == -1:
                cut = segment.rfind(" ")
            if cut == -1:
                cut = _safe_hard_cut(segment, max_chunk_chars)

            sentence = buffer[: cut + 1].strip()
            buffer = buffer[cut + 1 :]
            if sentence:
                yield sentence

    tail = buffer.strip()
    if tail:
        yield tail


async def generate_streaming_from_sentences(
    backend,
    sentence_stream: AsyncIterator[str],
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    trim_fn=None,
) -> AsyncIterator[Tuple[np.ndarray, int]]:
    """Fire per-sentence TTS as sentences arrive; yield ordered audio chunks.

    Each sentence is dispatched to ``backend.generate`` as an
    ``asyncio.create_task`` the moment it lands, so sentence *N+1*'s TTS
    starts running while sentence *N*'s audio is still being awaited by the
    caller. The tasks are awaited in submission order, so the yielded
    ``(audio, sample_rate)`` tuples arrive in the same order the sentences
    were produced — the caller doesn't have to reorder anything.
    """
    pending: List[asyncio.Task] = []
    index = 0

    async def _drain_ready() -> AsyncIterator[Tuple[np.ndarray, int]]:
        while pending and pending[0].done():
            task = pending.pop(0)
            audio, sr = task.result()
            if trim_fn is not None:
                audio = trim_fn(audio, sr)
            yield np.asarray(audio, dtype=np.float32), sr

    async for sentence in sentence_stream:
        # Vary the seed per sentence to avoid correlated RNG artefacts but
        # keep it deterministic for a given (seed, sentence-index).
        chunk_seed = (seed + index) if seed is not None else None
        pending.append(
            asyncio.create_task(
                backend.generate(sentence, voice_prompt, language, chunk_seed, instruct)
            )
        )
        index += 1

        async for out in _drain_ready():
            yield out

    for task in pending:
        audio, sr = await task
        if trim_fn is not None:
            audio = trim_fn(audio, sr)
        yield np.asarray(audio, dtype=np.float32), sr


async def generate_streaming_chunked(
    llm_backend,
    tts_backend,
    prompt: str,
    voice_prompt: dict,
    *,
    system: str | None = None,
    llm_max_tokens: int = 512,
    llm_temperature: float = 0.7,
    llm_model_size: str | None = None,
    llm_examples: list[tuple[str, str]] | None = None,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
) -> Tuple[np.ndarray, int]:
    """End-to-end LLM stream → per-sentence TTS → concatenated audio.

    Pulls text from ``llm_backend.generate_stream``, batches into sentences,
    dispatches each sentence to ``tts_backend.generate`` as an asyncio task,
    then concatenates the ordered audio chunks with a crossfade to remove
    boundary clicks. First-audio-latency drops to roughly the LLM's
    time-to-first-sentence plus one TTS chunk instead of the sum of the
    two, which is the whole point of the streaming path.
    """
    llm_stream = llm_backend.generate_stream(
        prompt,
        system=system,
        max_tokens=llm_max_tokens,
        temperature=llm_temperature,
        model_size=llm_model_size,
        examples=llm_examples,
    )
    sentence_stream = stream_sentences(llm_stream, max_chunk_chars=max_chunk_chars)

    audio_chunks: List[np.ndarray] = []
    sample_rate: int | None = None
    async for audio, sr in generate_streaming_from_sentences(
        tts_backend,
        sentence_stream,
        voice_prompt,
        language=language,
        seed=seed,
        instruct=instruct,
        trim_fn=trim_fn,
    ):
        audio_chunks.append(audio)
        if sample_rate is None:
            sample_rate = sr

    if sample_rate is None:
        # Streaming produced no audio (empty LLM output). Callers that
        # already validate against empty LLM output will have caught this
        # upstream; returning an empty array keeps the type honest for
        # anyone who didn't.
        return np.array([], dtype=np.float32), 0

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate
