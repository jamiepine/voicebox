"""
Server-side driver for ``POST /speak/stream``.

Wraps the streaming LLM+TTS pipeline in an SSE-friendly async generator
so the client hears each sentence as soon as it lands instead of waiting
for the full audio file to be concatenated and saved. The event shape
mirrors OpenAI's streaming style (a ``data: {...}`` frame per event, a
final ``data: [DONE]`` sentinel) so the browser side needs no server-
specific framing code.

The generator yields SSE-formatted ``str`` values, one per frame,
ready to be fed straight into a ``StreamingResponse``. Callers should
not JSON-encode or wrap the output — that's already done here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import traceback
import uuid
from typing import AsyncIterator, Optional

import numpy as np

from .. import config, models
from ..database import get_db
from ..utils.audio import save_audio

logger = logging.getLogger(__name__)


async def stream_speak_events(
    *,
    profile_id: str,
    text: str,
    engine: str,
    language: str,
    personality: bool,
    profile_personality: Optional[str],
    max_chunk_chars: int,
) -> AsyncIterator[str]:
    """Drive one /speak/stream request and yield SSE frames.

    Frames (in order):
      1. ``meta`` — sample_rate + channels + generation_id.
      2. ``audio`` — one per sentence, base64-encoded PCM float32.
      3. ``complete`` — total duration + final audio_path.
      4. ``[DONE]`` sentinel.

    On failure at any step the generator yields an ``error`` frame and
    the terminal sentinel; it never leaves the SSE connection open on a
    partial state.
    """
    from ..backends import (
        engine_needs_trim,
        get_llm_backend,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import (
        concatenate_audio_chunks,
        generate_streaming_from_sentences,
        stream_sentences,
    )
    from ..utils.audio import trim_tts_output
    from . import history, profiles
    from .personality import rewrite_as_profile_stream

    generation_id = str(uuid.uuid4())

    try:
        tts_backend = get_tts_backend_for_engine(engine)
        await load_engine_model(engine)

        bg_db = next(get_db())
        try:
            voice_prompt = await profiles.create_voice_prompt_for_profile(
                profile_id,
                bg_db,
                use_cache=True,
                engine=engine,
            )
        finally:
            bg_db.close()

        # Build the sentence stream — either from the streaming LLM path
        # (when the active backend can produce deltas AND personality is on)
        # or from a single-sentence generator wrapping the raw user text.
        llm_backend = get_llm_backend()
        can_stream_llm = bool(
            personality
            and profile_personality
            and getattr(llm_backend, "supports_streaming", lambda: False)()
        )
        streaming_llm = can_stream_llm

        # Materialise the text that TTS will actually render, so the
        # generations row stored below matches the audio on disk. On the
        # personality paths that means the rewritten reply; on the plain
        # path it's the caller's own text unchanged.
        if can_stream_llm:
            llm_stream = rewrite_as_profile_stream(profile_personality, text)
            # Every branch runs through ``stream_sentences`` so long
            # inputs stay chunked to ``max_chunk_chars`` instead of
            # falling into a single oversized ``backend.generate`` call
            # on the fallback paths.
            sentence_stream = stream_sentences(llm_stream, max_chunk_chars=max_chunk_chars)
        elif personality and profile_personality:
            # Personality requested but the LLM can't stream — fall back
            # to the non-streaming rewrite, then pipe the whole reply
            # through the same sentence splitter so TTS still runs one
            # sentence at a time.
            from .personality import rewrite_as_profile

            rewritten = await rewrite_as_profile(profile_personality, text)
            sentence_stream = stream_sentences(
                _as_single_chunk(rewritten.text.strip()),
                max_chunk_chars=max_chunk_chars,
            )
        else:
            sentence_stream = stream_sentences(
                _as_single_chunk(text),
                max_chunk_chars=max_chunk_chars,
            )

        trim_fn = trim_tts_output if engine_needs_trim(engine) else None

        audio_chunks: list[np.ndarray] = []
        sentence_texts: list[str] = []
        sample_rate: Optional[int] = None
        sentence_index = 0

        async for audio, sr, sentence in generate_streaming_from_sentences(
            tts_backend,
            sentence_stream,
            voice_prompt,
            language=language,
            trim_fn=trim_fn,
        ):
            if sample_rate is None:
                sample_rate = sr
                yield _sse_data(
                    models.SpeakStreamMeta(
                        generation_id=generation_id,
                        sample_rate=sr,
                        channels=1,
                        streaming_llm=streaming_llm,
                    ).model_dump()
                )

            audio_chunks.append(audio)
            sentence_texts.append(sentence)
            pcm_b64 = base64.b64encode(np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii")
            yield _sse_data(
                models.SpeakStreamAudioChunk(
                    sentence_index=sentence_index,
                    pcm_base64=pcm_b64,
                    text=sentence,
                ).model_dump()
            )
            sentence_index += 1

        if sample_rate is None:
            yield _sse_data(
                models.SpeakStreamError(
                    generation_id=generation_id,
                    message="Pipeline produced no audio.",
                ).model_dump()
            )
            yield _sse_done()
            return

        final_audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=50)
        final_path = config.get_generations_dir() / f"{generation_id}.wav"
        save_audio(final_audio, str(final_path), sample_rate)

        # The persisted ``text`` has to match what was actually spoken —
        # ``data.text`` is the raw user input, so on the personality path
        # it would disagree with the audio otherwise. Join what the
        # streaming pipeline handed to TTS.
        spoken_text = " ".join(s for s in sentence_texts if s).strip() or text

        # Persist a generations row so History reflects streamed speech
        # the same way fire-and-forget /speak does. Failing to persist
        # is non-fatal — the audio is already on disk and the client has
        # the whole thing in memory.
        try:
            bg_db = next(get_db())
            try:
                await history.create_generation(
                    profile_id=profile_id,
                    text=spoken_text,
                    language=language,
                    audio_path=config.to_storage_path(final_path),
                    duration=len(final_audio) / sample_rate,
                    seed=None,
                    db=bg_db,
                    generation_id=generation_id,
                    status="completed",
                    engine=engine,
                    source="stream_speak",
                )
            finally:
                bg_db.close()
        except Exception:
            logger.warning("History row for streamed speak failed", exc_info=True)

        yield _sse_data(
            models.SpeakStreamComplete(
                generation_id=generation_id,
                duration=len(final_audio) / sample_rate,
                # Match the ``audio_path`` shape the generations row was
                # persisted with (storage-relative, via
                # ``config.to_storage_path``) so a client looking the row
                # up by ``generation_id`` finds the same path in both
                # places rather than an absolute filesystem path here and
                # a storage-relative one in History.
                audio_path=config.to_storage_path(final_path),
            ).model_dump()
        )
        yield _sse_done()

    except Exception as exc:
        traceback.print_exc()
        yield _sse_data(
            models.SpeakStreamError(
                generation_id=generation_id,
                message=str(exc),
            ).model_dump()
        )
        yield _sse_done()


async def _as_single_chunk(text: str) -> AsyncIterator[str]:
    """Adapt a materialised string into a one-element async iterator.

    Feeds the plain and non-streaming-personality paths into
    ``stream_sentences`` so long inputs still get sentence-level chunking
    instead of collapsing into one oversized TTS call.
    """
    cleaned = text.strip()
    if cleaned:
        yield cleaned


def _sse_data(payload: dict) -> str:
    """Format a dict as one SSE ``data:`` frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    """Terminal OpenAI-style sentinel."""
    return "data: [DONE]\n\n"
