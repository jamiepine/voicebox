"""Hybrid multi-engine generation: routes paralinguistic tags to Chatterbox Turbo."""

import gc
import logging
from typing import Optional, Tuple

import numpy as np

from .tag_router import parse_tagged_text
from .chunked_tts import generate_chunked, concatenate_audio_chunks

logger = logging.getLogger(__name__)


async def generate_hybrid(
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: Optional[int] = None,
    instruct: Optional[str] = None,
    sampling_params: Optional[dict] = None,
    max_chunk_chars: int = 800,
    crossfade_ms: int = 50,
    jitter_ms: int = 0,
    primary_engine: str = "qwen",
    primary_model_size: str = "1.7B",
) -> Tuple[np.ndarray, int]:
    """
    Generate audio with hybrid engine routing.

    Text segments go to the primary engine (Qwen).
    Paralinguistic tag segments go to Chatterbox Turbo.
    Results are merged in order with crossfade.
    """
    from ..backends import (
        load_engine_model,
        get_tts_backend_for_engine,
    )

    segments = parse_tagged_text(text)

    # Separate segments by engine
    text_segments = [(i, seg) for i, seg in enumerate(segments) if seg.type == "text"]
    tag_segments = [(i, seg) for i, seg in enumerate(segments) if seg.type == "tag"]

    if not tag_segments:
        # No tags — fast path, use primary engine only
        backend = get_tts_backend_for_engine(primary_engine)
        return await generate_chunked(
            backend, text, voice_prompt, language, seed, instruct,
            max_chunk_chars=max_chunk_chars, crossfade_ms=crossfade_ms,
            sampling_params=sampling_params, jitter_ms=jitter_ms,
        )

    logger.info(
        "Hybrid generation: %d text segments, %d tag segments",
        len(text_segments),
        len(tag_segments),
    )

    # Results array indexed by original segment position
    audio_results: dict[int, np.ndarray] = {}
    sample_rate = 24000

    # Step 1: Generate all text segments with primary engine (Qwen)
    if text_segments:
        await load_engine_model(primary_engine, primary_model_size)
        backend = get_tts_backend_for_engine(primary_engine)

        for idx, seg in text_segments:
            chunk_seed = (seed + idx) if seed is not None else None
            audio, sr = await backend.generate(
                seg.content, voice_prompt, language, chunk_seed, instruct,
                sampling_params,
            )
            audio_results[idx] = np.asarray(audio, dtype=np.float32)
            sample_rate = sr
            logger.info("Generated text segment %d: %.2fs", idx, len(audio) / sr)

    # Step 2: Unload primary engine, load Chatterbox Turbo for tag segments
    if tag_segments:
        try:
            primary_backend = get_tts_backend_for_engine(primary_engine)
            if hasattr(primary_backend, "unload_model"):
                primary_backend.unload_model()
                logger.info(
                    "Unloaded %s to free memory for Chatterbox Turbo", primary_engine
                )
        except Exception:
            pass

        gc.collect()

        await load_engine_model("chatterbox_turbo")
        cb_backend = get_tts_backend_for_engine("chatterbox_turbo")

        for idx, seg in tag_segments:
            chunk_seed = (seed + idx) if seed is not None else None
            audio, sr = await cb_backend.generate(
                seg.content, voice_prompt, "en", chunk_seed,
            )
            audio_results[idx] = np.asarray(audio, dtype=np.float32)
            logger.info(
                "Generated tag segment %d ('%s'): %.2fs",
                idx,
                seg.content,
                len(audio) / sr,
            )

        # Unload Chatterbox after use
        try:
            cb_backend.unload_model()
            logger.info("Unloaded Chatterbox Turbo")
        except Exception:
            pass
        gc.collect()

    # Step 3: Reassemble in original order
    ordered_chunks = [audio_results[i] for i in sorted(audio_results.keys())]

    if len(ordered_chunks) == 1:
        return ordered_chunks[0], sample_rate

    combined = concatenate_audio_chunks(ordered_chunks, sample_rate, crossfade_ms, jitter_ms)
    return combined, sample_rate
