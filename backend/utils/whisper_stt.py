"""
Long-form Whisper STT helpers.

Whisper models accept ~30 seconds of audio per forward pass. Callers must
chunk longer inputs and stitch the text back together.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

WHISPER_SAMPLE_RATE = 16_000
WHISPER_CHUNK_SECONDS = 30
WHISPER_STRIDE_SECONDS = 5
WHISPER_CHUNK_SAMPLES = WHISPER_CHUNK_SECONDS * WHISPER_SAMPLE_RATE
WHISPER_STRIDE_SAMPLES = WHISPER_STRIDE_SECONDS * WHISPER_SAMPLE_RATE
# Ignore trailing fragments shorter than half a second.
MIN_CHUNK_SAMPLES = WHISPER_SAMPLE_RATE // 2


def iter_whisper_chunks(audio: np.ndarray) -> Iterator[np.ndarray]:
    """
    Yield 30-second windows with 5-second overlap for audio longer than one chunk.

    Short audio (<= 30 s) is yielded as a single chunk unchanged.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n <= WHISPER_CHUNK_SAMPLES:
        yield audio
        return

    step = WHISPER_CHUNK_SAMPLES - WHISPER_STRIDE_SAMPLES
    start = 0
    while start < n:
        end = min(start + WHISPER_CHUNK_SAMPLES, n)
        chunk = audio[start:end]
        if len(chunk) < MIN_CHUNK_SAMPLES and start > 0:
            break
        yield chunk
        if end >= n:
            break
        start += step


def join_whisper_chunk_texts(texts: list[str]) -> str:
    """Join per-chunk transcripts with a single space."""
    return " ".join(t.strip() for t in texts if t and t.strip())


def transcribe_whisper_pytorch(
    audio: np.ndarray,
    *,
    processor,
    model,
    device: str,
    language: Optional[str] = None,
) -> str:
    """Run PyTorch Whisper on arbitrary-length audio via overlapping chunks."""
    import torch

    texts: list[str] = []
    for chunk in iter_whisper_chunks(audio):
        inputs = processor(
            chunk,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generate_kwargs = {}
        if language:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
            )
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                **generate_kwargs,
            )

        text = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]
        if text:
            texts.append(text.strip())

    return join_whisper_chunk_texts(texts)
