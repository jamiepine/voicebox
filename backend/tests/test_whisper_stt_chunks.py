"""Tests for long-form Whisper chunking."""

import numpy as np

from backend.utils.whisper_stt import (
    WHISPER_CHUNK_SAMPLES,
    WHISPER_STRIDE_SAMPLES,
    iter_whisper_chunks,
    join_whisper_chunk_texts,
)


def test_iter_whisper_chunks_short_audio_single_window():
    audio = np.zeros(16000 * 10, dtype=np.float32)
    chunks = list(iter_whisper_chunks(audio))
    assert len(chunks) == 1
    assert len(chunks[0]) == len(audio)


def test_iter_whisper_chunks_long_audio_multiple_windows():
    audio = np.zeros(16000 * 65, dtype=np.float32)
    chunks = list(iter_whisper_chunks(audio))
    assert len(chunks) == 3
    assert all(len(c) <= WHISPER_CHUNK_SAMPLES for c in chunks)
    assert len(chunks[0]) == WHISPER_CHUNK_SAMPLES
    # Last chunk covers the tail after overlapping windows.
    assert sum(len(c) for c in chunks) > len(audio)


def test_join_whisper_chunk_texts_skips_empty():
    assert join_whisper_chunk_texts(["hello", "", "  ", "world"]) == "hello world"
