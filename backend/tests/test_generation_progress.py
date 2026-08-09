"""
Tests for real-time audio generation progress tracking and SSE event streaming.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from utils.generation_progress import GenerationProgressManager, get_generation_progress_manager
from utils.chunked_tts import generate_chunked, split_text_into_chunks


def test_generation_progress_manager_basic():
    """Test GenerationProgressManager update, complete, and error lifecycle."""
    mgr = GenerationProgressManager()
    gen_id = "test-gen-123"

    mgr.update_progress(
        generation_id=gen_id,
        progress=25.0,
        current_chunk=2,
        total_chunks=8,
        status="generating",
        message="Sentence 2 of 8",
    )

    data = mgr.get_progress(gen_id)
    assert data is not None
    assert data["id"] == gen_id
    assert data["progress"] == 25.0
    assert data["current_chunk"] == 2
    assert data["total_chunks"] == 8
    assert data["status"] == "generating"
    assert data["message"] == "Sentence 2 of 8"

    mgr.mark_complete(gen_id)
    completed_data = mgr.get_progress(gen_id)
    assert completed_data["status"] == "completed"
    assert completed_data["progress"] == 100.0

    mgr.mark_error(gen_id, "Test error message")
    error_data = mgr.get_progress(gen_id)
    assert error_data["status"] == "failed"
    assert error_data["error"] == "Test error message"


@pytest.mark.asyncio
async def test_generation_progress_sse_subscription():
    """Test SSE streaming subscription for generation progress updates."""
    mgr = GenerationProgressManager()
    gen_id = "test-sse-456"
    received_events = []

    async def sse_client():
        async for event in mgr.subscribe(gen_id):
            if event.startswith("data: "):
                payload = json.loads(event[6:])
                received_events.append(payload)
                if payload.get("status") in ("completed", "failed"):
                    break

    async def producer():
        await asyncio.sleep(0.05)
        mgr.update_progress(gen_id, progress=10.0, current_chunk=1, total_chunks=3, status="generating")
        await asyncio.sleep(0.05)
        mgr.update_progress(gen_id, progress=50.0, current_chunk=2, total_chunks=3, status="generating")
        await asyncio.sleep(0.05)
        mgr.mark_complete(gen_id)

    await asyncio.gather(sse_client(), producer())

    assert len(received_events) >= 2
    assert received_events[-1]["status"] == "completed"
    assert received_events[-1]["progress"] == 100.0


@pytest.mark.asyncio
async def test_generate_chunked_progress_callback():
    """Test that generate_chunked invokes progress_callback for each text chunk."""
    mock_backend = MagicMock()
    import numpy as np
    mock_backend.generate = AsyncMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))

    calls = []

    def callback(current: int, total: int, text: str):
        calls.append((current, total, text))

    # Long text to force multiple chunks
    long_text = "First sentence here. " * 30 + "Second sentence here. " * 30 + "Third sentence here. " * 30

    chunks = split_text_into_chunks(long_text, max_chars=200)
    assert len(chunks) > 1

    audio, sr = await generate_chunked(
        backend=mock_backend,
        text=long_text,
        voice_prompt={},
        max_chunk_chars=200,
        progress_callback=callback,
    )

    assert sr == 16000
    assert len(audio) > 0
    assert len(calls) == len(chunks)
    for idx, call in enumerate(calls):
        assert call[0] == idx + 1
        assert call[1] == len(chunks)
        assert isinstance(call[2], str)


def test_split_text_into_chunks_sentences():
    """Test that multi-sentence text is split per sentence boundary into sentence chunks."""
    user_prompt = (
        "Voicebox is an open source AI voice generator. "
        "It supports real-time progress streaming. "
        "Sentence three introduces chunked audio generation. "
        "Sentence four processes each line sequentially. "
        "Sentence five shows sentence completion in real-time. "
        "Sentence six tests the new progress bar component. "
        "Sentence seven completes the multi-chunk audio output."
    )

    chunks = split_text_into_chunks(user_prompt)
    assert len(chunks) == 7
    assert chunks[0] == "Voicebox is an open source AI voice generator."
    assert chunks[6] == "Sentence seven completes the multi-chunk audio output."

