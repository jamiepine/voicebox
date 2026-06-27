"""Unit tests for live audio streaming helpers."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio import audio_to_pcm16_bytes, streaming_wav_header
from utils.chunked_tts import generate_chunked_stream


class StreamingBackend:
    async def generate_stream(
        self,
        text,
        voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
        yield np.array([0.0, 0.5], dtype=np.float32), 24000
        yield np.array([-0.5, 1.0], dtype=np.float32), 24000

    async def generate(self, *args, **kwargs):
        raise AssertionError("streaming backend should not use buffered generate")


class BufferedBackend:
    async def generate(
        self,
        text,
        voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
        return np.array([0.25, -0.25], dtype=np.float32), 22050


@pytest.mark.asyncio
async def test_generate_chunked_stream_uses_backend_stream():
    chunks = []

    async for audio, sample_rate in generate_chunked_stream(
        StreamingBackend(),
        "hello",
        {},
    ):
        chunks.append((audio, sample_rate))

    assert len(chunks) == 2
    assert chunks[0][1] == 24000
    assert np.allclose(chunks[0][0], [0.0, 0.5])
    assert np.allclose(chunks[1][0], [-0.5, 1.0])


@pytest.mark.asyncio
async def test_generate_chunked_stream_falls_back_to_buffered_generate():
    chunks = []

    async for audio, sample_rate in generate_chunked_stream(
        BufferedBackend(),
        "hello",
        {},
    ):
        chunks.append((audio, sample_rate))

    assert len(chunks) == 1
    assert chunks[0][1] == 22050
    assert np.allclose(chunks[0][0], [0.25, -0.25])


def test_streaming_wav_header_uses_unknown_lengths():
    header = streaming_wav_header(24000)

    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"
    assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", header[24:28])[0] == 24000


def test_audio_to_pcm16_bytes_clips_and_packs_little_endian():
    pcm = audio_to_pcm16_bytes(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))

    assert pcm == struct.pack("<hhhhh", -32767, -32767, 0, 32767, 32767)
