"""PCM helpers for the voice pipeline.

Discord delivers 20 ms frames of 48 kHz, 16-bit, stereo PCM. Whisper wants a
mono file. Everything here is numpy rather than `audioop`, which was removed
from the standard library in Python 3.13.

No resampling happens on this side: a 48 kHz mono WAV is handed to Voicebox
and librosa resamples it during load. Downsampling here would only add a
second lossy step.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo 16-bit
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH


def to_mono(pcm: bytes) -> np.ndarray:
    """Interleaved stereo int16 bytes -> mono int16 samples."""
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return samples
    if samples.size % CHANNELS:
        samples = samples[: samples.size - (samples.size % CHANNELS)]
    stereo = samples.reshape(-1, CHANNELS).astype(np.int32)
    return stereo.mean(axis=1).astype(np.int16)


def rms(pcm: bytes) -> float:
    """Normalised loudness of a PCM buffer, 0.0-1.0."""
    mono = to_mono(pcm)
    if mono.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) / 32768.0)


def pcm_to_wav(pcm: bytes, *, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap PCM as a mono 16-bit WAV file."""
    mono = to_mono(pcm)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(mono.tobytes())
    return buffer.getvalue()


def duration_seconds(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SECOND


@dataclass
class SpeakerBuffer:
    """Accumulates one speaker's frames until their utterance ends."""

    user_id: int
    chunks: list[bytes] = field(default_factory=list)
    total_bytes: int = 0
    last_frame_at: float = 0.0
    started_at: float = 0.0
    loud_bytes: int = 0

    def add(self, pcm: bytes, now: float, *, silence_floor: float = 0.004) -> None:
        if not self.chunks:
            self.started_at = now
        self.chunks.append(pcm)
        self.total_bytes += len(pcm)
        self.last_frame_at = now
        if rms(pcm) >= silence_floor:
            self.loud_bytes += len(pcm)

    @property
    def duration(self) -> float:
        return self.total_bytes / BYTES_PER_SECOND

    @property
    def speech_ratio(self) -> float:
        """Fraction of the buffer that was above the silence floor.

        A buffer that is mostly background hiss isn't worth a Whisper call —
        it comes back as a hallucinated "Thank you." or "Bye." more often
        than as anything real.
        """
        if not self.total_bytes:
            return 0.0
        return self.loud_bytes / self.total_bytes

    def drain(self) -> bytes:
        pcm = b"".join(self.chunks)
        self.chunks.clear()
        self.total_bytes = 0
        self.loud_bytes = 0
        self.started_at = 0.0
        return pcm
