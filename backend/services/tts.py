"""
TTS inference module - delegates to backend abstraction layer.
"""

import io

import numpy as np
import soundfile as sf

from ..backends import TTSBackend, get_tts_backend


def get_tts_model() -> TTSBackend:
    """
    Get TTS backend instance (MLX or PyTorch based on platform).

    Returns:
        TTS backend instance
    """
    return get_tts_backend()


def unload_tts_model():
    """Unload TTS model to free memory."""
    backend = get_tts_backend()
    backend.unload_model()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
