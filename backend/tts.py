"""
TTS inference module - delegates to backend abstraction layer.
"""

from typing import Optional
import numpy as np
import io
import soundfile as sf

from .backends import get_tts_backend, get_chatterbox_backend, TTSBackend, get_pytorch_tts_backend


def get_tts_model() -> TTSBackend:
    """
    Get TTS backend instance (MLX or PyTorch based on platform).

    Returns:
        TTS backend instance
    """
    return get_tts_backend()


def unload_tts_model():
    """Unload TTS model to free memory.

    Also unloads the PyTorch adapter backend if it was loaded separately
    (on Apple Silicon where MLX is the default, a separate PyTorch backend
    is used for LoRA adapter inference).
    """
    backend = get_tts_backend()
    backend.unload_model()

    # Also unload separate PyTorch backend if it exists
    from .backends import _pytorch_tts_backend
    if _pytorch_tts_backend is not None and _pytorch_tts_backend is not backend:
        _pytorch_tts_backend.unload_model()


def get_chatterbox_model():
    """Get Chatterbox TTS backend instance (for Hebrew)."""
    return get_chatterbox_backend()


def unload_chatterbox_model():
    """Unload Chatterbox TTS model to free memory."""
    backend = get_chatterbox_backend()
    backend.unload_model()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
