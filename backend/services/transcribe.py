"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from typing import Optional
from ..backends import get_stt_backend, STTBackend
from ..backends import fourier_stt

_fourier = fourier_stt.FourierSTTBackend()


def get_transcription_model(model_size: Optional[str] = None) -> STTBackend:
    return _fourier if model_size == fourier_stt.MODEL else get_whisper_model()


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).
    
    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()
