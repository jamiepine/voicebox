"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from typing import Optional
from .backends import get_stt_backend, get_pytorch_stt_backend, STTBackend


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).

    Returns:
        STT backend instance
    """
    return get_stt_backend()


def get_pytorch_whisper_model() -> STTBackend:
    """
    Get PyTorch STT backend instance.

    Used for models that mlx_audio cannot load (e.g. ivrit-ai models).
    On non-MLX platforms this returns the same instance as get_whisper_model().

    Returns:
        PyTorch STT backend instance
    """
    return get_pytorch_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()


def unload_pytorch_whisper_model():
    """Unload PyTorch Whisper model to free memory."""
    backend = get_pytorch_stt_backend()
    backend.unload_model()
