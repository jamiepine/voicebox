"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.

Supports both Whisper and NVIDIA Parakeet via a single backend instance that
branches internally on the model family.
"""

from typing import Optional
from ..backends import get_stt_backend, STTBackend


def get_stt_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).

    The backend supports both Whisper and Parakeet; the model family is chosen
    by the ``model_name`` passed to ``load_model`` / ``transcribe``.

    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_stt_model():
    """Unload the currently-loaded STT model (Whisper or Parakeet)."""
    backend = get_stt_backend()
    backend.unload_model()


# Legacy aliases — kept so the dispatch helpers in backends/__init__.py and
# any external callers continue to work after the Parakeet additions.
def get_whisper_model() -> STTBackend:
    """Deprecated alias for :func:`get_stt_model`."""
    return get_stt_model()


def unload_whisper_model():
    """Deprecated alias for :func:`unload_stt_model`."""
    unload_stt_model()
