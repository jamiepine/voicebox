"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from ..backends import STTBackend, get_stt_backend

if TYPE_CHECKING:
    from ..backends.orukeet_backend import OrukeetSTTBackend


@lru_cache(maxsize=1)
def get_orukeet_model() -> "OrukeetSTTBackend":
    """Get the optional local Orukeet transcription backend."""
    from ..backends.orukeet_backend import OrukeetSTTBackend

    return OrukeetSTTBackend()


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
