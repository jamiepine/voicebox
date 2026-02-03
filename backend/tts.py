"""
TTS inference module - delegates to provider abstraction layer.
"""

from typing import Optional
import numpy as np
import io
import soundfile as sf

from .backends import TTSBackend
from .providers import get_provider_manager
from .providers.base import TTSProvider


def get_tts_model() -> TTSProvider:
    """
    Get TTS provider instance (via ProviderManager).
    
    Returns:
        TTS provider instance
    """
    manager = get_provider_manager()
    # Note: This is async but we need sync interface for backward compatibility
    # In practice, this will be called from async contexts
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, but can't await here
            # Return a wrapper that will use the provider manager
            return manager._get_default_provider()
        else:
            return loop.run_until_complete(manager.get_active_provider())
    except RuntimeError:
        # No event loop, return default
        return manager._get_default_provider()


async def get_tts_model_async() -> TTSProvider:
    """
    Get TTS provider instance asynchronously.
    
    Returns:
        TTS provider instance
    """
    manager = get_provider_manager()
    return await manager.get_active_provider()


def unload_tts_model():
    """Unload TTS model to free memory."""
    manager = get_provider_manager()
    provider = manager._get_default_provider()
    provider.unload_model()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
