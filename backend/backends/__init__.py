"""
Backend abstraction layer for TTS and STT.

Provides a unified interface for MLX and PyTorch backends.
"""

import threading
from typing import Protocol, Optional, Tuple, List
from typing_extensions import runtime_checkable
import numpy as np

from ..platform_detect import get_backend_type


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol for TTS backend implementations."""
    
    async def load_model(self, model_size: str) -> None:
        """Load TTS model."""
        ...
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.
        
        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        ...
    
    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.
        
        Returns:
            Tuple of (combined_audio_array, combined_text)
        """
        ...
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ...
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
    
    def _get_model_path(self, model_size: str) -> str:
        """
        Get model path for a given size.
        
        Returns:
            Model path or HuggingFace Hub ID
        """
        ...


@runtime_checkable
class STTBackend(Protocol):
    """Protocol for STT (Speech-to-Text) backend implementations."""
    
    async def load_model(self, model_size: str) -> None:
        """Load STT model."""
        ...
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Returns:
            Transcribed text
        """
        ...
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


# Global backend instances
_tts_backend: Optional[TTSBackend] = None
_tts_backends: dict[str, TTSBackend] = {}
_tts_backends_lock = threading.Lock()
_stt_backend: Optional[STTBackend] = None

# Supported TTS engines
TTS_ENGINES = {
    "qwen": "Qwen TTS",
    "luxtts": "LuxTTS",
    "chatterbox": "Chatterbox TTS",
    "chatterbox_turbo": "Chatterbox Turbo",
}


def get_tts_backend() -> TTSBackend:
    """
    Get or create the default (Qwen) TTS backend instance based on platform.
    
    Returns:
        TTS backend instance (MLX or PyTorch)
    """
    return get_tts_backend_for_engine("qwen")


def get_tts_backend_for_engine(engine: str) -> TTSBackend:
    """
    Get or create a TTS backend for the given engine.
    
    Args:
        engine: Engine name ("qwen" or "luxtts")
    
    Returns:
        TTS backend instance
    """
    global _tts_backends
    
    # Fast path: check without lock
    if engine in _tts_backends:
        return _tts_backends[engine]
    
    # Slow path: create with lock to avoid duplicate instantiation
    with _tts_backends_lock:
        # Double-check after acquiring lock
        if engine in _tts_backends:
            return _tts_backends[engine]
        
        if engine == "qwen":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .mlx_backend import MLXTTSBackend
                backend = MLXTTSBackend()
            else:
                from .pytorch_backend import PyTorchTTSBackend
                backend = PyTorchTTSBackend()
        elif engine == "luxtts":
            from .luxtts_backend import LuxTTSBackend
            backend = LuxTTSBackend()
        elif engine == "chatterbox":
            from .chatterbox_backend import ChatterboxTTSBackend
            backend = ChatterboxTTSBackend()
        elif engine == "chatterbox_turbo":
            from .chatterbox_turbo_backend import ChatterboxTurboTTSBackend
            backend = ChatterboxTurboTTSBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES.keys())}")
        
        _tts_backends[engine] = backend
        return backend


def get_stt_backend() -> STTBackend:
    """
    Get or create STT backend instance based on platform.
    
    Returns:
        STT backend instance (MLX or PyTorch)
    """
    global _stt_backend
    
    if _stt_backend is None:
        backend_type = get_backend_type()
        
        if backend_type == "mlx":
            from .mlx_backend import MLXSTTBackend
            _stt_backend = MLXSTTBackend()
        else:
            from .pytorch_backend import PyTorchSTTBackend
            _stt_backend = PyTorchSTTBackend()
    
    return _stt_backend


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _tts_backend, _tts_backends, _stt_backend
    _tts_backend = None
    _tts_backends.clear()
    _stt_backend = None
