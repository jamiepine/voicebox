"""
Backend abstraction layer for TTS and STT.

Provides a unified interface for MLX and PyTorch backends.
"""

from typing import Protocol, Optional, Tuple, List
from typing_extensions import runtime_checkable
import numpy as np

from ..platform_detect import get_backend_type

# Shared model name mapping for STT backends (MLX + PyTorch).
# Maps short model size keys to HuggingFace repo IDs.
STT_MODEL_MAP = {
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "ivrit-v3": "ivrit-ai/whisper-large-v3",
    "ivrit-v3-turbo": "ivrit-ai/whisper-large-v3-turbo",
}


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

    async def generate_with_adapter(
        self,
        text: str,
        voice_prompt: dict,
        adapter_path: str,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using a LoRA adapter.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary
            adapter_path: Path to LoRA adapter directory
            language: Language code
            seed: Random seed
            instruct: Natural language instruction

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
_stt_backend: Optional[STTBackend] = None
_chatterbox_backend = None  # Optional[ChatterboxTTSBackend]
_pytorch_tts_backend: Optional[TTSBackend] = None  # For adapter-based generation


def get_tts_backend() -> TTSBackend:
    """
    Get or create TTS backend instance based on platform.
    
    Returns:
        TTS backend instance (MLX or PyTorch)
    """
    global _tts_backend
    
    if _tts_backend is None:
        backend_type = get_backend_type()
        
        if backend_type == "mlx":
            from .mlx_backend import MLXTTSBackend
            _tts_backend = MLXTTSBackend()
        else:
            from .pytorch_backend import PyTorchTTSBackend
            _tts_backend = PyTorchTTSBackend()
    
    return _tts_backend


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


def get_pytorch_tts_backend() -> TTSBackend:
    """
    Get or create a PyTorch TTS backend instance.

    Used specifically for adapter-based generation, since LoRA adapters
    are trained with PyTorch+PEFT and must be loaded in PyTorch.
    On Apple Silicon, the default backend is MLX which doesn't support
    PEFT adapters, so we maintain a separate PyTorch backend.
    """
    global _pytorch_tts_backend

    if _pytorch_tts_backend is None:
        # Check if the default backend is already PyTorch — reuse it
        backend_type = get_backend_type()
        if backend_type != "mlx":
            _pytorch_tts_backend = get_tts_backend()
        else:
            from .pytorch_backend import PyTorchTTSBackend
            _pytorch_tts_backend = PyTorchTTSBackend()

    return _pytorch_tts_backend


def get_chatterbox_backend():
    """
    Get or create Chatterbox TTS backend instance (for Hebrew).

    Returns:
        ChatterboxTTSBackend instance
    """
    global _chatterbox_backend

    if _chatterbox_backend is None:
        from .chatterbox_backend import ChatterboxTTSBackend
        _chatterbox_backend = ChatterboxTTSBackend()

    return _chatterbox_backend


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _tts_backend, _stt_backend, _chatterbox_backend, _pytorch_tts_backend
    _tts_backend = None
    _stt_backend = None
    _chatterbox_backend = None
    _pytorch_tts_backend = None
