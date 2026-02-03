"""
Bundled provider that wraps existing MLX/PyTorch backends.
"""

from typing import Optional, Tuple
import numpy as np
import platform

from .base import TTSProvider
from .types import ProviderHealth, ProviderStatus
from ..backends import get_tts_backend, TTSBackend
from ..platform_detect import get_backend_type


class BundledProvider:
    """Provider that wraps the existing bundled TTS backend."""
    
    def __init__(self):
        self._backend: Optional[TTSBackend] = None
    
    def _get_backend(self) -> TTSBackend:
        """Get or create backend instance."""
        if self._backend is None:
            self._backend = get_tts_backend()
        return self._backend
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech audio."""
        backend = self._get_backend()
        return await backend.generate(text, voice_prompt, language, seed, instruct)
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Create voice prompt from reference audio."""
        backend = self._get_backend()
        return await backend.create_voice_prompt(audio_path, reference_text, use_cache)
    
    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> Tuple[np.ndarray, str]:
        """Combine multiple voice prompts."""
        backend = self._get_backend()
        return await backend.combine_voice_prompts(audio_paths, reference_texts)
    
    async def load_model_async(self, model_size: str) -> None:
        """Load TTS model."""
        backend = self._get_backend()
        if hasattr(backend, 'load_model_async'):
            await backend.load_model_async(model_size)
        else:
            await backend.load_model(model_size)

    # Alias for compatibility
    load_model = load_model_async
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        backend = self._get_backend()
        backend.unload_model()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        backend = self._get_backend()
        return backend.is_loaded()
    
    def _get_model_path(self, model_size: str) -> str:
        """Get model path for a given size."""
        backend = self._get_backend()
        return backend._get_model_path(model_size)
    
    async def health(self) -> ProviderHealth:
        """Get provider health status."""
        backend = self._get_backend()
        backend_type = get_backend_type()

        model_size = None
        if backend.is_loaded():
            # Try to get current model size from backend
            if hasattr(backend, '_current_model_size') and backend._current_model_size:
                model_size = backend._current_model_size

        device = None
        if backend_type == "mlx":
            device = "metal"
        elif hasattr(backend, 'device'):
            device = backend.device

        # Use apple-mlx for MLX backend, pytorch-cpu for PyTorch
        provider_name = "apple-mlx" if backend_type == "mlx" else "pytorch-cpu"

        return ProviderHealth(
            status="healthy",
            provider=provider_name,
            version=None,  # Provider versioning not implemented yet
            model=model_size,
            device=device,
        )
    
    async def status(self) -> ProviderStatus:
        """Get provider model status."""
        backend = self._get_backend()
        backend_type = get_backend_type()
        
        model_size = None
        if backend.is_loaded():
            if hasattr(backend, '_current_model_size') and backend._current_model_size:
                model_size = backend._current_model_size
        
        available_sizes = ["1.7B"]
        if backend_type == "pytorch":
            available_sizes.append("0.6B")
        
        gpu_available = None
        vram_used_mb = None
        
        if backend_type == "pytorch":
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    vram_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
        
        return ProviderStatus(
            model_loaded=backend.is_loaded(),
            model_size=model_size,
            available_sizes=available_sizes,
            gpu_available=gpu_available,
            vram_used_mb=int(vram_used_mb) if vram_used_mb else None,
        )
