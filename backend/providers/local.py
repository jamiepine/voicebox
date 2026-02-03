"""
Local provider that communicates with standalone provider servers via HTTP.
"""

from typing import Optional, Tuple
import base64
import io
import numpy as np
import httpx
import soundfile as sf

from .base import TTSProvider
from .types import ProviderHealth, ProviderStatus


class LocalProvider:
    """Provider that communicates with local subprocess via HTTP."""
    
    def __init__(self, base_url: str):
        """
        Initialize local provider.
        
        Args:
            base_url: Base URL of the provider server (e.g., "http://localhost:8000")
        """
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for generation
        self._current_model_size = "1.7B"  # Default model size
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech audio."""
        response = await self.client.post(
            f"{self.base_url}/tts/generate",
            json={
                "text": text,
                "voice_prompt": voice_prompt,
                "language": language,
                "seed": seed,
                "model_size": self._current_model_size,
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(data["audio"])
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_buffer)
        
        return audio, data["sample_rate"]
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Create voice prompt from reference audio."""
        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # Send multipart form data
        files = {
            "audio": ("audio.wav", audio_data, "audio/wav")
        }
        data = {
            "reference_text": reference_text,
            "use_cache": str(use_cache).lower(),
        }
        
        response = await self.client.post(
            f"{self.base_url}/tts/create_voice_prompt",
            files=files,
            data=data,
        )
        response.raise_for_status()
        result = response.json()
        
        return result["voice_prompt"], result.get("was_cached", False)
    
    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.
        
        Note: This is not implemented in the provider API yet.
        For now, we'll combine locally by concatenating audio.
        """
        import numpy as np
        from ..utils.audio import load_audio, normalize_audio
        
        combined_audio = []
        for audio_path in audio_paths:
            audio, sr = load_audio(audio_path)
            audio = normalize_audio(audio)
            combined_audio.append(audio)
        
        # Concatenate audio
        mixed = np.concatenate(combined_audio)
        mixed = normalize_audio(mixed)
        
        # Combine texts
        combined_text = " ".join(reference_texts)
        
        return mixed, combined_text
    
    async def load_model_async(self, model_size: str) -> None:
        """Load TTS model."""
        # Track the requested model size - the provider server will load it
        # when generate() is called with this size
        self._current_model_size = model_size

    # Alias for compatibility
    load_model = load_model_async
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        # Model unloading is handled by the provider server
        # This is a no-op for local providers
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        # We can't know this without querying the provider
        # Return True optimistically
        return True
    
    def _get_model_path(self, model_size: str) -> str:
        """Get model path for a given size."""
        # For local providers, model paths are handled by the provider server
        # Return a placeholder
        return f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base"
    
    async def health(self) -> ProviderHealth:
        """Get provider health status."""
        try:
            response = await self.client.get(f"{self.base_url}/tts/health")
            response.raise_for_status()
            data = response.json()
            return ProviderHealth(
                status=data["status"],
                provider=data["provider"],
                version=data.get("version"),
                model=data.get("model"),
                device=data.get("device"),
            )
        except Exception as e:
            return ProviderHealth(
                status="unhealthy",
                provider="local",
                version=None,
                model=None,
                device=None,
            )
    
    async def status(self) -> ProviderStatus:
        """Get provider model status."""
        try:
            response = await self.client.get(f"{self.base_url}/tts/status")
            response.raise_for_status()
            data = response.json()
            return ProviderStatus(
                model_loaded=data["model_loaded"],
                model_size=data.get("model_size"),
                available_sizes=data.get("available_sizes", []),
                gpu_available=data.get("gpu_available"),
                vram_used_mb=data.get("vram_used_mb"),
            )
        except Exception as e:
            return ProviderStatus(
                model_loaded=False,
                model_size=None,
                available_sizes=[],
                gpu_available=None,
                vram_used_mb=None,
            )
    
    async def stop(self) -> None:
        """Stop the provider (close HTTP client)."""
        await self.client.aclose()
