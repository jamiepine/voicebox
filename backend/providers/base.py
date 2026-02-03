"""
Base protocol for TTS providers.
"""

from typing import Protocol, Optional, Tuple
from typing_extensions import runtime_checkable
import numpy as np

from .types import ProviderHealth, ProviderStatus


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for TTS provider implementations."""
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from text.
        
        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary
            language: Language code
            seed: Random seed for reproducibility
            instruct: Delivery instructions
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ...
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.
        
        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of the audio
            use_cache: Whether to use cached prompts
            
        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        ...
    
    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.
        
        Args:
            audio_paths: List of audio file paths
            reference_texts: List of reference texts
            
        Returns:
            Tuple of (combined_audio_array, combined_text)
        """
        ...
    
    async def load_model_async(self, model_size: str) -> None:
        """Load TTS model."""
        ...
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
    
    def _get_model_path(self, model_size: str) -> str:
        """Get model path for a given size."""
        ...
    
    async def health(self) -> ProviderHealth:
        """Get provider health status."""
        ...
    
    async def status(self) -> ProviderStatus:
        """Get provider model status."""
        ...
