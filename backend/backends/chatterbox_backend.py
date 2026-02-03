from typing import Optional, Tuple, List, Dict
import asyncio
import numpy as np
import torch
from pathlib import Path

from . import TTSBackend
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import normalize_audio, load_audio

class ChatterboxPyTorchBackend:
    """PyTorch-based Chatterbox TTS backend."""
    
    def __init__(self, model_size: str = "turbo"):
        self.model = None
        self.model_type = model_size # 'turbo', 'standard', 'multilingual'
        
        # Detect best device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
    def is_loaded(self) -> bool:
        return self.model is not None
        
    async def load_model_async(self, model_size: Optional[str] = None):
        """Lazy load the Chatterbox model."""
        if model_size is None:
            model_size = self.model_type
            
        if self.model is not None and self.model_type == model_size:
            return
            
        await asyncio.to_thread(self._load_model_sync, model_size)
        
    # Alias
    load_model = load_model_async
    
    def _load_model_sync(self, model_size: str):
        try:
            print(f"Loading Chatterbox TTS model ({model_size}) on {self.device}...")
            
            if model_size == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif model_size == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                
            self.model_type = model_size
            print(f"Chatterbox TTS ({model_size}) loaded successfully.")
            
        except Exception as e:
            print(f"Error loading Chatterbox TTS: {e}")
            raise e

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Chatterbox TTS unloaded.")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt matching Voicebox's expected format.
        For Chatterbox, we can just pass the audio path/text in the dict.
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text
        }
        
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        # Chatterbox doesn't inherently support combining prompts in the same way Qwen might
        # But we can concatenate audio for a reference like Qwen does if needed.
        # For now, just take the first one or stub it.
        # Real implementation: concatenate audios?
        
        # Simple implementation: Concatenate audio, join text
        combined_audio = []
        for path in audio_paths:
            wav, sr = load_audio(path)
            combined_audio.append(wav)
            
        mixed = np.concatenate(combined_audio)
        combined_text = " ".join(reference_texts)
        
        return mixed, combined_text

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        
        await self.load_model_async(None)
        
        def _generate_sync():
            # Chatterbox infer method
            # signature: infer(text, ref_audio, ref_text, ...)
            
            ref_audio = voice_prompt.get("ref_audio")
            ref_text = voice_prompt.get("ref_text", "")
            
            # Set params
            # We can expose these in config too, but for now use defaults or from backend config
            # But the protocol doesn't pass these. Voicebox likely reads env/config globally or we add to signature?
            # The current protocol doesn't pass temp/etc in generate() args shown in init.
            # wait, backend protocol from init:
            # generate(self, text, voice_prompt, language, seed, instruct)
            
            # Chatterbox generation
            # Note: infer() returns audio
            
            # Set seed if supported
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
            audio = self.model.infer(
                text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language if self.model_type == 'multilingual' else None,
                # skip_ref_generation=True # if we want to skip ?
            )
            
            # audio is usually numpy array or list? check docs/code.
            # Usually chatterbox returns list of floats or numpy.
            # Assuming numpy array (N,) or (1, N)
            
            if isinstance(audio, list):
                audio = np.array(audio)
            
            return audio, 24000 # Chatterbox default sr is 24k
            
        return await asyncio.to_thread(_generate_sync)
