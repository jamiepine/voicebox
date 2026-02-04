from typing import Optional, Tuple, List, Dict
import asyncio
import numpy as np
import torch
from pathlib import Path

from . import TTSBackend
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import normalize_audio, load_audio, save_audio
import types

def patch_chatterbox():
    """
    Monkey-patch chatterbox to force float32 on MPS.
    MPS doesn't support float64, and chatterbox sometimes defaults to it.
    """
    try:
        # Patch torch.Tensor.to globally to catch all float64 -> MPS moves
        original_to = torch.Tensor.to
        
        def patched_to(self, *args, **kwargs):
            device = None
            dtype = None
            
            # Extract device and dtype from args/kwargs
            if args:
                if isinstance(args[0], (torch.device, str)):
                    device = args[0]
                elif isinstance(args[0], torch.dtype):
                    dtype = args[0]
                
                if len(args) > 1 and isinstance(args[1], torch.dtype):
                    dtype = args[1]
            
            device = kwargs.get('device', device)
            dtype = kwargs.get('dtype', dtype)
            
            # Trigger: moving to MPS or already on MPS with float64
            is_mps = False
            if device:
                is_mps = (isinstance(device, str) and 'mps' in device) or \
                         (isinstance(device, torch.device) and device.type == 'mps')
            else:
                is_mps = (self.device.type == 'mps')
                
            if is_mps:
                if dtype == torch.float64:
                    kwargs['dtype'] = torch.float32
                elif dtype is None and self.dtype == torch.float64:
                    # If no dtype specified but we are float64, force float32
                    kwargs['dtype'] = torch.float32
            
            return original_to(self, *args, **kwargs)
            
        torch.Tensor.to = patched_to
        print("Chatterbox: Global torch.Tensor.to monkey-patch applied.")

    except Exception as e:
        print(f"Warning: Failed to monkey-patch chatterbox: {e}")

    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to monkey-patch chatterbox: {e}")

# Apply patch immediately
patch_chatterbox()

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
    
    def _get_model_path(self, model_size: str) -> str:
        """
        Get the HuggingFace Hub model ID for Chatterbox models.
        
        Args:
            model_size: Model size (turbo, standard, multilingual)
            
        Returns:
            HuggingFace Hub model ID
        """
        # Chatterbox models are hosted on HuggingFace
        hf_model_map = {
            "turbo": "ResembleAI/chatterbox-turbo",
            "standard": "ResembleAI/chatterbox",
            "multilingual": "ResembleAI/chatterbox-multilingual",
        }
        
        return hf_model_map.get(model_size, "ResembleAI/chatterbox-turbo")
    
    def _is_model_cached(self, model_size: str) -> bool:
        """
        Check if Chatterbox model is cached in HuggingFace Hub.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model files are found in cache
        """
        model_path = self._get_model_path(model_size)
        
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            
            # Find the repo in cache
            for repo in cache_info.repos:
                if repo.repo_id == model_path:
                    # Check if there are any revisions with files
                    if repo.revisions:
                        # For extra safety, check for .incomplete files
                        for revision in repo.revisions:
                            for file in revision.files:
                                if file.file_name.endswith(".incomplete"):
                                    return False
                                # Chatterbox uses .safetensors or .bin
                                if file.file_name.endswith(".safetensors") or file.file_name.endswith(".bin"):
                                    return True
            return False
        except Exception:
            return False
        
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
            
        mixed = np.concatenate(combined_audio).astype(np.float32)
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
            # Chatterbox generate method
            # signatures:
            # Turbo: (text, repetition_penalty=1.2, min_p=0.0, top_p=0.95, audio_prompt_path=None, exaggeration=0.0, cfg_weight=0.0, temperature=0.8, top_k=1000, norm_loudness=True)
            # Standard: (text, repetition_penalty=1.2, min_p=0.05, top_p=1.0, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
            # Multilingual: (text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=2.0, min_p=0.05, top_p=1.0)
            
            audio_prompt_path = voice_prompt.get("ref_audio")
            
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
            try:
                if self.model_type == 'multilingual':
                    # Map language code if needed, but chatterbox-multilingual uses specific IDs
                    # For now, pass language directly as language_id
                    audio = self.model.generate(
                        text,
                        language_id=language,
                        audio_prompt_path=audio_prompt_path,
                    )
                else:
                    audio = self.model.generate(
                        text,
                        audio_prompt_path=audio_prompt_path,
                    )
            except Exception as e:
                import traceback
                print("--- Chatterbox Generation Error ---")
                traceback.print_exc()
                print("-----------------------------------")
                raise e
            
            if isinstance(audio, list):
                audio = np.array(audio).astype(np.float32)
            elif isinstance(audio, np.ndarray):
                audio = audio.astype(np.float32)
            
            return audio, 24000 # Chatterbox default sr is 24k
            
        return await asyncio.to_thread(_generate_sync)
