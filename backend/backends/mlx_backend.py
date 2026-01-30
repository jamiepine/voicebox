"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

from typing import Optional, List, Tuple
import asyncio
import numpy as np
from pathlib import Path

from . import TTSBackend, STTBackend
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from ..utils.tasks import get_task_manager


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""
    
    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.
        
        Args:
            model_size: Model size (1.7B or 0.6B)
            
        Returns:
            HuggingFace Hub model ID for MLX
        """
        # MLX model mapping
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            # 0.6B not yet converted to MLX format
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",  # Fallback to 1.7B
        }
        
        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")
        
        hf_model_id = mlx_model_map[model_size]
        print(f"Will download MLX model from HuggingFace Hub: {hf_model_id}")
        
        return hf_model_id
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX TTS model.
        
        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size
            
        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return
        
        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()
        
        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)
    
    # Alias for compatibility
    load_model = load_model_async
    
    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        try:
            from mlx_audio.tts import load
            
            # Get model path
            model_path = self._get_model_path(model_size)
            
            # Set up progress tracking
            progress_manager = get_progress_manager()
            model_name = f"qwen-tts-{model_size}"
            
            # Start tracking download task
            task_manager = get_task_manager()
            task_manager.start_download(model_name)
            
            print(f"Loading MLX TTS model {model_size}...")
            
            # Initialize progress state
            progress_manager.update_progress(
                model_name=model_name,
                current=0,
                total=1,
                filename="",
                status="downloading",
            )
            
            # Set up progress callback
            progress_callback = create_hf_progress_callback(model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback)
            
            # Use progress tracker during download
            with tracker.patch_download():
                # Load MLX model (downloads automatically)
                self.model = load(model_path)
            
            self._current_model_size = model_size
            self.model_size = model_size
            
            # Mark as complete
            progress_manager.mark_complete(model_name)
            task_manager.complete_download(model_name)
            
            print(f"MLX TTS model {model_size} loaded successfully")
            
        except ImportError as e:
            print(f"Error: mlx_audio package not found. Install with: pip install mlx-audio")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = f"qwen-tts-{model_size}"
            progress_manager.mark_error(model_name, str(e))
            task_manager.error_download(model_name, str(e))
            raise
        except Exception as e:
            print(f"Error loading MLX TTS model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = f"qwen-tts-{model_size}"
            progress_manager.mark_error(model_name, str(e))
            task_manager.error_download(model_name, str(e))
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            print("MLX TTS model unloaded")
    
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.
        
        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.
        
        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available
            
        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)
        
        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None:
                # Return cached prompt (should be dict format)
                if isinstance(cached_prompt, dict):
                    return cached_prompt, True
        
        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        
        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)
        
        return voice_prompt_items, False
    
    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple reference samples for better quality.
        
        Args:
            audio_paths: List of audio file paths
            reference_texts: List of reference texts
            
        Returns:
            Tuple of (combined_audio, combined_text)
        """
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
    
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        print(f"Generating audio for text: {text}")

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            # MLX generate() returns a generator yielding GenerationResult objects
            audio_chunks = []
            sample_rate = 24000
            
            # Set seed if provided (MLX uses numpy random)
            if seed is not None:
                import mlx.core as mx
                np.random.seed(seed)
                mx.random.seed(seed)
            
            # Extract voice prompt info
            ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
            ref_text = voice_prompt.get("ref_text", "")
            
            # Check if model supports voice cloning via generate method
            # MLX API may support ref_audio parameter directly
            try:
                # Try with voice cloning parameters if supported
                if ref_audio:
                    # Check if generate accepts ref_audio parameter
                    import inspect
                    sig = inspect.signature(self.model.generate)
                    if "ref_audio" in sig.parameters:
                        # Generate with voice cloning
                        for result in self.model.generate(text, ref_audio=ref_audio, ref_text=ref_text):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                    else:
                        # Fallback: generate without voice cloning
                        for result in self.model.generate(text):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                else:
                    # No voice prompt, generate normally
                    for result in self.model.generate(text):
                        audio_chunks.append(np.array(result.audio))
                        sample_rate = result.sample_rate
            except Exception as e:
                # If voice cloning fails, try without it
                print(f"Warning: Voice cloning failed, generating without voice prompt: {e}")
                for result in self.model.generate(text):
                    audio_chunks.append(np.array(result.audio))
                    sample_rate = result.sample_rate
            
            # Concatenate all chunks
            if audio_chunks:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
            else:
                # Fallback: empty audio
                audio = np.array([], dtype=np.float32)
            
            return audio, sample_rate

        # Run blocking inference in thread pool
        audio, sample_rate = await asyncio.to_thread(_generate_sync)

        return audio, sample_rate


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size
        
        if self.model is not None and self.model_size == model_size:
            return
        
        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)
    
    # Alias for compatibility
    load_model = load_model_async
    
    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        try:
            # IMPORTANT: Set up progress tracking BEFORE importing mlx_audio
            # This ensures tqdm is patched before any HuggingFace Hub imports
            progress_manager = get_progress_manager()
            progress_model_name = f"whisper-{model_size}"

            # Set up progress callback and tracker
            progress_callback = create_hf_progress_callback(progress_model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback)

            # Patch tqdm BEFORE importing mlx_audio
            # This is critical because mlx_audio imports huggingface_hub which imports tqdm
            print("[DEBUG] Starting tqdm patch BEFORE mlx_audio import")
            tracker_context = tracker.patch_download()
            tracker_context.__enter__()
            print("[DEBUG] tqdm patched, now importing mlx_audio")

            # NOW import mlx_audio - it will use our patched tqdm
            from mlx_audio.stt import load

            # MLX Whisper uses the standard OpenAI models
            model_name = f"openai/whisper-{model_size}"
            
            # Start tracking download task
            task_manager = get_task_manager()
            task_manager.start_download(progress_model_name)

            print(f"Loading MLX Whisper model {model_size}...")

            # Initialize progress state
            progress_manager.update_progress(
                model_name=progress_model_name,
                current=0,
                total=1,
                filename="",
                status="downloading",
            )

            # Load the model (tqdm is already patched from above)
            try:
                self.model = load(model_name)
            finally:
                # Exit the patch context
                tracker_context.__exit__(None, None, None)
            
            self.model_size = model_size
            
            # Mark as complete
            progress_manager.mark_complete(progress_model_name)
            task_manager.complete_download(progress_model_name)
            
            print(f"MLX Whisper model {model_size} loaded successfully")
            
        except ImportError as e:
            print(f"Error: mlx_audio package not found. Install with: pip install mlx-audio")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"
            progress_manager.mark_error(progress_model_name, str(e))
            task_manager.error_download(progress_model_name, str(e))
            raise
        except Exception as e:
            print(f"Error loading MLX Whisper model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"
            progress_manager.mark_error(progress_model_name, str(e))
            task_manager.error_download(progress_model_name, str(e))
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            print("MLX Whisper model unloaded")
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language hint (en or zh)
            
        Returns:
            Transcribed text
        """
        await self.load_model_async(None)
        
        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # Load audio
            audio, sr = load_audio(audio_path, sample_rate=16000)
            
            # MLX Whisper transcription
            # The API may vary - check mlx-audio documentation
            # For now, assuming similar API to PyTorch Whisper
            result = self.model.transcribe(audio, language=language)
            
            # Extract text from result (format may vary)
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get("text", "").strip()
            else:
                # Try to get text attribute
                return str(result).strip()
        
        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
