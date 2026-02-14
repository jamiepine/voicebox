"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

import warnings
# Suppress upstream tokenizer warnings from mlx-audio/transformers
warnings.filterwarnings("ignore", message="You are using a model of type.*to instantiate a model of type")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*fix_mistral_regex.*")

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
    
    def _is_model_cached(self, model_size: str) -> bool:
        """
        Check if the model is already cached locally AND fully downloaded.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is fully cached, False if missing or incomplete
        """
        try:
            from huggingface_hub import constants as hf_constants
            model_path = self._get_model_path(model_size)
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_path.replace("/", "--"))
            
            if not repo_cache.exists():
                return False
            
            # Check for .incomplete files - if any exist, download is still in progress
            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                print(f"[_is_model_cached] Found .incomplete files for {model_size}, treating as not cached")
                return False
            
            # Check that actual model weight files exist in snapshots
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                has_weights = (
                    any(snapshots_dir.rglob("*.safetensors")) or
                    any(snapshots_dir.rglob("*.bin")) or
                    any(snapshots_dir.rglob("*.npz"))
                )
                if not has_weights:
                    print(f"[_is_model_cached] No model weights found for {model_size}, treating as not cached")
                    return False
            
            return True
        except Exception as e:
            print(f"[_is_model_cached] Error checking cache for {model_size}: {e}")
            return False
    
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
            # Get model path BEFORE importing mlx_audio
            model_path = self._get_model_path(model_size)
            
            # Set up progress tracking
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            model_name = f"qwen-tts-{model_size}"
            
            # Check if model is already cached
            is_cached = self._is_model_cached(model_size)
            
            # Set up progress callback
            # If cached: filter out non-download progress
            # If not cached: report all progress (we're actually downloading)
            progress_callback = create_hf_progress_callback(model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback, filter_non_downloads=is_cached)
            
            print(f"Loading MLX TTS model {model_size}...")
            
            # Only track download progress if model is NOT cached
            if not is_cached:
                # Start tracking download task
                task_manager.start_download(model_name)
                
                # Initialize progress state so SSE endpoint has initial data to send
                # This provides immediate feedback while HuggingFace fetches metadata
                progress_manager.update_progress(
                    model_name=model_name,
                    current=0,
                    total=0,  # Will be updated once actual total is known
                    filename="Connecting to HuggingFace...",
                    status="downloading",
                )
            
            # IMPORTANT: Patch tqdm BEFORE importing mlx_audio
            # Otherwise mlx_audio caches reference to original tqdm
            tracker_context = tracker.patch_download()
            tracker_context.__enter__()
            
            # Import mlx_audio AFTER patching tqdm
            from mlx_audio.tts import load
            
            # Load MLX model (downloads automatically)
            try:
                self.model = load(model_path)
            finally:
                # Exit the patch context
                tracker_context.__exit__(None, None, None)
            
            # Only mark download as complete if we were tracking it
            if not is_cached:
                progress_manager.mark_complete(model_name)
                task_manager.complete_download(model_name)
            
            self._current_model_size = model_size
            self.model_size = model_size
            
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
                    # Validate that the cached audio file still exists
                    cached_audio_path = cached_prompt.get("ref_audio") or cached_prompt.get("ref_audio_path")
                    if cached_audio_path and Path(cached_audio_path).exists():
                        return cached_prompt, True
                    else:
                        # Cached file no longer exists, invalidate cache
                        print(f"Cached audio file not found: {cached_audio_path}, regenerating prompt")
        
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
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)
            progress_callback: Optional callback(progress_pct: float) where 0.0-100.0

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

            # Estimate total chunks for progress reporting
            estimated_chunks = max(1, len(text) // 2)
            chunk_count = 0

            # Set seed if provided (MLX uses numpy random)
            if seed is not None:
                import mlx.core as mx
                np.random.seed(seed)
                mx.random.seed(seed)

            # Extract voice prompt info
            ref_audio_path = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
            ref_text = voice_prompt.get("ref_text", "")
            ref_audio = None

            # Validate that the audio file exists and load it
            if ref_audio_path and Path(ref_audio_path).exists():
                try:
                    # Load audio as array - model.generate expects mx.array, not numpy
                    # Qwen3-TTS uses 24kHz sample rate
                    import mlx.core as mx
                    ref_audio_data, _ = load_audio(ref_audio_path, sample_rate=24000)
                    # Convert to mx.array - the model expects this format
                    ref_audio = mx.array(np.array(ref_audio_data, dtype=np.float32))
                    print(f"Loaded reference audio: {ref_audio.shape} samples")
                except Exception as e:
                    print(f"Warning: Failed to load reference audio: {e}")
                    ref_audio = None
            elif ref_audio_path:
                print(f"Warning: Audio file not found: {ref_audio_path}")
                print("This may be due to a cached voice prompt referencing a deleted temp file.")
                print("Regenerating without voice prompt.")

            # Check if model supports voice cloning via generate method
            # MLX API may support ref_audio parameter directly
            try:
                # Try with voice cloning parameters if supported
                if ref_audio is not None:
                    # Check if generate accepts ref_audio parameter
                    import inspect
                    sig = inspect.signature(self.model.generate)
                    if "ref_audio" in sig.parameters:
                        # Generate with voice cloning - pass loaded array
                        for result in self.model.generate(text, ref_audio=ref_audio, ref_text=ref_text):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                            chunk_count += 1
                            if progress_callback:
                                pct = min(95.0, (chunk_count / estimated_chunks) * 100.0)
                                progress_callback(pct)
                    else:
                        # Fallback: generate without voice cloning
                        for result in self.model.generate(text):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                            chunk_count += 1
                            if progress_callback:
                                pct = min(95.0, (chunk_count / estimated_chunks) * 100.0)
                                progress_callback(pct)
                else:
                    # No voice prompt, generate normally
                    for result in self.model.generate(text):
                        audio_chunks.append(np.array(result.audio))
                        sample_rate = result.sample_rate
                        chunk_count += 1
                        if progress_callback:
                            pct = min(95.0, (chunk_count / estimated_chunks) * 100.0)
                            progress_callback(pct)
            except Exception as e:
                # If voice cloning fails, try without it
                print(f"Warning: Voice cloning failed, generating without voice prompt: {e}")
                import traceback
                traceback.print_exc()
                for result in self.model.generate(text):
                    audio_chunks.append(np.array(result.audio))
                    sample_rate = result.sample_rate
                    chunk_count += 1
                    if progress_callback:
                        pct = min(95.0, (chunk_count / estimated_chunks) * 100.0)
                        progress_callback(pct)

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
    
    def __init__(self, model_size: Optional[str] = None):
        self.model = None
        if model_size is None:
            model_size = "base"
        self.model_size = model_size
        self._current_model_size = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    @staticmethod
    def get_mlx_whisper_model_map() -> dict:
        """
        Get the mapping of model sizes to MLX model IDs.
        
        Returns:
            Dictionary mapping model size to HuggingFace model ID
        """
        # Use the new ASR-specific models that work with mlx_audio.stt.load()
        return {
            "tiny": "mlx-community/whisper-tiny-asr-fp16",
            "base": "mlx-community/whisper-base-asr-fp16",
            "small": "mlx-community/whisper-small-asr-fp16",
            "medium": "mlx-community/whisper-medium-asr-fp16",
            "large": "mlx-community/whisper-large-asr-fp16",
            "large-v2": "mlx-community/whisper-large-v2-asr-fp16",
            "large-v3": "mlx-community/whisper-large-v3-asr-fp16",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
        }
    
    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX Whisper model path.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo)
            
        Returns:
            HuggingFace Hub model ID for MLX Whisper
        """
        mlx_model_map = self.get_mlx_whisper_model_map()
        
        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown Whisper model size: {model_size}. Available sizes: {list(mlx_model_map.keys())}")
        
        hf_model_id = mlx_model_map[model_size]
        print(f"Will download MLX Whisper model from HuggingFace Hub: {hf_model_id}")
        
        return hf_model_id
    
    def _is_model_cached(self, model_size: str) -> bool:
        """
        Check if the Whisper model is already cached locally AND fully downloaded.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is fully cached, False if missing or incomplete
        """
        try:
            from huggingface_hub import constants as hf_constants
            model_path = self._get_model_path(model_size)
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_path.replace("/", "--"))
            
            if not repo_cache.exists():
                return False
            
            # Check for .incomplete files - if any exist, download is still in progress
            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                print(f"[_is_model_cached] Found .incomplete files for whisper-{model_size}, treating as not cached")
                return False
            
            # Check that actual model weight files exist in snapshots
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                has_weights = (
                    any(snapshots_dir.rglob("*.safetensors")) or
                    any(snapshots_dir.rglob("*.bin")) or
                    any(snapshots_dir.rglob("*.npz"))
                )
                if not has_weights:
                    print(f"[_is_model_cached] No model weights found for whisper-{model_size}, treating as not cached")
                    return False
            
            return True
        except Exception as e:
            print(f"[_is_model_cached] Error checking cache for whisper-{model_size}: {e}")
            return False
    
    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo)
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
            # Get model path BEFORE importing mlx_audio
            model_path = self._get_model_path(model_size)
            
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = f"whisper-{model_size}"

            # Check if model is already cached
            is_cached = self._is_model_cached(model_size)

            # Set up progress callback and tracker
            # If cached: filter out non-download progress
            # If not cached: report all progress (we're actually downloading)
            progress_callback = create_hf_progress_callback(progress_model_name, progress_manager)
            tracker = HFProgressTracker(progress_callback, filter_non_downloads=is_cached)

            # Patch tqdm BEFORE importing mlx_audio
            tracker_context = tracker.patch_download()
            tracker_context.__enter__()

            # Import the proper load function (from_pretrained is deprecated and doesn't load the processor)
            from mlx_audio.stt import load as stt_load

            print(f"Loading MLX Whisper model {model_size}...")

            # Only track download progress if model is NOT cached
            if not is_cached:
                # Start tracking download task
                task_manager.start_download(progress_model_name)
                
                # Initialize progress state so SSE endpoint has initial data to send
                progress_manager.update_progress(
                    model_name=progress_model_name,
                    current=0,
                    total=0,
                    filename="Connecting to HuggingFace...",
                    status="downloading",
                )

            # Load the model using stt.load() which properly initializes the HuggingFace processor
            # This is required for the tokenizer/generate functionality to work
            try:
                self.model = stt_load(model_path)
            finally:
                # Exit the patch context
                tracker_context.__exit__(None, None, None)
            
            # Only mark download as complete if we were tracking it
            if not is_cached:
                progress_manager.mark_complete(progress_model_name)
                task_manager.complete_download(progress_model_name)
            
            self._current_model_size = model_size
            self.model_size = model_size
            
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
            self._current_model_size = None
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
            import numpy as np
            
            # Load audio ourselves to handle any format (mp3, wav, etc.)
            # Whisper expects 16kHz mono audio
            audio, sr = load_audio(audio_path, sample_rate=16000)
            
            # Ensure it's a numpy array (model.generate expects numpy or mx.array)
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # MLX Whisper transcription using generate method
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Pass audio array instead of path to avoid format detection issues
            result = self.model.generate(audio, **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get("text", "").strip()
            elif hasattr(result, "text"):
                return result.text.strip()
            else:
                return str(result).strip()

        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
