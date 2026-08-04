"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

from typing import Optional, List, Tuple
import asyncio
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# MLX's Metal backend keeps a per-thread stream registry. Loading a model on
# one worker thread (via asyncio.to_thread, which round-robins across the
# default executor's pool) and then generating on a different worker thread
# raises "There is no Stream(gpu, N) in current thread." All MLX calls in
# this module must therefore run on the SAME OS thread for the process
# lifetime — route them through this single-worker executor instead of
# asyncio.to_thread's shared multi-worker pool.
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")


def _run_on_mlx_thread(func, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_mlx_executor, func, *args)

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import patch_huggingface_hub_offline, ensure_original_qwen_config_cached

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import is_model_cached, combine_voice_prompts as _combine_voice_prompts, model_load_progress
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None
        # Guards the whole load-then-use sequence in generate()/create_voice_prompt()
        # so a concurrent request for a different model_size can't swap self.model
        # out from under an in-flight request between its load and its inference.
        self._op_lock = asyncio.Lock()

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
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        }

        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        hf_model_id = mlx_model_map[model_size]
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

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

        # Unload (if needed) and load as ONE callable on the MLX worker thread.
        # Doing this as two separate _run_on_mlx_thread calls would run the
        # unload on whichever thread issues the second call — usually still
        # correct, but a caller-side await gap between them would let another
        # coroutine slip a conflicting load in between. One callable removes
        # the gap.
        await _run_on_mlx_thread(self._reload_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _reload_sync(self, model_size: str):
        """Unload a mismatched model and load the requested one, in one MLX-thread op."""
        if self.model is not None and self._current_model_size != model_size:
            self._unload_model_sync()
        self._load_model_sync(model_size)

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_path = self._get_model_path(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts import load

            logger.info("Loading MLX TTS model %s...", model_size)

            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TTS model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory.

        Safe to call from any thread (e.g. the FastAPI event loop, from the
        /models/unload routes): the actual teardown is submitted to the MLX
        worker thread and awaited synchronously here, so Metal resources are
        always released on the same OS thread that created them. Do not call
        this from within a callable already running ON the MLX worker thread
        (e.g. from _reload_sync) — use _unload_model_sync directly there, or
        this would deadlock the single-worker executor waiting on itself.
        """
        # Submit unconditionally rather than checking self.model here first:
        # that check would race against a load already queued on the worker
        # thread (this call could see None, skip, and leave a model that
        # finishes loading a moment later still resident). The loaded check
        # belongs inside _unload_model_sync, where it runs atomically with
        # the teardown itself.
        _mlx_executor.submit(self._unload_model_sync).result()

    def _unload_model_sync(self):
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("MLX TTS model unloaded")

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
        async with self._op_lock:
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
                        logger.warning("Cached audio file not found: %s, regenerating prompt", cached_audio_path)

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

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts)

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
        logger.info("Generating audio for text: %s", text)

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            # MLX generate() returns a generator yielding GenerationResult objects
            audio_chunks = []
            sample_rate = 24000
            lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            # Set seed if provided (MLX uses numpy random)
            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            # Extract voice prompt info
            ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
            ref_text = voice_prompt.get("ref_text", "")

            # Validate that the audio file exists
            if ref_audio and not Path(ref_audio).exists():
                logger.warning("Audio file not found: %s", ref_audio)
                logger.warning("This may be due to a cached voice prompt referencing a deleted temp file.")
                logger.warning("Regenerating without voice prompt.")
                ref_audio = None

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state. Forcing offline here (previously used to avoid lazy
            # mlx_audio lookups hanging when the network drops mid-inference,
            # issue #462) regressed online users because libraries make
            # legitimate metadata calls during generation.
            try:
                if ref_audio:
                    # Check if generate accepts ref_audio parameter
                    import inspect

                    sig = inspect.signature(self.model.generate)
                    if "ref_audio" in sig.parameters:
                        # Generate with voice cloning
                        for result in self.model.generate(text, ref_audio=ref_audio, ref_text=ref_text, lang_code=lang):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                    else:
                        # Fallback: generate without voice cloning
                        for result in self.model.generate(text, lang_code=lang):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                else:
                    # No voice prompt, generate normally
                    for result in self.model.generate(text, lang_code=lang):
                        audio_chunks.append(np.array(result.audio))
                        sample_rate = result.sample_rate
            except Exception as e:
                # If voice cloning fails, try without it
                logger.warning("Voice cloning failed, generating without voice prompt: %s", e)
                for result in self.model.generate(text, lang_code=lang):
                    audio_chunks.append(np.array(result.audio))
                    sample_rate = result.sample_rate

            # Concatenate all chunks
            if audio_chunks:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
            else:
                # Fallback: empty audio
                audio = np.array([], dtype=np.float32)

            return audio, sample_rate

        def _reload_and_generate_sync():
            """Ensure the configured model is loaded, then generate — as ONE
            MLX-worker submission. Two separate submissions (load, then
            generate) leave a gap after the load future resolves and before
            the generate future is submitted; an unload_model() call from
            another thread could land in that gap and tear down the model
            this call is about to use. Folding both into one callable closes
            the gap: the executor's own FIFO ordering is the only guarantee
            this needs, and the reload check here is self-healing even if an
            unload happened to run just before this callable started.
            """
            if self.model is None or self._current_model_size != self.model_size:
                self._reload_sync(self.model_size)
            return _generate_sync()

        async with self._op_lock:
            audio, sample_rate = await _run_on_mlx_thread(_reload_and_generate_sync)

        return audio, sample_rate


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size
        # See MLXTTSBackend._op_lock — same reason.
        self._op_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

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
        await _run_on_mlx_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            self.model = load(model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory.

        Safe to call from any thread — see MLXTTSBackend.unload_model for why,
        including why this submits unconditionally instead of checking
        self.model first.
        """
        _mlx_executor.submit(self._unload_model_sync).result()

    def _unload_model_sync(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("MLX Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        resolved_size = model_size if model_size is not None else self.model_size

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # MLX Whisper transcription using generate method
            # The generate method accepts audio path directly
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — see the comment in MLXTTSBackend.generate for the
            # regression this revert fixes (issue #462).
            result = self.model.generate(str(audio_path), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get("text", "").strip()
            elif hasattr(result, "text"):
                return result.text.strip()
            else:
                return str(result).strip()

        def _reload_and_transcribe_sync():
            """Ensure the requested model is loaded, then transcribe — as ONE
            MLX-worker submission. See MLXTTSBackend._reload_and_generate_sync
            for why this needs to be a single callable rather than a separate
            load-then-transcribe pair.
            """
            if self.model is None or self.model_size != resolved_size:
                self._load_model_sync(resolved_size)
            return _transcribe_sync()

        async with self._op_lock:
            return await _run_on_mlx_thread(_reload_and_transcribe_sync)
