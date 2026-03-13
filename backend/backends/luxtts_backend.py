"""
LuxTTS backend implementation.

Wraps the LuxTTS (ZipVoice) model for zero-shot voice cloning.
~1GB VRAM, 48kHz output, 150x realtime on CPU.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import TTSBackend
from ..utils.audio import normalize_audio, load_audio
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

# HuggingFace repo for model weight detection
LUXTTS_HF_REPO = "YatharthS/LuxTTS"


class LuxTTSBackend:
    """LuxTTS backend for zero-shot voice cloning."""

    def __init__(self):
        self.model = None
        self.model_size = "default"  # LuxTTS has only one model size
        self._device = None

    def _get_device(self) -> str:
        """Get the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def _get_model_path(self, model_size: str) -> str:
        return LUXTTS_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if LuxTTS model weights are cached locally."""
        try:
            from huggingface_hub import constants as hf_constants

            repo_cache = (
                Path(hf_constants.HF_HUB_CACHE)
                / ("models--" + LUXTTS_HF_REPO.replace("/", "--"))
            )

            if not repo_cache.exists():
                return False

            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                return False

            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                has_weights = any(snapshots_dir.rglob("*.pt")) or any(
                    snapshots_dir.rglob("*.safetensors")
                ) or any(snapshots_dir.rglob("*.onnx")) or any(
                    snapshots_dir.rglob("*.bin")
                )
                return has_weights

            return False
        except Exception as e:
            logger.warning(f"Error checking LuxTTS cache: {e}")
            return False

    async def load_model(self, model_size: str = "default") -> None:
        """Load the LuxTTS model."""
        if self.model is not None:
            return

        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback

        progress_manager = get_progress_manager()
        task_manager = get_task_manager()
        model_name = "luxtts"

        is_cached = self._is_model_cached()

        # Set up HF progress tracking (intercepts tqdm for file-level progress)
        progress_callback = create_hf_progress_callback(model_name, progress_manager)
        tracker = HFProgressTracker(progress_callback, filter_non_downloads=is_cached)
        tracker_context = tracker.patch_download()
        tracker_context.__enter__()

        if not is_cached:
            task_manager.start_download(model_name)
            progress_manager.update_progress(
                model_name=model_name,
                current=0,
                total=0,
                filename="Connecting to HuggingFace...",
                status="downloading",
            )

        try:
            from zipvoice.luxvoice import LuxTTS

            device = self.device
            logger.info(f"Loading LuxTTS on {device}...")

            # LuxTTS constructor downloads model and loads everything
            try:
                if device == "cpu":
                    import os
                    threads = os.cpu_count() or 4
                    self.model = LuxTTS(
                        model_path=LUXTTS_HF_REPO,
                        device="cpu",
                        threads=min(threads, 8),
                    )
                else:
                    self.model = LuxTTS(
                        model_path=LUXTTS_HF_REPO,
                        device=device,
                    )
            finally:
                tracker_context.__exit__(None, None, None)

            if not is_cached:
                progress_manager.mark_complete(model_name)
                task_manager.complete_download(model_name)

            logger.info("LuxTTS loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LuxTTS: {e}")
            if not is_cached:
                progress_manager.mark_error(model_name, str(e))
                task_manager.error_download(model_name, str(e))
            raise

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("LuxTTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        LuxTTS uses its own encode_prompt() which runs Whisper ASR internally
        to transcribe the reference. The reference_text parameter is not used
        by LuxTTS itself, but we include it in the cache key for consistency.
        """
        await self.load_model()

        # Compute cache key once for both lookup and storage
        cache_key = ("luxtts_" + get_cache_key(audio_path, reference_text)) if use_cache else None

        if cache_key:
            cached = get_cached_voice_prompt(cache_key)
            if cached is not None and isinstance(cached, dict):
                return cached, True

        def _encode_sync():
            return self.model.encode_prompt(
                prompt_audio=str(audio_path),
                duration=5,
                rms=0.01,
            )

        encoded = await asyncio.to_thread(_encode_sync)

        if cache_key:
            cache_voice_prompt(cache_key, encoded)

        return encoded, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple reference samples.

        LuxTTS doesn't have native multi-prompt support, so we concatenate
        the audio and let encode_prompt handle the combined clip.
        """
        combined_audio = []
        for path in audio_paths:
            audio, _sr = load_audio(path, sample_rate=24000)
            audio = normalize_audio(audio)
            combined_audio.append(audio)

        mixed = np.concatenate(combined_audio)
        mixed = normalize_audio(mixed)
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
        Generate audio from text using LuxTTS.

        Args:
            text: Text to synthesize
            voice_prompt: Encoded prompt dict from encode_prompt()
            language: Language code (LuxTTS is English-focused)
            seed: Random seed for reproducibility
            instruct: Not supported by LuxTTS (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            wav = self.model.generate_speech(
                text=text,
                encode_dict=voice_prompt,
                num_steps=4,
                guidance_scale=3.0,
                t_shift=0.5,
                speed=1.0,
                return_smooth=False,  # 48kHz output
            )

            # LuxTTS returns a tensor (may be on GPU/MPS), move to CPU first
            audio = wav.detach().cpu().numpy().squeeze()
            return audio, 48000

        return await asyncio.to_thread(_generate_sync)
