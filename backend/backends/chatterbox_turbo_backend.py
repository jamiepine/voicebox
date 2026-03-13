"""
Chatterbox Turbo TTS backend implementation.

Wraps ChatterboxTurboTTS from chatterbox-tts for fast, English-only
voice cloning with paralinguistic tag support ([laugh], [cough], etc.).
Forces CPU on macOS due to known MPS tensor issues.
"""

import asyncio
import logging
import platform
import threading
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from . import TTSBackend
from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

CHATTERBOX_TURBO_HF_REPO = "ResembleAI/chatterbox-turbo"

# Files that must be present for the turbo model
_TURBO_WEIGHT_FILES = [
    "t3_turbo_v1.safetensors",
    "s3gen_meanflow.safetensors",
    "ve.safetensors",
]


class ChatterboxTurboTTSBackend:
    """Chatterbox Turbo TTS backend — fast, English-only, with paralinguistic tags."""

    # Class-level lock for torch.load monkey-patching
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Get the best available device. Forces CPU on macOS (MPS issue)."""
        if platform.system() == "Darwin":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return CHATTERBOX_TURBO_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if the Chatterbox Turbo model is cached locally."""
        try:
            from huggingface_hub import constants as hf_constants

            repo_cache = Path(hf_constants.HF_HUB_CACHE) / (
                "models--" + CHATTERBOX_TURBO_HF_REPO.replace("/", "--")
            )

            if not repo_cache.exists():
                return False

            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                return False

            # Check for turbo weight files
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                for fname in _TURBO_WEIGHT_FILES:
                    if not any(snapshots_dir.rglob(fname)):
                        return False
                return True

            return False
        except Exception as e:
            logger.warning(f"Error checking Chatterbox Turbo cache: {e}")
            return False

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Chatterbox Turbo model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback

        progress_manager = get_progress_manager()
        task_manager = get_task_manager()
        model_name = "chatterbox-turbo"

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
            device = self._get_device()
            self._device = device

            logger.info(f"Loading Chatterbox Turbo TTS on {device}...")

            import torch
            from huggingface_hub import snapshot_download
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            # Download model files ourselves so we can pass token=None
            # (upstream from_pretrained passes token=True which requires
            # a stored HF token even though the repo is public).
            try:
                local_path = snapshot_download(
                    repo_id=CHATTERBOX_TURBO_HF_REPO,
                    token=None,
                    allow_patterns=[
                        "*.safetensors", "*.json", "*.txt", "*.pt", "*.model",
                    ],
                )
            finally:
                tracker_context.__exit__(None, None, None)

            # Monkey-patch torch.load for CPU loading. The model's .pt files
            # were saved on CUDA; from_local() doesn't pass map_location
            # so loading on CPU fails without this.
            if device == "cpu":
                _orig_torch_load = torch.load

                def _patched_load(*args, **kwargs):
                    kwargs.setdefault("map_location", "cpu")
                    return _orig_torch_load(*args, **kwargs)

                with ChatterboxTurboTTSBackend._load_lock:
                    torch.load = _patched_load
                    try:
                        self.model = ChatterboxTurboTTS.from_local(
                            local_path, device,
                        )
                    finally:
                        torch.load = _orig_torch_load
            else:
                self.model = ChatterboxTurboTTS.from_local(
                    local_path, device,
                )

            if not is_cached:
                progress_manager.mark_complete(model_name)
                task_manager.complete_download(model_name)

            logger.info("Chatterbox Turbo TTS loaded successfully")

        except ImportError as e:
            logger.error(
                "chatterbox-tts package not found. "
                "Install with: pip install chatterbox-tts"
            )
            if not is_cached:
                progress_manager.mark_error(model_name, str(e))
                task_manager.error_download(model_name, str(e))
            raise
        except Exception as e:
            logger.error(f"Failed to load Chatterbox Turbo: {e}")
            if not is_cached:
                progress_manager.mark_error(model_name, str(e))
                task_manager.error_download(model_name, str(e))
            raise

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
            logger.info("Chatterbox Turbo unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Chatterbox Turbo processes reference audio at generation time, so the
        prompt just stores the file path.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """Combine multiple reference samples."""
        combined_audio = []
        for path in audio_paths:
            audio, _sr = load_audio(path)
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
        Generate audio using Chatterbox Turbo TTS.

        Supports paralinguistic tags in text: [laugh], [cough], [chuckle], etc.

        Args:
            text: Text to synthesize (may include paralinguistic tags)
            voice_prompt: Dict with ref_audio path
            language: Ignored (Turbo is English-only)
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)

            logger.info("[Chatterbox Turbo] Generating (English)")

            wav = self.model.generate(
                text,
                audio_prompt_path=ref_audio,
                temperature=0.8,
                top_k=1000,
                top_p=0.95,
                repetition_penalty=1.2,
            )

            # Convert tensor -> numpy
            if isinstance(wav, torch.Tensor):
                audio = wav.squeeze().cpu().numpy().astype(np.float32)
            else:
                audio = np.asarray(wav, dtype=np.float32)

            sample_rate = (
                getattr(self.model, "sr", None)
                or getattr(self.model, "sample_rate", 24000)
            )

            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
