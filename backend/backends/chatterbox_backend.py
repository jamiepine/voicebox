"""
Chatterbox TTS backend implementation.

Wraps ChatterboxMultilingualTTS from chatterbox-tts for zero-shot
voice cloning. Supports 23 languages including Hebrew. Forces CPU
on macOS due to known MPS tensor issues.
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

CHATTERBOX_HF_REPO = "ResembleAI/chatterbox"

# Files that must be present for the multilingual model
_MTL_WEIGHT_FILES = [
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "ve.pt",
]


class ChatterboxTTSBackend:
    """Chatterbox Multilingual TTS backend for voice cloning."""

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
        return CHATTERBOX_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if the Chatterbox multilingual model is cached locally."""
        try:
            from huggingface_hub import constants as hf_constants

            repo_cache = Path(hf_constants.HF_HUB_CACHE) / (
                "models--" + CHATTERBOX_HF_REPO.replace("/", "--")
            )

            if not repo_cache.exists():
                return False

            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                return False

            # Check for multilingual weight files
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                for fname in _MTL_WEIGHT_FILES:
                    if not any(snapshots_dir.rglob(fname)):
                        return False
                return True

            return False
        except Exception as e:
            logger.warning(f"Error checking Chatterbox cache: {e}")
            return False

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Chatterbox multilingual model."""
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
        model_name = "chatterbox-tts"

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

            logger.info(f"Loading Chatterbox Multilingual TTS on {device}...")

            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            # Load into a local variable first, apply all patches, then
            # assign to self.model.  This avoids leaving a half-initialised
            # model on self.model if any patch step raises an exception.
            #
            # Monkey-patch torch.load for CPU loading. The model's .pt files
            # were saved on CUDA; from_pretrained() doesn't pass map_location
            # so loading on CPU fails without this.
            try:
                if device == "cpu":
                    _orig_torch_load = torch.load

                    def _patched_load(*args, **kwargs):
                        kwargs.setdefault("map_location", "cpu")
                        return _orig_torch_load(*args, **kwargs)

                    with ChatterboxTTSBackend._load_lock:
                        torch.load = _patched_load
                        try:
                            model = ChatterboxMultilingualTTS.from_pretrained(
                                device=device,
                            )
                        finally:
                            torch.load = _orig_torch_load
                else:
                    model = ChatterboxMultilingualTTS.from_pretrained(
                        device=device,
                    )
            finally:
                tracker_context.__exit__(None, None, None)

            # Fix: transformers >= 4.36 defaults LlamaModel to sdpa attention
            # which doesn't support output_attentions=True (needed by
            # Chatterbox's AlignmentStreamAnalyzer). Force eager attention.
            t3_tfmr = model.t3.tfmr
            if hasattr(t3_tfmr, "config") and hasattr(
                t3_tfmr.config, "_attn_implementation"
            ):
                t3_tfmr.config._attn_implementation = "eager"
                for layer in getattr(t3_tfmr, "layers", []):
                    if hasattr(layer, "self_attn"):
                        layer.self_attn._attn_implementation = "eager"

            if not is_cached:
                progress_manager.mark_complete(model_name)
                task_manager.complete_download(model_name)

            # Patch float64 → float32 dtype mismatches in upstream chatterbox.
            # librosa.load returns float64 numpy; multiple upstream code paths
            # convert it to a torch tensor via torch.from_numpy() without
            # casting, then matmul it against float32 model weights.
            import types

            # Patch S3Tokenizer (used by s3gen.tokenizer)
            _tokzr = model.s3gen.tokenizer
            _orig_log_mel = _tokzr.log_mel_spectrogram.__func__

            def _f32_log_mel(self_tokzr, audio, padding=0):
                import torch as _torch
                if _torch.is_tensor(audio):
                    audio = audio.float()
                return _orig_log_mel(self_tokzr, audio, padding)

            _tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, _tokzr)

            # Patch VoiceEncoder
            _ve = model.ve
            _orig_ve_forward = _ve.forward.__func__

            def _f32_ve_forward(self_ve, mels):
                return _orig_ve_forward(self_ve, mels.float())

            _ve.forward = types.MethodType(_f32_ve_forward, _ve)

            # All patches applied successfully — publish the model
            self.model = model

            logger.info("Chatterbox Multilingual TTS loaded successfully")

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
            logger.error(f"Failed to load Chatterbox: {e}")
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
            logger.info("Chatterbox unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Chatterbox processes reference audio at generation time, so the
        prompt just stores the file path. The actual audio is loaded by
        model.generate() via audio_prompt_path.
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

    # Per-language generation defaults. Lower temp + higher cfg = clearer speech.
    _LANG_DEFAULTS: ClassVar[dict] = {
        "he": {
            "exaggeration": 0.4,
            "cfg_weight": 0.7,
            "temperature": 0.65,
            "repetition_penalty": 2.5,
        },
    }
    _GLOBAL_DEFAULTS: ClassVar[dict] = {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 2.0,
    }

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using Chatterbox Multilingual TTS.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path
            language: BCP-47 language code
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

        # Merge language-specific defaults with global defaults
        lang_defaults = self._LANG_DEFAULTS.get(language, self._GLOBAL_DEFAULTS)

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)

            logger.info(f"[Chatterbox] Generating: lang={language}")

            wav = self.model.generate(
                text,
                language_id=language,
                audio_prompt_path=ref_audio,
                exaggeration=lang_defaults["exaggeration"],
                cfg_weight=lang_defaults["cfg_weight"],
                temperature=lang_defaults["temperature"],
                repetition_penalty=lang_defaults["repetition_penalty"],
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
