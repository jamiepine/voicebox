"""
Chatterbox TTS backend implementation.

Wraps ChatterboxMultilingualTTS from chatterbox-tts for zero-shot
voice cloning. Supports 23 languages including Hebrew. Forces CPU
on macOS due to known MPS tensor issues.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
    patch_chatterbox_f32,
)

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
        return get_torch_device(force_cpu_on_mac=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return CHATTERBOX_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(CHATTERBOX_HF_REPO, required_files=_MTL_WEIGHT_FILES)

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
        model_name = "chatterbox-tts"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading Chatterbox Multilingual TTS on {device}...")

            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            if device == "cpu":
                _orig_torch_load = torch.load

                def _patched_load(*args, **kwargs):
                    kwargs.setdefault("map_location", "cpu")
                    return _orig_torch_load(*args, **kwargs)

                with ChatterboxTTSBackend._load_lock:
                    torch.load = _patched_load
                    try:
                        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                    finally:
                        torch.load = _orig_torch_load
            else:
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)

            # Fix sdpa attention for output_attentions support
            t3_tfmr = model.t3.tfmr
            if hasattr(t3_tfmr, "config") and hasattr(t3_tfmr.config, "_attn_implementation"):
                t3_tfmr.config._attn_implementation = "eager"
                for layer in getattr(t3_tfmr, "layers", []):
                    if hasattr(layer, "self_attn"):
                        layer.self_attn._attn_implementation = "eager"

            patch_chatterbox_f32(model)
            self.model = model

        logger.info("Chatterbox Multilingual TTS loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
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
        return await _combine_voice_prompts(audio_paths, reference_texts)

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
                manual_seed(seed, self._device)

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

            sample_rate = getattr(self.model, "sr", None) or getattr(self.model, "sample_rate", 24000)

            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
