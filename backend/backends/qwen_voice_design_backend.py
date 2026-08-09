"""
Qwen3-TTS VoiceDesign backend implementation.

Wraps the Qwen3-TTS-12Hz VoiceDesign model, which synthesises a voice from a
natural-language description instead of a reference recording or a preset
speaker id. Uses the same qwen_tts library as the Base model
(pytorch_backend.py) but loads a different checkpoint and calls
generate_voice_design() instead of generate_voice_clone().

Key differences from the CustomVoice engine:
  - The voice identity is a free-text description carried on the profile
    (voice_type "designed", design_prompt column), not one of 9 preset ids
  - Only one checkpoint exists upstream: 1.7B. There is no 0.6B VoiceDesign.

Languages supported: zh, en, ja, ko, de, fr, ru, pt, es, it
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import torch

from . import TTSBackend, LANGUAGE_CODE_TO_NAME
from .base import (
    is_model_cached,
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# Used when an engine-level voice prompt is requested without a designed
# profile behind it — mirrors QWEN_CV_DEFAULT_SPEAKER in the CustomVoice
# backend. Deliberately plain so it reads as "unstyled" rather than as a
# character.
QWEN_VD_DEFAULT_DESIGN = "A clear, neutral voice with natural pacing."

# HuggingFace repo IDs per model size. VoiceDesign ships 1.7B only.
QWEN_VD_HF_REPOS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


class QwenVoiceDesignBackend:
    """Qwen3-TTS VoiceDesign backend — voices described in natural language."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = self._get_device()
        self._current_model_size: Optional[str] = None
        # Without this, two concurrent generations can both see model is None
        # and each load the 3.5 GB checkpoint. Same pattern as the Chatterbox
        # and TADA backends.
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_VD_HF_REPOS:
            raise ValueError(f"Unknown model size: {model_size}")
        return QWEN_VD_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        size = model_size or self.model_size
        return is_model_cached(self._get_model_path(size))

    async def load_model_async(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        async with self._model_load_lock:
            # Re-check: another request may have finished loading while this
            # one waited for the lock.
            if self.model is not None and self._current_model_size == model_size:
                return

            if self.model is not None and self._current_model_size != model_size:
                self.unload_model()

            await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility with the TTSBackend protocol
    load_model = load_model_async

    def _load_model_sync(self, model_size: str) -> None:
        model_name = f"qwen-voice-design-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from qwen_tts import Qwen3TTSModel

            # generate_voice_design landed after the 0.0.x line. Fail with
            # something readable instead of an AttributeError mid-generation.
            if not hasattr(Qwen3TTSModel, "generate_voice_design"):
                raise RuntimeError(
                    "The installed qwen-tts is too old for VoiceDesign. "
                    "Upgrade with: pip install -U 'qwen-tts>=0.1.1'"
                )

            model_path = self._get_model_path(model_size)
            logger.info("Loading Qwen VoiceDesign %s on %s...", model_size, self.device)

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen VoiceDesign %s loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen VoiceDesign unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt for VoiceDesign.

        VoiceDesign doesn't use reference audio — the voice is described in
        text. When called for a cloned profile (fallback), uses the neutral
        default description. For designed profiles, the voice_prompt dict is
        built by the profile service and bypasses this method entirely.
        """
        return {
            "voice_type": "designed",
            "design_prompt": QWEN_VD_DEFAULT_DESIGN,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Qwen VoiceDesign.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with design_prompt (the voice description)
            language: Language code (zh, en, ja, ko, etc.)
            seed: Random seed for reproducibility
            instruct: Per-generation style instruction, layered on top of the
                      profile's design_prompt rather than replacing it

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        design_prompt = voice_prompt.get("design_prompt") or QWEN_VD_DEFAULT_DESIGN

        # For this model `instruct` carries the voice identity itself, so the
        # profile's description has to lead. A per-request instruct is appended
        # so style tweaks layer on rather than replacing the designed voice.
        if instruct:
            logger.debug("Layering per-request instruct on top of design_prompt")
            combined_instruct = f"{design_prompt.rstrip('. ')}. {instruct}"
        else:
            combined_instruct = design_prompt

        def _generate_sync():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state. Forcing offline here (issue #462) regressed online
            # users whose libraries issue legitimate metadata lookups
            # during generation.
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                instruct=combined_instruct,
                language=lang_name.capitalize() if lang_name != "auto" else "Auto",
            )
            return wavs[0], sample_rate

        audio, sample_rate = await asyncio.to_thread(_generate_sync)
        return audio, sample_rate
