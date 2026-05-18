"""
Qwen3-TTS VoiceDesign backend implementation.

VoiceDesign creates a synthetic voice from a natural-language description
instead of cloning a reference audio file or selecting a preset speaker.
It uses the same qwen_tts package as Base and CustomVoice, but loads the
VoiceDesign checkpoint and calls generate_voice_design().
"""

import asyncio
import logging
import threading
from typing import Optional

import numpy as np
import torch

from . import LANGUAGE_CODE_TO_NAME
from .base import (
    get_torch_device,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

QWEN_VD_HF_REPOS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


class QwenVoiceDesignBackend:
    """Qwen3-TTS VoiceDesign backend - text-designed voices with instruct control."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = get_torch_device(allow_xpu=True, allow_directml=True)
        self._current_model_size: Optional[str] = None
        self._model_load_lock = asyncio.Lock()
        self._state_lock = threading.RLock()

    def is_loaded(self) -> bool:
        with self._state_lock:
            return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_VD_HF_REPOS:
            raise ValueError(f"Unknown Qwen VoiceDesign model size: {model_size}")
        return QWEN_VD_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        size = model_size or self.model_size
        return is_model_cached(self._get_model_path(size))

    async def load_model_async(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        async with self._model_load_lock:
            with self._state_lock:
                if self.model is not None and self._current_model_size == model_size:
                    return

                if self.model is not None and self._current_model_size != model_size:
                    self._unload_model_locked()

            await asyncio.to_thread(self._load_model_sync, model_size)

    load_model = load_model_async

    def _load_model_sync(self, model_size: str) -> None:
        with self._state_lock:
            model_name = f"qwen-voice-design-{model_size}"
            is_cached = self._is_model_cached(model_size)

            with model_load_progress(model_name, is_cached):
                from qwen_tts import Qwen3TTSModel

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
        with self._state_lock:
            self._unload_model_locked()

    def _unload_model_locked(self) -> None:
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
        """Create a VoiceDesign prompt from text for protocol compatibility."""
        if audio_path:
            logger.warning(
                "Qwen VoiceDesign ignores audio_path while creating a designed voice prompt: %s",
                audio_path,
            )
        if use_cache:
            logger.debug(
                "Qwen VoiceDesign uses text design prompts directly; use_cache is ignored."
            )
        return {
            "voice_type": "designed",
            "design_prompt": reference_text,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        raise NotImplementedError("Qwen VoiceDesign does not support combining audio voice prompts.")

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        await self.load_model_async(None)

        design_prompt = (voice_prompt.get("design_prompt") or "").strip()
        delivery_prompt = (instruct or "").strip()
        if not design_prompt:
            raise ValueError("Qwen VoiceDesign requires a design_prompt on the voice profile")

        effective_instruct = design_prompt
        if delivery_prompt:
            effective_instruct = f"{design_prompt}. Delivery: {delivery_prompt}"

        def _generate_sync():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "auto")
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                language=lang_name.capitalize() if lang_name != "auto" else "Auto",
                instruct=effective_instruct,
                temperature=temperature,
            )
            return wavs[0], sample_rate

        audio, sample_rate = await asyncio.to_thread(_generate_sync)
        return audio, sample_rate
