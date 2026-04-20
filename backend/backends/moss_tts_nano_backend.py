"""
MOSS-TTS-Nano backend implementation.

Wraps NanoTTSService from moss-tts-nano for zero-shot voice cloning.
0.1B parameters, 48 kHz stereo output, 20 languages, CPU-friendly.
WeTextProcessing is optional (Chinese text normalization); if absent,
the tts_robust_normalizer_single_script fallback is used instead.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

MOSS_TTS_NANO_HF_REPO = "OpenMOSS-Team/MOSS-TTS-Nano"
MOSS_AUDIO_TOKENIZER_HF_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"

_REQUIRED_WEIGHT_FILES = ["model.safetensors"]


class MOSSTTSNanoBackend:
    """MOSS-TTS-Nano backend — 0.1B, 48 kHz stereo, 20 languages, CPU-friendly."""

    def __init__(self):
        self._service = None
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device(allow_mps=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self._service is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return MOSS_TTS_NANO_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(
            MOSS_TTS_NANO_HF_REPO,
            required_files=_REQUIRED_WEIGHT_FILES,
        ) and is_model_cached(MOSS_AUDIO_TOKENIZER_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        """Load MOSS-TTS-Nano model."""
        if self._service is not None:
            return
        async with self._model_load_lock:
            if self._service is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading."""
        model_name = "moss-tts-nano"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading MOSS-TTS-Nano on {device}...")

            from moss_tts_nano_runtime import NanoTTSService

            service = NanoTTSService(
                checkpoint_path=MOSS_TTS_NANO_HF_REPO,
                audio_tokenizer_path=MOSS_AUDIO_TOKENIZER_HF_REPO,
                device=device,
                dtype="auto",
            )
            service.load()
            self._service = service

        logger.info("MOSS-TTS-Nano loaded successfully")

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._service is not None:
            device = self._device
            del self._service
            self._service = None
            self._device = None
            empty_device_cache(device)
            logger.info("MOSS-TTS-Nano unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        MOSS-TTS-Nano processes reference audio at generation time
        (voice_clone mode), so the prompt stores the file path.
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

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using MOSS-TTS-Nano voice cloning.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict with ref_audio path.
            language: BCP-47 language code (ignored at inference; language
                      is inferred from the model's multilingual tokenizer).
            seed: Optional random seed for reproducibility.
            instruct: Unused (protocol compatibility).

        Returns:
            Tuple of (mono_audio_float32, sample_rate).
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None
            ref_text = None

        def _generate_sync() -> Tuple[np.ndarray, int]:
            if seed is not None:
                manual_seed(seed, self._device)

            logger.info(f"[MOSS-TTS-Nano] Generating: lang={language}")

            result = self._service.synthesize(
                text=text,
                prompt_audio_path=ref_audio,
                prompt_text=ref_text,
                mode="voice_clone" if ref_audio else "continuation",
                seed=seed,
            )

            waveform: np.ndarray = result["waveform_numpy"]
            sample_rate: int = int(result["sample_rate"])

            # Convert stereo (samples, 2) → mono (samples,)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)

            audio = waveform.astype(np.float32)
            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
