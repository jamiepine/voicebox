"""
Kokoro TTS backend implementation.

Wraps the Kokoro-82M model for fast, lightweight text-to-speech.
82M parameters, CPU realtime, 24kHz output, Apache 2.0 license.

Kokoro uses pre-built voice style vectors (not traditional zero-shot cloning
from arbitrary audio). Voice prompts are stored as deferred references to
HF-hosted voice .pt files.

Languages supported (via misaki G2P):
  - American English (a), British English (b)
  - Spanish (e), French (f), Hindi (h), Italian (i), Portuguese (p)
  - Japanese (j) — requires misaki[ja]
  - Chinese (z) — requires misaki[zh]
"""

import asyncio
import logging
import os
from typing import Optional

import numpy as np

from . import TTSBackend
from .base import (
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# HuggingFace repo for model + voice detection
KOKORO_HF_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000

# Default voice if none specified
KOKORO_DEFAULT_VOICE = "af_heart"

# All available Kokoro voices: (voice_id, display_name, gender, lang_code)
KOKORO_VOICES = [
    # American English female
    ("af_alloy", "Alloy", "female", "en"),
    ("af_aoede", "Aoede", "female", "en"),
    ("af_bella", "Bella", "female", "en"),
    ("af_heart", "Heart", "female", "en"),
    ("af_jessica", "Jessica", "female", "en"),
    ("af_kore", "Kore", "female", "en"),
    ("af_nicole", "Nicole", "female", "en"),
    ("af_nova", "Nova", "female", "en"),
    ("af_river", "River", "female", "en"),
    ("af_sarah", "Sarah", "female", "en"),
    ("af_sky", "Sky", "female", "en"),
    # American English male
    ("am_adam", "Adam", "male", "en"),
    ("am_echo", "Echo", "male", "en"),
    ("am_eric", "Eric", "male", "en"),
    ("am_fenrir", "Fenrir", "male", "en"),
    ("am_liam", "Liam", "male", "en"),
    ("am_michael", "Michael", "male", "en"),
    ("am_onyx", "Onyx", "male", "en"),
    ("am_puck", "Puck", "male", "en"),
    ("am_santa", "Santa", "male", "en"),
    # British English female
    ("bf_alice", "Alice", "female", "en"),
    ("bf_emma", "Emma", "female", "en"),
    ("bf_isabella", "Isabella", "female", "en"),
    ("bf_lily", "Lily", "female", "en"),
    # British English male
    ("bm_daniel", "Daniel", "male", "en"),
    ("bm_fable", "Fable", "male", "en"),
    ("bm_george", "George", "male", "en"),
    ("bm_lewis", "Lewis", "male", "en"),
    # Spanish
    ("ef_dora", "Dora", "female", "es"),
    ("em_alex", "Alex", "male", "es"),
    ("em_santa", "Santa", "male", "es"),
    # French
    ("ff_siwis", "Siwis", "female", "fr"),
    # Hindi
    ("hf_alpha", "Alpha", "female", "hi"),
    ("hf_beta", "Beta", "female", "hi"),
    ("hm_omega", "Omega", "male", "hi"),
    ("hm_psi", "Psi", "male", "hi"),
    # Italian
    ("if_sara", "Sara", "female", "it"),
    ("im_nicola", "Nicola", "male", "it"),
    # Japanese
    ("jf_alpha", "Alpha", "female", "ja"),
    ("jf_gongitsune", "Gongitsune", "female", "ja"),
    ("jf_nezumi", "Nezumi", "female", "ja"),
    ("jf_tebukuro", "Tebukuro", "female", "ja"),
    ("jm_kumo", "Kumo", "male", "ja"),
    # Portuguese
    ("pf_dora", "Dora", "female", "pt"),
    ("pm_alex", "Alex", "male", "pt"),
    ("pm_santa", "Santa", "male", "pt"),
    # Chinese
    ("zf_xiaobei", "Xiaobei", "female", "zh"),
    ("zf_xiaoni", "Xiaoni", "female", "zh"),
    ("zf_xiaoxiao", "Xiaoxiao", "female", "zh"),
    ("zf_xiaoyi", "Xiaoyi", "female", "zh"),
]

# Map our ISO language codes to Kokoro lang_code characters
LANG_CODE_MAP = {
    "en": "a",  # American English
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}


class KokoroTTSBackend:
    """Kokoro-82M TTS backend — tiny, fast, CPU-friendly."""

    def __init__(self):
        self._model = None
        self._pipelines: dict = {}  # lang_code -> KPipeline
        self._device: Optional[str] = None
        self.model_size = "default"

    def _get_device(self) -> str:
        """Select device. Kokoro supports CUDA and CPU. MPS needs fallback env var."""
        device = get_torch_device(allow_mps=False)
        # Kokoro can use MPS but requires PYTORCH_ENABLE_MPS_FALLBACK=1
        # For now, skip MPS to avoid user confusion — CPU is already realtime
        return device

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model_path(self, model_size: str) -> str:
        return KOKORO_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if Kokoro model files are cached locally."""
        from .base import is_model_cached

        return is_model_cached(
            KOKORO_HF_REPO,
            required_files=["config.json", "kokoro-v1_0.pth"],
        )

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Kokoro model."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "kokoro"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            from kokoro import KModel

            device = self.device
            logger.info(f"Loading Kokoro-82M on {device}...")

            self._model = KModel(repo_id=KOKORO_HF_REPO).to(device).eval()

        logger.info("Kokoro-82M loaded successfully")

    def _get_pipeline(self, lang_code: str):
        """Get or create a KPipeline for the given language code."""
        kokoro_lang = LANG_CODE_MAP.get(lang_code, "a")

        if kokoro_lang not in self._pipelines:
            from kokoro import KPipeline

            # Create pipeline with our existing model (no redundant model loading)
            self._pipelines[kokoro_lang] = KPipeline(
                lang_code=kokoro_lang,
                repo_id=KOKORO_HF_REPO,
                model=self._model,
            )

        return self._pipelines[kokoro_lang]

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._pipelines.clear()

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Kokoro unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt for Kokoro.

        Kokoro doesn't do traditional voice cloning from arbitrary audio.
        When called for a cloned profile (fallback), uses the default voice.
        For preset profiles, the voice_prompt dict is built by the profile
        service and bypasses this method entirely.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": KOKORO_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """Combine voice prompts — uses base implementation for audio concatenation."""
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=KOKORO_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using Kokoro.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with kokoro_voice key
            language: Language code
            seed: Random seed for reproducibility
            instruct: Not supported by Kokoro (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        voice_name = voice_prompt.get("preset_voice_id") or voice_prompt.get("kokoro_voice") or KOKORO_DEFAULT_VOICE

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            pipeline = self._get_pipeline(language)

            # Generate all chunks and concatenate
            audio_chunks = []
            for result in pipeline(text, voice=voice_name, speed=1.0):
                if result.audio is not None:
                    chunk = result.audio
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.detach().cpu().numpy()
                    audio_chunks.append(chunk.squeeze())

            if not audio_chunks:
                # Return 1 second of silence as fallback
                return np.zeros(KOKORO_SAMPLE_RATE, dtype=np.float32), KOKORO_SAMPLE_RATE

            audio = np.concatenate(audio_chunks)
            return audio.astype(np.float32), KOKORO_SAMPLE_RATE

        return await asyncio.to_thread(_generate_sync)
