"""
MMS-TTS backend implementation.

Wraps Meta's Massively Multilingual Speech (MMS) TTS checkpoints — one VITS
model per language (``facebook/mms-tts-{iso3}``). Pure PyTorch via
``transformers``, CPU realtime, 16kHz output, CC-BY-NC 4.0 license.

MMS has no concept of voice cloning: each checkpoint is a single preset
speaker, so profiles follow the preset-voice pattern (like Kokoro).

Languages supported:
  - Romanian (ro) — ``facebook/mms-tts-ron``

Adding a language requires an entry in ``MMS_HF_REPOS`` and ``MMS_VOICES``,
a ``ModelConfig`` registration, and threading the requested language through
``_get_model_path``/``_load_model_sync`` — today those resolve to
``MMS_DEFAULT_LANGUAGE`` because only one checkpoint ships.
"""

import asyncio
import logging
import unicodedata

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# Our ISO 639-1 language codes -> MMS per-language HF checkpoints (ISO 639-3)
MMS_HF_REPOS = {"ro": "facebook/mms-tts-ron"}
MMS_DEFAULT_LANGUAGE = "ro"

# Confirmed against model.config.sampling_rate for mms-tts-ron
MMS_SAMPLE_RATE = 16000

# Default voice if none specified
MMS_DEFAULT_VOICE = "mms_ro_default"

# All available MMS voices: (voice_id, display_name, gender, lang_code).
# One entry per language checkpoint — MMS ships a single speaker each.
MMS_VOICES = [
    ("mms_ro_default", "Romanian (MMS)", "male", "ro"),
]

# The mms-tts-ron vocab mixes Romanian diacritic conventions: it contains
# comma-below ș (U+0219) but cedilla ţ (U+0163), and lacks their
# counterparts ş (U+015F) and ț (U+021B). The character-level VitsTokenizer
# silently drops characters outside its vocab, so both real-world variants
# must be mapped onto the trained form. Uppercase variants are included —
# the tokenizer lowercases after this mapping (Ș -> ș, Ţ -> ţ).
_RO_DIACRITICS_TRANSLATION = str.maketrans(
    {
        "ş": "ș",  # ş (s-cedilla) -> ș (s-comma-below)
        "Ş": "Ș",  # Ş (S-cedilla) -> Ș (S-comma-below)
        "ț": "ţ",  # ț (t-comma-below) -> ţ (t-cedilla)
        "Ț": "Ţ",  # Ț (T-comma-below) -> Ţ (T-cedilla)
    }
)


def normalize_romanian_text(text: str) -> str:
    """Normalize Romanian text to the diacritic forms in the MMS vocab.

    Applies NFC normalization first (composing any decomposed
    letter + combining-mark sequences), then maps cedilla/comma-below
    s and t variants onto the forms the ``facebook/mms-tts-ron``
    checkpoint was trained on, so no diacritic is silently dropped
    by the tokenizer.
    """
    return unicodedata.normalize("NFC", text).translate(_RO_DIACRITICS_TRANSLATION)


class MMSTTSBackend:
    """Meta MMS-TTS backend — per-language VITS checkpoints, single preset voice."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        self.model_size = "default"

    @property
    def device(self) -> str:
        if self._device is None:
            # CPU is realtime for a ~100M-param VITS; skip MPS like Kokoro.
            self._device = get_torch_device(allow_mps=False)
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model_path(self, model_size: str) -> str:
        return MMS_HF_REPOS[MMS_DEFAULT_LANGUAGE]

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if MMS model files are cached locally."""
        return is_model_cached(MMS_HF_REPOS[MMS_DEFAULT_LANGUAGE])

    async def load_model(self, model_size: str = "default") -> None:
        """Load the MMS model and tokenizer."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "mms-tts-ron"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            from transformers import AutoTokenizer, VitsModel  # lazy: heavy import

            repo = MMS_HF_REPOS[MMS_DEFAULT_LANGUAGE]
            device = self.device
            logger.info("Loading MMS-TTS (%s) on %s...", repo, device)

            self._tokenizer = AutoTokenizer.from_pretrained(repo)
            self._model = VitsModel.from_pretrained(repo).to(device).eval()

        logger.info("MMS-TTS loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            empty_device_cache(self.device)
            logger.info("MMS-TTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt for MMS.

        MMS doesn't do voice cloning — each checkpoint is one fixed speaker.
        When called for a cloned profile (fallback), uses the default voice.
        For preset profiles, the voice_prompt dict is built by the profile
        service and bypasses this method entirely.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "mms",
            "preset_voice_id": MMS_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """Combine voice prompts — uses base implementation for audio concatenation."""
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=MMS_SAMPLE_RATE)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "ro",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using MMS-TTS.

        Args:
            text: Text to synthesize
            voice_prompt: Preset voice dict (single speaker — informational only)
            language: Language code
            seed: Random seed for reproducibility (VITS sampling is stochastic)
            instruct: Not supported by MMS (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        def _generate_sync():
            import torch  # lazy: heavy import

            if seed is not None:
                manual_seed(seed, self.device)

            normalized = normalize_romanian_text(text) if language == "ro" else unicodedata.normalize("NFC", text)

            inputs = self._tokenizer(normalized, return_tensors="pt").to(self.device)
            sample_rate = getattr(self._model.config, "sampling_rate", MMS_SAMPLE_RATE)

            # Text made entirely of out-of-vocab characters tokenizes to an
            # empty sequence — return 1 second of silence as fallback.
            if inputs["input_ids"].shape[-1] == 0:
                return np.zeros(sample_rate, dtype=np.float32), sample_rate

            with torch.no_grad():
                waveform = self._model(**inputs).waveform

            audio = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
