"""
OpenVoice V2 backend — multi-lingual voice cloning via ToneColorConverter.

Architecture:
  The OpenVoice V2 pipeline separates TTS from voice conversion.
  This backend uses Kokoro (lightweight, always available) as the base TTS,
  then applies OpenVoice's ToneColorConverter to transfer the target
  voice's timbre onto the generated audio.

  Source style embeddings for each language are pre-extracted and stored
  on disk — no base-speaker TTS model needed.

Pipeline (generate):
  1. Generate base audio via Kokoro TTS (lightweight, CPU-friendly)
  2. Load source style embedding for the target language
  3. Load target style embedding (from reference audio via create_voice_prompt)
  4. ToneColorConverter.convert(source_audio, src_se, tgt_se) → cloned audio

Languages supported: EN, ES, FR, ZH, JP, KR
"""

import asyncio
import logging
import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch

from . import TTSBackend
from .base import (
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────

MODELS_DIR = "/mnt/480ssd/voice-models/openvoicev2"
OPENVOICE_REPO = "/mnt/480ssd/voice-models/openvoice-repo"
BASE_SPEAKERS_SES = os.path.join(MODELS_DIR, "base_speakers", "ses")
CONVERTER_DIR = os.path.join(MODELS_DIR, "converter")

# Language → base speaker filename (style embedding)
LANG_SOURCE_SE = {
    "ES": "es",
    "EN": "en-default",
    "EN-US": "en-us",
    "EN-UK": "en-newest",
    "EN-IN": "en-india",
    "EN-AU": "en-au",
    "FR": "fr",
    "ZH": "zh",
    "JP": "jp",
    "KR": "kr",
}

# Voicebox language codes → OpenVoice source key
LANG_CODE_MAP = {
    "es": "ES",
    "en": "EN",
    "en-us": "EN-US",
    "en-uk": "EN-UK",
    "en-in": "EN-IN",
    "fr": "FR",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

SUPPORTED_LANGUAGES = list(LANG_CODE_MAP.keys())

OPENVOICE_SAMPLE_RATE = 22050


class OpenVoiceBackend:
    """OpenVoice V2 backend — multi-lingual voice cloning."""

    def __init__(self):
        self._tone_color_converter = None
        self._device = None
        self._model_load_lock = asyncio.Lock()
        self.model_size = "default"

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        return self._tone_color_converter is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return "myshell-ai/OpenVoiceV2"

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if ToneColorConverter checkpoint exists."""
        ckpt = os.path.join(CONVERTER_DIR, "checkpoint.pth")
        cfg = os.path.join(CONVERTER_DIR, "config.json")
        return os.path.isfile(ckpt) and os.path.isfile(cfg) and os.path.isdir(BASE_SPEAKERS_SES)

    async def load_model(self, model_size: str = "default") -> None:
        """Load the ToneColorConverter model."""
        if self._tone_color_converter is not None:
            return
        async with self._model_load_lock:
            if self._tone_color_converter is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "openvoice-v2"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info("Loading OpenVoice V2 ToneColorConverter on %s...", device)

            # Add openvoice repo to sys.path if not already there
            if os.path.exists(OPENVOICE_REPO) and OPENVOICE_REPO not in sys.path:
                sys.path.insert(0, OPENVOICE_REPO)

            from openvoice.api import ToneColorConverter

            config_path = os.path.join(CONVERTER_DIR, "config.json")
            ckpt_path = os.path.join(CONVERTER_DIR, "checkpoint.pth")

            self._tone_color_converter = ToneColorConverter(
                config_path, device=device
            )
            self._tone_color_converter.load_ckpt(ckpt_path)
            # Disable watermark to avoid wavmark dependency
            self._tone_color_converter.watermark_model = None
            self._tone_color_converter.add_watermark = lambda audio, msg: audio

        logger.info("OpenVoice V2 loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._tone_color_converter is not None:
            device = self._device
            del self._tone_color_converter
            self._tone_color_converter = None
            self._device = None
            empty_device_cache(device)
            logger.info("OpenVoice V2 unloaded")

    def _load_source_se(self, lang_key: str) -> Optional[torch.Tensor]:
        """Load pre-extracted source style embedding for a language."""
        speaker_file = LANG_SOURCE_SE.get(lang_key, "en-default")
        path = os.path.join(BASE_SPEAKERS_SES, f"{speaker_file}.pth")
        if os.path.exists(path):
            return torch.load(path, map_location=self._device)
        logger.warning("Source SE not found for %s, falling back to en-default", lang_key)
        fallback = os.path.join(BASE_SPEAKERS_SES, "en-default.pth")
        if os.path.exists(fallback):
            return torch.load(fallback, map_location=self._device)
        return None

    async def _generate_base_audio(
        self, text: str, language: str, seed: Optional[int] = None
    ) -> tuple[np.ndarray, int]:
        """
        Generate base audio using Kokoro TTS.

        Kokoro is used because:
        - It's lightweight (82M params, always cached)
        - It generates directly from text + preset voice (no voice prompt needed)
        - It supports multiple languages including Spanish
        """
        from ..backends import get_tts_backend_for_engine

        kokoro = get_tts_backend_for_engine("kokoro")
        await kokoro.load_model()

        # Use a neutral voice for the base audio
        # Kokoro voices: american female "af_heart", american male "am_liam"
        # spanish: "ef_dora" (female), "em_alex" (male)
        lang_to_voice = {
            "es": "ef_dora",
            "en": "af_heart",
            "fr": "ff_siwis",
            "zh": "zf_xiaoxiao",
            "ja": "jf_alpha",
            "ko": "jf_alpha",  # Kokoro doesn't have Korean; fallback to Japanese
        }

        voice_id = lang_to_voice.get(language, "af_heart")
        kokoro_voice_prompt = {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": voice_id,
        }

        audio, sr = await kokoro.generate(
            text, kokoro_voice_prompt, language=language, seed=seed
        )
        return audio, sr

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Extract style embedding from reference audio for voice cloning.

        The embedding is stored in the voice_prompt dict for later use
        during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript (not used by OpenVoice, kept for protocol)
            use_cache: Whether to reuse cached embeddings

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model()

        def _extract_se():
            from openvoice import se_extractor

            return se_extractor.get_se(
                audio_path,
                self._tone_color_converter,
                target_dir=os.path.join(MODELS_DIR, "processed_se"),
                vad=True,
            )

        try:
            target_se, audio_name = await asyncio.to_thread(_extract_se)
        except Exception as e:
            logger.warning("SE extraction failed with VAD, trying without: %s", e)
            try:
                def _extract_se_no_vad():
                    from openvoice import se_extractor
                    return se_extractor.get_se(
                        audio_path,
                        self._tone_color_converter,
                        target_dir=os.path.join(MODELS_DIR, "processed_se"),
                        vad=False,
                    )
                target_se, audio_name = await asyncio.to_thread(_extract_se_no_vad)
            except Exception as e2:
                logger.error("SE extraction completely failed: %s", e2)
                raise RuntimeError(f"Failed to extract voice embedding: {e2}") from e2

        voice_prompt = {"target_se": target_se}
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """Combine multiple reference voice prompts."""
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=OPENVOICE_SAMPLE_RATE
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
        Generate voice-cloned audio using OpenVoice V2 pipeline.

        Pipeline:
        1. Generate base audio via Kokoro TTS (lightweight, CPU-friendly)
        2. Load source SE for the target language
        3. Run ToneColorConverter to transfer voice timbre

        Args:
            text: Text to synthesize
            voice_prompt: Dict with target_se from create_voice_prompt.
                         If target_se is absent, falls back to base speaker.
            language: Language code (en, es, fr, zh, ja, ko)
            seed: Random seed for reproducibility
            instruct: Not supported by OpenVoice (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Map language code
        lang_key = LANG_CODE_MAP.get(language)
        if lang_key is None:
            for vbox_code, ov_key in LANG_CODE_MAP.items():
                if language.startswith(vbox_code):
                    lang_key = ov_key
                    break
        if lang_key is None:
            logger.warning("Unsupported language %s, falling back to EN", language)
            lang_key = "EN"

        await self.load_model()

        # ── Step 1: Generate base audio with Kokoro TTS ──
        logger.info("[OpenVoice] Generating base TTS audio via Kokoro...")
        base_audio, base_sr = await self._generate_base_audio(text, language, seed)
        logger.info(
            "[OpenVoice] Base TTS generated: %d samples @ %dHz",
            len(base_audio),
            base_sr,
        )

        # ── Step 2 & 3 & 4: sync conversion ──
        def _convert_sync():
            import soundfile as sf

            if seed is not None:
                manual_seed(seed, self._device)

            tmp_path = tempfile.mktemp(suffix=".wav")
            out_path = tempfile.mktemp(suffix=".wav")

            try:
                sf.write(tmp_path, base_audio, base_sr)

                # Source SE
                source_se = self._load_source_se(lang_key)
                if source_se is None:
                    raise RuntimeError(
                        f"No source style embedding for {lang_key}. "
                        f"Available: {list(LANG_SOURCE_SE.keys())}"
                    )

                # Target SE — if no reference provided, use source = base speaker
                target_se = voice_prompt.get("target_se")
                if target_se is None:
                    logger.info("[OpenVoice] No reference voice, using base speaker")
                    target_se = source_se

                # Convert
                logger.info("[OpenVoice] Running ToneColorConverter...")
                self._tone_color_converter.convert(
                    audio_src_path=tmp_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=out_path,
                    tau=0.3,
                    message="voicebox-openvoice",
                )

                # Read result
                out_audio, out_sr = sf.read(out_path)
                out_audio = out_audio.astype(np.float32)

                logger.info(
                    "[OpenVoice] Voice clone done: %d samples @ %dHz",
                    len(out_audio),
                    out_sr,
                )
                return out_audio, out_sr

            finally:
                for p in [tmp_path, out_path]:
                    if os.path.exists(p):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

        return await asyncio.to_thread(_convert_sync)
