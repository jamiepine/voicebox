"""
OmniVoice backend implementation.

Wraps k2-fsa/OmniVoice, a diffusion-language-model zero-shot TTS covering 600+
languages. ~3.3 GB of weights, 24 kHz output, RTF down to 0.025 on CUDA.

Two things set this engine apart from the rest of the roster:

  - It needs ``transformers.HiggsAudioV2TokenizerModel`` as its audio codec,
    which only exists in transformers >= 5.3 while Voicebox is pinned to
    <= 4.57.6. ``backend/utils/transformers5_compat.py`` grafts a vendored
    copy onto the transformers namespace; it must run before ``omnivoice`` is
    imported.
  - Its ``instruct`` parameter is a closed attribute vocabulary (gender, age,
    pitch, whisper, English accent, Chinese dialect), not the free-form prose
    Qwen CustomVoice accepts. Anything outside it raises upstream.

Languages: the 22 codes Voicebox already knows that OmniVoice also covers,
plus Arabic via the ISO 639-3 code ``arb``. OmniVoice supports far more, but
exposing them means widening the language enum across the API and frontend.
"""

import asyncio
import gc
import logging
import sys
from typing import Any, Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)
from ..utils.cache import cache_voice_prompt, get_cache_key, get_cached_voice_prompt

logger = logging.getLogger(__name__)

OMNIVOICE_HF_REPO = "k2-fsa/OmniVoice"

# OmniVoice speaks ISO 639-3. Most of Voicebox's two-letter codes are valid
# 639-3 codes too and pass straight through; Arabic is the exception, because
# OmniVoice enumerates the individual varieties rather than the "ar"
# macrolanguage. "arb" is Modern Standard Arabic.
LANGUAGE_CODE_OVERRIDES = {"ar": "arb"}

# Output sample rate, from the audio tokenizer's feature extractor. Read off
# the loaded model when available; this is the fallback.
DEFAULT_SAMPLE_RATE = 24000

# Diffusion steps. Upstream defaults to 32; 16 roughly halves latency at some
# quality cost.
NUM_DIFFUSION_STEPS = 32

_VOICE_CLONE_PROMPT_FORMAT_VERSION = 1


def _clear_codec_layer_cache() -> None:
    """Release the Higgs codec's per-instance layer cache.

    ``HiggsAudioV2TokenizerPreTrainedModel._get_conv1d_layers`` carries an
    ``@lru_cache`` on an instance method. The cache key includes ``self``, and
    the cache lives on the function object — which lives on the class — so the
    first call pins the tokenizer instance for the lifetime of the process.
    Encoding a voice prompt triggers exactly one call, after which unloading
    the engine strands ~0.8 GB of codec weights on the device.

    Upstream bug, vendored verbatim from transformers rather than patched, so
    a re-sync does not silently drop the fix. Clearing here costs one recompute
    on the next load.
    """
    module = sys.modules.get(
        "backend.vendor.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer"
    )
    if module is None:
        return

    cached_method = getattr(
        getattr(module, "HiggsAudioV2TokenizerPreTrainedModel", None),
        "_get_conv1d_layers",
        None,
    )
    cache_clear = getattr(cached_method, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


class OmniVoiceBackend:
    """OmniVoice backend for zero-shot voice cloning and voice design."""

    def __init__(self):
        self.model = None
        self.model_size = "default"  # OmniVoice ships a single checkpoint
        self._device = None
        self._asr_loaded = False

    def _get_device(self) -> str:
        # MPS is allowed: from_pretrained keeps the audio tokenizer on CPU
        # there by itself, because the codec exceeds the 65536 output-channel
        # limit MPS imposes.
        return get_torch_device(allow_mps=True, allow_xpu=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        return OMNIVOICE_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(OMNIVOICE_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return

        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        model_name = "omnivoice"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            import torch

            # Must precede the omnivoice import: omnivoice does
            # `from transformers import HiggsAudioV2TokenizerModel` at module
            # scope, and on transformers 4.57.x that name does not exist.
            from ..utils.transformers5_compat import install as install_transformers5_compat

            install_transformers5_compat()

            from omnivoice import OmniVoice

            device = self.device
            # fp16 on accelerators, fp32 on CPU and MPS. MPS fp16 underflows
            # in the diffusion loop; CPU fp16 is slower than fp32.
            dtype = (
                torch.float16
                if device in ("cuda", "xpu")
                else torch.float32
            )

            logger.info("Loading OmniVoice on %s (%s)...", device, dtype)

            self.model = OmniVoice.from_pretrained(
                OMNIVOICE_HF_REPO,
                device_map=device,
                dtype=dtype,
            )

        logger.info("OmniVoice loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self.device
            del self.model
            self.model = None
            self._device = None
            self._asr_loaded = False

            _clear_codec_layer_cache()
            gc.collect()
            empty_device_cache(device)

            logger.info("OmniVoice unloaded")

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model, "sampling_rate", DEFAULT_SAMPLE_RATE))

    def _ensure_asr(self) -> None:
        """Load OmniVoice's Whisper for auto-transcribing a reference clip.

        Only needed when a profile carries no reference text. The default is
        openai/whisper-large-v3-turbo, the same repo behind Voicebox's own
        "turbo" STT model, so it is often already in the HF cache.
        """
        if self._asr_loaded:
            return
        logger.info("Loading OmniVoice ASR for reference transcription...")
        self.model.load_asr_model()
        self._asr_loaded = True

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Encode reference audio into a reusable voice clone prompt.

        Returns the fields of OmniVoice's ``VoiceClonePrompt`` as a plain dict
        rather than the dataclass itself: the on-disk cache reads back with
        ``torch.load(weights_only=True)``, which refuses arbitrary classes.
        The layout matches ``VoiceClonePrompt.save()`` exactly.
        """
        await self.load_model()

        cache_key = (
            ("omnivoice_" + get_cache_key(audio_path, reference_text)) if use_cache else None
        )

        if cache_key:
            cached = get_cached_voice_prompt(cache_key)
            if isinstance(cached, dict) and "ref_audio_tokens" in cached:
                return cached, True

        def _encode_sync() -> dict:
            ref_text = reference_text or None
            if ref_text is None:
                self._ensure_asr()

            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=ref_text,
            )
            return {
                "format_version": _VOICE_CLONE_PROMPT_FORMAT_VERSION,
                "ref_audio_tokens": prompt.ref_audio_tokens.detach().cpu(),
                "ref_text": prompt.ref_text,
                "ref_rms": float(prompt.ref_rms),
            }

        encoded = await asyncio.to_thread(_encode_sync)

        if cache_key:
            cache_voice_prompt(cache_key, encoded)

        return encoded, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=DEFAULT_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using OmniVoice.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict produced by create_voice_prompt(). Empty or
                missing tokens fall back to voice design / auto voice.
            language: Voicebox two-letter code, translated to ISO 639-3.
            seed: Random seed for reproducibility.
            instruct: Voice design attributes, comma-separated — e.g.
                "female, low pitch, british accent". This is a closed
                vocabulary, not free-form prose.

        Returns:
            Tuple of (audio_array, sample_rate).

        Raises:
            ValueError: if instruct contains attributes OmniVoice does not
                recognise. The upstream message names the offending items.
        """
        await self.load_model()

        clone_prompt = self._rebuild_clone_prompt(voice_prompt)
        resolved_language = LANGUAGE_CODE_OVERRIDES.get(language, language)

        def _generate_sync() -> Tuple[np.ndarray, int]:
            if seed is not None:
                manual_seed(seed, self.device)

            kwargs: dict[str, Any] = {
                "text": text,
                "language": resolved_language,
                "num_step": NUM_DIFFUSION_STEPS,
            }
            if clone_prompt is not None:
                kwargs["voice_clone_prompt"] = clone_prompt
            if instruct:
                kwargs["instruct"] = instruct

            try:
                audios = self.model.generate(**kwargs)
            except ValueError as exc:
                if instruct and "instruct" in str(exc).lower():
                    raise ValueError(
                        "OmniVoice voice design accepts a fixed set of "
                        "attributes (gender, age, pitch, whisper, English "
                        f"accent, Chinese dialect), not free text. {exc}"
                    ) from exc
                raise

            return audios[0], self.sample_rate

        return await asyncio.to_thread(_generate_sync)

    def _rebuild_clone_prompt(self, voice_prompt: dict):
        """Turn a cached prompt dict back into OmniVoice's dataclass.

        Returns None when the profile carries no cloned reference, which puts
        generation into voice-design or auto-voice mode.
        """
        if not voice_prompt or "ref_audio_tokens" not in voice_prompt:
            return None

        from omnivoice import VoiceClonePrompt

        version = voice_prompt.get("format_version")
        if version != _VOICE_CLONE_PROMPT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported OmniVoice voice prompt format version: {version}"
            )

        return VoiceClonePrompt(
            ref_audio_tokens=voice_prompt["ref_audio_tokens"],
            ref_text=voice_prompt["ref_text"],
            ref_rms=voice_prompt["ref_rms"],
        )
