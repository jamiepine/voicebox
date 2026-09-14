"""VoxCPM2 TTS backend implementation.

Wraps the native ``voxcpm`` package for local reference-audio cloning.  The
backend deliberately disables VoxCPM's optional denoiser so Voicebox can keep
the dependency surface focused on synthesis; reference audio is already
validated and normalized by the profile pipeline.
"""

import asyncio
import inspect
import logging
from pathlib import Path

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

VOXCPM_HF_REPO = "openbmb/VoxCPM2"
VOXCPM_MODEL_FILES = ("config.json", "model.safetensors", "audiovae.pth")


class VoxCPMBackend:
    """Native VoxCPM2 backend for cloned voice profiles."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device: str | None = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # Current VoxCPM2 runs on MPS, but a smoke test showed it substantially
        # slower than CPU on Apple Silicon and upstream has had MPS-specific
        # AudioVAE failures. Match Voicebox's Chatterbox/TADA policy: use CUDA
        # where available and a predictable CPU fallback on macOS.
        return get_torch_device(force_cpu_on_mac=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return VOXCPM_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(VOXCPM_HF_REPO, required_files=list(VOXCPM_MODEL_FILES))

    async def load_model(self, model_size: str = "default") -> None:
        """Download and load VoxCPM2 lazily on the selected device."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    async def load_model_async(self, model_size: str = "default") -> None:
        """Compatibility alias used by a few model-management call sites."""
        await self.load_model(model_size)

    def _load_model_sync(self) -> None:
        model_name = "voxcpm2"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            from voxcpm import VoxCPM

            device = self.device
            logger.info("Loading VoxCPM2 on %s...", device)
            self.model = VoxCPM.from_pretrained(
                VOXCPM_HF_REPO,
                load_denoiser=False,
                optimize=device == "cuda",
                device=device,
            )

        logger.info("VoxCPM2 loaded successfully on %s", self.device)

    def unload_model(self) -> None:
        """Unload VoxCPM2 and release accelerator cache."""
        if self.model is None:
            return
        device = self._device
        del self.model
        self.model = None
        self._device = None
        empty_device_cache(device)
        logger.info("VoxCPM2 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """Store reference paths; VoxCPM2 builds its prompt at generation time."""
        del use_cache
        return {
            "ref_audio": str(audio_path),
            "ref_text": str(reference_text or "").strip(),
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        # Voicebox's profile service persists combined prompts at 24 kHz.  Keep
        # that convention; VoxCPM2 resamples references to its 16 kHz encoder
        # rate internally.
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate 48 kHz speech with reference-audio voice cloning."""
        del language
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = str(voice_prompt.get("ref_text") or "").strip()
        if ref_audio and not Path(ref_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        def _generate_sync():
            if seed is not None:
                manual_seed(seed, self.device)

            kwargs = {}
            if ref_audio:
                # With delivery instructions, use controllable cloning so the
                # parenthesized style prefix can take effect. Without an
                # instruction, Voicebox's accurate reference transcript enables
                # VoxCPM2's highest-fidelity continuation/cloning mode.
                kwargs["reference_wav_path"] = str(ref_audio)
                if ref_text and not (instruct and instruct.strip()):
                    kwargs["prompt_wav_path"] = str(ref_audio)
                    kwargs["prompt_text"] = ref_text

            generation_text = text
            if instruct and instruct.strip():
                generation_text = f"({instruct.strip()}){text}"

            generation_kwargs = {
                "text": generation_text,
                "retry_badcase": True,
                **kwargs,
            }
            # PyPI 2.0.3 predates the explicit seed parameter now available on
            # VoxCPM main. Voicebox still seeds torch above, and forwards the
            # argument automatically once a newer VoxCPM release exposes it.
            try:
                generate_parameters = inspect.signature(
                    getattr(self.model, "_generate", self.model.generate)
                ).parameters
            except (TypeError, ValueError):
                generate_parameters = {}
            if seed is not None and "seed" in generate_parameters:
                generation_kwargs["seed"] = seed

            wav = self.model.generate(**generation_kwargs)
            audio = np.asarray(wav, dtype=np.float32).squeeze()
            if audio.ndim != 1 or audio.size == 0:
                raise ValueError("VoxCPM2 returned empty audio")
            sample_rate = int(getattr(getattr(self.model, "tts_model", None), "sample_rate", 48000))
            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
