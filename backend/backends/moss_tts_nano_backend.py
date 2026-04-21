"""
MOSS-TTS-Nano backend implementation.

Wraps NanoTTSService from moss-tts-nano for zero-shot voice cloning.
0.1B parameters, 48 kHz stereo output, 20 languages, CPU-friendly.
WeTextProcessing is optional (Chinese text normalization); if absent,
the tts_robust_normalizer_single_script fallback is used instead.

Concurrency model
-----------------
``_service_lock``       - guards reads/writes of ``_service`` and ``_device``,
                          and serialises ``_load_model_sync``.
``_active_generations`` - count of synthesis calls currently in flight.
``_unloading``          - flag set by ``unload_model`` to close the generation
                          gate before waiting for in-flight calls to drain.
``_idle``               - Condition (backed by ``_service_lock``) that
                          ``unload_model`` waits on until the count reaches zero.

Concurrent generations run without holding ``_service_lock`` for the duration
of the actual ``synthesize()`` call, so throughput is not artificially capped.
``unload_model`` sets ``_unloading`` first, then waits for the counter to reach
zero, preventing new generations from starting after teardown begins.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

MOSS_TTS_NANO_HF_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
MOSS_AUDIO_TOKENIZER_HF_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"

_REQUIRED_WEIGHT_FILES = ["model.safetensors"]


class MOSSTTSNanoBackend:
    """MOSS-TTS-Nano backend - 0.1B, 48 kHz stereo, 20 languages, CPU-friendly."""

    def __init__(self):
        """Initialise locks, counters, and service state; no model is loaded yet."""
        self._service = None
        self._device = None
        self._model_load_lock = asyncio.Lock()
        self._service_lock = threading.RLock()
        self._active_generations: int = 0
        self._unloading: bool = False
        self._idle = threading.Condition(self._service_lock)

    def _get_device(self) -> str:
        """Return the best available torch device (CUDA > MPS > XPU > CPU)."""
        return get_torch_device(allow_mps=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        """Return True if the NanoTTSService has been initialised."""
        return self._service is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        """Return the HuggingFace repo ID used as the checkpoint path."""
        return MOSS_TTS_NANO_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """
        Return True only when both the model checkpoint and the audio
        tokenizer are present in the local HuggingFace cache.
        """
        return is_model_cached(
            MOSS_TTS_NANO_HF_REPO,
            required_files=_REQUIRED_WEIGHT_FILES,
        ) and is_model_cached(MOSS_AUDIO_TOKENIZER_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        """
        Async entry-point for model loading.

        Uses an asyncio.Lock to serialise concurrent load requests so that
        only one worker thread initialises the service.

        Raises:
            RuntimeError: If ``unload_model`` is currently draining in-flight
                generations; callers should retry after the unload completes.
        """
        with self._idle:
            if self._unloading:
                raise RuntimeError(
                    "MOSS-TTS-Nano is currently unloading; retry after unload completes"
                )
            if self._service is not None:
                return
        async with self._model_load_lock:
            with self._idle:
                if self._unloading:
                    raise RuntimeError(
                        "MOSS-TTS-Nano is currently unloading; retry after unload completes"
                    )
                if self._service is not None:
                    return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading, protected by _service_lock."""
        model_name = "moss-tts-nano"
        is_cached = self._is_model_cached()

        with self._service_lock:
            if self._service is not None:
                return

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
        """
        Unload the model and free device memory.

        Sets ``_unloading`` to close the generation gate, then waits for all
        in-flight ``synthesize`` calls to finish before tearing down
        ``_service``, so that no new generations can start and no running
        synthesis is interrupted mid-call.
        """
        with self._idle:
            if self._service is None:
                return
            self._unloading = True
            try:
                self._idle.wait_for(lambda: self._active_generations == 0)
                device = self._device
                self._service = None
                self._device = None
            finally:
                self._unloading = False
                self._idle.notify_all()

        empty_device_cache(device)
        logger.info("MOSS-TTS-Nano unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
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
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """
        Merge multiple reference audio clips into a single voice prompt.

        Delegates to the shared base helper so that all backends handle
        multi-sample prompts consistently.
        """
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
        Generate audio using MOSS-TTS-Nano voice cloning.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict with ref_audio path and optional ref_text.
            language: BCP-47 language code (ignored at inference; language
                      is inferred from the model's multilingual tokenizer).
            seed: Optional random seed passed directly to synthesize().
            instruct: Unused (protocol compatibility).

        Returns:
            Tuple of (mono_audio_float32, sample_rate).

        Raises:
            FileNotFoundError: If ref_audio is provided but the file is missing.
            RuntimeError: If the service is unloaded or unloading.
        """
        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            raise FileNotFoundError(
                f"MOSS-TTS-Nano reference audio not found: {ref_audio}"
            )

        await self.load_model()

        def _generate_sync() -> tuple[np.ndarray, int]:
            """
            Run synthesis on a thread-pool worker.

            Checks ``_unloading`` and increments ``_active_generations`` under
            ``_idle``, then releases the lock and calls ``synthesize`` without
            holding it (so concurrent requests run in parallel). Re-acquires
            the lock to decrement the counter and notify any waiting
            ``unload_model`` call.
            """
            with self._idle:
                service = self._service
                device = self._device
                if service is None or self._unloading:
                    raise RuntimeError("MOSS-TTS-Nano service is not loaded")
                self._active_generations += 1

            try:
                logger.info(f"[MOSS-TTS-Nano] Generating: lang={language}")

                # synthesize() runs WITHOUT holding _service_lock so that
                # concurrent requests are not serialised. seed is forwarded
                # to the service rather than calling process-global manual_seed,
                # which would corrupt RNG state of concurrent generations.
                result = service.synthesize(
                    text=text,
                    prompt_audio_path=ref_audio,
                    prompt_text=ref_text,
                    mode="voice_clone" if ref_audio else "continuation",
                    seed=seed,
                )
            finally:
                with self._idle:
                    self._active_generations -= 1
                    self._idle.notify_all()

            waveform: np.ndarray = result["waveform_numpy"]
            sample_rate: int = int(result["sample_rate"])

            # Convert stereo (samples, 2) -> mono (samples,)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)

            audio = waveform.astype(np.float32)
            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
