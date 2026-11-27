"""
Higgs Audio v3 TTS backend (Boson AI), MLX (Metal) only.

Qwen3-backed conversational TTS with zero-shot voice cloning, 100+ languages and inline
control tokens (``<|emotion:amusement|>``, ``<|sfx:laughter|>``...). It runs on the Apple
Silicon GPU through mlx-audio's ``higgs_audio_v3`` port, which loads the upstream bf16
checkpoint ``bosonai/higgs-tts-3-4b`` directly (no pre-converted mlx-community repo exists).

Weights are under the Boson Higgs Audio v3 research and non-commercial licence.
"""

import asyncio
import logging
from pathlib import Path

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
)
from .mlx_backend import _run_on_mlx_thread

logger = logging.getLogger(__name__)

HIGGS_HF_REPO = "bosonai/higgs-tts-3-4b"
_WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer.json"]


class HiggsMLXTTSBackend:
    """Higgs Audio v3 TTS backend for voice cloning, on MLX/Metal."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        # Guards the load-then-use sequence, as MLXTTSBackend._op_lock does.
        self._op_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return HIGGS_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(HIGGS_HF_REPO, required_files=_WEIGHT_FILES)

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Higgs Audio v3 model."""
        if self.model is not None:
            return
        async with self._op_lock:
            if self.model is not None:
                return
            # MLX streams are thread-local: every MLX call in the process
            # shares the single worker in mlx_backend (issue #699).
            await _run_on_mlx_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        is_cached = self._is_model_cached()

        with model_load_progress("higgs-audio-v3", is_cached):
            from huggingface_hub import snapshot_download  # lazy: heavy import
            from mlx_audio.tts.utils import load  # lazy: heavy import

            logger.info("Loading Higgs Audio v3 TTS on MLX (Metal)...")
            ckpt_dir = snapshot_download(HIGGS_HF_REPO, token=None)
            # mlx-audio 0.5.9 drops its higgs_multimodal_qwen3 -> higgs_audio_v3 remap when the
            # path is a local snapshot dir, so the architecture has to be named explicitly.
            self.model = load(ckpt_dir, model_type="higgs_audio_v3")

        logger.info("Higgs Audio v3 TTS (MLX) loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is None:
            return

        del self.model
        self.model = None
        try:
            import mlx.core as mx  # lazy: heavy import

            mx.clear_cache()
        except Exception:
            logger.debug("mlx cache not cleared", exc_info=True)
        logger.info("Higgs Audio v3 (MLX) unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Higgs encodes the reference audio at generation time, so the prompt just stores
        the file path and its transcript (both are required for cloning).
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
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Higgs Audio v3 on MLX.

        Args:
            text: Text to synthesize (may contain Higgs inline control tokens)
            voice_prompt: Dict with ref_audio path and ref_text
            language: Unused: Higgs infers the language from the text
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None
        if not ref_audio:
            ref_text = None

        def _generate_sync():
            # Load (if needed) and generate as ONE worker submission, as in
            # MLXTTSBackend._reload_and_generate_sync, so an unload cannot
            # land between the two; bind the model locally for the same reason.
            if self.model is None:
                self._load_model_sync()
            model = self.model

            logger.info("[Higgs/MLX] Generating (cloned=%s)", bool(ref_audio))

            # mlx-audio yields GenerationResult chunks; the whole clip is their concatenation.
            chunks = [
                np.asarray(result.audio).squeeze()
                for result in model.generate(
                    text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    seed=seed,
                    verbose=False,
                )
            ]
            audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
            return audio, int(model.sample_rate)

        async with self._op_lock:
            return await _run_on_mlx_thread(_generate_sync)
