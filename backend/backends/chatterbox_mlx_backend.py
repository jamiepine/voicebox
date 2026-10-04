"""
Chatterbox multilingual TTS backend, MLX (Metal) flavour.

Same zero-shot voice cloning and same 23 languages as ``chatterbox_backend``, but running
on the Apple Silicon GPU through mlx-audio instead of PyTorch on the CPU.

The PyTorch path is pinned to the CPU on macOS (see ``chatterbox_backend``), which costs
roughly 4x realtime. This backend uses the pre-converted weights published as
``mlx-community/chatterbox-multilingual-v3`` and renders the same sentences at about 0.5x
realtime on an M4 Max with a cloned pt-BR profile.

This mirrors the split the qwen engine already makes between ``mlx_backend`` and
``pytorch_backend``: MLX where it is available, PyTorch everywhere else.
"""

import asyncio
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from . import CHATTERBOX_MLX_S3_TOKENIZER_REPO
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
)
from .mlx_backend import _run_on_mlx_thread

logger = logging.getLogger(__name__)

CHATTERBOX_MLX_HF_REPO = "mlx-community/chatterbox-multilingual-v3"
# mlx-audio's Model.from_pretrained fetches the S3 speech tokenizer from this
# second repo (~470 MB), so the engine is only "downloaded" once both are cached.
S3_TOKENIZER_HF_REPO = CHATTERBOX_MLX_S3_TOKENIZER_REPO

# Files that must be present for the MLX multilingual model
_MLX_WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer.json"]
_S3_TOKENIZER_FILES = ["model.safetensors", "config.json"]


class ChatterboxMLXTTSBackend:
    """Chatterbox Multilingual TTS backend for voice cloning, on MLX/Metal."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        # Guards the load-then-use sequence, as MLXTTSBackend._op_lock does.
        self._op_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return CHATTERBOX_MLX_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        model_cached = is_model_cached(CHATTERBOX_MLX_HF_REPO, required_files=_MLX_WEIGHT_FILES)
        tokenizer_cached = is_model_cached(S3_TOKENIZER_HF_REPO, required_files=_S3_TOKENIZER_FILES)
        return model_cached and tokenizer_cached

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Chatterbox multilingual MLX model."""
        if self.model is not None:
            return
        async with self._op_lock:
            if self.model is not None:
                return
            # MLX streams are thread-local: every MLX call in the process
            # shares the single worker in mlx_backend (issue #699), so load
            # and generate always land on the same OS thread.
            await _run_on_mlx_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        is_cached = self._is_model_cached()

        with model_load_progress("chatterbox-tts", is_cached):
            from huggingface_hub import snapshot_download  # lazy: heavy import
            from mlx_audio.tts.models.chatterbox.chatterbox import Model  # lazy: heavy import

            logger.info("Loading Chatterbox Multilingual TTS on MLX (Metal)...")
            ckpt_dir = snapshot_download(CHATTERBOX_MLX_HF_REPO)
            self.model = Model.from_pretrained(ckpt_dir, s3_tokenizer_repo=S3_TOKENIZER_HF_REPO)

        logger.info("Chatterbox Multilingual TTS (MLX) loaded successfully")

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
        logger.info("Chatterbox (MLX) unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Chatterbox conditions on the reference audio at generation time, so the prompt
        just stores the file path.
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

    # The MLX port carries its own sampling defaults, validated by ear against the PyTorch
    # output on a cloned profile. The PyTorch tuning (repetition_penalty=2.0) is not
    # transferable: the two implementations weight it differently.
    _DEFAULTS: ClassVar[dict] = {
        "exaggeration": 0.1,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
    }

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Chatterbox Multilingual TTS on MLX.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path
            language: BCP-47 language code
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None

        def _generate_sync():
            import mlx.core as mx  # lazy: heavy import

            # Load (if needed) and generate as ONE worker submission, as in
            # MLXTTSBackend._reload_and_generate_sync, so an unload cannot
            # land between the two; bind the model locally for the same reason.
            if self.model is None:
                self._load_model_sync()
            model = self.model

            if seed is not None:
                mx.random.seed(seed)

            logger.info(f"[Chatterbox/MLX] Generating: lang={language}")

            # mlx-audio yields GenerationResult chunks; the whole clip is their concatenation.
            chunks = [
                np.asarray(result.audio).squeeze()
                for result in model.generate(
                    text,
                    ref_audio=ref_audio,
                    lang_code=language,
                    verbose=False,
                    **self._DEFAULTS,
                )
            ]
            audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

            sample_rate = getattr(model, "sr", None) or 24000
            return audio, int(sample_rate)

        async with self._op_lock:
            return await _run_on_mlx_thread(_generate_sync)
