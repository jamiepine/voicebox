"""
CosyVoice2 / CosyVoice3 TTS backend implementation.

Wraps the upstream FunAudioLLM/CosyVoice library for zero-shot voice cloning
with instruct support (emotions, speed, volume, dialects).  The CosyVoice repo
is cloned at setup time (``just setup-python``) and added to ``sys.path`` at
import time.

Model variants:
    - CosyVoice2-0.5B: ``inference_instruct2()`` for 9-language cloning + instruct
    - Fun-CosyVoice3-0.5B: improved robustness, prosody, and Chinese dialects

Both variants share a single ``cosyvoice`` engine key; the ``model_size``
field selects which HuggingFace checkpoint to download.
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# ── HuggingFace repos ─────────────────────────────────────────────────

COSYVOICE_HF_REPOS = {
    "v2": "FunAudioLLM/CosyVoice2-0.5B",
    "v3": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}

# Files that must be present for CosyVoice2 / CosyVoice3
_REQUIRED_FILES = {
    "v2": ["llm.pt", "flow.pt", "hift.pt", "cosyvoice2.yaml", "campplus.onnx"],
    "v3": ["llm.pt", "flow.pt", "hift.pt", "cosyvoice3.yaml", "campplus.onnx"],
}

# Model name → variant key
_MODEL_NAME_TO_VARIANT = {
    "cosyvoice2-0.5b": "v2",
    "cosyvoice3-0.5b": "v3",
}

# Default sample rate (both models produce 24 kHz audio)
COSYVOICE_SAMPLE_RATE = 24000


def _ensure_cosyvoice_on_path() -> None:
    """Add the cloned CosyVoice repo + Matcha-TTS to sys.path if not already present."""
    backend_dir = Path(__file__).resolve().parent.parent  # backend/
    cosyvoice_root = backend_dir / "vendors" / "CosyVoice"

    if not cosyvoice_root.exists():
        raise RuntimeError(
            f"CosyVoice source not found at {cosyvoice_root}. "
            "Run `just setup-python` to clone it."
        )

    cosyvoice_str = str(cosyvoice_root)
    matcha_str = str(cosyvoice_root / "third_party" / "Matcha-TTS")

    if cosyvoice_str not in sys.path:
        sys.path.insert(0, cosyvoice_str)
    if os.path.isdir(matcha_str) and matcha_str not in sys.path:
        sys.path.insert(0, matcha_str)


def _shim_training_only_modules() -> None:
    """
    Pre-populate ``sys.modules`` with lightweight stubs for modules that
    the CosyVoice YAML configs reference but are only needed for training.

    ``hyperpyyaml`` resolves every ``!name:`` / ``!new:`` tag via
    ``pydoc.locate`` which eagerly imports the target module.  The YAML
    references ``cosyvoice.dataset.processor`` (12 times) which pulls in
    ``pyarrow``, ``pyworld``, etc. at module level.

    Several ``matcha.utils.*`` submodules also import
    ``lightning.pytorch`` at module level.  We stub those so the real
    ``matcha.utils`` package can still expose ``audio.py`` and ``model.py``
    for inference.
    """
    import types
    import logging as _logging

    _noop = lambda *a, **kw: None

    def get_pylogger(name: str = __name__) -> _logging.Logger:
        return _logging.getLogger(name)

    # ── matcha.utils submodules that import lightning ──────────────
    fake_pylogger = types.ModuleType("matcha.utils.pylogger")
    fake_pylogger.get_pylogger = get_pylogger  # type: ignore[attr-defined]

    fake_logging_utils = types.ModuleType("matcha.utils.logging_utils")
    fake_logging_utils.log_hyperparameters = _noop  # type: ignore[attr-defined]

    fake_rich_utils = types.ModuleType("matcha.utils.rich_utils")
    fake_rich_utils.enforce_tags = _noop  # type: ignore[attr-defined]
    fake_rich_utils.print_config_tree = _noop  # type: ignore[attr-defined]

    fake_instantiators = types.ModuleType("matcha.utils.instantiators")
    fake_instantiators.instantiate_callbacks = lambda *a, **kw: []  # type: ignore[attr-defined]
    fake_instantiators.instantiate_loggers = lambda *a, **kw: []  # type: ignore[attr-defined]

    fake_utils_utils = types.ModuleType("matcha.utils.utils")
    fake_utils_utils.extras = _noop  # type: ignore[attr-defined]
    fake_utils_utils.get_metric_value = _noop  # type: ignore[attr-defined]
    fake_utils_utils.task_wrapper = lambda fn: fn  # type: ignore[attr-defined]

    sys.modules["matcha.utils.pylogger"] = fake_pylogger
    sys.modules["matcha.utils.logging_utils"] = fake_logging_utils
    sys.modules["matcha.utils.rich_utils"] = fake_rich_utils
    sys.modules["matcha.utils.instantiators"] = fake_instantiators
    sys.modules["matcha.utils.utils"] = fake_utils_utils

    # ── cosyvoice.dataset.processor (training data pipeline) ──────
    # Referenced 12 times in cosyvoice2.yaml / cosyvoice3.yaml via
    # !name: tags.  Imports pyarrow, pyworld, whisper at module level.
    fake_dataset = types.ModuleType("cosyvoice.dataset")
    fake_dataset.__path__ = []  # type: ignore[attr-defined]
    fake_processor = types.ModuleType("cosyvoice.dataset.processor")
    for _fn in (
        "parquet_opener", "tokenize", "filter", "resample", "truncate",
        "compute_fbank", "compute_whisper_fbank", "compute_f0",
        "parse_embedding", "shuffle", "sort", "batch", "padding",
    ):
        setattr(fake_processor, _fn, _noop)

    sys.modules.setdefault("cosyvoice.dataset", fake_dataset)
    sys.modules["cosyvoice.dataset.processor"] = fake_processor




def _patch_modelscope_to_hf() -> None:
    """
    Monkey-patch ``modelscope.snapshot_download`` → ``huggingface_hub.snapshot_download``
    so that CosyVoice's ``__init__`` downloads from HuggingFace instead of ModelScope.

    Also passes ``token=None`` to avoid HF auth prompts on public repos.
    """
    import types
    from huggingface_hub import snapshot_download as hf_snapshot_download

    def _hf_download(model_id, **kwargs):
        kwargs.pop("revision", None)
        kwargs.pop("model_version", None)
        return hf_snapshot_download(model_id, token=None, **kwargs)

    # Create a fake "modelscope" module so ``from modelscope import snapshot_download`` works.
    fake_ms = types.ModuleType("modelscope")
    fake_ms.snapshot_download = _hf_download
    sys.modules["modelscope"] = fake_ms


def _patch_torchaudio_load() -> None:
    """
    Replace ``torchaudio.load`` with a soundfile-backed implementation.

    torchaudio >= 2.9 unconditionally delegates to TorchCodec and ignores
    the ``backend`` parameter.  CosyVoice calls ``torchaudio.load(wav,
    backend='soundfile')`` which now fails unless ``torchcodec`` is
    installed.  We swap in a lightweight wrapper that reads via soundfile
    and returns the same ``(Tensor, sample_rate)`` tuple.
    """
    import torch
    import torchaudio
    import soundfile as sf

    def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, buffer_size=4096,
                 backend=None):
        data, sr = sf.read(uri, start=frame_offset,
                           stop=None if num_frames < 0 else frame_offset + num_frames,
                           dtype="float32", always_2d=True)
        # data shape: (frames, channels) → tensor
        tensor = torch.from_numpy(data)
        if channels_first:
            tensor = tensor.T  # (channels, frames)
        return tensor, sr

    torchaudio.load = _sf_load


class CosyVoiceTTSBackend:
    """CosyVoice2 / CosyVoice3 TTS backend for voice cloning with instruct support."""

    # Class-level lock for import patching
    _import_lock: ClassVar[threading.Lock] = threading.Lock()
    _patched: ClassVar[bool] = False

    def __init__(self):
        self.model = None
        self._variant: Optional[str] = None  # "v2" or "v3"
        self._device: Optional[str] = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # CosyVoice has no MPS support — force CPU on macOS
        return get_torch_device(force_cpu_on_mac=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "v2") -> str:
        return COSYVOICE_HF_REPOS.get(model_size, COSYVOICE_HF_REPOS["v2"])

    def _is_model_cached(self, model_size: str = "v2") -> bool:
        variant = model_size if model_size in COSYVOICE_HF_REPOS else "v2"
        repo = COSYVOICE_HF_REPOS[variant]
        required = _REQUIRED_FILES[variant]
        return is_model_cached(repo, required_files=required)

    async def load_model(self, model_size: str = "v2") -> None:
        """Load a CosyVoice model variant.

        Args:
            model_size: ``"v2"`` for CosyVoice2-0.5B or ``"v3"`` for CosyVoice3-0.5B.
        """
        variant = model_size if model_size in COSYVOICE_HF_REPOS else "v2"

        # If already loaded with the right variant, skip
        if self.model is not None and self._variant == variant:
            return

        async with self._model_load_lock:
            if self.model is not None and self._variant == variant:
                return
            # Unload previous variant if switching
            if self.model is not None:
                self.unload_model()
            await asyncio.to_thread(self._load_model_sync, variant)

    def _load_model_sync(self, variant: str) -> None:
        """Synchronous model loading."""
        model_name = f"cosyvoice{'2' if variant == 'v2' else '3'}-0.5b"
        is_cached = self._is_model_cached(variant)

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            hf_repo = COSYVOICE_HF_REPOS[variant]
            logger.info(
                "Loading CosyVoice %s (%s) on %s...",
                "2" if variant == "v2" else "3",
                hf_repo,
                device,
            )

            # 1. Ensure cosyvoice source is on sys.path
            _ensure_cosyvoice_on_path()

            # 2. Patch imports (thread-safe, once)
            with CosyVoiceTTSBackend._import_lock:
                if not CosyVoiceTTSBackend._patched:
                    _shim_training_only_modules()
                    _patch_modelscope_to_hf()
                    _patch_torchaudio_load()
                    CosyVoiceTTSBackend._patched = True

            # 3. Patch torch.load to force map_location on CPU
            import torch

            if device == "cpu":
                _orig_torch_load = torch.load

                def _patched_load(*args, **kwargs):
                    kwargs.setdefault("map_location", "cpu")
                    return _orig_torch_load(*args, **kwargs)

                torch.load = _patched_load

            try:
                if variant == "v2":
                    from cosyvoice.cli.cosyvoice import CosyVoice2

                    model = CosyVoice2(hf_repo)
                else:
                    from cosyvoice.cli.cosyvoice import CosyVoice3

                    model = CosyVoice3(hf_repo)
            finally:
                # Restore original torch.load
                if device == "cpu":
                    torch.load = _orig_torch_load

            self.model = model
            self._variant = variant

        logger.info("CosyVoice %s loaded successfully", "2" if variant == "v2" else "3")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._variant = None
            self._device = None
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
            logger.info("CosyVoice unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        CosyVoice processes the reference at generation time via
        ``frontend_zero_shot`` / ``frontend_instruct2``, so we just
        store the path + text for later use.
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
        Generate audio using CosyVoice instruct2 (with cloning) or zero-shot.

        If ``instruct`` is provided, uses ``inference_instruct2()`` which
        supports emotion, speed, volume, and dialect control.
        Otherwise falls back to ``inference_zero_shot()``.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict with ``ref_audio`` path and ``ref_text``.
            language: BCP-47 language code (unused by CosyVoice directly,
                      but kept for protocol compatibility).
            seed: Random seed for reproducibility.
            instruct: Instruct text for style control, e.g.
                      ``"Read with a happy tone, slowly."``.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        await self.load_model(self._variant or "v2")

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text", "")

        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Reference audio not found: %s", ref_audio)
            ref_audio = None

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)

            # Collect all chunks from the generator
            audio_chunks = []

            if instruct and ref_audio:
                # instruct2: text + instruct + reference audio → cloned + styled
                logger.info("[CosyVoice] instruct2: lang=%s instruct=%s", language, instruct[:60])
                for chunk in self.model.inference_instruct2(
                    tts_text=text,
                    instruct_text=instruct,
                    prompt_wav=ref_audio,
                    stream=False,
                    speed=1.0,
                ):
                    audio_chunks.append(chunk["tts_speech"])
            elif ref_audio:
                # zero-shot voice cloning
                logger.info("[CosyVoice] zero_shot: lang=%s", language)
                for chunk in self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=ref_text,
                    prompt_wav=ref_audio,
                    stream=False,
                    speed=1.0,
                ):
                    audio_chunks.append(chunk["tts_speech"])
            else:
                # cross-lingual (no reference audio, shouldn't normally happen
                # in voicebox since profiles always have samples, but handle it)
                logger.info("[CosyVoice] cross_lingual fallback: lang=%s", language)
                for chunk in self.model.inference_cross_lingual(
                    tts_text=text,
                    prompt_wav=ref_audio or "",
                    stream=False,
                    speed=1.0,
                ):
                    audio_chunks.append(chunk["tts_speech"])

            # Concatenate all chunks
            if not audio_chunks:
                return np.zeros(COSYVOICE_SAMPLE_RATE, dtype=np.float32), COSYVOICE_SAMPLE_RATE

            full_audio = torch.cat(audio_chunks, dim=-1)
            audio_np = full_audio.squeeze().cpu().numpy().astype(np.float32)

            return audio_np, COSYVOICE_SAMPLE_RATE

        return await asyncio.to_thread(_generate_sync)
