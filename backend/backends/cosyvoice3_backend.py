"""Fun-CosyVoice3 backend.

This backend intentionally treats CosyVoice as an optional development
dependency. Upstream is not packaged as a normal PyPI library and currently
expects callers to import from a cloned repository plus the Matcha-TTS
submodule. Set COSYVOICE_REPO_PATH to that clone before using this engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts
from .base import empty_device_cache, is_model_cached, model_load_progress

logger = logging.getLogger(__name__)

COSYVOICE3_HF_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE3_MODEL_NAME = "cosyvoice3-0.5B"
COSYVOICE3_REQUIRED_FILES = [
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "spk2info.pt",
    "CosyVoice-BlankEN/*",
]


class CosyVoice3Backend:
    """Fun-CosyVoice3 0.5B backend for zero-shot voice cloning."""

    def __init__(self):
        self.model = None
        self.model_size = "0.5B"
        self._device = "cpu"
        self._model_load_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "0.5B") -> str:
        return COSYVOICE3_HF_REPO

    def _is_model_cached(self, model_size: str = "0.5B") -> bool:
        return is_model_cached(
            COSYVOICE3_HF_REPO,
            required_files=["cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx"],
        )

    async def load_model(self, model_size: str = "0.5B") -> None:
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        cosyvoice_repo = os.environ.get("COSYVOICE_REPO_PATH")
        if not cosyvoice_repo:
            raise RuntimeError(
                "CosyVoice3 requires COSYVOICE_REPO_PATH to point at a cloned "
                "FunAudioLLM/CosyVoice repository with submodules initialized."
            )

        repo_path = Path(cosyvoice_repo).expanduser().resolve()
        matcha_path = repo_path / "third_party" / "Matcha-TTS"
        if not (repo_path / "cosyvoice").exists() or not matcha_path.exists():
            raise RuntimeError(
                f"COSYVOICE_REPO_PATH is invalid: {repo_path}. Expected a "
                "CosyVoice clone with third_party/Matcha-TTS."
            )

        for path in (str(repo_path), str(matcha_path)):
            if path not in sys.path:
                sys.path.insert(0, path)

        is_cached = self._is_model_cached()
        with model_load_progress(COSYVOICE3_MODEL_NAME, is_cached):
            from huggingface_hub import snapshot_download
            from cosyvoice.cli.cosyvoice import AutoModel

            model_dir = snapshot_download(
                COSYVOICE3_HF_REPO,
                token=None,
                allow_patterns=COSYVOICE3_REQUIRED_FILES,
            )
            logger.info("Loading Fun-CosyVoice3 from %s", model_dir)
            _patch_torchaudio_load()
            self.model = AutoModel(model_dir=model_dir, load_trt=False, load_vllm=False, fp16=False)
            self.model_size = "0.5B"
            self._device = "cuda" if _torch_cuda_available() else "cpu"

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            empty_device_cache(self._device)
            logger.info("CosyVoice3 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=16000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        if not ref_audio or not Path(ref_audio).exists():
            raise RuntimeError("CosyVoice3 requires a valid reference audio sample.")

        ref_text = voice_prompt.get("ref_text") or ""
        prompt_text = _build_prompt_text(ref_text, language, instruct)
        tts_text, use_instruct = _prepare_text(text, language, instruct)

        def _generate_sync() -> tuple[np.ndarray, int]:
            import torch

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            if use_instruct:
                outputs = self.model.inference_instruct2(
                    tts_text,
                    prompt_text,
                    ref_audio,
                    stream=False,
                )
            else:
                outputs = self.model.inference_zero_shot(
                    tts_text,
                    prompt_text,
                    ref_audio,
                    stream=False,
                )

            chunks: list[np.ndarray] = []
            for item in outputs:
                wav = item["tts_speech"]
                if isinstance(wav, torch.Tensor):
                    chunks.append(wav.squeeze().detach().cpu().numpy().astype(np.float32))
                else:
                    chunks.append(np.asarray(wav, dtype=np.float32).squeeze())

            if not chunks:
                raise RuntimeError("CosyVoice3 returned no audio.")

            return np.concatenate(chunks), int(getattr(self.model, "sample_rate", 24000))

        return await asyncio.to_thread(_generate_sync)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _patch_torchaudio_load() -> None:
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_voicebox_soundfile_patch", False):
        return

    def _load_with_soundfile(uri, *args, **kwargs):
        data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        frame_offset = kwargs.get("frame_offset", 0) or 0
        num_frames = kwargs.get("num_frames", -1)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames is not None and num_frames >= 0:
            tensor = tensor[:, :num_frames]
        return tensor, sample_rate

    _load_with_soundfile._voicebox_soundfile_patch = True  # type: ignore[attr-defined]
    torchaudio.load = _load_with_soundfile


def _build_prompt_text(reference_text: str, language: str, instruct: Optional[str]) -> str:
    prefix = "You are a helpful assistant."
    if instruct:
        prefix = f"{prefix} {instruct.strip()}"
    elif language == "yue":
        prefix = f"{prefix} 请用广东话表达。"
    return f"{prefix}<|endofprompt|>{reference_text}".strip()


def _prepare_text(text: str, language: str, instruct: Optional[str]) -> tuple[str, bool]:
    if instruct or language == "yue":
        return text, True
    if language in {"zh", "en", "ja", "ko", "yue"}:
        return text, False
    return text, True
