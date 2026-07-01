"""
F5-TTS backend implementation.

Wraps the F5-TTS model for zero-shot voice cloning.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# Monkeypatch torchaudio.load to use soundfile instead of torchcodec
# TorchAudio 2.6 removed legacy soundfile fallback and strictly requires
# TorchCodec, which crashes without full FFmpeg DLLs on Windows.
import torchaudio
import soundfile as sf
import torch

def _mock_torchaudio_load(uri, **kwargs):
    audio, sr = sf.read(uri, dtype="float32")
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    else:
        audio = audio.T
    return torch.from_numpy(audio), sr

torchaudio.load = _mock_torchaudio_load
VOCOS_REPO = "charactr/vocos-mel-24khz"

MODELS = {
    "f5-tts-ro": {
        "repo": "cdorob/f5-tts-romanian",
        "ckpt_file": "model_last.pt",
        "vocab_file": "vocab.txt",
    },
}


class F5TTSBackend:
    """F5-TTS backend for voice cloning."""

    def __init__(self):
        self.model = None
        self.vocoder = None
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device(allow_mps=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self.model is not None and self.vocoder is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        if model_size == "default":
            model_size = "f5-tts-ro"
        return MODELS[model_size]["repo"]

    def _is_model_cached(self, model_size: str = "default") -> bool:
        if model_size == "default":
            model_size = "f5-tts-ro"
        model_info = MODELS[model_size]
        required = [model_info["ckpt_file"]]
        if model_info["vocab_file"]:
            required.append(model_info["vocab_file"])
            
        model_cached = is_model_cached(model_info["repo"], required_files=required)
        vocoder_cached = is_model_cached(VOCOS_REPO, required_files=["pytorch_model.bin", "config.yaml"])
        return model_cached and vocoder_cached

    async def load_model(self, model_size: str = "default") -> None:
        """Load the F5-TTS model and Vocos vocoder."""
        if self.is_loaded():
            return
        async with self._model_load_lock:
            if self.is_loaded():
                return
            await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        if model_size == "default":
            model_size = "f5-tts-ro"
            
        model_name = model_size
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading F5-TTS on {device}...")

            import torch
            from huggingface_hub import hf_hub_download
            
            # F5-TTS imports
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model, load_vocoder

            # Download models if needed
            model_info = MODELS[model_size]
            ckpt_path = hf_hub_download(repo_id=model_info["repo"], filename=model_info["ckpt_file"])
            
            vocab_file = ""
            if model_info["vocab_file"]:
                vocab_file = hf_hub_download(repo_id=model_info["repo"], filename=model_info["vocab_file"])
            
            # Load Vocoder (downloads if needed internally via huggingface_hub)
            self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            
            self.model = load_model(
                model_cls=DiT,
                model_cfg=model_cfg,
                ckpt_path=ckpt_path,
                mel_spec_type="vocos",
                vocab_file=vocab_file,
                ode_method="euler",
                use_ema=True,
                device=device,
            )

        logger.info("F5-TTS loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.is_loaded():
            device = self._device
            del self.model
            del self.vocoder
            self.model = None
            self.vocoder = None
            self._device = None
            empty_device_cache(device)
            logger.info("F5-TTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.
        
        Using deferred file paths pattern. The actual processing and inference
        happens at generation time. We bypass pydub trimming, so users should
        provide well-trimmed audio.
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
        # F5-TTS works at 24kHz
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using F5-TTS.
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")

        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None

        def _generate_sync():
            import torch
            from f5_tts.infer.utils_infer import infer_process

            if seed is not None:
                manual_seed(seed, self._device)

            logger.info("[F5-TTS] Generating...")
            
            # Use their infer_process wrapper
            audio_array, sample_rate, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                model_obj=self.model,
                vocoder=self.vocoder,
                mel_spec_type="vocos",
                show_info=logger.info,
                progress=None,
                device=self._device,
            )

            # Convert to numpy if it's a tensor (infer_process already returns numpy array)
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.squeeze().cpu().numpy().astype(np.float32)
            else:
                audio_array = np.asarray(audio_array, dtype=np.float32)

            return audio_array, sample_rate

        return await asyncio.to_thread(_generate_sync)
