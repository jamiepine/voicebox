"""English Qwen3-ASR transcription with the offline sherpa-onnx runtime."""

import asyncio
import os
import threading
from pathlib import Path

import numpy as np

MODEL = "oruk-fourier-2b"
REPO = "oruk/oruk-fourier-2b"
REVISION = "3fda023e157ce7986f137f76fcf326ed6e08fa73"
SUBDIR = "onnx/sherpa-int8"
FILES = (
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "encoder.int8.onnx.data",
    "decoder.int8.onnx",
    "decoder.int8.onnx.data",
    "tokenizer/config.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)
SAMPLE_RATE = 16000


def cached_directory() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / "voicebox" / MODEL / REVISION / SUBDIR


def is_cached() -> bool:
    root = cached_directory()
    return all((root / f).is_file() for f in FILES)


def supports_language(language):
    return not language or language.lower() in ("en", "english", "auto") or language.lower().startswith("en-")


def audio_chunks(audio):
    """Use short windows, cutting long recordings at the quietest 20 ms near the end."""
    limit = 28 * SAMPLE_RATE
    while len(audio) > limit:
        start = limit - 2 * SAMPLE_RATE
        frames = audio[start:limit].reshape(-1, SAMPLE_RATE // 50)
        quietest = int(np.argmin(np.mean(frames * frames, axis=1)))
        cut = start + quietest * (SAMPLE_RATE // 50) + SAMPLE_RATE // 100
        yield audio[:cut]
        audio = audio[cut:]
    if len(audio):
        yield audio


class FourierSTTBackend:
    model_size = MODEL

    def __init__(self):
        self._recognizer = None
        self._lock = threading.RLock()

    def is_loaded(self):
        return self._recognizer is not None

    def _is_model_cached(self, model_size):
        return model_size == MODEL and is_cached()

    def _load(self):
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        with self._lock:
            if self.is_loaded():
                return
            root = cached_directory()
            if not is_cached():
                snapshot_download(
                    REPO,
                    revision=REVISION,
                    local_dir=root.parent.parent,
                    allow_patterns=[f"{SUBDIR}/{f}" for f in FILES],
                )
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=str(root / "conv_frontend.onnx"),
                encoder=str(root / "encoder.int8.onnx"),
                decoder=str(root / "decoder.int8.onnx"),
                tokenizer=str(root / "tokenizer"),
                provider="cpu",
                num_threads=min(4, os.cpu_count() or 1),
            )

    async def load_model_async(self, model_size=None):
        await asyncio.to_thread(self._load)

    load_model = load_model_async

    def unload_model(self):
        with self._lock:
            self._recognizer = None

    async def transcribe(self, audio_path, language=None, model_size=None):
        if not supports_language(language):
            raise ValueError("Oruk Fourier 2B supports English only.")
        return await asyncio.to_thread(self._transcribe_file, audio_path)

    def _transcribe_file(self, audio_path):
        from ..utils.audio import load_audio

        audio, _ = load_audio(audio_path, sample_rate=SAMPLE_RATE)
        words = []
        with self._lock:
            self._load()
            for chunk in audio_chunks(audio):
                if not np.any(chunk):
                    continue
                stream = self._recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, chunk)
                self._recognizer.decode_stream(stream)
                words.append(stream.result.text.strip())
        return " ".join(filter(None, words))
