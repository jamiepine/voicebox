"""Optional CPU transcription with the pinned Orukeet INT8 ONNX export."""

import asyncio
import hashlib
import threading
from pathlib import Path

import numpy as np

REPO_ID = "oruk/orukeet"
REVISION = "1751fce6ecde442f14543cf1804800c49b3e415c"
SUBFOLDER = "onnx/combined-v0.1.0-int8"
LANGUAGES = frozenset(
    [
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "es",
        "sv",
        "uk",
    ]
)
FILE_HASHES = {
    "encoder-model.int8.onnx": "7b55f2a504a20a8e462899f5befd45f4a1784948d76ed0127902d9cf39405487",
    "decoder_joint-model.int8.onnx": "95d3b1f53f9aadc5ef58e63664a3681a2184ee228b5010e1ef975a1c4ea8318a",
    "vocab.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    "config.json": "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
}
LICENSE_FILES = ("LICENSE-WEIGHTS", "NOTICE.md", "LICENSE-CONVERTER.txt", "LICENSE-PREPROCESSOR.txt")


def get_model_directory() -> Path:
    """Resolve the complete pinned cache or download and verify the model.

    The runtime's required config.json is included in Hugging Face's ordinary
    download accounting. Cached recognition makes no network requests.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    required = (*FILE_HASHES, *LICENSE_FILES)
    paths = [try_to_load_from_cache(REPO_ID, f"{SUBFOLDER}/{name}", revision=REVISION) for name in required]
    if all(isinstance(path, str) for path in paths):
        directory = Path(paths[0]).parent
    else:
        snapshot = snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            allow_patterns=[f"{SUBFOLDER}/{name}" for name in required],
        )
        directory = Path(snapshot) / SUBFOLDER

    for filename, expected in FILE_HASHES.items():
        with (directory / filename).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Orukeet checksum mismatch for {filename}. Remove the cached model and download it again."
            )
    return directory


class OrukeetSTTBackend:
    """Serialize model initialization and recognition without blocking the event loop."""

    def __init__(self) -> None:
        """Create a lazy recognizer with one lock for loading and inference."""
        self._model = None
        self._lock = threading.Lock()

    async def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> str:
        """Transcribe mono PCM with automatic language detection.

        Args:
            audio: Mono floating point samples, as decoded by the upload route.
            sample_rate: Sample rate in Hz. The runtime resamples to 16 kHz.
            language: Optional supported language hint; Orukeet detects language
                automatically and does not force a decoder language.
        """
        if language is not None and language not in LANGUAGES:
            raise ValueError(
                f"Orukeet does not support language '{language}'. Supported: {', '.join(sorted(LANGUAGES))}"
            )
        if sample_rate <= 0 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Orukeet requires finite mono audio and a positive sample rate.")
        if audio.size == 0:
            return ""
        return await asyncio.to_thread(self._transcribe, np.asarray(audio, dtype=np.float32), sample_rate)

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Load the verified CPU model once and recognize under the shared lock."""
        with self._lock:
            if self._model is None:
                try:
                    import onnx_asr  # lazy: heavy import
                except ImportError as error:
                    raise ImportError(
                        "Orukeet requires the optional backend/requirements-orukeet.txt dependencies."
                    ) from error
                model_path = get_model_directory()
                self._model = onnx_asr.load_model(
                    "nemo-conformer-tdt",
                    path=model_path,
                    quantization="int8",
                    providers=["CPUExecutionProvider"],
                )
            return self._model.recognize(audio, sample_rate=sample_rate)
