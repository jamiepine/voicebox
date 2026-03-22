"""
Backend abstraction layer for TTS and STT.

Provides a unified interface for MLX and PyTorch backends,
and a model config registry that eliminates per-engine dispatch maps.
"""

import threading
from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, List
from typing_extensions import runtime_checkable
import numpy as np

from ..utils.platform_detect import get_backend_type

LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

WHISPER_HF_REPOS = {
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}


@dataclass
class ModelConfig:
    """Declarative config for a downloadable model variant."""

    model_name: str  # e.g. "luxtts", "chatterbox-tts"
    display_name: str  # e.g. "LuxTTS (Fast, CPU-friendly)"
    engine: str  # e.g. "luxtts", "chatterbox"
    hf_repo_id: str  # e.g. "YatharthS/LuxTTS"
    model_size: str = "default"
    size_mb: int = 0
    needs_trim: bool = False
    supports_instruct: bool = False
    languages: list[str] = field(default_factory=lambda: ["en"])


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol for TTS backend implementations."""

    # Each backend class should define MODEL_CONFIGS as a class variable:
    # MODEL_CONFIGS: list[ModelConfig]

    async def load_model(self, model_size: str) -> None:
        """Load TTS model."""
        ...

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        ...

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.

        Returns:
            Tuple of (combined_audio_array, combined_text)
        """
        ...

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    def _get_model_path(self, model_size: str) -> str:
        """
        Get model path for a given size.

        Returns:
            Model path or HuggingFace Hub ID
        """
        ...


@runtime_checkable
class STTBackend(Protocol):
    """Protocol for STT (Speech-to-Text) backend implementations."""

    async def load_model(self, model_size: str) -> None:
        """Load STT model."""
        ...

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Returns:
            Transcribed text
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


# Global backend instances
_tts_backend: Optional[TTSBackend] = None
_tts_backends: dict[str, TTSBackend] = {}
_tts_backends_lock = threading.Lock()
_stt_backend: Optional[STTBackend] = None

# Supported TTS engines — keyed by engine name, value is the backend class import path.
# The factory function uses this for the if/elif chain; the model configs live on the backend classes.
TTS_ENGINES = {
    "qwen": "Qwen TTS",
    "qwen_custom_voice": "Qwen CustomVoice",
    "luxtts": "LuxTTS",
    "chatterbox": "Chatterbox TTS",
    "chatterbox_turbo": "Chatterbox Turbo",
    "tada": "TADA",
    "kokoro": "Kokoro",
}


def _get_qwen_model_configs() -> list[ModelConfig]:
    """Return Qwen model configs with backend-aware HF repo IDs."""
    backend_type = get_backend_type()
    if backend_type == "mlx":
        repo_1_7b = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        repo_0_6b = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"  # 0.6B not available in MLX, falls back
    else:
        repo_1_7b = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        repo_0_6b = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    return [
        ModelConfig(
            model_name="qwen-tts-1.7B",
            display_name="Qwen TTS 1.7B",
            engine="qwen",
            hf_repo_id=repo_1_7b,
            model_size="1.7B",
            size_mb=3500,
            supports_instruct=False,  # Base model drops instruct silently
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
        ModelConfig(
            model_name="qwen-tts-0.6B",
            display_name="Qwen TTS 0.6B",
            engine="qwen",
            hf_repo_id=repo_0_6b,
            model_size="0.6B",
            size_mb=1200,
            supports_instruct=False,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
    ]


def _get_qwen_custom_voice_configs() -> list[ModelConfig]:
    """Return Qwen CustomVoice model configs."""
    return [
        ModelConfig(
            model_name="qwen-custom-voice-1.7B",
            display_name="Qwen CustomVoice 1.7B",
            engine="qwen_custom_voice",
            hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            model_size="1.7B",
            size_mb=3500,
            supports_instruct=True,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
        ModelConfig(
            model_name="qwen-custom-voice-0.6B",
            display_name="Qwen CustomVoice 0.6B",
            engine="qwen_custom_voice",
            hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            model_size="0.6B",
            size_mb=1200,
            supports_instruct=True,
            languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        ),
    ]


def _get_non_qwen_tts_configs() -> list[ModelConfig]:
    """Return model configs for non-Qwen TTS engines.

    These are static — no backend-type branching needed.
    """
    return [
        ModelConfig(
            model_name="luxtts",
            display_name="LuxTTS (Fast, CPU-friendly)",
            engine="luxtts",
            hf_repo_id="YatharthS/LuxTTS",
            size_mb=300,
            languages=["en"],
        ),
        ModelConfig(
            model_name="chatterbox-tts",
            display_name="Chatterbox TTS (Multilingual)",
            engine="chatterbox",
            hf_repo_id="ResembleAI/chatterbox",
            size_mb=3200,
            needs_trim=True,
            languages=[
                "zh",
                "en",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "he",
                "ar",
                "da",
                "el",
                "fi",
                "hi",
                "ms",
                "nl",
                "no",
                "pl",
                "sv",
                "sw",
                "tr",
            ],
        ),
        ModelConfig(
            model_name="chatterbox-turbo",
            display_name="Chatterbox Turbo (English, Tags)",
            engine="chatterbox_turbo",
            hf_repo_id="ResembleAI/chatterbox-turbo",
            size_mb=1500,
            needs_trim=True,
            languages=["en"],
        ),
        ModelConfig(
            model_name="tada-1b",
            display_name="TADA 1B (English)",
            engine="tada",
            hf_repo_id="HumeAI/tada-1b",
            model_size="1B",
            size_mb=4000,
            languages=["en"],
        ),
        ModelConfig(
            model_name="tada-3b-ml",
            display_name="TADA 3B Multilingual",
            engine="tada",
            hf_repo_id="HumeAI/tada-3b-ml",
            model_size="3B",
            size_mb=8000,
            languages=["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
        ),
        ModelConfig(
            model_name="kokoro",
            display_name="Kokoro 82M",
            engine="kokoro",
            hf_repo_id="hexgrad/Kokoro-82M",
            size_mb=350,
            languages=["en", "es", "fr", "hi", "it", "pt", "ja", "zh"],
        ),
    ]


def _get_whisper_configs() -> list[ModelConfig]:
    """Return Whisper STT model configs."""
    return [
        ModelConfig(
            model_name="whisper-base",
            display_name="Whisper Base",
            engine="whisper",
            hf_repo_id="openai/whisper-base",
            model_size="base",
        ),
        ModelConfig(
            model_name="whisper-small",
            display_name="Whisper Small",
            engine="whisper",
            hf_repo_id="openai/whisper-small",
            model_size="small",
        ),
        ModelConfig(
            model_name="whisper-medium",
            display_name="Whisper Medium",
            engine="whisper",
            hf_repo_id="openai/whisper-medium",
            model_size="medium",
        ),
        ModelConfig(
            model_name="whisper-large",
            display_name="Whisper Large",
            engine="whisper",
            hf_repo_id="openai/whisper-large-v3",
            model_size="large",
        ),
        ModelConfig(
            model_name="whisper-turbo",
            display_name="Whisper Turbo",
            engine="whisper",
            hf_repo_id="openai/whisper-large-v3-turbo",
            model_size="turbo",
        ),
    ]


def get_all_model_configs() -> list[ModelConfig]:
    """Return the full list of model configs (TTS + STT)."""
    return _get_qwen_model_configs() + _get_qwen_custom_voice_configs() + _get_non_qwen_tts_configs() + _get_whisper_configs()


def get_tts_model_configs() -> list[ModelConfig]:
    """Return only TTS model configs."""
    return _get_qwen_model_configs() + _get_qwen_custom_voice_configs() + _get_non_qwen_tts_configs()


# Lookup helpers — these replace the if/elif chains in main.py


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Look up a model config by model_name."""
    for cfg in get_all_model_configs():
        if cfg.model_name == model_name:
            return cfg
    return None


def engine_needs_trim(engine: str) -> bool:
    """Whether this engine's output should be run through trim_tts_output."""
    for cfg in get_tts_model_configs():
        if cfg.engine == engine:
            return cfg.needs_trim
    return False


def engine_has_model_sizes(engine: str) -> bool:
    """Whether this engine supports multiple model sizes (only Qwen currently)."""
    configs = [c for c in get_tts_model_configs() if c.engine == engine]
    return len(configs) > 1


async def load_engine_model(engine: str, model_size: str = "default") -> None:
    """Load a model for the given engine, handling engines with multiple model sizes."""
    backend = get_tts_backend_for_engine(engine)
    if engine in ("qwen", "qwen_custom_voice"):
        await backend.load_model_async(model_size)
    elif engine == "tada":
        await backend.load_model(model_size)
    else:
        await backend.load_model()


async def ensure_model_cached_or_raise(engine: str, model_size: str = "default") -> None:
    """Check if a model is cached, raise HTTPException if not. Used by streaming endpoint."""
    from fastapi import HTTPException

    backend = get_tts_backend_for_engine(engine)
    cfg = None
    for c in get_tts_model_configs():
        if c.engine == engine and c.model_size == model_size:
            cfg = c
            break

    if engine in ("qwen", "qwen_custom_voice", "tada"):
        if not backend._is_model_cached(model_size):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_size} is not downloaded yet. Use /generate to trigger a download.",
            )
    else:
        if not backend._is_model_cached():
            display = cfg.display_name if cfg else engine
            raise HTTPException(
                status_code=400,
                detail=f"{display} model is not downloaded yet. Use /generate to trigger a download.",
            )


def unload_model_by_config(config: ModelConfig) -> bool:
    """Unload a model given its config. Returns True if it was loaded, False otherwise."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe

    if config.engine == "whisper":
        whisper_model = transcribe.get_whisper_model()
        if whisper_model.is_loaded() and whisper_model.model_size == config.model_size:
            transcribe.unload_whisper_model()
            return True
        return False

    if config.engine == "qwen":
        tts_model = tts.get_tts_model()
        loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
        if tts_model.is_loaded() and loaded_size == config.model_size:
            tts.unload_tts_model()
            return True
        return False

    if config.engine == "qwen_custom_voice":
        backend = get_tts_backend_for_engine(config.engine)
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        if backend.is_loaded() and loaded_size == config.model_size:
            backend.unload_model()
            return True
        return False

    # All other TTS engines
    backend = get_tts_backend_for_engine(config.engine)
    if backend.is_loaded():
        backend.unload_model()
        return True
    return False


def check_model_loaded(config: ModelConfig) -> bool:
    """Check if a model is currently loaded."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe

    try:
        if config.engine == "whisper":
            whisper_model = transcribe.get_whisper_model()
            return whisper_model.is_loaded() and getattr(whisper_model, "model_size", None) == config.model_size

        if config.engine == "qwen":
            tts_model = tts.get_tts_model()
            loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
            return tts_model.is_loaded() and loaded_size == config.model_size

        if config.engine == "qwen_custom_voice":
            backend = get_tts_backend_for_engine(config.engine)
            loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
            return backend.is_loaded() and loaded_size == config.model_size

        backend = get_tts_backend_for_engine(config.engine)
        return backend.is_loaded()
    except Exception:
        return False


def get_model_load_func(config: ModelConfig):
    """Return a callable that loads/downloads the model."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe

    if config.engine == "whisper":
        return lambda: transcribe.get_whisper_model().load_model(config.model_size)

    if config.engine == "qwen":
        return lambda: tts.get_tts_model().load_model(config.model_size)

    if config.engine == "qwen_custom_voice":
        return lambda: get_tts_backend_for_engine(config.engine).load_model(config.model_size)

    return lambda: get_tts_backend_for_engine(config.engine).load_model()


def get_tts_backend() -> TTSBackend:
    """
    Get or create the default (Qwen) TTS backend instance based on platform.

    Returns:
        TTS backend instance (MLX or PyTorch)
    """
    return get_tts_backend_for_engine("qwen")


def get_tts_backend_for_engine(engine: str) -> TTSBackend:
    """
    Get or create a TTS backend for the given engine.

    Args:
        engine: Engine name (e.g. "qwen", "luxtts", "chatterbox", "chatterbox_turbo")

    Returns:
        TTS backend instance
    """
    global _tts_backends

    # Fast path: check without lock
    if engine in _tts_backends:
        return _tts_backends[engine]

    # Slow path: create with lock to avoid duplicate instantiation
    with _tts_backends_lock:
        # Double-check after acquiring lock
        if engine in _tts_backends:
            return _tts_backends[engine]

        if engine == "qwen":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .mlx_backend import MLXTTSBackend

                backend = MLXTTSBackend()
            else:
                from .pytorch_backend import PyTorchTTSBackend

                backend = PyTorchTTSBackend()
        elif engine == "luxtts":
            from .luxtts_backend import LuxTTSBackend

            backend = LuxTTSBackend()
        elif engine == "chatterbox":
            from .chatterbox_backend import ChatterboxTTSBackend

            backend = ChatterboxTTSBackend()
        elif engine == "chatterbox_turbo":
            from .chatterbox_turbo_backend import ChatterboxTurboTTSBackend

            backend = ChatterboxTurboTTSBackend()
        elif engine == "tada":
            from .hume_backend import HumeTadaBackend

            backend = HumeTadaBackend()
        elif engine == "kokoro":
            from .kokoro_backend import KokoroTTSBackend

            backend = KokoroTTSBackend()
        elif engine == "qwen_custom_voice":
            from .qwen_custom_voice_backend import QwenCustomVoiceBackend

            backend = QwenCustomVoiceBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES.keys())}")

        _tts_backends[engine] = backend
        return backend


def get_stt_backend() -> STTBackend:
    """
    Get or create STT backend instance based on platform.

    Returns:
        STT backend instance (MLX or PyTorch)
    """
    global _stt_backend

    if _stt_backend is None:
        backend_type = get_backend_type()

        if backend_type == "mlx":
            from .mlx_backend import MLXSTTBackend

            _stt_backend = MLXSTTBackend()
        else:
            from .pytorch_backend import PyTorchSTTBackend

            _stt_backend = PyTorchSTTBackend()

    return _stt_backend


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _tts_backend, _tts_backends, _stt_backend
    _tts_backend = None
    _tts_backends.clear()
    _stt_backend = None
