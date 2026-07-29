"""
MiniMax cloud TTS backend implementation.

Unlike the local engines, this backend does not download a model — it calls
the MiniMax Text-to-Audio (T2A) HTTP API and decodes the returned audio to a
numpy float32 array. It plugs into the same ``TTSBackend`` protocol as the
local engines, so long-text chunking, trimming and profile handling all work
unchanged.

Regions:
  - ``global_en`` (default): https://api.minimax.io/v1/t2a_v2
  - ``cn_zh``:               https://api.minimaxi.com/v1/t2a_v2
  Select with the ``MINIMAX_API_REGION`` environment variable.

Speech models (``MINIMAX_TTS_MODELS``) all share the same request shape; the
default is ``speech-2.8-hd``. Audio is requested as hex-encoded data in the
synchronous response (``data.audio``) and decoded locally — ``pcm`` is decoded
directly to float32 with no codec dependency, while ``mp3`` / ``wav`` / ``flac``
are decoded through soundfile.

Voices are preset MiniMax system voices. Voice cloning is intentionally out of
scope for this backend.

Requirements:
  - ``MINIMAX_API_KEY`` environment variable (or ``~/.env.local``)
  - httpx (already in requirements.txt)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

# Regional API endpoints — the global and mainland-China hosts expose the same
# request/response shape and differ only in the base host.
MINIMAX_ENDPOINTS: dict[str, str] = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2",
}
MINIMAX_DEFAULT_REGION = "global_en"

# Speech models, newest first. All use the same T2A request shape.
MINIMAX_TTS_MODELS = [
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
]
MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"

# Supported synchronous-response audio formats.
MINIMAX_AUDIO_FORMATS = ["mp3", "wav", "flac", "pcm"]
MINIMAX_DEFAULT_FORMAT = "pcm"

# 32 kHz mono is a good default for speech and is supported by every model.
MINIMAX_DEFAULT_SAMPLE_RATE = 32000

MINIMAX_DEFAULT_VOICE = "English_Graceful_Lady"

# Preset system voices: (voice_id, display_name, gender, language).
MINIMAX_VOICES: list[tuple[str, str, str, str]] = [
    ("English_Graceful_Lady", "Graceful Lady", "female", "en"),
    ("English_radiant_girl", "Radiant Girl", "female", "en"),
    ("English_expressive_narrator", "Expressive Narrator", "female", "en"),
    ("English_Insightful_Speaker", "Insightful Speaker", "male", "en"),
    ("English_Persuasive_Man", "Persuasive Man", "male", "en"),
    ("English_Lucky_Robot", "Lucky Robot", "male", "en"),
    ("Chinese_Gentle_and_Clear", "Gentle and Clear", "female", "zh"),
    ("Chinese_Elegant_Lady", "Elegant Lady", "female", "zh"),
    ("Chinese_Intellectual_Female", "Intellectual Female", "female", "zh"),
    ("Chinese_Energetic_Boy", "Energetic Boy", "male", "zh"),
    ("Chinese_Magnetic_Male", "Magnetic Male", "male", "zh"),
]

# Map our ISO language codes to MiniMax ``language_boost`` names. Unmapped
# languages fall back to "auto" so MiniMax detects the language itself.
LANGUAGE_BOOST_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def resolve_region(region: str | None = None) -> str:
    """Resolve the MiniMax API region, honouring ``MINIMAX_API_REGION``.

    Falls back to the global endpoint for unknown or missing values.
    """
    candidate = region or os.environ.get("MINIMAX_API_REGION") or MINIMAX_DEFAULT_REGION
    candidate = candidate.strip().lower()
    if candidate not in MINIMAX_ENDPOINTS:
        return MINIMAX_DEFAULT_REGION
    return candidate


def get_endpoint(region: str | None = None) -> str:
    """Return the T2A endpoint URL for the resolved region."""
    return MINIMAX_ENDPOINTS[resolve_region(region)]


def _load_api_key() -> str | None:
    """Load ``MINIMAX_API_KEY`` from the environment or ``~/.env.local``."""
    key = os.environ.get("MINIMAX_API_KEY")
    if key:
        return key

    env_local = os.path.expanduser("~/.env.local")
    if os.path.exists(env_local):
        try:
            with open(env_local) as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if line.startswith("MINIMAX_API_KEY="):
                        val = line[len("MINIMAX_API_KEY=") :].strip().strip("\"'")
                        if val:
                            return val
        except OSError:
            pass

    return None


def _decode_audio(audio_bytes: bytes, audio_format: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Decode MiniMax audio bytes into a float32 array.

    ``pcm`` is signed 16-bit little-endian and is decoded directly. Container
    formats (mp3/wav/flac) are decoded through soundfile, which reports the
    real sample rate from the stream.
    """
    if not audio_bytes:
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    if audio_format == "pcm":
        audio_int16 = np.frombuffer(audio_bytes, dtype="<i2")
        return audio_int16.astype(np.float32) / 32768.0, sample_rate

    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


class MiniMaxTTSBackend:
    """MiniMax cloud TTS backend — no local model, one HTTP call per request.

    The backend is "loaded" whenever the API key is available; ``load_model``
    only validates that the key is present so callers get an early, clear error
    instead of an opaque HTTP 401.
    """

    def __init__(self) -> None:
        self._api_key: str | None = None
        self.model_size = "default"

    # -- Protocol helpers -----------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True once the API key has been located."""
        if self._api_key is None:
            self._api_key = _load_api_key()
        return bool(self._api_key)

    def _get_model_path(self, model_size: str = "default") -> str:
        """No local model — report the active API endpoint instead."""
        return get_endpoint()

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """For a cloud API, "cached" means the API key is configured."""
        return self.is_loaded()

    def unload_model(self) -> None:
        """No-op: a cloud backend has nothing to unload."""

    # -- Model loading --------------------------------------------------------

    async def load_model(self, model_size: str = "default") -> None:
        """Validate that the API key is configured; nothing is downloaded."""
        if self._api_key is None:
            self._api_key = _load_api_key()
        if not self._api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set. Add it to your environment or to ~/.env.local.")
        logger.info("MiniMax TTS backend ready (cloud API, no local model)")

    # -- Voice prompt API -----------------------------------------------------

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """Return the default preset voice.

        MiniMax uses preset system voices rather than reference audio. Preset
        profiles build the voice prompt in the profile service and never reach
        this method; it exists only as a safe fallback.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "minimax",
            "preset_voice_id": MINIMAX_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """Combine reference clips — delegates to the shared audio utility."""
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=MINIMAX_DEFAULT_SAMPLE_RATE)

    # -- Core generation ------------------------------------------------------

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech through the MiniMax T2A API.

        Args:
            text: Text to synthesize.
            voice_prompt: Dict carrying at least ``preset_voice_id``. Optional
                keys ``tts_model``, ``audio_format``, ``sample_rate``, ``speed``,
                ``vol`` and ``pitch`` override the request controls.
            language: ISO language code, mapped to ``language_boost``.
            seed: Ignored — the T2A API does not expose a seed.
            instruct: Ignored — not supported by the T2A API.

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        await self.load_model()

        voice_id = voice_prompt.get("preset_voice_id") or MINIMAX_DEFAULT_VOICE

        model = voice_prompt.get("tts_model") or MINIMAX_TTS_DEFAULT_MODEL
        if model not in MINIMAX_TTS_MODELS:
            model = MINIMAX_TTS_DEFAULT_MODEL

        audio_format = voice_prompt.get("audio_format") or MINIMAX_DEFAULT_FORMAT
        if audio_format not in MINIMAX_AUDIO_FORMATS:
            audio_format = MINIMAX_DEFAULT_FORMAT

        sample_rate = int(voice_prompt.get("sample_rate") or MINIMAX_DEFAULT_SAMPLE_RATE)
        payload = self._build_payload(text, voice_id, model, audio_format, sample_rate, language, voice_prompt)

        audio_bytes = await asyncio.to_thread(self._generate_sync, payload)
        return await asyncio.to_thread(_decode_audio, audio_bytes, audio_format, sample_rate)

    def _build_payload(
        self,
        text: str,
        voice_id: str,
        model: str,
        audio_format: str,
        sample_rate: int,
        language: str,
        voice_prompt: dict,
    ) -> dict:
        """Assemble the T2A request body from the supported request controls."""
        return {
            "model": model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "language_boost": LANGUAGE_BOOST_MAP.get(language, "auto"),
            "voice_setting": {
                "voice_id": voice_id,
                "speed": float(voice_prompt.get("speed", 1.0)),
                "vol": float(voice_prompt.get("vol", 1.0)),
                "pitch": int(voice_prompt.get("pitch", 0)),
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "format": audio_format,
                "channel": 1,
            },
        }

    def _generate_sync(self, payload: dict) -> bytes:
        """Blocking T2A call. Returns the raw (decoded-from-hex) audio bytes."""
        import httpx

        api_key = self._api_key or _load_api_key()
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = get_endpoint()

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()

        return self._parse_response(body)

    @staticmethod
    def _parse_response(body: dict) -> bytes:
        """Validate ``base_resp`` and decode the hex audio from ``data.audio``."""
        base_resp = body.get("base_resp") or {}
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            msg = base_resp.get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax TTS API error {status_code}: {msg}")

        data = body.get("data") or {}
        audio_hex = data.get("audio")
        if not audio_hex:
            raise RuntimeError(
                "MiniMax TTS API returned no audio data. Check your MINIMAX_API_KEY and request parameters."
            )

        try:
            audio_bytes = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise RuntimeError("MiniMax TTS API returned malformed audio data.") from exc

        logger.debug("MiniMax TTS: decoded %d bytes of audio", len(audio_bytes))
        return audio_bytes
