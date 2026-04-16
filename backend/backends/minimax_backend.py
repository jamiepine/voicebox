"""
MiniMax TTS backend implementation.

Uses the MiniMax Text-to-Audio (T2A) cloud API to generate speech.
Unlike other backends, this does not download a local model — it sends
requests to https://api.minimax.io/v1/t2a_v2 and streams back PCM audio
encoded as hex in SSE events.  PCM (signed 16-bit LE) is decoded directly
to numpy float32 without any codec dependency.

Requirements:
  - MINIMAX_API_KEY environment variable (or ~/.env.local)
  - httpx (already in requirements.txt)

Models:
  - speech-2.8-hd  (default, highest quality)
  - speech-2.8-turbo  (faster)

Voice IDs are preset MiniMax system voices — voice cloning is not supported.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MINIMAX_TTS_BASE_URL = "https://api.minimax.io"
MINIMAX_TTS_ENDPOINT = "/v1/t2a_v2"
MINIMAX_TTS_SAMPLE_RATE = 32000
MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_TTS_DEFAULT_VOICE = "English_Graceful_Lady"

# Available MiniMax system voice IDs
MINIMAX_VOICES: list[tuple[str, str, str]] = [
    # (voice_id, display_name, language)
    ("English_Graceful_Lady", "Graceful Lady", "en"),
    ("English_Insightful_Speaker", "Insightful Speaker", "en"),
    ("English_radiant_girl", "Radiant Girl", "en"),
    ("English_Persuasive_Man", "Persuasive Man", "en"),
    ("English_Lucky_Robot", "Lucky Robot", "en"),
    ("English_expressive_narrator", "Expressive Narrator", "en"),
    ("Chinese_Gentle_and_Clear", "Gentle and Clear", "zh"),
    ("Chinese_Energetic_Boy", "Energetic Boy", "zh"),
    ("Chinese_Elegant_Lady", "Elegant Lady", "zh"),
    ("Chinese_Intellectual_Female", "Intellectual Female", "zh"),
    ("Chinese_Magnetic_Male", "Magnetic Male", "zh"),
]

# Available TTS model IDs
MINIMAX_TTS_MODELS = [
    "speech-2.8-hd",
    "speech-2.8-turbo",
]


def _load_api_key() -> Optional[str]:
    """Load MINIMAX_API_KEY from environment or ~/.env.local."""
    key = os.environ.get("MINIMAX_API_KEY")
    if key:
        return key

    env_local = os.path.expanduser("~/.env.local")
    if os.path.exists(env_local):
        try:
            with open(env_local) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MINIMAX_API_KEY="):
                        val = line[len("MINIMAX_API_KEY="):].strip().strip("\"'")
                        if val:
                            return val
        except OSError:
            pass

    return None


def _pcm_bytes_to_numpy(pcm_bytes: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Convert raw PCM bytes (signed 16-bit little-endian) to float32 numpy array.

    MiniMax streams PCM audio as hex-encoded signed 16-bit LE samples.
    We convert directly to float32 in [-1.0, 1.0] without any codec dependency.
    """
    if not pcm_bytes:
        return np.zeros(sample_rate, dtype=np.float32), sample_rate

    # int16 LE → float32
    audio_int16 = np.frombuffer(pcm_bytes, dtype="<i2")
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32, sample_rate


class MiniMaxTTSBackend:
    """
    MiniMax cloud TTS backend.

    This backend is always "loaded" — there is no local model to download.
    ``load_model`` is a no-op; ``is_loaded`` returns True whenever the
    API key is available.
    """

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self.model_size = "default"

    # ── Protocol helpers ─────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        """Return True if the API key is configured."""
        if self._api_key is None:
            self._api_key = _load_api_key()
        return bool(self._api_key)

    def _get_model_path(self, model_size: str) -> str:
        """MiniMax has no local model path — return the API endpoint."""
        return MINIMAX_TTS_BASE_URL + MINIMAX_TTS_ENDPOINT

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """For a cloud API 'cached' means the API key is present."""
        return self.is_loaded()

    def unload_model(self) -> None:
        """No-op: nothing to unload for a cloud API backend."""

    # ── Model loading ────────────────────────────────────────────────────────

    async def load_model(self, model_size: str = "default") -> None:
        """
        Validate API key availability.

        For a cloud API there is nothing to download, but we still check
        that MINIMAX_API_KEY is set so the user gets an early, clear error
        rather than a cryptic HTTP 401 later.
        """
        if self._api_key is None:
            self._api_key = _load_api_key()
        if not self._api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not set. "
                "Add it to your environment or to ~/.env.local."
            )
        logger.info("MiniMax TTS backend ready (cloud API, no local model)")

    # ── Voice prompt API ─────────────────────────────────────────────────────

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        MiniMax uses preset system voices, not user-provided reference audio.

        When this method is called (e.g., for a cloned profile fallback) we
        return the default voice.  In normal usage the voice prompt is built
        by the profile service from preset_voice_id and never calls this.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "minimax",
            "preset_voice_id": MINIMAX_TTS_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine voice prompts — delegates to the shared audio utility.

        MiniMax does not support voice cloning, so this is only called if
        a cloned profile somehow reaches this backend.
        """
        from .base import combine_voice_prompts as _combine

        return await _combine(audio_paths, reference_texts, sample_rate=MINIMAX_TTS_SAMPLE_RATE)

    # ── Core generation ──────────────────────────────────────────────────────

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech via the MiniMax TTS API.

        The API returns audio as hex-encoded MP3 in SSE events.  We collect
        all ``status=1`` chunks, concatenate them, and decode to numpy.

        Args:
            text: Text to synthesize (max 10,000 characters per request).
            voice_prompt: Dict with ``preset_voice_id`` key.
            language: ISO language code (informational; MiniMax handles it).
            seed: Ignored — MiniMax API does not expose a seed parameter.
            instruct: Ignored — not supported by the TTS API.

        Returns:
            Tuple of (audio_array float32, sample_rate).
        """
        await self.load_model()

        voice_id = (
            voice_prompt.get("preset_voice_id")
            or MINIMAX_TTS_DEFAULT_VOICE
        )
        tts_model = voice_prompt.get("tts_model") or MINIMAX_TTS_DEFAULT_MODEL

        mp3_bytes = await asyncio.to_thread(
            self._generate_sync, text, voice_id, tts_model
        )
        audio, sr = await asyncio.to_thread(_pcm_bytes_to_numpy, mp3_bytes, MINIMAX_TTS_SAMPLE_RATE)
        return audio, sr

    def _generate_sync(self, text: str, voice_id: str, model: str) -> bytes:
        """Blocking HTTP call to the MiniMax TTS API."""
        import httpx

        api_key = self._api_key or _load_api_key()
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is not configured.")

        payload = {
            "model": model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": MINIMAX_TTS_SAMPLE_RATE,
                "format": "pcm",
                "channel": 1,
            },
            "stream_options": {
                "exclude_aggregated_audio": True,
            },
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = MINIMAX_TTS_BASE_URL + MINIMAX_TTS_ENDPOINT
        audio_hex_parts: list[str] = []

        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()

                buffer = ""
                for chunk in resp.iter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()  # keep incomplete last line

                    for line in lines:
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        json_str = line[len("data:"):].strip()
                        if not json_str or json_str == "[DONE]":
                            continue

                        import json

                        try:
                            event = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        # status=1: audio chunk; status=2: aggregated (we skip it)
                        event_data = event.get("data", {})
                        status = event_data.get("status")
                        audio_hex = event_data.get("audio", "")

                        if status == 1 and audio_hex:
                            audio_hex_parts.append(audio_hex)

                        # Surface API-level errors
                        base_resp = event.get("base_resp", {})
                        status_code = base_resp.get("status_code", 0)
                        if status_code != 0:
                            msg = base_resp.get("status_msg", "Unknown error")
                            raise RuntimeError(
                                f"MiniMax TTS API error {status_code}: {msg}"
                            )

        if not audio_hex_parts:
            raise RuntimeError(
                "MiniMax TTS API returned no audio data. "
                "Check your MINIMAX_API_KEY and request parameters."
            )

        pcm_bytes = bytes.fromhex("".join(audio_hex_parts))
        logger.debug("MiniMax TTS: received %d bytes of PCM audio", len(pcm_bytes))
        return pcm_bytes
