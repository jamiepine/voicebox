"""
MiniMax Cloud TTS backend implementation.

Wraps MiniMax's Text-to-Speech API for cloud-based voice synthesis.
Two model variants:
  - speech-2.8-hd: High-quality, maximized timbre similarity (default)
  - speech-2.8-turbo: Faster, more affordable version

Unlike local backends, this requires a MINIMAX_API_KEY environment variable
and makes HTTP requests to the MiniMax API. No local model downloads needed.

24kHz output, PCM audio format.
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MINIMAX_API_BASE = "https://api.minimax.io/v1"
MINIMAX_TTS_ENDPOINT = f"{MINIMAX_API_BASE}/t2a_v2"

MINIMAX_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_SAMPLE_RATE = 24000

# Available preset voice IDs
MINIMAX_VOICES = [
    ("English_Graceful_Lady", "Graceful Lady", "female", "en"),
    ("English_Insightful_Speaker", "Insightful Speaker", "male", "en"),
    ("English_radiant_girl", "Radiant Girl", "female", "en"),
    ("English_Persuasive_Man", "Persuasive Man", "male", "en"),
    ("English_Lucky_Robot", "Lucky Robot", "male", "en"),
    ("Wise_Woman", "Wise Woman", "female", "en"),
    ("cute_boy", "Cute Boy", "male", "en"),
    ("lovely_girl", "Lovely Girl", "female", "en"),
    ("Friendly_Person", "Friendly Person", "male", "en"),
    ("Inspirational_girl", "Inspirational Girl", "female", "en"),
    ("Deep_Voice_Man", "Deep Voice Man", "male", "en"),
    ("sweet_girl", "Sweet Girl", "female", "en"),
]

DEFAULT_VOICE_ID = "English_Graceful_Lady"


class MiniMaxTTSBackend:
    """MiniMax Cloud TTS backend for cloud-based voice synthesis."""

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model: str = MINIMAX_DEFAULT_MODEL
        self._ready = False

    def is_loaded(self) -> bool:
        return self._ready

    def _get_model_path(self, model_size: str = "default") -> str:
        return MINIMAX_DEFAULT_MODEL

    def _is_model_cached(self, model_size: str = "default") -> bool:
        # Cloud backend — always "cached" (no download needed)
        return True

    async def load_model(self, model_size: str = "default") -> None:
        """Validate API key availability. No model download needed."""
        if self._ready:
            return

        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY environment variable is required for MiniMax TTS. "
                "Get your API key from https://platform.minimax.io"
            )
        self._api_key = api_key
        self._ready = True
        logger.info("MiniMax Cloud TTS ready (model: %s)", self._model)

    def unload_model(self) -> None:
        """Clear API key reference."""
        self._api_key = None
        self._ready = False
        logger.info("MiniMax Cloud TTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        MiniMax TTS uses preset voice IDs, not reference audio cloning.

        Returns a preset voice prompt using the default voice ID.
        The reference audio is ignored.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "minimax",
            "preset_voice_id": DEFAULT_VOICE_ID,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """Not supported — MiniMax uses preset voices, not audio cloning."""
        raise NotImplementedError(
            "MiniMax Cloud TTS uses preset voice IDs and does not support "
            "voice cloning from reference audio."
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio via MiniMax TTS API.

        Args:
            text: Text to synthesize (max 10,000 chars)
            voice_prompt: Dict with voice_type and preset_voice_id
            language: Language code (MiniMax auto-detects language)
            seed: Not supported by MiniMax TTS (ignored)
            instruct: Not supported by MiniMax TTS (ignored)

        Returns:
            Tuple of (audio_array, sample_rate=24000)
        """
        await self.load_model()

        voice_id = DEFAULT_VOICE_ID
        if isinstance(voice_prompt, dict):
            voice_id = voice_prompt.get("preset_voice_id", DEFAULT_VOICE_ID)

        def _generate_sync():
            payload = {
                "model": self._model,
                "text": text,
                "stream": False,
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "format": "pcm",
                    "sample_rate": MINIMAX_SAMPLE_RATE,
                },
            }

            req = urllib.request.Request(
                MINIMAX_TTS_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )

            logger.info(
                "[MiniMax TTS] Generating (%s), voice: %s, text length: %d",
                language,
                voice_id,
                len(text),
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"MiniMax TTS API error ({e.code}): {error_body}"
                ) from e

            # Check for API-level errors
            base_resp = body.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                raise RuntimeError(
                    f"MiniMax TTS API error: {base_resp.get('status_msg', 'unknown')}"
                )

            # Extract hex-encoded audio
            audio_hex = body.get("data", {}).get("audio", "")
            if not audio_hex:
                raise RuntimeError("MiniMax TTS API returned empty audio data")

            # Decode hex → raw PCM bytes → float32 numpy array
            audio_bytes = bytes.fromhex(audio_hex)
            audio = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            return audio, MINIMAX_SAMPLE_RATE

        return await asyncio.to_thread(_generate_sync)
