"""Async client for the local Voicebox API.

Voicebox does the heavy lifting for both directions of the audio loop:
Whisper transcription on the way in, cloned-voice TTS on the way out. This
module is the only place that knows the HTTP shapes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

# Voicebox answers 202 while a model downloads in the background. That isn't an
# error, it's "come back shortly", so those calls retry rather than fail.
MODEL_DOWNLOAD_RETRIES = 20
MODEL_DOWNLOAD_DELAY = 15.0


class VoiceboxError(RuntimeError):
    pass


class ModelDownloading(VoiceboxError):
    """Raised when a required model is still being fetched."""


# Engines that actually perform zero-shot cloning from a reference sample.
# `qwen_custom_voice` is NOT one of them — it synthesises from a fixed preset
# speaker and ignores reference audio entirely, so pointing a cloned profile
# at it silently produces a stranger's voice. That mismatch is the single
# most confusing failure mode in this integration, so engine choice is
# derived from the profile rather than left to a global default.
CLONING_ENGINES = {"qwen", "chatterbox", "chatterbox_turbo", "luxtts"}

# Engines that accept the natural-language `instruct` delivery hint.
INSTRUCT_ENGINES = {"qwen_custom_voice", "tada"}

DEFAULT_CLONE_ENGINE = "qwen"


@dataclass
class Profile:
    id: str
    name: str
    language: str = "en"
    voice_type: str = "cloned"
    preset_engine: str | None = None
    default_engine: str | None = None
    sample_count: int = 0

    def engine_for_speech(self, configured: str | None = None) -> str:
        """Pick an engine that can actually voice this profile."""
        if self.voice_type == "preset" and self.preset_engine:
            return self.preset_engine
        if self.default_engine:
            return self.default_engine
        # A cloned profile must use a cloning engine, whatever the global
        # default says.
        if configured and configured in CLONING_ENGINES:
            return configured
        return DEFAULT_CLONE_ENGINE


class VoiceboxClient:
    def __init__(self, base_url: str, *, whisper_model: str = "turbo", tts_engine: str = DEFAULT_CLONE_ENGINE):
        self.base_url = base_url.rstrip("/")
        self.whisper_model = whisper_model
        self.tts_engine = tts_engine
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300, sock_connect=10),
                headers={"X-Voicebox-Client-Id": "voxguard-discord"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health(self) -> bool:
        try:
            sess = await self._sess()
            async with sess.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
        except Exception:
            return False

    # -- speech in ----------------------------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "clip.wav",
        language: str | None = None,
        model: str | None = None,
    ) -> str:
        """Transcribe raw audio bytes. Returns the text (possibly empty)."""
        sess = await self._sess()
        for attempt in range(MODEL_DOWNLOAD_RETRIES):
            form = aiohttp.FormData()
            form.add_field("file", audio, filename=filename, content_type="application/octet-stream")
            form.add_field("model", model or self.whisper_model)
            if language:
                form.add_field("language", language)

            async with sess.post(f"{self.base_url}/transcribe", data=form) as r:
                if r.status == 202:
                    if attempt == 0:
                        log.info(
                            "Whisper model '%s' is downloading; transcription will start when it lands.",
                            model or self.whisper_model,
                        )
                    await asyncio.sleep(MODEL_DOWNLOAD_DELAY)
                    continue
                if r.status != 200:
                    raise VoiceboxError(f"transcribe failed ({r.status}): {(await r.text())[:300]}")
                payload = await r.json()
                return (payload.get("text") or "").strip()

        raise ModelDownloading("Whisper model is still downloading. Try again in a minute.")

    # -- profiles / cloning -------------------------------------------------

    async def list_profiles(self) -> list[Profile]:
        sess = await self._sess()
        async with sess.get(f"{self.base_url}/profiles") as r:
            if r.status != 200:
                raise VoiceboxError(f"list profiles failed ({r.status})")
            return [
                Profile(
                    id=p["id"],
                    name=p["name"],
                    language=p.get("language", "en"),
                    voice_type=p.get("voice_type", "cloned"),
                    preset_engine=p.get("preset_engine"),
                    default_engine=p.get("default_engine"),
                    sample_count=p.get("sample_count", 0),
                )
                for p in await r.json()
            ]

    async def get_profile(self, profile_id: str) -> Profile | None:
        sess = await self._sess()
        async with sess.get(f"{self.base_url}/profiles/{profile_id}") as r:
            if r.status != 200:
                return None
            p = await r.json()
            return Profile(
                id=p["id"],
                name=p["name"],
                language=p.get("language", "en"),
                voice_type=p.get("voice_type", "cloned"),
                preset_engine=p.get("preset_engine"),
                default_engine=p.get("default_engine"),
                sample_count=p.get("sample_count", 0),
            )

    async def find_profile(self, name_or_id: str) -> Profile | None:
        needle = name_or_id.strip().casefold()
        for profile in await self.list_profiles():
            if profile.id == name_or_id or profile.name.casefold() == needle:
                return profile
        return None

    async def create_profile(
        self,
        name: str,
        *,
        description: str | None = None,
        language: str = "en",
        personality: str | None = None,
    ) -> Profile:
        sess = await self._sess()
        body = {
            "name": name[:100],
            "description": (description or "")[:500] or None,
            "language": language,
            "voice_type": "cloned",
        }
        if personality:
            body["personality"] = personality[:2000]

        async with sess.post(f"{self.base_url}/profiles", json=body) as r:
            if r.status not in (200, 201):
                raise VoiceboxError(f"create profile failed ({r.status}): {(await r.text())[:300]}")
            data = await r.json()
            return Profile(
                id=data["id"],
                name=data["name"],
                language=data.get("language", "en"),
                voice_type=data.get("voice_type", "cloned"),
                default_engine=data.get("default_engine"),
            )

    async def add_sample(
        self, profile_id: str, audio: bytes, *, filename: str, reference_text: str
    ) -> dict:
        """Attach a reference sample — this is what makes the clone.

        `reference_text` should be a transcript of the sample; the zero-shot
        engines use it to align the reference audio.
        """
        sess = await self._sess()
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=filename, content_type="application/octet-stream")
        form.add_field("reference_text", reference_text)

        async with sess.post(f"{self.base_url}/profiles/{profile_id}/samples", data=form) as r:
            if r.status not in (200, 201):
                raise VoiceboxError(f"add sample failed ({r.status}): {(await r.text())[:300]}")
            return await r.json()

    async def delete_profile(self, profile_id: str) -> bool:
        sess = await self._sess()
        async with sess.delete(f"{self.base_url}/profiles/{profile_id}") as r:
            return r.status in (200, 204)

    # -- speech out ---------------------------------------------------------

    async def synthesize(
        self,
        profile_id: str,
        text: str,
        *,
        language: str = "en",
        instruct: str | None = None,
        engine: str | None = None,
        personality: bool = False,
        profile: "Profile | None" = None,
    ) -> bytes:
        """Generate speech and return WAV bytes.

        Uses /generate/stream so nothing is written to the Voicebox history —
        a bot that chats in VC would otherwise fill it with thousands of rows.
        `instruct` is the natural-language delivery hint ("sound amused,
        speak quickly"), which is how the agent expresses emotion.
        """
        sess = await self._sess()

        # Resolve the engine from the profile so a cloned voice never gets
        # routed to a preset-only engine (see CLONING_ENGINES).
        if engine is None:
            if profile is None:
                profile = await self.get_profile(profile_id)
            engine = (
                profile.engine_for_speech(self.tts_engine)
                if profile is not None
                else DEFAULT_CLONE_ENGINE
            )

        body: dict = {
            "profile_id": profile_id,
            "text": text[:5000],
            "language": language,
            "engine": engine,
            "personality": personality,
            "normalize": True,
        }
        # Sending `instruct` to an engine that doesn't implement it is at best
        # ignored and at worst a 422, so only include it where it means
        # something.
        if instruct and engine in INSTRUCT_ENGINES:
            body["instruct"] = instruct[:500]

        for attempt in range(MODEL_DOWNLOAD_RETRIES):
            async with sess.post(f"{self.base_url}/generate/stream", json=body) as r:
                if r.status == 202:
                    if attempt == 0:
                        log.info("TTS model is downloading; speech will start when it lands.")
                    await asyncio.sleep(MODEL_DOWNLOAD_DELAY)
                    continue
                if r.status != 200:
                    raise VoiceboxError(f"synthesize failed ({r.status}): {(await r.text())[:300]}")
                return await r.read()

        raise ModelDownloading("TTS model is still downloading. Try again in a minute.")
