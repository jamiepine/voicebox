"""Speaking into a voice channel.

Voicebox returns a WAV; discord.py wants a PCM source. FFmpeg bridges the two
over a pipe, so nothing hits disk. Playback is serialised per guild — two
overlapping `play()` calls on one voice client would otherwise drop the first.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil

import discord

from .voicebox_client import VoiceboxClient, VoiceboxError

log = logging.getLogger(__name__)

FFMPEG_MISSING = (
    "FFmpeg wasn't found on PATH. Voice playback needs it — install it with "
    "`apt install ffmpeg`, `brew install ffmpeg`, or `winget install Gyan.FFmpeg`."
)


class Speaker:
    """Serialised TTS playback, one lock per guild."""

    def __init__(self, voicebox: VoiceboxClient) -> None:
        self.voicebox = voicebox
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self, guild_id: int) -> asyncio.Lock:
        lock = self._locks.get(guild_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[guild_id] = lock
        return lock

    async def speak(
        self,
        voice_client: discord.VoiceClient,
        profile_id: str,
        text: str,
        *,
        language: str = "en",
        instruct: str | None = None,
        personality: bool = False,
    ) -> None:
        """Synthesize `text` and play it, returning when playback finishes."""
        if shutil.which("ffmpeg") is None:
            raise VoiceboxError(FFMPEG_MISSING)

        wav = await self.voicebox.synthesize(
            profile_id,
            text,
            language=language,
            instruct=instruct,
            personality=personality,
        )
        await self.play_wav(voice_client, wav)

    async def play_wav(self, voice_client: discord.VoiceClient, wav: bytes) -> None:
        if not voice_client.is_connected():
            return

        async with self._lock(voice_client.guild.id):
            # Wait out anything already playing on this client.
            for _ in range(600):
                if not voice_client.is_playing():
                    break
                await asyncio.sleep(0.1)
            else:
                voice_client.stop()

            done = asyncio.Event()
            loop = asyncio.get_running_loop()

            def finished(error: Exception | None) -> None:
                if error:
                    log.warning("Voice playback error: %s", error)
                loop.call_soon_threadsafe(done.set)

            source = discord.FFmpegPCMAudio(
                io.BytesIO(wav),
                pipe=True,
                before_options="-hide_banner -loglevel error",
            )
            try:
                voice_client.play(source, after=finished)
            except discord.ClientException as exc:
                log.warning("Could not start playback: %s", exc)
                return

            try:
                await asyncio.wait_for(done.wait(), timeout=180)
            except asyncio.TimeoutError:
                voice_client.stop()
