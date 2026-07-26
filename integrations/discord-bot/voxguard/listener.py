"""Live voice capture, segmentation and transcription.

Discord hands us a continuous stream of 20 ms frames tagged by speaker. To get
usable transcripts we need utterances, not frames, so frames are buffered per
speaker and flushed when that speaker stops talking. Silence is the signal:
Discord simply stops sending frames for a user who isn't speaking, so a gap
longer than `silence_gap` ends the utterance.

Each finished utterance is transcribed and handed to every registered handler.
Moderation and conversation both consume the same stream — the bot transcribes
once per utterance no matter how many features are listening.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import discord

from . import audio
from .voicebox_client import VoiceboxClient, VoiceboxError

log = logging.getLogger(__name__)

try:
    from discord.ext import voice_recv
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "discord-ext-voice-recv is required for voice capture.\n"
        "Install it with: pip install discord-ext-voice-recv"
    ) from exc


# An utterance ends after this much silence from the speaker.
SILENCE_GAP = 0.8
# Ignore blips — a cough or a keyboard click isn't worth a Whisper call.
MIN_UTTERANCE = 0.45
# Force a flush on long monologues so moderation latency stays bounded.
MAX_UTTERANCE = 14.0
# Below this fraction of audible audio, the buffer is treated as noise.
MIN_SPEECH_RATIO = 0.18
# Concurrent transcription calls per session.
MAX_INFLIGHT = 3

# Whisper emits these for silence and background noise. Acting on a
# hallucinated phrase would be an unexplainable moderation action, so short
# transcripts that consist only of these are dropped.
HALLUCINATIONS = {
    "you",
    "thank you",
    "thanks for watching",
    "thanks for watching!",
    "bye",
    "bye.",
    "thank you.",
    "thank you for watching",
    ".",
    "...",
    "[blank_audio]",
    "[silence]",
    "subs by www.zeoranger.co.uk",
}


@dataclass
class Utterance:
    guild_id: int
    channel_id: int
    user_id: int
    text: str
    started_at: float
    duration: float


UtteranceHandler = Callable[[Utterance], Awaitable[None]]


class SegmentingSink(voice_recv.AudioSink):
    """Buffers decoded PCM per speaker and hands segments to the session."""

    def __init__(self, session: "VoiceSession") -> None:
        super().__init__()
        self.session = session

    def wants_opus(self) -> bool:
        return False

    def write(self, user: discord.abc.User | None, data) -> None:  # noqa: ANN001
        # Called on the voice receive thread — hop to the event loop rather
        # than touching session state from here.
        if user is None or not data.pcm:
            return
        self.session.submit_frame(user.id, data.pcm)

    def cleanup(self) -> None:
        return None


class VoiceSession:
    """One connected voice channel, its buffers, and its transcription loop."""

    def __init__(
        self,
        voice_client: "voice_recv.VoiceRecvClient",
        voicebox: VoiceboxClient,
        *,
        language: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.voice_client = voice_client
        self.voicebox = voicebox
        self.language = language
        self.loop = loop or asyncio.get_event_loop()

        self.guild_id = voice_client.guild.id
        self.channel_id = voice_client.channel.id

        self._buffers: dict[int, audio.SpeakerBuffer] = {}
        self._handlers: list[UtteranceHandler] = []
        self._flusher: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(MAX_INFLIGHT)
        self._pending: set[asyncio.Task] = set()
        self._paused_users: set[int] = set()
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.voice_client.listen(SegmentingSink(self))
        self._flusher = self.loop.create_task(self._flush_loop())
        log.info("Listening in guild=%s channel=%s", self.guild_id, self.channel_id)

    async def stop(self) -> None:
        self._running = False
        if self._flusher:
            self._flusher.cancel()
            try:
                await self._flusher
            except asyncio.CancelledError:
                pass
            self._flusher = None
        try:
            self.voice_client.stop_listening()
        except Exception:
            pass
        for task in list(self._pending):
            task.cancel()
        self._pending.clear()
        self._buffers.clear()

    def add_handler(self, handler: UtteranceHandler) -> None:
        self._handlers.append(handler)

    def remove_handler(self, handler: UtteranceHandler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    @property
    def handler_count(self) -> int:
        return len(self._handlers)

    def pause_user(self, user_id: int) -> None:
        """Stop capturing a user — used while the bot itself is speaking."""
        self._paused_users.add(user_id)

    def resume_user(self, user_id: int) -> None:
        self._paused_users.discard(user_id)
        self._buffers.pop(user_id, None)

    # -- capture ------------------------------------------------------------

    def submit_frame(self, user_id: int, pcm: bytes) -> None:
        """Thread-safe entry point from the sink."""
        if not self._running or user_id in self._paused_users:
            return
        self.loop.call_soon_threadsafe(self._append, user_id, pcm, time.monotonic())

    def _append(self, user_id: int, pcm: bytes, now: float) -> None:
        buffer = self._buffers.get(user_id)
        if buffer is None:
            buffer = audio.SpeakerBuffer(user_id=user_id)
            self._buffers[user_id] = buffer
        buffer.add(pcm, now)

        if buffer.duration >= MAX_UTTERANCE:
            self._dispatch(buffer)

    async def _flush_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(0.15)
                now = time.monotonic()
                for buffer in list(self._buffers.values()):
                    if not buffer.chunks:
                        continue
                    if now - buffer.last_frame_at >= SILENCE_GAP:
                        self._dispatch(buffer)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice flush loop crashed")

    def _dispatch(self, buffer: audio.SpeakerBuffer) -> None:
        duration = buffer.duration
        speech_ratio = buffer.speech_ratio
        user_id = buffer.user_id
        started_at = buffer.started_at
        pcm = buffer.drain()

        if duration < MIN_UTTERANCE or speech_ratio < MIN_SPEECH_RATIO:
            return

        task = self.loop.create_task(self._transcribe(user_id, pcm, started_at, duration))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _transcribe(
        self, user_id: int, pcm: bytes, started_at: float, duration: float
    ) -> None:
        async with self._semaphore:
            try:
                wav = audio.pcm_to_wav(pcm)
                text = await self.voicebox.transcribe(
                    wav, filename=f"vc-{user_id}.wav", language=self.language
                )
            except VoiceboxError as exc:
                log.warning("Transcription failed: %s", exc)
                return
            except Exception:
                log.exception("Unexpected transcription error")
                return

        cleaned = text.strip()
        if not cleaned:
            return
        if len(cleaned) < 30 and cleaned.casefold().strip(" .!?") in {
            h.strip(" .!?") for h in HALLUCINATIONS
        }:
            return

        utterance = Utterance(
            guild_id=self.guild_id,
            channel_id=self.channel_id,
            user_id=user_id,
            text=cleaned,
            started_at=started_at,
            duration=duration,
        )

        for handler in list(self._handlers):
            try:
                await handler(utterance)
            except Exception:
                log.exception("Utterance handler failed")


class SessionManager:
    """Tracks one VoiceSession per guild."""

    def __init__(self, voicebox: VoiceboxClient) -> None:
        self.voicebox = voicebox
        self._sessions: dict[int, VoiceSession] = {}

    def get(self, guild_id: int) -> VoiceSession | None:
        return self._sessions.get(guild_id)

    async def join(
        self, channel: discord.VoiceChannel | discord.StageChannel, *, language: str | None = None
    ) -> VoiceSession:
        guild_id = channel.guild.id
        existing = self._sessions.get(guild_id)

        if existing is not None:
            client = existing.voice_client
            if client.is_connected():
                if client.channel and client.channel.id == channel.id:
                    return existing
                await client.move_to(channel)
                existing.channel_id = channel.id
                return existing
            await self.leave(guild_id)

        client = await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30.0)
        session = VoiceSession(client, self.voicebox, language=language)
        session.start()
        self._sessions[guild_id] = session
        return session

    async def leave(self, guild_id: int) -> bool:
        session = self._sessions.pop(guild_id, None)
        if session is None:
            return False
        await session.stop()
        try:
            await session.voice_client.disconnect(force=True)
        except Exception:
            pass
        return True

    async def shutdown(self) -> None:
        for guild_id in list(self._sessions):
            await self.leave(guild_id)
