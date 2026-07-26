"""`/vctalk` — live voice conversation with the agent.

Reuses the same VoiceSession utterance stream as moderation (one Whisper call
per utterance, not two), and reuses the same ServerAgent as text chat and
`/roam`, so a personality set once applies everywhere the bot talks. What's
specific to this mode is turn-taking: while the bot's reply is being spoken,
the speaker who triggered it is paused so the bot doesn't transcribe its own
turn's crosstalk into the next one.
"""

from __future__ import annotations

import logging

import discord

from .agent import ServerAgent
from .listener import SessionManager, Utterance
from .tts import Speaker
from .voicebox_client import VoiceboxError

log = logging.getLogger(__name__)

MIN_UTTERANCE_CHARS = 2


class VCTalkController:
    def __init__(self, sessions: SessionManager, speaker: Speaker, agent: ServerAgent) -> None:
        self.sessions = sessions
        self.speaker = speaker
        self.agent = agent
        # guild_id -> channel to log conversation turns to
        self._active: dict[int, discord.abc.Messageable | None] = {}
        # guild_id -> (session, handler) currently attached, so start() can
        # tell a live session from a stale one (e.g. after a disconnect that
        # skipped /vctalk stop) and stop() can detach exactly the right handler.
        self._handlers: dict[int, tuple[object, object]] = {}

    async def start(
        self,
        guild: discord.Guild,
        voice_channel: discord.VoiceChannel | discord.StageChannel,
        *,
        text_log_channel: discord.abc.Messageable | None,
        language: str | None,
    ) -> None:
        session = await self.sessions.join(voice_channel, language=language)
        self._active[guild.id] = text_log_channel

        existing = self._handlers.get(guild.id)
        if existing is not None and existing[0] is session:
            return

        if existing is not None:
            existing[0].remove_handler(existing[1])  # type: ignore[attr-defined]

        handler = self._make_handler(guild.id)
        session.add_handler(handler)
        self._handlers[guild.id] = (session, handler)

    async def stop(self, guild_id: int) -> bool:
        if guild_id not in self._active:
            return False
        del self._active[guild_id]
        existing = self._handlers.pop(guild_id, None)
        if existing is not None:
            existing[0].remove_handler(existing[1])  # type: ignore[attr-defined]
        return True

    def _make_handler(self, guild_id: int):
        async def handle(utterance: Utterance) -> None:
            if guild_id not in self._active:
                return
            if len(utterance.text.strip()) < MIN_UTTERANCE_CHARS:
                return
            await self._handle_utterance(guild_id, utterance)

        return handle

    # bot.py supplies the actual context resolver; see set_resolver().
    _resolver = None

    def set_resolver(self, resolver) -> None:
        """resolver(guild_id) -> (discord.Guild, agent.AgentContext, voice_client) | None"""
        self._resolver = resolver

    async def _handle_utterance(self, guild_id: int, utterance: Utterance) -> None:
        if self._resolver is None:
            return
        resolved = self._resolver(guild_id)
        if resolved is None:
            return
        guild, ctx, voice_client = resolved
        speaker_member = guild.get_member(utterance.user_id)
        author_name = speaker_member.display_name if speaker_member else "Someone"

        log.info("[vctalk] %s: %s", author_name, utterance.text)

        session = self.sessions.get(guild_id)
        if session:
            session.pause_user(utterance.user_id)
        try:
            reply = await self.agent.respond(
                ctx, author_name, utterance.text, use_tools="chat" in ctx.allowed_tiers
            )
            if not reply.text.strip():
                return

            profile_id = ctx.config.get("ai", {}).get("voice_profile_id")
            if not profile_id:
                log.info("[vctalk] (no voice profile bound — reply not spoken: %s)", reply.text)
                return

            try:
                await self.speaker.speak(
                    voice_client,
                    profile_id,
                    reply.text,
                    instruct=reply.delivery if ctx.config.get("ai", {}).get("emotion", True) else None,
                    personality=False,
                )
            except VoiceboxError as exc:
                log.warning("[vctalk] speech synthesis failed: %s", exc)
        finally:
            if session:
                session.resume_user(utterance.user_id)
