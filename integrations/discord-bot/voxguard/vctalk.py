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
HANDLER_KEY = "vctalk"


class VCTalkController:
    def __init__(self, sessions: SessionManager, speaker: Speaker, agent: ServerAgent) -> None:
        self.sessions = sessions
        self.speaker = speaker
        self.agent = agent
        # guild_id -> channel to log conversation turns to
        self._active: dict[int, discord.abc.Messageable | None] = {}

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
        # Registering under a stable key makes this idempotent, and the key
        # is carried across a session replaced by a reconnect.
        session.add_handler(HANDLER_KEY, self._make_handler(guild.id))

    async def stop(self, guild_id: int) -> bool:
        if guild_id not in self._active:
            return False
        del self._active[guild_id]
        session = self.sessions.get(guild_id)
        if session is not None:
            session.remove_handler(HANDLER_KEY)
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
