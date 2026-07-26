"""Live voice conversation and spoken commands.

Reuses the same VoiceSession utterance stream as moderation (one Whisper call
per utterance, not two) and the same ServerAgent as text chat and `/roam`, so
a personality set once applies everywhere the bot talks.

What's specific to this mode is turn-taking and authority:

* While a reply is being spoken, the speaker who triggered it is paused, so
  the bot doesn't transcribe its own crosstalk into the next turn.
* Every utterance runs with the tiers its *speaker* is entitled to — see
  `voicecommands.speaker_tiers`. Saying "ban them" only bans if the person who
  said it could have run `/ban`.
"""

from __future__ import annotations

import logging

import discord

from .agent import ServerAgent
from .listener import SessionManager, Utterance
from .tts import Speaker
from .voicebox_client import VoiceboxError
from .voicecommands import VoiceCommandRouter, build_context

log = logging.getLogger(__name__)

MIN_UTTERANCE_CHARS = 2
HANDLER_KEY = "vctalk"


class VCTalkController:
    def __init__(
        self,
        sessions: SessionManager,
        speaker: Speaker,
        agent: ServerAgent,
        router: VoiceCommandRouter,
    ) -> None:
        self.sessions = sessions
        self.speaker = speaker
        self.agent = agent
        self.router = router
        # guild_id -> channel used for transcript/approval posts
        self._active: dict[int, discord.abc.Messageable | None] = {}
        self._resolver = None

    def set_resolver(self, resolver) -> None:
        """resolver(guild_id) -> (guild, config, voice_client, channel) | None"""
        self._resolver = resolver

    def is_active(self, guild_id: int) -> bool:
        return guild_id in self._active

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
        # Stable key: idempotent across repeat /vctalk, and carried onto a
        # replacement session after a reconnect.
        session.add_handler(HANDLER_KEY, self._make_handler(guild.id, conversation=True))

    async def stop(self, guild_id: int) -> bool:
        if guild_id not in self._active:
            return False
        del self._active[guild_id]
        session = self.sessions.get(guild_id)
        if session is not None:
            session.remove_handler(HANDLER_KEY)
        return True

    def attach_command_listener(self, guild_id: int) -> bool:
        """Listen for wake-word commands without full conversation mode.

        Used by `/join` so a channel the bot is moderating can also take
        spoken instructions, without it replying to every sentence.
        """
        session = self.sessions.get(guild_id)
        if session is None:
            return False
        session.add_handler("voicecmd", self._make_handler(guild_id, conversation=False))
        return True

    def detach_command_listener(self, guild_id: int) -> bool:
        session = self.sessions.get(guild_id)
        return bool(session and session.remove_handler("voicecmd"))

    def _make_handler(self, guild_id: int, *, conversation: bool):
        async def handle(utterance: Utterance) -> None:
            if conversation and guild_id not in self._active:
                return
            if len(utterance.text.strip()) < MIN_UTTERANCE_CHARS:
                return
            await self._handle(guild_id, utterance, conversation)

        return handle

    async def _handle(self, guild_id: int, utterance: Utterance, conversation: bool) -> None:
        if self._resolver is None:
            return
        resolved = self._resolver(guild_id)
        if resolved is None:
            return
        guild, config, voice_client, channel = resolved

        member = guild.get_member(utterance.user_id)
        author = member.display_name if member else "Someone"
        bot_name = guild.me.display_name if guild.me else "VoxGuard"

        decision = self.router.evaluate(
            utterance.text, member, guild, config, bot_name, conversation_mode=conversation
        )
        if decision is None:
            return
        _, cleaned, tiers = decision

        log.info("[voice] %s -> %s (tiers: %s)", author, cleaned, ",".join(tiers))

        session = self.sessions.get(guild_id)
        if session:
            session.pause_user(utterance.user_id)
        try:
            ctx = build_context(guild, channel, member, config, tiers, voice_client)
            reply = await self.agent.respond(ctx, author, cleaned)

            if reply.actions:
                log.info("[voice] actions: %s", "; ".join(reply.actions))
            await self._post_transcript(guild_id, author, cleaned, reply)

            if reply.text.strip():
                await self._speak(guild, config, voice_client, reply)
        except Exception:
            log.exception("Voice turn failed in guild=%s", guild_id)
        finally:
            if session:
                session.resume_user(utterance.user_id)

    async def _speak(self, guild, config, voice_client, reply) -> None:
        profile_id = config.get("ai", {}).get("voice_profile_id")
        if not profile_id:
            log.info("[voice] no voice profile bound; reply not spoken: %s", reply.text)
            return
        if voice_client is None or not voice_client.is_connected():
            return
        try:
            await self.speaker.speak(
                voice_client,
                profile_id,
                reply.text,
                instruct=reply.delivery if config.get("ai", {}).get("emotion", True) else None,
            )
        except VoiceboxError as exc:
            log.warning("[voice] synthesis failed: %s", exc)

    async def _post_transcript(self, guild_id: int, author: str, said: str, reply) -> None:
        channel = self._active.get(guild_id)
        if channel is None:
            return
        lines = [f"**{author}:** {said}"]
        if reply.text.strip():
            lines.append(f"**Bot:** {reply.text}")
        for action in reply.actions:
            lines.append(f"`{action}`")
        for proposal in reply.proposals:
            lines.append(f"*{proposal}*")
        try:
            await channel.send(
                "\n".join(lines)[:1900], allowed_mentions=discord.AllowedMentions.none()
            )
        except discord.HTTPException:
            pass
