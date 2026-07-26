"""`/roam` — the agent's unprompted presence in text channels.

Two behaviours live here, both gated by `roam.enabled` and the channel
allowlist:

* **Mention/reply triggers** — any message that pings the bot or replies to
  one of its messages gets a response, regardless of idle timing.
* **Idle interjection** — a slim chance per eligible message that the bot
  chimes in unprompted, throttled to `idle_reply_seconds` per channel so it
  doesn't dominate the conversation.

Tool access in roam still passes through the same tier gate as everywhere
else (`agent.invoke`), so "free will" here means free to *decide* to act, not
free of the checks on *what* it's allowed to act on.
"""

from __future__ import annotations

import logging
import random
import time

import discord

from .agent import AgentContext, ServerAgent

log = logging.getLogger(__name__)

IDLE_CHANCE = 0.06  # per eligible message, once the cooldown has elapsed
MIN_MESSAGE_CHARS = 6


class RoamController:
    def __init__(self, agent: ServerAgent) -> None:
        self.agent = agent
        self._last_spoke: dict[tuple[int, int], float] = {}

    def channel_eligible(self, config: dict, channel_id: int) -> bool:
        roam = config.get("roam", {})
        if not roam.get("enabled", False):
            return False
        allowlist = roam.get("channels") or []
        return not allowlist or str(channel_id) in {str(c) for c in allowlist}

    def should_reply(self, message: discord.Message, config: dict, bot_user_id: int) -> bool:
        if not self.channel_eligible(config, message.channel.id):
            return False

        mentioned = bot_user_id in {m.id for m in message.mentions}
        replied_to_bot = False
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            replied_to_bot = message.reference.resolved.author.id == bot_user_id
        if mentioned or replied_to_bot:
            return True

        if len(message.content.strip()) < MIN_MESSAGE_CHARS:
            return False

        key = (message.guild.id, message.channel.id)
        cooldown = int(config.get("roam", {}).get("idle_reply_seconds", 300))
        last = self._last_spoke.get(key, 0)
        if time.time() - last < cooldown:
            return False

        return random.random() < IDLE_CHANCE

    def mark_spoke(self, guild_id: int, channel_id: int) -> None:
        self._last_spoke[(guild_id, channel_id)] = time.time()

    async def handle_message(
        self, message: discord.Message, config: dict, allowed_tiers: tuple[str, ...]
    ) -> None:
        if not isinstance(message.author, discord.Member) or message.author.bot:
            return

        clean_content = message.clean_content.strip()
        if not clean_content:
            return

        ctx = AgentContext(
            guild=message.guild,
            channel=message.channel,
            invoker=message.author,
            config=config,
            allowed_tiers=allowed_tiers,
            approval_channel=self._audit_channel(message.guild, config) or message.channel,
        )

        async with message.channel.typing():
            reply = await self.agent.respond(ctx, message.author.display_name, clean_content)

        self.mark_spoke(message.guild.id, message.channel.id)
        if reply.text.strip():
            for chunk in _split(reply.text, 1900):
                await message.channel.send(chunk)
        if reply.proposals:
            log.info("[roam] proposed: %s", "; ".join(reply.proposals))

    @staticmethod
    def _audit_channel(guild: discord.Guild, config: dict) -> discord.abc.Messageable | None:
        channel_id = config.get("roam", {}).get("audit_channel_id")
        if not channel_id:
            return None
        channel = guild.get_channel(int(channel_id))
        return channel if isinstance(channel, discord.abc.Messageable) else None


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, current = [], []
    length = 0
    for word in text.split(" "):
        if length + len(word) + 1 > limit:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
