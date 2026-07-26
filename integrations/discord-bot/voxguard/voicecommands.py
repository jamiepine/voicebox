"""Spoken commands.

You talk in the voice channel, the bot transcribes you, and the agent carries
out what you asked — then answers back in the cloned voice.

The whole feature turns on one question: **who said it?** Discord gives us the
speaker per audio packet, so every utterance arrives already attributed. That
attribution is what makes this safe to build at all — without it, "ban
everyone" from any random member in the channel would be indistinguishable
from the same words spoken by the owner.

So the tiers an utterance runs with are the *intersection* of what the guild
enabled and what that specific speaker could do by typing the equivalent slash
command. A member with no moderation permissions gets a conversational agent.
An admin gets the full toolset. Nobody gains authority by speaking rather than
typing.
"""

from __future__ import annotations

import logging
import re

import discord

from . import guardrails
from .agent import AgentContext

log = logging.getLogger(__name__)

# Phrasings that mean "this was aimed at the bot". Matched on the normalised
# transcript, so punctuation and casing don't matter.
ADDRESS_PATTERNS = [
    r"^(hey|ok|okay|yo|hi|hello)\s+{name}\b",
    r"^{name}\b",
    r"\b{name}[,\s]+(can you|could you|please|would you)\b",
]

# Verbs that indicate an instruction rather than chatter. Used only to decide
# whether to offer tools, never to authorise anything.
COMMAND_HINTS = re.compile(
    r"\b(ban|kick|mute|timeout|warn|purge|delete|create|make|rename|lock|unlock|"
    r"slowmode|assign|give|remove|role|channel|thread|ticket|announce|say|"
    r"set|enable|disable|start|stop|close|open)\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.casefold()).strip()


def addressed_to(text: str, wake_words: list[str]) -> bool:
    """Was this utterance directed at the bot?"""
    clean = normalise(text)
    for raw in wake_words:
        name = re.escape(normalise(raw))
        if not name:
            continue
        for pattern in ADDRESS_PATTERNS:
            if re.search(pattern.format(name=name), clean):
                return True
    return False


def strip_wake_word(text: str, wake_words: list[str]) -> str:
    """Remove the leading address so the agent sees just the request."""
    result = text.strip()
    for raw in sorted(wake_words, key=len, reverse=True):
        pattern = re.compile(
            rf"^\s*(hey|ok|okay|yo|hi|hello)?\s*{re.escape(raw)}\s*[,:!.]*\s*", re.IGNORECASE
        )
        stripped = pattern.sub("", result, count=1)
        if stripped != result:
            return stripped.strip() or result
    return result


def looks_like_command(text: str) -> bool:
    return bool(COMMAND_HINTS.search(text))


def speaker_tiers(
    member: discord.Member | None,
    guild: discord.Guild,
    config: dict,
    owner_ids: set[int],
) -> tuple[str, ...]:
    """Tiers this speaker may drive, capped by their own Discord permissions.

    Configured tiers are a ceiling, not a grant. Speaking is not a privilege
    escalation path: a member who cannot ban by typing `/ban` cannot ban by
    saying it out loud either.
    """
    configured = set(config.get("roam", {}).get("tiers") or ["chat"])
    configured.add("chat")

    if member is None:
        return ("chat",)

    allowed = {"chat"}
    perms = member.guild_permissions

    if guardrails.is_operator(member, guild, owner_ids):
        allowed |= {"manage", "moderate"}
    else:
        if perms.manage_channels or perms.manage_roles:
            allowed.add("manage")
        if perms.ban_members or perms.kick_members or perms.moderate_members:
            allowed.add("moderate")

    effective = configured & allowed
    return tuple(t for t in ("chat", "manage", "moderate") if t in effective)


class VoiceCommandRouter:
    """Decides what an utterance is, and with what authority it runs."""

    def __init__(self, owner_ids: set[int]) -> None:
        self.owner_ids = owner_ids

    def wake_words(self, config: dict, bot_name: str) -> list[str]:
        cfg = config.get("voice_commands", {})
        words = [w for w in (cfg.get("wake_words") or []) if w.strip()]
        if not words:
            words = [bot_name]
        # The bound voice profile's name is a natural thing to call it.
        if persona := config.get("ai", {}).get("voice_profile_name"):
            words.append(persona)
        return words

    def evaluate(
        self,
        text: str,
        member: discord.Member | None,
        guild: discord.Guild,
        config: dict,
        bot_name: str,
        *,
        conversation_mode: bool,
    ) -> tuple[bool, str, tuple[str, ...]] | None:
        """Returns (should_respond, cleaned_text, tiers) or None to ignore."""
        cfg = config.get("voice_commands", {})
        words = self.wake_words(config, bot_name)
        was_addressed = addressed_to(text, words)

        # In conversation mode (/vctalk) the bot answers everything; otherwise
        # it only acts when spoken to by name, so it isn't executing fragments
        # of a conversation between two other people.
        if not conversation_mode and not was_addressed:
            return None
        if not conversation_mode and not cfg.get("enabled", False):
            return None

        cleaned = strip_wake_word(text, words) if was_addressed else text.strip()
        if not cleaned:
            return None

        tiers = speaker_tiers(member, guild, config, self.owner_ids)

        # Only widen past chat when the utterance actually reads like an
        # instruction and the speaker was addressing the bot. Idle chatter
        # never gets handed destructive tools.
        if not (was_addressed and looks_like_command(cleaned)):
            tiers = ("chat",)
        elif not cfg.get("allow_actions", True):
            tiers = ("chat",)

        return True, cleaned, tiers


def build_context(
    guild: discord.Guild,
    channel: discord.abc.Messageable | None,
    member: discord.Member | None,
    config: dict,
    tiers: tuple[str, ...],
    voice_client: discord.VoiceClient | None,
) -> AgentContext:
    return AgentContext(
        guild=guild,
        channel=channel,
        invoker=member,
        config=config,
        allowed_tiers=tiers,
        voice_client=voice_client,
        approval_channel=channel,
    )
