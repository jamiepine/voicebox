"""Text-channel automod and anti-nuke.

Two separate jobs that share a file because they share a shape: watch a
stream of events, score them, act within the same guardrails as everything
else.

**Automod** is the text-side counterpart of the voice filter (Dyno/Carl-bot
territory): invites, link spam, mass mentions, message flooding, all-caps,
and the blocked-word list.

**Anti-nuke** is the one that matters most on a bot holding Administrator.
It watches for a *single actor* mass-deleting channels or roles, or mass
banning, and strips their privileged roles. That's aimed at a compromised
admin account or a rogue moderator — including, deliberately, this bot's own
AI agent if it ever gets talked into a rampage.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import discord

from ..matching import Matcher
from ..store import Store

log = logging.getLogger(__name__)

INVITE_RE = re.compile(r"(?:discord\.(?:gg|io|me|li)|discordapp\.com/invite)/[a-z0-9-]+", re.I)
URL_RE = re.compile(r"https?://([^\s/]+)", re.I)


@dataclass
class Trigger:
    rule: str
    detail: str
    action: str


class TextAutomod:
    def __init__(self, store: Store) -> None:
        self.store = store
        # (guild, user) -> recent message timestamps, for flood detection.
        self._recent: dict[tuple[int, int], deque[float]] = defaultdict(lambda: deque(maxlen=20))

    def check(self, message: discord.Message, config: dict, matcher: Matcher | None) -> Trigger | None:
        cfg = config.get("automod", {})
        if not cfg.get("enabled", False):
            return None
        rules = cfg.get("rules", {})
        content = message.content or ""

        invites = rules.get("invites", {})
        if invites.get("enabled") and INVITE_RE.search(content):
            return Trigger("invites", "Discord invite link", invites.get("action", "delete"))

        links = rules.get("links", {})
        if links.get("enabled"):
            allowed = {d.lower().lstrip("www.") for d in links.get("allowed_domains", [])}
            for host in URL_RE.findall(content):
                bare = host.lower().split(":")[0].lstrip("www.")
                if bare not in allowed:
                    return Trigger("links", f"link to {bare}", links.get("action", "delete"))

        mentions = rules.get("mass_mentions", {})
        if mentions.get("enabled"):
            limit = int(mentions.get("limit", 5))
            total = len(message.mentions) + len(message.role_mentions)
            if total >= limit:
                return Trigger(
                    "mass_mentions", f"{total} mentions (limit {limit})",
                    mentions.get("action", "timeout"),
                )

        spam = rules.get("spam", {})
        if spam.get("enabled"):
            window = float(spam.get("seconds", 5))
            limit = int(spam.get("messages", 5))
            key = (message.guild.id, message.author.id)
            now = time.time()
            bucket = self._recent[key]
            bucket.append(now)
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                bucket.clear()
                return Trigger(
                    "spam", f"{limit} messages in {window:g}s", spam.get("action", "timeout")
                )

        caps = rules.get("caps", {})
        if caps.get("enabled") and len(content) >= int(caps.get("min_length", 10)):
            letters = [c for c in content if c.isalpha()]
            if letters:
                ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                threshold = int(caps.get("percent", 70)) / 100
                if ratio >= threshold:
                    return Trigger(
                        "caps", f"{ratio:.0%} caps", caps.get("action", "delete")
                    )

        words = rules.get("words", {})
        if words.get("enabled") and matcher is not None and len(matcher):
            hits = matcher.scan(content, min_confidence=0.7)
            if hits:
                return Trigger("words", f"blocked term '{hits[0].term}'", words.get("action", "delete"))

        return None


class AntiNuke:
    """Detects one actor performing destructive actions in bulk."""

    TRACKED = {
        "channel_delete": "channel_delete_limit",
        "role_delete": "role_delete_limit",
        "ban": "ban_limit",
        "kick": "kick_limit",
    }

    def __init__(self, store: Store) -> None:
        self.store = store
        self._actions: dict[tuple[int, int, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._recently_tripped: dict[tuple[int, int], float] = {}

    def record(
        self, guild_id: int, actor_id: int, kind: str, config: dict
    ) -> tuple[int, int] | None:
        """Record an action; returns (count, limit) if it breached the limit."""
        cfg = config.get("antinuke", {})
        if not cfg.get("enabled", False) or kind not in self.TRACKED:
            return None
        if str(actor_id) in {str(u) for u in cfg.get("whitelist", [])}:
            return None

        window = float(cfg.get("window_seconds", 30))
        limit = int(cfg.get(self.TRACKED[kind], 3))
        now = time.time()

        bucket = self._actions[(guild_id, actor_id, kind)]
        bucket.append(now)
        while bucket and now - bucket[0] > window:
            bucket.popleft()

        if len(bucket) < limit:
            return None

        # One response per actor per window.
        last = self._recently_tripped.get((guild_id, actor_id), 0)
        if now - last < window:
            return None
        self._recently_tripped[(guild_id, actor_id)] = now
        return len(bucket), limit

    async def respond(
        self, guild: discord.Guild, actor: discord.Member, kind: str, count: int, config: dict
    ) -> str:
        cfg = config.get("antinuke", {})
        response = cfg.get("response", "strip_roles")

        self.store.audit(
            guild.id, "antinuke", f"detected:{kind}", str(actor.id), f"count={count}"
        )
        self.store.bump_metric(guild.id, "antinuke_triggers")

        if actor.id == guild.owner_id:
            return f"⚠️ {actor.mention} ({kind} ×{count}) — server owner, not actioned."
        if response != "strip_roles":
            return f"⚠️ Anti-nuke: {actor.mention} performed {kind} ×{count}."

        # Remove every role that carries a dangerous permission and that we
        # actually outrank. This stops the bleeding without a ban, which
        # matters because the usual cause is a compromised account, not a
        # malicious person.
        dangerous = []
        for role in actor.roles:
            if role.is_default() or role >= guild.me.top_role:
                continue
            perms = role.permissions
            if (
                perms.administrator or perms.manage_guild or perms.manage_channels
                or perms.manage_roles or perms.ban_members or perms.kick_members
            ):
                dangerous.append(role)

        if not dangerous:
            return f"⚠️ Anti-nuke: {actor.mention} did {kind} ×{count} — no strippable roles."

        try:
            await actor.remove_roles(
                *dangerous, reason=f"VoxGuard anti-nuke: {kind} ×{count}"
            )
        except discord.HTTPException as exc:
            return f"⚠️ Anti-nuke tripped on {actor.mention} but role removal failed: {exc}"

        self.store.audit(guild.id, "antinuke", "strip_roles", str(actor.id), f"{len(dangerous)} roles")
        return (
            f"🛡️ **Anti-nuke**: stripped {len(dangerous)} privileged role(s) from "
            f"{actor.mention} after {kind} ×{count} in the detection window."
        )
