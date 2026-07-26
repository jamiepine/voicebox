"""Safety rails shared by every automated action.

An automated moderator with Administrator is one bad match away from banning
the wrong person, and it acts on input anyone in the server can produce —
speech in a voice channel, text in a chat, a filename. Three checks stand
between a detection and an irreversible action:

* **Immunity** — staff, the guild owner, the bot itself, and anyone above the
  bot in the role hierarchy are never actioned automatically.
* **Feasibility** — Discord's own hierarchy and permission rules are checked
  before the call, so failures are reported rather than swallowed.
* **A circuit breaker** — if automated enforcement exceeds a per-hour budget,
  it drops to log-only and shouts. A filter matching far more than expected
  is the signature of a bad word list or a false-positive storm, and the safe
  response is to stop acting, not to keep going faster.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import discord

from .store import Store


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = Verdict(True)


def is_operator(member: discord.abc.User, guild: discord.Guild, owner_ids: set[int]) -> bool:
    """May this user change enforcement settings?"""
    if member.id in owner_ids or member.id == guild.owner_id:
        return True
    perms = getattr(member, "guild_permissions", None)
    return bool(perms and (perms.administrator or perms.manage_guild))


def is_immune(member: discord.Member, config: dict) -> Verdict:
    """Is this member shielded from automated enforcement?"""
    guard = config.get("guardrails", {})

    if member.bot:
        return Verdict(False, "target is a bot")
    if member.id == member.guild.owner_id:
        return Verdict(False, "target is the guild owner")
    if str(member.id) in {str(u) for u in guard.get("immune_users", [])}:
        return Verdict(False, "target is on the immunity list")

    immune_roles = {str(r) for r in guard.get("immune_roles", [])}
    if immune_roles and any(str(role.id) in immune_roles for role in member.roles):
        return Verdict(False, "target holds an immune role")

    perms = member.guild_permissions
    for name in guard.get("immune_permissions", []):
        if getattr(perms, name, False):
            return Verdict(False, f"target has the `{name}` permission")

    return ALLOWED


def can_action(guild: discord.Guild, member: discord.Member, action: str) -> Verdict:
    """Can the bot actually perform `action` on `member` right now?"""
    me = guild.me
    if me is None:
        return Verdict(False, "bot member record unavailable")
    if member.id == me.id:
        return Verdict(False, "refusing to action myself")

    if me.top_role <= member.top_role:
        return Verdict(False, "target's highest role is at or above mine")

    perms = me.guild_permissions
    needed = {
        "timeout": ("moderate_members", perms.moderate_members),
        "kick": ("kick_members", perms.kick_members),
        "ban": ("ban_members", perms.ban_members),
    }
    if action in needed:
        name, held = needed[action]
        if not held:
            return Verdict(False, f"I'm missing the `{name}` permission")

    return ALLOWED


class RateLimiter:
    """Per-guild circuit breaker over automated actions.

    Counts are kept in memory and backed by the audit table, so a restart
    mid-storm doesn't hand the bot a fresh budget.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self._tripped: dict[int, float] = {}

    def is_tripped(self, guild_id: int) -> bool:
        until = self._tripped.get(guild_id)
        if until is None:
            return False
        if time.time() >= until:
            del self._tripped[guild_id]
            return False
        return True

    def trip(self, guild_id: int, minutes: int = 30) -> None:
        self._tripped[guild_id] = time.time() + minutes * 60

    def reset(self, guild_id: int) -> None:
        self._tripped.pop(guild_id, None)

    def check(self, guild_id: int, config: dict, actor: str = "auto") -> Verdict:
        if self.is_tripped(guild_id):
            return Verdict(
                False,
                "enforcement is paused — the hourly action limit was hit. "
                "Review the word list, then re-enable with `/guard resume`.",
            )

        limit = int(config.get("guardrails", {}).get("max_actions_per_hour", 20))
        if limit <= 0:
            return ALLOWED

        used = self.store.audit_count_since(guild_id, time.time() - 3600, actor=actor)
        if used >= limit:
            self.trip(guild_id)
            return Verdict(
                False,
                f"hourly automated-action limit ({limit}) reached — enforcement paused. "
                "This usually means a word list is matching far more than intended.",
            )
        return ALLOWED


def dry_run(config: dict) -> bool:
    return bool(config.get("guardrails", {}).get("dry_run", False))
