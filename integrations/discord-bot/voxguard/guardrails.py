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


def may_action(member: discord.Member, config: dict) -> Verdict:
    """May automated enforcement act on this member?

    Truthy means "go ahead"; falsy carries the reason they're shielded. Named
    positively on purpose — an `is_immune()` that returns False when someone
    *is* immune inverts at every call site and makes an accidental
    enforcement bypass a one-character mistake.
    """
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
    """Per-guild, per-actor circuit breaker over automated actions.

    Budgets are tracked separately for each actor ("voice-mod", "roam", ...)
    because they have independent limits and independent failure modes: a
    runaway word list should pause the voice filter without also disarming
    the agent, and vice versa.

    Counts are backed by the audit table, so a restart mid-storm doesn't hand
    the bot a fresh budget.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self._tripped: dict[tuple[int, str], float] = {}

    def is_tripped(self, guild_id: int, actor: str = "auto") -> bool:
        key = (guild_id, actor)
        until = self._tripped.get(key)
        if until is None:
            return False
        if time.time() >= until:
            del self._tripped[key]
            return False
        return True

    def any_tripped(self, guild_id: int) -> list[str]:
        """Actors currently paused in this guild — for status display."""
        now = time.time()
        return [actor for (gid, actor), until in self._tripped.items() if gid == guild_id and now < until]

    def trip(self, guild_id: int, actor: str = "auto", minutes: int = 30) -> None:
        self._tripped[(guild_id, actor)] = time.time() + minutes * 60

    def reset(self, guild_id: int, actor: str | None = None) -> None:
        if actor is not None:
            self._tripped.pop((guild_id, actor), None)
            return
        for key in [k for k in self._tripped if k[0] == guild_id]:
            del self._tripped[key]

    def check(self, guild_id: int, config: dict, actor: str = "auto") -> Verdict:
        if self.is_tripped(guild_id, actor):
            return Verdict(
                False,
                f"`{actor}` enforcement is paused — its hourly action limit was hit. "
                "Review the configuration, then re-enable with `/guard resume`.",
            )

        limit = int(config.get("guardrails", {}).get("max_actions_per_hour", 20))
        if limit <= 0:
            return ALLOWED

        used = self.store.audit_count_since(guild_id, time.time() - 3600, actor=actor)
        if used >= limit:
            self.trip(guild_id, actor)
            return Verdict(
                False,
                f"hourly automated-action limit ({limit}) reached for `{actor}` — paused. "
                "This usually means a rule is matching far more than intended.",
            )
        return ALLOWED


def dry_run(config: dict) -> bool:
    return bool(config.get("guardrails", {}).get("dry_run", False))
