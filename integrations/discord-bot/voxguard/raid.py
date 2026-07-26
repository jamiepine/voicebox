"""Raid detection.

A join burst on its own is a poor signal — a server can get one from a
successful post. What distinguishes a raid is the *shape* of the burst: many
accounts arriving at once that are freshly created, have no avatar, and have
names generated from the same template.

Each signal contributes to a 0-100 score and the response fires on the total,
so a genuine popularity spike (old accounts, varied names) stays below the
line while a botnet clears it on the first few joins.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import discord
from rapidfuzz import fuzz

from . import guardrails
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class Joiner:
    user_id: int
    name: str
    joined_at: float
    account_age_days: float
    default_avatar: bool


@dataclass
class RaidEvent:
    guild_id: int
    score: int
    joiners: list[Joiner]
    signals: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return ", ".join(f"{k}: {v}" for k, v in self.signals.items())


def _name_similarity(names: list[str]) -> float:
    """Mean pairwise similarity, 0.0-1.0. Sampled above 12 names."""
    if len(names) < 3:
        return 0.0
    sample = names[:12]
    scores: list[float] = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            scores.append(fuzz.ratio(sample[i].casefold(), sample[j].casefold()) / 100)
    return sum(scores) / len(scores) if scores else 0.0


class RaidDetector:
    def __init__(self, store: Store, limiter: guardrails.RateLimiter) -> None:
        self.store = store
        self.limiter = limiter
        self._joins: dict[int, deque[Joiner]] = {}
        self._lockdowns: dict[int, float] = {}
        self._last_alert: dict[int, float] = {}

    # -- scoring ------------------------------------------------------------

    def record_join(self, member: discord.Member, config: dict) -> RaidEvent | None:
        cfg = config.get("raid", {})
        if not cfg.get("enabled", False):
            return None

        window = float(cfg.get("join_window_seconds", 60))
        now = time.time()

        created = member.created_at or dt.datetime.now(dt.timezone.utc)
        age_days = (dt.datetime.now(dt.timezone.utc) - created).total_seconds() / 86400

        queue = self._joins.setdefault(member.guild.id, deque(maxlen=200))
        queue.append(
            Joiner(
                user_id=member.id,
                name=member.name,
                joined_at=now,
                account_age_days=age_days,
                default_avatar=member.avatar is None,
            )
        )
        while queue and now - queue[0].joined_at > window:
            queue.popleft()

        recent = list(queue)
        if len(recent) < 3:
            return None

        threshold = max(2, int(cfg.get("join_threshold", 8)))
        new_days = float(cfg.get("new_account_days", 7))

        rate_factor = min(1.0, len(recent) / threshold)
        new_fraction = sum(1 for j in recent if j.account_age_days <= new_days) / len(recent)
        avatar_fraction = sum(1 for j in recent if j.default_avatar) / len(recent)
        similarity = _name_similarity([j.name for j in recent])

        score = int(
            rate_factor * 40 + new_fraction * 25 + avatar_fraction * 15 + similarity * 20
        )

        signals = {
            "joins": f"{len(recent)} in {int(window)}s (threshold {threshold})",
            "new accounts": f"{new_fraction:.0%} under {new_days:g}d old",
            "no avatar": f"{avatar_fraction:.0%}",
            "name similarity": f"{similarity:.0%}",
        }

        if score < int(cfg.get("score_threshold", 60)):
            return None

        # One response per window — don't re-fire on every subsequent join.
        last = self._last_alert.get(member.guild.id, 0)
        if now - last < window:
            return None
        self._last_alert[member.guild.id] = now

        return RaidEvent(member.guild.id, score, recent, signals)

    # -- response -----------------------------------------------------------

    async def respond(self, guild: discord.Guild, config: dict, event: RaidEvent) -> list[str]:
        """Carry out the configured raid response. Returns a report."""
        cfg = config.get("raid", {})
        action = cfg.get("action", "lockdown")
        report: list[str] = []

        self.store.audit(
            guild.id, "raid", "detected", None, f"score={event.score} {event.summary}"
        )

        if guardrails.dry_run(config):
            report.append(f"Dry run — would have run `{action}`.")
            return report

        if action in ("alert", "lockdown"):
            if action == "lockdown":
                minutes = int(cfg.get("lockdown_minutes", 15))
                ok, note = await self.lockdown(guild, True, reason=f"raid score {event.score}")
                report.append(note)
                if ok:
                    self._lockdowns[guild.id] = time.time() + minutes * 60
            return report

        # kick / ban only touch the accounts in the detection window, and
        # only ones that pass the same immunity checks as any other action.
        actioned, skipped = 0, 0
        for joiner in event.joiners:
            member = guild.get_member(joiner.user_id)
            if member is None:
                continue
            if not guardrails.is_immune(member, config):
                skipped += 1
                continue
            if not guardrails.can_action(guild, member, action):
                skipped += 1
                continue
            try:
                reason = f"VoxGuard raid response (score {event.score})"
                if action == "ban":
                    await member.ban(reason=reason, delete_message_seconds=3600)
                else:
                    await member.kick(reason=reason)
                actioned += 1
                self.store.audit(guild.id, "raid", action, str(member.id), reason)
            except discord.HTTPException as exc:
                log.warning("Raid %s failed for %s: %s", action, member.id, exc)
                skipped += 1

        report.append(f"{action.title()}ed {actioned} account(s); skipped {skipped}.")
        return report

    async def lockdown(
        self, guild: discord.Guild, on: bool, *, reason: str = "raid response"
    ) -> tuple[bool, str]:
        """Deny @everyone send/connect across the server, or restore it."""
        everyone = guild.default_role
        changed, failed = 0, 0

        for channel in guild.channels:
            overwrite = channel.overwrites_for(everyone)
            if isinstance(channel, discord.TextChannel | discord.ForumChannel):
                overwrite.send_messages = False if on else None
            elif isinstance(channel, discord.VoiceChannel | discord.StageChannel):
                overwrite.connect = False if on else None
            else:
                continue
            try:
                await channel.set_permissions(everyone, overwrite=overwrite, reason=reason)
                changed += 1
            except discord.HTTPException:
                failed += 1

        try:
            await guild.edit(
                verification_level=(
                    discord.VerificationLevel.high if on else discord.VerificationLevel.medium
                ),
                reason=reason,
            )
        except discord.HTTPException:
            pass

        if not on:
            self._lockdowns.pop(guild.id, None)

        state = "Locked down" if on else "Lifted lockdown on"
        note = f"{state} {changed} channel(s)."
        if failed:
            note += f" {failed} could not be changed (missing permissions)."
        self.store.audit(guild.id, "raid", "lockdown_on" if on else "lockdown_off", None, note)
        return changed > 0, note

    def expired_lockdowns(self) -> list[int]:
        now = time.time()
        return [gid for gid, until in self._lockdowns.items() if now >= until]

    def is_locked_down(self, guild_id: int) -> bool:
        return guild_id in self._lockdowns
