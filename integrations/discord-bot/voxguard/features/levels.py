"""XP and levelling, for text *and* voice.

Levelling is the single most-used feature of the big engagement bots (MEE6,
Arcane, Amari), and it's a natural fit here because this bot is already
listening to voice channels: time spent actually talking earns XP, not just
time parked idle in a channel with a muted mic.

The curve is the widely-used `5*(l^2) + 50*l + 100` per-level cost, so ranks
line up with what members expect from other servers.
"""

from __future__ import annotations

import logging
import time

import discord

from ..store import Store

log = logging.getLogger(__name__)


def xp_for_level(level: int) -> int:
    """Total XP required to reach `level` from zero."""
    total = 0
    for current in range(level):
        total += 5 * (current**2) + 50 * current + 100
    return total


def level_from_xp(xp: int) -> int:
    level = 0
    remaining = xp
    while True:
        cost = 5 * (level**2) + 50 * level + 100
        if remaining < cost:
            return level
        remaining -= cost
        level += 1


def level_progress(xp: int) -> tuple[int, int, int]:
    """(level, xp_into_level, xp_needed_for_next)."""
    level = level_from_xp(xp)
    consumed = xp_for_level(level)
    cost = 5 * (level**2) + 50 * level + 100
    return level, xp - consumed, cost


class LevelEngine:
    def __init__(self, store: Store) -> None:
        self.store = store
        # user -> (guild, channel, joined_at) for members currently in voice.
        self._voice_since: dict[tuple[int, int], float] = {}

    # -- text ---------------------------------------------------------------

    async def on_message(self, message: discord.Message, config: dict) -> None:
        cfg = config.get("levels", {})
        if not cfg.get("enabled", False) or message.author.bot or message.guild is None:
            return
        if str(message.channel.id) in {str(c) for c in cfg.get("no_xp_channels", [])}:
            return

        row = self.store.get_level_row(message.guild.id, message.author.id)
        cooldown = int(cfg.get("message_cooldown_seconds", 60))
        if row and time.time() - float(row["last_award_at"]) < cooldown:
            return

        before = int(row["xp"]) if row else 0
        gain = int(cfg.get("xp_per_message", 15))
        after = self.store.add_xp(message.guild.id, message.author.id, gain, messages=1)
        await self._handle_levelup(message.guild, message.author, before, after, config, message.channel)

    # -- voice --------------------------------------------------------------

    def voice_joined(self, guild_id: int, user_id: int) -> None:
        self._voice_since[(guild_id, user_id)] = time.time()

    def voice_left(self, guild_id: int, user_id: int) -> float:
        """Returns seconds spent in voice, and clears the timer."""
        started = self._voice_since.pop((guild_id, user_id), None)
        return time.time() - started if started else 0.0

    async def award_voice(
        self, guild: discord.Guild, member: discord.Member, seconds: float, config: dict
    ) -> None:
        cfg = config.get("levels", {})
        if not cfg.get("enabled", False) or seconds < 60:
            return

        minutes = int(seconds // 60)
        gain = minutes * int(cfg.get("xp_per_voice_minute", 8))
        if gain <= 0:
            return

        row = self.store.get_level_row(guild.id, member.id)
        before = int(row["xp"]) if row else 0
        after = self.store.add_xp(guild.id, member.id, gain, voice_seconds=int(seconds))
        await self._handle_levelup(guild, member, before, after, config, None)

    def voice_tick(self, guild: discord.Guild, config: dict) -> list[tuple[discord.Member, float]]:
        """Flush accrued voice time for everyone still connected.

        Called on a timer so XP lands during long calls rather than only when
        someone disconnects — and so a crash doesn't lose an entire session.
        """
        cfg = config.get("levels", {})
        if not cfg.get("enabled", False):
            return []

        require_unmuted = cfg.get("voice_requires_unmuted", True)
        now = time.time()
        out: list[tuple[discord.Member, float]] = []

        for (guild_id, user_id), started in list(self._voice_since.items()):
            if guild_id != guild.id:
                continue
            member = guild.get_member(user_id)
            if member is None or member.voice is None:
                self._voice_since.pop((guild_id, user_id), None)
                continue
            # Alone in a channel, or muted/deafened, isn't participation.
            state = member.voice
            if require_unmuted and (state.self_mute or state.self_deaf or state.mute or state.deaf):
                self._voice_since[(guild_id, user_id)] = now
                continue
            if state.channel and len([m for m in state.channel.members if not m.bot]) < 2:
                self._voice_since[(guild_id, user_id)] = now
                continue

            elapsed = now - started
            if elapsed >= 60:
                self._voice_since[(guild_id, user_id)] = now
                out.append((member, elapsed))
        return out

    # -- shared -------------------------------------------------------------

    async def _handle_levelup(
        self,
        guild: discord.Guild,
        member: discord.abc.User,
        before_xp: int,
        after_xp: int,
        config: dict,
        channel: discord.abc.Messageable | None,
    ) -> None:
        before_level = level_from_xp(before_xp)
        after_level = level_from_xp(after_xp)
        if after_level <= before_level:
            return

        cfg = config.get("levels", {})
        self.store.bump_metric(guild.id, "levelups")

        if isinstance(member, discord.Member):
            await self._grant_rewards(guild, member, after_level, cfg)

        if not cfg.get("announce", True):
            return

        target: discord.abc.Messageable | None = channel
        if announce_id := cfg.get("announce_channel_id"):
            found = guild.get_channel(int(announce_id))
            if isinstance(found, discord.abc.Messageable):
                target = found
        if target is None:
            return

        try:
            await target.send(f"🎉 {member.mention} reached **level {after_level}**!")
        except discord.HTTPException:
            pass

    async def _grant_rewards(
        self, guild: discord.Guild, member: discord.Member, level: int, cfg: dict
    ) -> None:
        rewards = self.store.level_rewards(guild.id)
        if not rewards:
            return

        earned = [r for r in rewards if int(r["level"]) <= level]
        if not earned:
            return

        stack = cfg.get("stack_rewards", False)
        to_add: list[discord.Role] = []
        to_remove: list[discord.Role] = []

        # Without stacking, only the highest earned reward is kept — the
        # common configuration, so members don't accumulate every rank role.
        keep = earned if stack else earned[-1:]
        keep_ids = {int(r["role_id"]) for r in keep}

        for reward in earned:
            role = guild.get_role(int(reward["role_id"]))
            if role is None or role >= guild.me.top_role:
                continue
            if role.id in keep_ids and role not in member.roles:
                to_add.append(role)
            elif role.id not in keep_ids and role in member.roles:
                to_remove.append(role)

        try:
            if to_add:
                await member.add_roles(*to_add, reason=f"Level {level} reward")
            if to_remove:
                await member.remove_roles(*to_remove, reason=f"Superseded by level {level}")
        except discord.HTTPException as exc:
            log.warning("Could not update level roles for %s: %s", member.id, exc)
