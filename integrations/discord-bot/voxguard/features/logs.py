"""Server event logging.

The audit-log feature every large server runs (Dyno, Carl-bot, Logger).
Everything routes through one channel with colour-coded embeds so moderators
can reconstruct what happened without Discord's own audit log, which expires
and doesn't include message content.
"""

from __future__ import annotations

import datetime as dt
import logging

import discord

log = logging.getLogger(__name__)

COLOURS = {
    "delete": 0xE74C3C,
    "edit": 0xF39C12,
    "join": 0x2ECC71,
    "leave": 0x95A5A6,
    "voice": 0x3498DB,
    "mod": 0x9B59B6,
    "update": 0x1ABC9C,
}


class EventLogger:
    """Formats and dispatches guild events to the configured log channel."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _channel(guild: discord.Guild, config: dict) -> discord.abc.Messageable | None:
        cfg = config.get("logging", {})
        if not cfg.get("enabled", False) or not cfg.get("channel_id"):
            return None
        channel = guild.get_channel(int(cfg["channel_id"]))
        return channel if isinstance(channel, discord.abc.Messageable) else None

    async def _send(self, guild: discord.Guild, config: dict, embed: discord.Embed) -> None:
        channel = self._channel(guild, config)
        if channel is None:
            return
        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            pass

    def _base(self, title: str, kind: str) -> discord.Embed:
        return discord.Embed(
            title=title,
            colour=COLOURS.get(kind, 0x2C3E50),
            timestamp=dt.datetime.now(dt.timezone.utc),
        )

    async def message_deleted(self, message: discord.Message, config: dict) -> None:
        if not config.get("logging", {}).get("message_delete", True):
            return
        if message.guild is None or message.author.bot:
            return
        embed = self._base("Message deleted", "delete")
        embed.add_field(name="Author", value=f"{message.author.mention} (`{message.author.id}`)")
        embed.add_field(name="Channel", value=message.channel.mention)
        if message.content:
            embed.add_field(name="Content", value=f">>> {message.content[:1000]}", inline=False)
        if message.attachments:
            embed.add_field(
                name="Attachments",
                value="\n".join(a.filename for a in message.attachments[:5]),
                inline=False,
            )
        await self._send(message.guild, config, embed)

    async def message_edited(
        self, before: discord.Message, after: discord.Message, config: dict
    ) -> None:
        if not config.get("logging", {}).get("message_edit", True):
            return
        if after.guild is None or after.author.bot or before.content == after.content:
            return
        embed = self._base("Message edited", "edit")
        embed.add_field(name="Author", value=f"{after.author.mention} (`{after.author.id}`)")
        embed.add_field(name="Channel", value=after.channel.mention)
        embed.add_field(name="Before", value=f">>> {before.content[:500] or '(empty)'}", inline=False)
        embed.add_field(name="After", value=f">>> {after.content[:500] or '(empty)'}", inline=False)
        embed.add_field(name="Jump", value=f"[Go to message]({after.jump_url})", inline=False)
        await self._send(after.guild, config, embed)

    async def member_joined(self, member: discord.Member, config: dict) -> None:
        if not config.get("logging", {}).get("member_join", True):
            return
        embed = self._base("Member joined", "join")
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Member", value=f"{member.mention} (`{member.id}`)", inline=False)
        embed.add_field(
            name="Account created",
            value=discord.utils.format_dt(member.created_at, "R"),
            inline=True,
        )
        embed.add_field(name="Member count", value=str(member.guild.member_count), inline=True)
        await self._send(member.guild, config, embed)

    async def member_left(self, member: discord.Member, config: dict) -> None:
        if not config.get("logging", {}).get("member_leave", True):
            return
        embed = self._base("Member left", "leave")
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Member", value=f"{member} (`{member.id}`)", inline=False)
        if member.joined_at:
            embed.add_field(
                name="Joined", value=discord.utils.format_dt(member.joined_at, "R"), inline=True
            )
        roles = [r.mention for r in member.roles if not r.is_default()]
        if roles:
            embed.add_field(name="Roles", value=" ".join(roles[:10]), inline=False)
        await self._send(member.guild, config, embed)

    async def voice_state(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState, config: dict
    ) -> None:
        if not config.get("logging", {}).get("voice_state", False):
            return
        if before.channel == after.channel:
            return
        embed = self._base("Voice activity", "voice")
        embed.add_field(name="Member", value=f"{member.mention} (`{member.id}`)", inline=False)
        if before.channel is None:
            embed.description = f"Joined {after.channel.mention}"
        elif after.channel is None:
            embed.description = f"Left {before.channel.mention}"
        else:
            embed.description = f"Moved {before.channel.mention} → {after.channel.mention}"
        await self._send(member.guild, config, embed)

    async def member_updated(
        self, before: discord.Member, after: discord.Member, config: dict
    ) -> None:
        if not config.get("logging", {}).get("member_update", False):
            return
        changes: list[str] = []
        if before.nick != after.nick:
            changes.append(f"**Nickname**: `{before.nick or '—'}` → `{after.nick or '—'}`")
        added = set(after.roles) - set(before.roles)
        removed = set(before.roles) - set(after.roles)
        if added:
            changes.append("**Roles added**: " + " ".join(r.mention for r in added))
        if removed:
            changes.append("**Roles removed**: " + " ".join(r.mention for r in removed))
        if not changes:
            return
        embed = self._base("Member updated", "update")
        embed.add_field(name="Member", value=f"{after.mention} (`{after.id}`)", inline=False)
        embed.description = "\n".join(changes)
        await self._send(after.guild, config, embed)

    async def moderation(
        self,
        guild: discord.Guild,
        config: dict,
        *,
        case_number: int,
        action: str,
        target: str,
        moderator: str,
        reason: str | None,
    ) -> None:
        if not config.get("logging", {}).get("moderation", True):
            return
        embed = self._base(f"Case #{case_number} — {action}", "mod")
        embed.add_field(name="User", value=target, inline=True)
        embed.add_field(name="Moderator", value=moderator, inline=True)
        embed.add_field(name="Reason", value=reason or "No reason given", inline=False)
        await self._send(guild, config, embed)
