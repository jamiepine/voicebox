"""Core moderation: cases, ban/kick/timeout/warn, purge, channel control, threads."""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from .. import guardrails
from ..checks import require_operator

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

DURATION_RE = re.compile(r"(\d+)\s*([smhdw])", re.I)
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str | None) -> int | None:
    """Parse '10m', '2h30m', '7d' into seconds. None if unparseable."""
    if not text:
        return None
    total = sum(int(n) * UNIT_SECONDS[u.lower()] for n, u in DURATION_RE.findall(text))
    return total or None


class ModCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    async def _record(
        self,
        interaction: discord.Interaction,
        target: discord.abc.User,
        action: str,
        reason: str | None,
        expires_at: float | None = None,
    ) -> int:
        runtime = self.bot.runtime
        number = runtime.store.add_case(
            interaction.guild.id, target.id, interaction.user.id, action, reason, expires_at
        )
        runtime.store.bump_metric(interaction.guild.id, action)
        runtime.store.audit(
            interaction.guild.id, str(interaction.user.id), action, str(target.id), reason
        )
        await runtime.events.moderation(
            interaction.guild,
            runtime.config(interaction.guild.id),
            case_number=number,
            action=action,
            target=f"{target} (`{target.id}`)",
            moderator=f"{interaction.user}",
            reason=reason,
        )
        return number

    def _blocked(self, interaction: discord.Interaction, member: discord.Member, action: str) -> str | None:
        """Manual moderation still respects hierarchy — but not auto-immunity.

        A human with the permission is allowed to action a moderator; the
        immunity list exists to stop *automated* systems from doing it.
        """
        if member.id == interaction.user.id:
            return "You can't do that to yourself."
        feasible = guardrails.can_action(interaction.guild, member, action)
        if not feasible:
            return f"Can't {action}: {feasible.reason}."
        actor = interaction.user
        if (
            isinstance(actor, discord.Member)
            and actor.id != interaction.guild.owner_id
            and member.top_role >= actor.top_role
        ):
            return "That member's highest role is at or above yours."
        return None

    # -- punishments --------------------------------------------------------

    @app_commands.command(name="ban", description="Ban a member and open a case.")
    @app_commands.describe(
        member="Who to ban", reason="Why", delete_days="Days of their messages to delete (0-7)"
    )
    @require_operator()
    async def ban(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
        delete_days: app_commands.Range[int, 0, 7] = 0,
    ) -> None:
        if problem := self._blocked(interaction, member, "ban"):
            await interaction.response.send_message(problem, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            await member.ban(
                reason=f"{interaction.user}: {reason or 'no reason'}",
                delete_message_seconds=delete_days * 86400,
            )
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Ban failed: {exc}")
            return
        number = await self._record(interaction, member, "ban", reason)
        await interaction.followup.send(f"🔨 Banned **{member}** — case `#{number}`.")

    @app_commands.command(name="unban", description="Lift a ban by user ID.")
    @require_operator()
    async def unban(
        self, interaction: discord.Interaction, user_id: str, reason: str | None = None
    ) -> None:
        if not user_id.isdigit():
            await interaction.response.send_message("Give a numeric user ID.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            user = discord.Object(id=int(user_id))
            await interaction.guild.unban(user, reason=f"{interaction.user}: {reason or ''}")
        except discord.NotFound:
            await interaction.followup.send("That user isn't banned.")
            return
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Unban failed: {exc}")
            return
        number = await self._record(
            interaction, discord.Object(id=int(user_id)), "unban", reason  # type: ignore[arg-type]
        )
        await interaction.followup.send(f"Unbanned `{user_id}` — case `#{number}`.")

    @app_commands.command(name="kick", description="Kick a member and open a case.")
    @require_operator()
    async def kick(
        self, interaction: discord.Interaction, member: discord.Member, reason: str | None = None
    ) -> None:
        if problem := self._blocked(interaction, member, "kick"):
            await interaction.response.send_message(problem, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            await member.kick(reason=f"{interaction.user}: {reason or 'no reason'}")
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Kick failed: {exc}")
            return
        number = await self._record(interaction, member, "kick", reason)
        await interaction.followup.send(f"👢 Kicked **{member}** — case `#{number}`.")

    @app_commands.command(name="timeout", description="Time a member out (e.g. 10m, 2h, 7d).")
    @app_commands.describe(duration="How long: 30s, 10m, 2h, 7d (max 28d)")
    @require_operator()
    async def timeout(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        duration: str,
        reason: str | None = None,
    ) -> None:
        seconds = parse_duration(duration)
        if not seconds:
            await interaction.response.send_message(
                "Couldn't read that duration. Try `10m`, `2h`, or `7d`.", ephemeral=True
            )
            return
        if seconds > 28 * 86400:
            await interaction.response.send_message(
                "Discord caps timeouts at 28 days.", ephemeral=True
            )
            return
        if problem := self._blocked(interaction, member, "timeout"):
            await interaction.response.send_message(problem, ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
        try:
            await member.timeout(until, reason=f"{interaction.user}: {reason or 'no reason'}")
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Timeout failed: {exc}")
            return
        number = await self._record(interaction, member, "timeout", reason, until.timestamp())
        await interaction.followup.send(
            f"🔇 Timed out **{member}** until {discord.utils.format_dt(until, 'R')} — case `#{number}`."
        )

    @app_commands.command(name="untimeout", description="Remove a member's timeout.")
    @require_operator()
    async def untimeout(
        self, interaction: discord.Interaction, member: discord.Member
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            await member.timeout(None, reason=f"Cleared by {interaction.user}")
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Failed: {exc}")
            return
        await interaction.followup.send(f"Timeout cleared for **{member}**.")

    @app_commands.command(name="warn", description="Warn a member and open a case.")
    @require_operator()
    async def warn(
        self, interaction: discord.Interaction, member: discord.Member, reason: str
    ) -> None:
        await interaction.response.defer(thinking=True)
        number = await self._record(interaction, member, "warn", reason)
        delivered = True
        try:
            await member.send(
                f"You've been warned in **{interaction.guild.name}**: {reason}\n"
                f"(case #{number})"
            )
        except discord.HTTPException:
            delivered = False
        note = "" if delivered else " (couldn't DM them — they have DMs closed)"
        await interaction.followup.send(f"⚠️ Warned **{member}** — case `#{number}`.{note}")

    # -- case management ----------------------------------------------------

    case = app_commands.Group(name="case", description="Inspect and edit moderation cases.")

    @case.command(name="view", description="Show a case by number.")
    @require_operator()
    async def case_view(self, interaction: discord.Interaction, number: int) -> None:
        row = self.bot.runtime.store.get_case(interaction.guild.id, number)
        if row is None:
            await interaction.response.send_message(f"No case `#{number}`.", ephemeral=True)
            return
        embed = discord.Embed(title=f"Case #{number} — {row['action']}", colour=0x9B59B6)
        embed.add_field(name="User", value=f"<@{row['user_id']}> (`{row['user_id']}`)")
        embed.add_field(name="Moderator", value=f"<@{row['moderator_id']}>")
        embed.add_field(name="Active", value="yes" if row["active"] else "no")
        embed.add_field(name="Reason", value=row["reason"] or "No reason given", inline=False)
        embed.timestamp = dt.datetime.fromtimestamp(row["created_at"], dt.timezone.utc)
        await interaction.response.send_message(embed=embed)

    @case.command(name="reason", description="Set or correct a case's reason.")
    @require_operator()
    async def case_reason(
        self, interaction: discord.Interaction, number: int, reason: str
    ) -> None:
        ok = self.bot.runtime.store.set_case_reason(interaction.guild.id, number, reason)
        await interaction.response.send_message(
            f"Updated case `#{number}`." if ok else f"No case `#{number}`.", ephemeral=not ok
        )

    @case.command(name="history", description="Show a member's case history.")
    @require_operator()
    async def case_history(
        self, interaction: discord.Interaction, member: discord.Member
    ) -> None:
        rows = self.bot.runtime.store.user_cases(interaction.guild.id, member.id)
        if not rows:
            await interaction.response.send_message(
                f"**{member}** has no cases.", ephemeral=True
            )
            return
        lines = [
            f"`#{r['case_number']}` **{r['action']}** — {(r['reason'] or 'no reason')[:80]}"
            for r in rows
        ]
        embed = discord.Embed(
            title=f"Cases for {member}", description="\n".join(lines[:20]), colour=0x9B59B6
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="modlog", description="Show the most recent moderation cases.")
    @require_operator()
    async def modlog(
        self, interaction: discord.Interaction, limit: app_commands.Range[int, 1, 25] = 10
    ) -> None:
        rows = self.bot.runtime.store.recent_cases(interaction.guild.id, limit)
        if not rows:
            await interaction.response.send_message("No cases yet.", ephemeral=True)
            return
        lines = [
            f"`#{r['case_number']}` **{r['action']}** <@{r['user_id']}> — "
            f"{(r['reason'] or 'no reason')[:60]}"
            for r in rows
        ]
        embed = discord.Embed(title="Moderation log", description="\n".join(lines), colour=0x9B59B6)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- channel control ----------------------------------------------------

    @app_commands.command(name="purge", description="Bulk-delete recent messages.")
    @app_commands.describe(
        count="How many messages to scan (1-100)", member="Only delete messages from this member"
    )
    @require_operator()
    async def purge(
        self,
        interaction: discord.Interaction,
        count: app_commands.Range[int, 1, 100],
        member: discord.Member | None = None,
    ) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("Only works in text channels.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        check = (lambda m: m.author.id == member.id) if member else None
        try:
            deleted = await interaction.channel.purge(
                limit=count, check=check, reason=f"Purge by {interaction.user}"
            )
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Purge failed: {exc}")
            return
        self.bot.runtime.store.bump_metric(interaction.guild.id, "messages_purged", len(deleted))
        await interaction.followup.send(f"🧹 Deleted {len(deleted)} message(s).")

    @app_commands.command(name="lock", description="Stop @everyone sending in this channel.")
    @require_operator()
    async def lock(self, interaction: discord.Interaction, reason: str | None = None) -> None:
        channel = interaction.channel
        if not isinstance(channel, discord.TextChannel):
            await interaction.response.send_message("Only works in text channels.", ephemeral=True)
            return
        overwrite = channel.overwrites_for(interaction.guild.default_role)
        overwrite.send_messages = False
        await channel.set_permissions(
            interaction.guild.default_role, overwrite=overwrite, reason=reason
        )
        await interaction.response.send_message(f"🔒 Locked {channel.mention}.")

    @app_commands.command(name="unlock", description="Restore @everyone's ability to send here.")
    @require_operator()
    async def unlock(self, interaction: discord.Interaction) -> None:
        channel = interaction.channel
        if not isinstance(channel, discord.TextChannel):
            await interaction.response.send_message("Only works in text channels.", ephemeral=True)
            return
        overwrite = channel.overwrites_for(interaction.guild.default_role)
        overwrite.send_messages = None
        await channel.set_permissions(interaction.guild.default_role, overwrite=overwrite)
        await interaction.response.send_message(f"🔓 Unlocked {channel.mention}.")

    @app_commands.command(name="slowmode", description="Set this channel's slowmode delay.")
    @require_operator()
    async def slowmode(
        self, interaction: discord.Interaction, seconds: app_commands.Range[int, 0, 21600]
    ) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("Only works in text channels.", ephemeral=True)
            return
        await interaction.channel.edit(slowmode_delay=seconds)
        await interaction.response.send_message(
            f"Slowmode {'disabled' if seconds == 0 else f'set to {seconds}s'}."
        )

    # -- threads ------------------------------------------------------------

    thread = app_commands.Group(name="thread", description="Create and manage threads.")

    @thread.command(name="create", description="Open a thread in this channel.")
    @app_commands.describe(
        name="Thread name",
        private="Make it a private thread (invite-only)",
        archive_hours="Auto-archive after this many hours of inactivity",
    )
    @app_commands.choices(
        archive_hours=[
            app_commands.Choice(name="1 hour", value=60),
            app_commands.Choice(name="24 hours", value=1440),
            app_commands.Choice(name="3 days", value=4320),
            app_commands.Choice(name="1 week", value=10080),
        ]
    )
    async def thread_create(
        self,
        interaction: discord.Interaction,
        name: str,
        private: bool = False,
        archive_hours: app_commands.Choice[int] | None = None,
    ) -> None:
        channel = interaction.channel
        if not isinstance(channel, discord.TextChannel):
            await interaction.response.send_message(
                "Threads can only be created in text channels.", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            created = await channel.create_thread(
                name=name[:100],
                type=discord.ChannelType.private_thread
                if private
                else discord.ChannelType.public_thread,
                auto_archive_duration=archive_hours.value if archive_hours else 1440,
                reason=f"Created by {interaction.user}",
            )
            await created.add_user(interaction.user)
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Couldn't create the thread: {exc}")
            return
        self.bot.runtime.store.bump_metric(interaction.guild.id, "threads_created")
        await interaction.followup.send(f"🧵 Created {created.mention}.")

    @thread.command(name="archive", description="Archive the current thread.")
    async def thread_archive(self, interaction: discord.Interaction) -> None:
        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("Run this inside a thread.", ephemeral=True)
            return
        await interaction.response.send_message("Archiving this thread.")
        await interaction.channel.edit(archived=True)

    @thread.command(name="lock", description="Lock the current thread (mods only).")
    @require_operator()
    async def thread_lock(self, interaction: discord.Interaction) -> None:
        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("Run this inside a thread.", ephemeral=True)
            return
        await interaction.response.send_message("🔒 Locking this thread.")
        await interaction.channel.edit(locked=True, archived=True)

    @thread.command(name="auto", description="Auto-create a thread on every message in a channel.")
    @require_operator()
    async def thread_auto(
        self, interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        channels = {str(c) for c in config["threads"]["auto_thread_channels"]}
        if enabled:
            channels.add(str(channel.id))
        else:
            channels.discard(str(channel.id))
        config["threads"]["auto_thread_channels"] = list(channels)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Auto-threading {'enabled' if enabled else 'disabled'} for {channel.mention}."
        )


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(ModCmds(bot))
