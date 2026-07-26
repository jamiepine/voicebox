"""`/raid` — advanced join-raid detection and response."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

ACTION_CHOICES = [
    app_commands.Choice(name="Alert only", value="alert"),
    app_commands.Choice(name="Lockdown (deny sending/joining)", value="lockdown"),
    app_commands.Choice(name="Kick suspicious joiners", value="kick"),
    app_commands.Choice(name="Ban suspicious joiners", value="ban"),
]


class RaidCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    raid = app_commands.Group(name="raid", description="Configure and control raid detection.")

    @raid.command(name="toggle", description="Turn raid detection on or off.")
    @require_operator()
    async def raid_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["raid"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(f"Raid detection is now **{'on' if enabled else 'off'}**.")

    @raid.command(name="configure", description="Tune raid detection thresholds and response.")
    @app_commands.describe(
        join_threshold="Joins within the window that count as a burst",
        window_seconds="Rolling window size in seconds",
        score_threshold="Risk score (0-100) that triggers a response",
        action="What to do when a raid is detected",
        alert_channel="Where to post raid alerts",
        mention_role="Role to ping on a raid alert",
        lockdown_minutes="How long a lockdown response lasts before auto-lifting",
    )
    @app_commands.choices(action=ACTION_CHOICES)
    @require_operator()
    async def raid_configure(
        self,
        interaction: discord.Interaction,
        join_threshold: app_commands.Range[int, 2, 100] | None = None,
        window_seconds: app_commands.Range[int, 10, 600] | None = None,
        score_threshold: app_commands.Range[int, 10, 100] | None = None,
        action: app_commands.Choice[str] | None = None,
        alert_channel: discord.TextChannel | None = None,
        mention_role: discord.Role | None = None,
        lockdown_minutes: app_commands.Range[int, 1, 1440] | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        r = config["raid"]
        if join_threshold:
            r["join_threshold"] = join_threshold
        if window_seconds:
            r["join_window_seconds"] = window_seconds
        if score_threshold:
            r["score_threshold"] = score_threshold
        if action:
            r["action"] = action.value
        if alert_channel:
            r["alert_channel_id"] = alert_channel.id
        if mention_role:
            r["mention_role_id"] = mention_role.id
        if lockdown_minutes:
            r["lockdown_minutes"] = lockdown_minutes
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Raid detection settings updated.")

    @raid.command(name="lockdown", description="Manually lock the server down right now.")
    @require_operator()
    async def raid_lockdown(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        ok, note = await self.bot.runtime.raid.lockdown(interaction.guild, True, reason="manual lockdown")
        await interaction.followup.send(note)

    @raid.command(name="lift", description="Lift a manual or automatic lockdown.")
    @require_operator()
    async def raid_lift(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        ok, note = await self.bot.runtime.raid.lockdown(interaction.guild, False, reason="manual lift")
        await interaction.followup.send(note)

    @raid.command(name="status", description="Show current raid-detection configuration.")
    @require_operator()
    async def raid_status(self, interaction: discord.Interaction) -> None:
        r = self.bot.runtime.config(interaction.guild.id)["raid"]
        locked = self.bot.runtime.raid.is_locked_down(interaction.guild.id)
        embed = discord.Embed(title="Raid detection")
        embed.add_field(name="Enabled", value=str(r["enabled"]), inline=True)
        embed.add_field(name="Locked down", value=str(locked), inline=True)
        embed.add_field(name="Action", value=r["action"], inline=True)
        embed.add_field(
            name="Trigger",
            value=f"{r['join_threshold']} joins / {r['join_window_seconds']}s, score ≥ {r['score_threshold']}",
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(RaidCmds(bot))
