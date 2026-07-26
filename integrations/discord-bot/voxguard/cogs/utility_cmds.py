"""Utility, information, logging config, data retention, and the dashboard link."""

from __future__ import annotations

import datetime as dt
import platform
import time
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator

if TYPE_CHECKING:
    from ..bot import VoxGuardBot


def uptime_string(started_at: float) -> str:
    delta = int(time.time() - started_at)
    days, rem = divmod(delta, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


class UtilityCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    # -- information --------------------------------------------------------

    @app_commands.command(name="userinfo", description="Show details about a member.")
    async def userinfo(
        self, interaction: discord.Interaction, member: discord.Member | None = None
    ) -> None:
        target = member or interaction.user
        embed = discord.Embed(colour=target.colour or discord.Colour(0x5865F2))
        embed.set_author(name=str(target), icon_url=target.display_avatar.url)
        embed.set_thumbnail(url=target.display_avatar.url)
        embed.add_field(name="ID", value=f"`{target.id}`", inline=True)
        embed.add_field(
            name="Account created",
            value=discord.utils.format_dt(target.created_at, "R"),
            inline=True,
        )
        if target.joined_at:
            embed.add_field(
                name="Joined server",
                value=discord.utils.format_dt(target.joined_at, "R"),
                inline=True,
            )
        roles = [r.mention for r in reversed(target.roles) if not r.is_default()]
        embed.add_field(
            name=f"Roles ({len(roles)})",
            value=" ".join(roles[:15]) if roles else "None",
            inline=False,
        )
        cases = self.bot.runtime.store.user_cases(interaction.guild.id, target.id, limit=100)
        if cases:
            embed.add_field(name="Moderation cases", value=str(len(cases)), inline=True)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="serverinfo", description="Show details about this server.")
    async def serverinfo(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        embed = discord.Embed(title=guild.name, colour=0x5865F2)
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        embed.add_field(name="ID", value=f"`{guild.id}`", inline=True)
        embed.add_field(name="Owner", value=f"<@{guild.owner_id}>", inline=True)
        embed.add_field(
            name="Created", value=discord.utils.format_dt(guild.created_at, "R"), inline=True
        )
        embed.add_field(name="Members", value=f"{guild.member_count:,}", inline=True)
        embed.add_field(name="Channels", value=str(len(guild.channels)), inline=True)
        embed.add_field(name="Roles", value=str(len(guild.roles)), inline=True)
        embed.add_field(name="Boosts", value=str(guild.premium_subscription_count), inline=True)
        embed.add_field(
            name="Verification", value=str(guild.verification_level).title(), inline=True
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="avatar", description="Show a member's avatar.")
    async def avatar(
        self, interaction: discord.Interaction, member: discord.Member | None = None
    ) -> None:
        target = member or interaction.user
        embed = discord.Embed(title=f"{target.display_name}'s avatar", colour=0x5865F2)
        embed.set_image(url=target.display_avatar.url)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="stats", description="Bot health, uptime and activity.")
    async def stats(self, interaction: discord.Interaction) -> None:
        runtime = self.bot.runtime
        store = runtime.store
        guild_totals = store.metric_totals(interaction.guild.id)
        cases = store.case_counts(interaction.guild.id)

        embed = discord.Embed(title="VoxGuard", colour=0x5865F2)
        embed.add_field(name="Uptime", value=uptime_string(runtime.started_at), inline=True)
        embed.add_field(name="Servers", value=str(len(self.bot.guilds)), inline=True)
        embed.add_field(
            name="Latency", value=f"{self.bot.latency * 1000:.0f} ms", inline=True
        )
        embed.add_field(
            name="Moderation here",
            value=(
                f"bans **{cases.get('ban', 0)}** · kicks **{cases.get('kick', 0)}** · "
                f"timeouts **{cases.get('timeout', 0)}** · warns **{cases.get('warn', 0)}**"
            ),
            inline=False,
        )
        embed.add_field(
            name="Activity here",
            value=(
                f"level-ups **{guild_totals.get('levelups', 0)}** · "
                f"tickets **{guild_totals.get('tickets_opened', 0)}** · "
                f"threads **{guild_totals.get('threads_created', 0)}**"
            ),
            inline=False,
        )
        errors = store.error_count_since(time.time() - 86400)
        embed.set_footer(text=f"Python {platform.python_version()} · {errors} error(s) in 24h")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="dashboard", description="Link to the web stats dashboard.")
    @require_operator()
    async def dashboard(self, interaction: discord.Interaction) -> None:
        settings = self.bot.settings
        if not settings.dashboard_enabled:
            await interaction.response.send_message(
                "The web dashboard isn't enabled. Set `VOXGUARD_DASHBOARD=1` and a "
                "`VOXGUARD_DASHBOARD_TOKEN`, then restart the bot.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            f"📊 Dashboard: {settings.dashboard_public_url}\n"
            "It needs the access token from your `.env` to sign in.",
            ephemeral=True,
        )

    # -- logging ------------------------------------------------------------

    logs = app_commands.Group(name="logs", description="Server event logging.")

    @logs.command(name="configure", description="Choose the log channel and which events to record.")
    @require_operator()
    async def logs_configure(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        channel: discord.TextChannel | None = None,
        message_delete: bool | None = None,
        message_edit: bool | None = None,
        member_join: bool | None = None,
        member_leave: bool | None = None,
        voice_state: bool | None = None,
        member_update: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["logging"]
        for key, value in (
            ("enabled", enabled),
            ("message_delete", message_delete),
            ("message_edit", message_edit),
            ("member_join", member_join),
            ("member_leave", member_leave),
            ("voice_state", voice_state),
            ("member_update", member_update),
        ):
            if value is not None:
                cfg[key] = value
        if channel is not None:
            cfg["channel_id"] = channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Logging settings updated.")

    @logs.command(name="status", description="Show what's being logged.")
    @require_operator()
    async def logs_status(self, interaction: discord.Interaction) -> None:
        cfg = self.bot.runtime.config(interaction.guild.id)["logging"]
        target = f"<#{cfg['channel_id']}>" if cfg.get("channel_id") else "not set"
        rows = [
            ("Message deletions", cfg.get("message_delete")),
            ("Message edits", cfg.get("message_edit")),
            ("Member joins", cfg.get("member_join")),
            ("Member leaves", cfg.get("member_leave")),
            ("Voice activity", cfg.get("voice_state")),
            ("Member updates", cfg.get("member_update")),
            ("Moderation actions", cfg.get("moderation")),
        ]
        body = "\n".join(f"{'🟢' if on else '⚫'} {name}" for name, on in rows)
        embed = discord.Embed(
            title=f"Logging — {'enabled' if cfg.get('enabled') else 'disabled'}",
            description=f"**Channel:** {target}\n\n{body}",
            colour=0x5865F2,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- data retention -----------------------------------------------------

    data = app_commands.Group(
        name="data", description="Retention and erasure of stored member data."
    )

    @data.command(name="retention", description="How long recorded data is kept (0 = forever).")
    @app_commands.describe(
        transcript_days="Days before voice transcripts are blanked",
        infraction_days="Days before infraction rows are deleted",
        conversation_days="Days before AI conversation history is deleted",
        audit_days="Days before audit rows are deleted",
    )
    @require_operator()
    async def data_retention(
        self,
        interaction: discord.Interaction,
        transcript_days: app_commands.Range[int, 0, 3650] | None = None,
        infraction_days: app_commands.Range[int, 0, 3650] | None = None,
        conversation_days: app_commands.Range[int, 0, 3650] | None = None,
        audit_days: app_commands.Range[int, 0, 3650] | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["retention"]
        for key, value in (
            ("transcript_days", transcript_days),
            ("infraction_days", infraction_days),
            ("conversation_days", conversation_days),
            ("audit_days", audit_days),
        ):
            if value is not None:
                cfg[key] = value
        runtime.save_config(interaction.guild.id, config)

        summary = " · ".join(f"{k.replace('_days', '')}: {v}d" for k, v in cfg.items())
        await interaction.response.send_message(f"Retention updated — {summary}")

    @data.command(name="forget", description="Erase everything stored about a member.")
    @require_operator()
    async def data_forget(
        self, interaction: discord.Interaction, member: discord.Member, confirm: bool = False
    ) -> None:
        if not confirm:
            await interaction.response.send_message(
                f"This permanently erases all stored data for **{member}** — transcripts, "
                "infractions, cases, XP and AI memory. Re-run with `confirm: True`.",
                ephemeral=True,
            )
            return
        removed = self.bot.runtime.store.erase_user(interaction.guild.id, member.id)
        detail = ", ".join(f"{k}: {v}" for k, v in removed.items()) or "nothing stored"
        await interaction.response.send_message(
            f"Erased data for **{member}** — {detail}.", ephemeral=True
        )

    @data.command(name="purge-now", description="Immediately apply the retention policy.")
    @require_operator()
    async def data_purge_now(self, interaction: discord.Interaction) -> None:
        runtime = self.bot.runtime
        cfg = runtime.config(interaction.guild.id)["retention"]
        removed = runtime.store.purge_expired(cfg)
        scrubbed = runtime.store.scrub_transcripts(int(cfg.get("transcript_days", 0)))
        detail = ", ".join(f"{k}: {v}" for k, v in removed.items()) or "no expired rows"
        await interaction.response.send_message(
            f"Retention applied — {detail}; {scrubbed} transcript(s) blanked.", ephemeral=True
        )


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(UtilityCmds(bot))
