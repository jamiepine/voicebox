"""`/automod` and `/antinuke` configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

RULE_CHOICES = [
    app_commands.Choice(name="Discord invites", value="invites"),
    app_commands.Choice(name="Links", value="links"),
    app_commands.Choice(name="Mass mentions", value="mass_mentions"),
    app_commands.Choice(name="Message spam / flooding", value="spam"),
    app_commands.Choice(name="Excessive caps", value="caps"),
    app_commands.Choice(name="Blocked words", value="words"),
]

ACTION_CHOICES = [
    app_commands.Choice(name="Delete the message", value="delete"),
    app_commands.Choice(name="Delete + warn", value="warn"),
    app_commands.Choice(name="Delete + timeout", value="timeout"),
    app_commands.Choice(name="Delete + kick", value="kick"),
    app_commands.Choice(name="Delete + ban", value="ban"),
    app_commands.Choice(name="Log only", value="log"),
]


class AutomodCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    automod = app_commands.Group(name="automod", description="Text-channel automoderation.")

    @automod.command(name="toggle", description="Turn text automod on or off.")
    @require_operator()
    async def automod_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["automod"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Text automod is now **{'on' if enabled else 'off'}**."
        )

    @automod.command(name="rule", description="Enable or configure one automod rule.")
    @app_commands.describe(
        rule="Which rule",
        enabled="Turn this rule on or off",
        action="What to do when it triggers",
        threshold="Mention limit / spam message count / caps percent, depending on the rule",
    )
    @app_commands.choices(rule=RULE_CHOICES, action=ACTION_CHOICES)
    @require_operator()
    async def automod_rule(
        self,
        interaction: discord.Interaction,
        rule: app_commands.Choice[str],
        enabled: bool | None = None,
        action: app_commands.Choice[str] | None = None,
        threshold: app_commands.Range[int, 1, 100] | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        entry = config["automod"]["rules"].setdefault(rule.value, {})

        if enabled is not None:
            entry["enabled"] = enabled
        if action is not None:
            entry["action"] = action.value
        if threshold is not None:
            key = {
                "mass_mentions": "limit",
                "spam": "messages",
                "caps": "percent",
            }.get(rule.value)
            if key is None:
                await interaction.response.send_message(
                    f"The **{rule.name}** rule doesn't take a threshold.", ephemeral=True
                )
                return
            entry[key] = threshold

        runtime.save_config(interaction.guild.id, config)
        state = "on" if entry.get("enabled") else "off"
        await interaction.response.send_message(
            f"**{rule.name}**: {state}, action `{entry.get('action', 'delete')}`."
        )

    @automod.command(name="allow-domain", description="Allow a domain past the links rule.")
    @require_operator()
    async def automod_allow_domain(
        self, interaction: discord.Interaction, domain: str, remove: bool = False
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        entry = config["automod"]["rules"].setdefault("links", {})
        allowed = {d.lower() for d in entry.get("allowed_domains", [])}
        clean = domain.lower().strip().lstrip("www.")
        if remove:
            allowed.discard(clean)
        else:
            allowed.add(clean)
        entry["allowed_domains"] = sorted(allowed)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"`{clean}` {'removed from' if remove else 'added to'} the allowed domains."
        )

    @automod.command(name="log-channel", description="Where automod actions are reported.")
    @require_operator()
    async def automod_log(
        self, interaction: discord.Interaction, channel: discord.TextChannel
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["automod"]["log_channel_id"] = channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(f"Automod will log to {channel.mention}.")

    @automod.command(name="status", description="Show automod rules and their state.")
    @require_operator()
    async def automod_status(self, interaction: discord.Interaction) -> None:
        cfg = self.bot.runtime.config(interaction.guild.id)["automod"]
        lines = []
        for choice in RULE_CHOICES:
            entry = cfg["rules"].get(choice.value, {})
            mark = "🟢" if entry.get("enabled") else "⚫"
            lines.append(f"{mark} **{choice.name}** — `{entry.get('action', 'delete')}`")
        embed = discord.Embed(
            title=f"Automod — {'enabled' if cfg['enabled'] else 'disabled'}",
            description="\n".join(lines),
            colour=0x5865F2 if cfg["enabled"] else 0x95A5A6,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- anti-nuke ----------------------------------------------------------

    antinuke = app_commands.Group(
        name="antinuke", description="Protection against mass-destructive admin actions."
    )

    @antinuke.command(name="toggle", description="Turn anti-nuke protection on or off.")
    @require_operator()
    async def antinuke_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["antinuke"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)
        note = (
            "\nI'll strip privileged roles from anyone who mass-deletes channels/roles or "
            "mass-bans within the detection window. Make sure my role sits above the staff "
            "roles you want protected — I can't strip a role above my own."
            if enabled
            else ""
        )
        await interaction.response.send_message(
            f"Anti-nuke is now **{'on' if enabled else 'off'}**.{note}"
        )

    @antinuke.command(name="configure", description="Tune anti-nuke thresholds.")
    @app_commands.describe(
        window_seconds="Detection window",
        channel_deletes="Channel deletions in the window that trip it",
        role_deletes="Role deletions in the window that trip it",
        bans="Bans in the window that trip it",
        kicks="Kicks in the window that trip it",
        alert_channel="Where to post anti-nuke alerts",
    )
    @app_commands.choices(
        response=[
            app_commands.Choice(name="Strip the actor's privileged roles", value="strip_roles"),
            app_commands.Choice(name="Alert only", value="alert"),
        ]
    )
    @require_operator()
    async def antinuke_configure(
        self,
        interaction: discord.Interaction,
        window_seconds: app_commands.Range[int, 5, 600] | None = None,
        channel_deletes: app_commands.Range[int, 1, 50] | None = None,
        role_deletes: app_commands.Range[int, 1, 50] | None = None,
        bans: app_commands.Range[int, 1, 50] | None = None,
        kicks: app_commands.Range[int, 1, 50] | None = None,
        response: app_commands.Choice[str] | None = None,
        alert_channel: discord.TextChannel | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["antinuke"]
        if window_seconds is not None:
            cfg["window_seconds"] = window_seconds
        if channel_deletes is not None:
            cfg["channel_delete_limit"] = channel_deletes
        if role_deletes is not None:
            cfg["role_delete_limit"] = role_deletes
        if bans is not None:
            cfg["ban_limit"] = bans
        if kicks is not None:
            cfg["kick_limit"] = kicks
        if response is not None:
            cfg["response"] = response.value
        if alert_channel is not None:
            cfg["alert_channel_id"] = alert_channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Anti-nuke settings updated.")

    @antinuke.command(name="whitelist", description="Exempt a trusted admin from anti-nuke.")
    @require_operator()
    async def antinuke_whitelist(
        self, interaction: discord.Interaction, member: discord.Member, remove: bool = False
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        current = {str(u) for u in config["antinuke"]["whitelist"]}
        if remove:
            current.discard(str(member.id))
        else:
            current.add(str(member.id))
        config["antinuke"]["whitelist"] = list(current)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"{member.mention} {'removed from' if remove else 'added to'} the anti-nuke whitelist."
        )


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(AutomodCmds(bot))
