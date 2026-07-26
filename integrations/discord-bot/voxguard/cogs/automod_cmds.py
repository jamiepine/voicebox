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

AI_ACTION_CHOICES = [
    app_commands.Choice(name="Log only", value="log"),
    app_commands.Choice(name="Delete the message", value="delete"),
    app_commands.Choice(name="Delete + timeout", value="timeout"),
    app_commands.Choice(name="Delete + kick", value="kick"),
    app_commands.Choice(name="Delete + ban", value="ban"),
]

CATEGORY_CHOICES = [
    app_commands.Choice(name="Harassment", value="harassment"),
    app_commands.Choice(name="Hate speech", value="hate"),
    app_commands.Choice(name="Threats of violence", value="threats"),
    app_commands.Choice(name="Sexual content", value="sexual"),
    app_commands.Choice(name="Self-harm", value="self_harm"),
    app_commands.Choice(name="Scams and phishing", value="scam"),
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
            state = "on " if entry.get("enabled") else "off"
            lines.append(f"{state}  {choice.name:<26} {entry.get('action', 'delete')}")
        embed = discord.Embed(
            title=f"Automod — {'enabled' if cfg['enabled'] else 'disabled'}",
            description="```\n" + "\n".join(lines) + "```",
            colour=0x2B2D31,
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


    # -- AI text moderation -------------------------------------------------

    aimod = app_commands.Group(
        name="aimod", description="AI-powered text moderation using the local model."
    )

    @aimod.command(name="toggle", description="Turn AI text moderation on or off.")
    @require_operator()
    async def aimod_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["ai_moderation"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)

        note = ""
        if enabled:
            note = (
                "\nMessages that get past the regex rules are classified by the local "
                "model. Start with `/guard dry-run enabled:True` for a day and review "
                "`/aimod status` before letting it act."
            )
        await interaction.response.send_message(
            f"AI text moderation is now **{'on' if enabled else 'off'}**.{note}"
        )

    @aimod.command(name="configure", description="Tune AI moderation sensitivity and actions.")
    @app_commands.describe(
        min_confidence="Below this the verdict is logged, never enforced (0.5-1.0)",
        severity_1="Action for mild violations",
        severity_2="Action for clear violations",
        severity_3="Action for severe violations",
        max_per_minute="Cap on model calls per minute in this server",
        log_channel="Where AI moderation decisions are reported",
    )
    @app_commands.choices(
        severity_1=AI_ACTION_CHOICES, severity_2=AI_ACTION_CHOICES, severity_3=AI_ACTION_CHOICES
    )
    @require_operator()
    async def aimod_configure(
        self,
        interaction: discord.Interaction,
        min_confidence: app_commands.Range[float, 0.5, 1.0] | None = None,
        severity_1: app_commands.Choice[str] | None = None,
        severity_2: app_commands.Choice[str] | None = None,
        severity_3: app_commands.Choice[str] | None = None,
        max_per_minute: app_commands.Range[int, 1, 120] | None = None,
        log_channel: discord.TextChannel | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["ai_moderation"]
        if min_confidence is not None:
            cfg["min_confidence"] = float(min_confidence)
        for level, choice in (("1", severity_1), ("2", severity_2), ("3", severity_3)):
            if choice is not None:
                cfg["actions"][level] = choice.value
        if max_per_minute is not None:
            cfg["max_checks_per_minute"] = max_per_minute
        if log_channel is not None:
            cfg["log_channel_id"] = log_channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("AI moderation settings updated.")

    @aimod.command(name="category", description="Enable or disable one violation category.")
    @app_commands.choices(category=CATEGORY_CHOICES)
    @require_operator()
    async def aimod_category(
        self, interaction: discord.Interaction, category: app_commands.Choice[str], enabled: bool
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        current = set(config["ai_moderation"]["categories"])
        if enabled:
            current.add(category.value)
        else:
            current.discard(category.value)
        config["ai_moderation"]["categories"] = sorted(current)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"**{category.name}** is now {'watched' if enabled else 'ignored'}."
        )

    @aimod.command(name="status", description="Show AI moderation configuration.")
    @require_operator()
    async def aimod_status(self, interaction: discord.Interaction) -> None:
        cfg = self.bot.runtime.config(interaction.guild.id)["ai_moderation"]
        watched = set(cfg["categories"])
        lines = [
            f"{'on ' if c.value in watched else 'off'}  {c.name}" for c in CATEGORY_CHOICES
        ]
        actions = cfg["actions"]
        embed = discord.Embed(
            title=f"AI moderation — {'enabled' if cfg['enabled'] else 'disabled'}",
            description="```\n" + "\n".join(lines) + "```",
            colour=0x2B2D31,
        )
        embed.add_field(
            name="Actions",
            value=(
                f"mild: `{actions.get('1')}` · clear: `{actions.get('2')}` · "
                f"severe: `{actions.get('3')}`"
            ),
            inline=False,
        )
        embed.add_field(
            name="Confidence floor", value=f"{float(cfg['min_confidence']):.0%}", inline=True
        )
        embed.add_field(name="Rate cap", value=f"{cfg['max_checks_per_minute']}/min", inline=True)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @aimod.command(name="test", description="Run a sample message through the classifier.")
    @require_operator()
    async def aimod_test(self, interaction: discord.Interaction, message: str) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        config = self.bot.runtime.config(interaction.guild.id)
        # Force a check even if the feature is off, so it can be tuned safely.
        probe = {**config, "ai_moderation": {**config["ai_moderation"], "enabled": True}}
        verdict = await self.bot.runtime.ai_moderation.classify(
            message, interaction.guild.id, probe
        )
        if verdict is None:
            await interaction.followup.send(
                "No verdict — the message was too short, the rate cap was hit, or the "
                "model is unreachable."
            )
            return
        action = self.bot.runtime.ai_moderation.action_for(verdict, probe)
        embed = discord.Embed(title="Classifier result", colour=0x2B2D31)
        embed.add_field(name="Violation", value=str(verdict.violation), inline=True)
        embed.add_field(name="Category", value=verdict.category, inline=True)
        embed.add_field(name="Severity", value=str(verdict.severity), inline=True)
        embed.add_field(name="Confidence", value=f"{verdict.confidence:.0%}", inline=True)
        embed.add_field(name="Would do", value=f"`{action}`", inline=True)
        embed.add_field(name="Reason", value=verdict.reason or "—", inline=False)
        await interaction.followup.send(embed=embed)


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(AutomodCmds(bot))
