"""`/rank`, `/leaderboard`, `/levels` — XP configuration and display."""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator
from ..features.levels import level_progress

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

BAR_WIDTH = 18


def progress_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total) if total else 0
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def format_voice(seconds: int) -> str:
    hours, rem = divmod(int(seconds), 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


class LevelsCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    @app_commands.command(name="rank", description="Show your level, XP and voice time.")
    async def rank(
        self, interaction: discord.Interaction, member: discord.Member | None = None
    ) -> None:
        target = member or interaction.user
        store = self.bot.runtime.store
        row = store.get_level_row(interaction.guild.id, target.id)
        if row is None:
            await interaction.response.send_message(
                f"**{target.display_name}** hasn't earned any XP yet.", ephemeral=True
            )
            return

        xp = int(row["xp"])
        level, into, needed = level_progress(xp)
        rank = store.rank_of(interaction.guild.id, target.id)

        embed = discord.Embed(colour=target.colour or discord.Colour(0x5865F2))
        embed.set_author(name=target.display_name, icon_url=target.display_avatar.url)
        embed.add_field(name="Level", value=str(level), inline=True)
        embed.add_field(name="Rank", value=f"#{rank}", inline=True)
        embed.add_field(name="Total XP", value=f"{xp:,}", inline=True)
        embed.add_field(
            name=f"Progress — {into}/{needed} XP",
            value=f"`{progress_bar(into, needed)}`",
            inline=False,
        )
        embed.add_field(name="Messages", value=f"{int(row['messages']):,}", inline=True)
        embed.add_field(name="Voice time", value=format_voice(int(row["voice_seconds"])), inline=True)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="leaderboard", description="Top members by XP.")
    async def leaderboard(
        self, interaction: discord.Interaction, page: app_commands.Range[int, 1, 20] = 1
    ) -> None:
        limit = 10
        offset = (page - 1) * limit
        rows = self.bot.runtime.store.leaderboard(interaction.guild.id, limit, offset)
        if not rows:
            await interaction.response.send_message("Nothing on the leaderboard yet.", ephemeral=True)
            return

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        lines = []
        for index, row in enumerate(rows, start=offset + 1):
            level, _, _ = level_progress(int(row["xp"]))
            marker = medals.get(index, f"`#{index}`")
            lines.append(
                f"{marker} <@{row['user_id']}> — level **{level}** · {int(row['xp']):,} XP · "
                f"{format_voice(int(row['voice_seconds']))} in voice"
            )

        embed = discord.Embed(
            title=f"{interaction.guild.name} leaderboard",
            description="\n".join(lines),
            colour=0xF1C40F,
        )
        embed.set_footer(text=f"Page {page}")
        await interaction.response.send_message(embed=embed)

    levels = app_commands.Group(name="levels", description="Configure the levelling system.")

    @levels.command(name="toggle", description="Turn levelling on or off.")
    @require_operator()
    async def levels_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["levels"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Levelling is now **{'on' if enabled else 'off'}**."
        )

    @levels.command(name="configure", description="Tune XP rates and announcements.")
    @app_commands.describe(
        xp_per_message="XP for a message (default 15)",
        cooldown_seconds="Minimum gap between message XP awards",
        xp_per_voice_minute="XP per minute of active voice chat",
        voice_requires_unmuted="Only award voice XP when unmuted and not alone",
        announce_channel="Where level-up messages go (default: where it happened)",
        announce="Announce level-ups at all",
    )
    @require_operator()
    async def levels_configure(
        self,
        interaction: discord.Interaction,
        xp_per_message: app_commands.Range[int, 0, 200] | None = None,
        cooldown_seconds: app_commands.Range[int, 0, 3600] | None = None,
        xp_per_voice_minute: app_commands.Range[int, 0, 200] | None = None,
        voice_requires_unmuted: bool | None = None,
        announce_channel: discord.TextChannel | None = None,
        announce: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["levels"]
        if xp_per_message is not None:
            cfg["xp_per_message"] = xp_per_message
        if cooldown_seconds is not None:
            cfg["message_cooldown_seconds"] = cooldown_seconds
        if xp_per_voice_minute is not None:
            cfg["xp_per_voice_minute"] = xp_per_voice_minute
        if voice_requires_unmuted is not None:
            cfg["voice_requires_unmuted"] = voice_requires_unmuted
        if announce_channel is not None:
            cfg["announce_channel_id"] = announce_channel.id
        if announce is not None:
            cfg["announce"] = announce
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Levelling settings updated.")

    @levels.command(name="reward", description="Give a role automatically at a level.")
    @require_operator()
    async def levels_reward(
        self,
        interaction: discord.Interaction,
        level: app_commands.Range[int, 1, 500],
        role: discord.Role,
        remove: bool = False,
    ) -> None:
        store = self.bot.runtime.store
        if remove:
            store.remove_level_reward(interaction.guild.id, level)
            await interaction.response.send_message(f"Removed the level {level} reward.")
            return
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role, so I couldn't grant it.", ephemeral=True
            )
            return
        store.set_level_reward(interaction.guild.id, level, role.id)
        await interaction.response.send_message(
            f"Members reaching level **{level}** will get {role.mention}."
        )

    @levels.command(name="rewards", description="List configured level rewards.")
    async def levels_rewards(self, interaction: discord.Interaction) -> None:
        rows = self.bot.runtime.store.level_rewards(interaction.guild.id)
        if not rows:
            await interaction.response.send_message("No level rewards configured.", ephemeral=True)
            return
        lines = [f"Level **{r['level']}** → <@&{r['role_id']}>" for r in rows]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @levels.command(name="reset", description="Wipe all XP in this server. Cannot be undone.")
    @require_operator()
    async def levels_reset(self, interaction: discord.Interaction, confirm: bool = False) -> None:
        if not confirm:
            await interaction.response.send_message(
                "This erases every member's XP. Re-run with `confirm: True` to proceed.",
                ephemeral=True,
            )
            return
        count = self.bot.runtime.store.reset_levels(interaction.guild.id)
        await interaction.response.send_message(f"Reset XP for {count} member(s).")

    @levels.command(name="give", description="Manually grant XP to a member.")
    @require_operator()
    async def levels_give(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        xp: app_commands.Range[int, -1000000, 1000000],
    ) -> None:
        total = self.bot.runtime.store.add_xp(interaction.guild.id, member.id, xp)
        level, _, _ = level_progress(total)
        await interaction.response.send_message(
            f"{member.mention} now has **{total:,} XP** (level {level})."
        )


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(LevelsCmds(bot))
