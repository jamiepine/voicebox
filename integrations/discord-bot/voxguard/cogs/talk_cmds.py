"""`/talk-ai`, `/talk-here`, and `/help`.

`talk-ai` turns a text channel into a direct line to the agent: every message
is read, answered, and — where the sender has the standing for it — acted on.
`talk-here` bridges the other way: you type, the bot speaks it in the voice
channel in its cloned voice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator
from ..voicebox_client import VoiceboxError

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

HELP_SECTIONS = [
    (
        "Getting started",
        "setup",
        [
            ("/help", "This message. Pass a section name for detail."),
            ("/setup", "Guided first-run checklist for this server."),
            ("/stats", "Bot health, uptime and activity."),
            ("/dashboard", "Link to the web stats dashboard."),
        ],
    ),
    (
        "AI",
        "ai",
        [
            ("/chat", "Talk to the AI in text."),
            ("/talk-ai", "Make a channel a direct line to the AI — it reads and acts on every message."),
            ("/talk-here", "Type here, the AI speaks it aloud in your voice channel."),
            ("/personality-ai", "Set its persona, bound voice, model and emotion."),
            ("/roam", "Let it join conversations unprompted, with optional tool tiers."),
            ("/aimod", "AI-powered text moderation."),
        ],
    ),
    (
        "Voice",
        "voice",
        [
            ("/join, /leave", "Join a voice channel and moderate it in real time."),
            ("/voiceclone", "Clone a voice from an audio sample."),
            ("/vctalk", "Live spoken conversation with the AI."),
            ("/voice commands", "Let people give the bot spoken instructions."),
            ("/voice say", "Speak a message aloud in the voice channel."),
            ("/voice summary", "AI recap of the recent voice conversation."),
            ("/catch, /blacklist", "Configure voice word filtering and its response."),
        ],
    ),
    (
        "Music",
        "music",
        [
            ("/play", "Play a YouTube link, playlist, or search term."),
            ("/skip, /stop, /pause, /resume", "Playback control."),
            ("/queue, /nowplaying", "See what's playing and what's next."),
            ("/volume, /loop, /shuffle, /remove", "Queue and output control."),
        ],
    ),
    (
        "Moderation",
        "moderation",
        [
            ("/ban, /kick, /timeout, /warn", "Punishments, each opening a numbered case."),
            ("/unban, /untimeout", "Reverse them."),
            ("/case, /modlog", "Inspect and edit case history."),
            ("/purge, /lock, /unlock, /slowmode", "Channel control."),
            ("/automod, /antinuke, /raid", "Automatic protection."),
            ("/guard", "Safety controls: dry-run, immunity, circuit breaker."),
        ],
    ),
    (
        "Community",
        "community",
        [
            ("/rank, /leaderboard, /levels", "Text and voice XP."),
            ("/role, /buttonroles, /autorole", "Role management and self-assign pickers."),
            ("/welcome, /logs", "Greetings and the server audit trail."),
            ("/ticket, /starboard, /giveaway, /poll, /tag", "Engagement features."),
            ("/thread", "Create and manage threads."),
            ("/userinfo, /serverinfo, /avatar", "Information."),
        ],
    ),
]


class TalkCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    # -- /talk-ai -----------------------------------------------------------

    @app_commands.command(
        name="talk-ai",
        description="Make a channel a direct line to the AI — it reads and acts on every message.",
    )
    @app_commands.describe(
        enabled="Turn it on or off",
        channel="Which channel (defaults to this one)",
        allow_actions="Let it run commands, not just reply",
    )
    @require_operator()
    async def talk_ai(
        self,
        interaction: discord.Interaction,
        enabled: bool,
        channel: discord.TextChannel | None = None,
        allow_actions: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["talk_ai"]
        target = channel or interaction.channel

        channels = {str(c) for c in cfg["channels"]}
        if enabled:
            channels.add(str(target.id))
        else:
            channels.discard(str(target.id))
        cfg["channels"] = sorted(channels)
        cfg["enabled"] = bool(channels)
        if allow_actions is not None:
            cfg["allow_actions"] = allow_actions
        runtime.save_config(interaction.guild.id, config)

        if not enabled:
            await interaction.response.send_message(
                f"The AI will no longer read {target.mention}."
            )
            return

        note = (
            "It will reply to everything sent there and can run commands on behalf of "
            "whoever asked — each person's requests are capped by their own permissions."
            if cfg["allow_actions"]
            else "It will reply but won't run any commands (`allow_actions: False`)."
        )
        await interaction.response.send_message(
            f"The AI is now reading {target.mention}.\n{note}"
        )

    # -- /talk-here ---------------------------------------------------------

    @app_commands.command(
        name="talk-here",
        description="Type in this channel and the AI speaks it aloud in your voice channel.",
    )
    @app_commands.describe(
        enabled="Turn it on or off",
        channel="Text channel to read from (defaults to this one)",
        speak_replies="Speak the AI's answer aloud, not just relay what you typed",
    )
    @require_operator()
    async def talk_here(
        self,
        interaction: discord.Interaction,
        enabled: bool,
        channel: discord.TextChannel | None = None,
        speak_replies: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["talk_here"]
        target = channel or interaction.channel

        if enabled and not config["ai"].get("voice_profile_id"):
            await interaction.response.send_message(
                "No voice is bound yet. Run `/voiceclone` and then "
                "`/personality-ai voice:<name>` first.",
                ephemeral=True,
            )
            return

        channels = {str(c) for c in cfg["channels"]}
        if enabled:
            channels.add(str(target.id))
        else:
            channels.discard(str(target.id))
        cfg["channels"] = sorted(channels)
        cfg["enabled"] = bool(channels)
        if speak_replies is not None:
            cfg["speak_replies"] = speak_replies
        runtime.save_config(interaction.guild.id, config)

        if not enabled:
            await interaction.response.send_message(f"Stopped speaking messages from {target.mention}.")
            return

        voice_note = ""
        if interaction.guild.voice_client is None:
            voice_note = "\nI'm not in a voice channel yet — run `/join` or `/play` to bring me in."

        mode = (
            "I'll answer out loud in my cloned voice."
            if cfg["speak_replies"]
            else "I'll read your messages aloud verbatim."
        )
        await interaction.response.send_message(
            f"Listening to {target.mention}. {mode}{voice_note}"
        )

    # -- /help --------------------------------------------------------------

    @app_commands.command(name="help", description="Show what this bot can do.")
    @app_commands.describe(section="Show one section in detail")
    @app_commands.choices(
        section=[app_commands.Choice(name=title, value=key) for title, key, _ in HELP_SECTIONS]
    )
    async def help_command(
        self, interaction: discord.Interaction, section: app_commands.Choice[str] | None = None
    ) -> None:
        if section is not None:
            match = next((s for s in HELP_SECTIONS if s[1] == section.value), None)
            if match is None:
                await interaction.response.send_message("Unknown section.", ephemeral=True)
                return
            title, _, entries = match
            embed = discord.Embed(
                title=title,
                description="\n".join(f"**{cmd}**\n{desc}" for cmd, desc in entries),
                colour=0x2B2D31,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        config = self.bot.runtime.config(interaction.guild.id)
        active = [
            name
            for name, block in (
                ("Voice moderation", "voice"), ("AI moderation", "ai_moderation"),
                ("Text automod", "automod"), ("Raid detection", "raid"),
                ("Anti-nuke", "antinuke"), ("Levelling", "levels"),
                ("Logging", "logging"), ("Tickets", "tickets"),
                ("Starboard", "starboard"), ("Talk-AI", "talk_ai"),
            )
            if config.get(block, {}).get("enabled")
        ]

        embed = discord.Embed(
            title="VoxGuard",
            description=(
                "Voice moderation, an AI agent that speaks in a cloned voice, a full "
                "moderation suite, and a music player.\n\n"
                "Use `/help section:<name>` for the commands in each area."
            ),
            colour=0x2B2D31,
        )
        for title, key, entries in HELP_SECTIONS:
            embed.add_field(
                name=title,
                value=" ".join(f"`{cmd.split(',')[0]}`" for cmd, _ in entries),
                inline=False,
            )
        embed.add_field(
            name="Active here",
            value=", ".join(active) if active else "Nothing enabled yet — try `/setup`.",
            inline=False,
        )
        embed.set_footer(text=f"{len(self.bot.tree.get_commands())} commands available")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- /setup -------------------------------------------------------------

    @app_commands.command(name="setup", description="Check what's configured and what's missing.")
    @require_operator()
    async def setup_command(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        guild = interaction.guild

        checks: list[tuple[bool, str, str]] = []

        voicebox_up = await runtime.voicebox.health()
        checks.append((
            voicebox_up, "Voicebox reachable",
            "Transcription and speech need it running at "
            f"`{self.bot.settings.voicebox_url}`.",
        ))

        ollama_up = await runtime.ollama.is_up()
        checks.append((
            ollama_up, "Ollama reachable",
            "The AI features need it. It starts automatically if `ollama` is installed.",
        ))

        words = len(runtime.matchers.get(guild.id, "voice"))
        checks.append((
            words > 0, f"Word list configured ({words} terms)",
            "Voice moderation flags nothing until you run `/blacklist add`.",
        ))

        voice_bound = bool(config["ai"].get("voice_profile_id"))
        checks.append((
            voice_bound, "Cloned voice bound",
            "Run `/voiceclone`, then `/personality-ai voice:<name>` so it can speak.",
        ))

        perms = guild.me.guild_permissions
        needed = {
            "Moderate Members": perms.moderate_members,
            "Kick Members": perms.kick_members,
            "Ban Members": perms.ban_members,
            "Manage Roles": perms.manage_roles,
            "Manage Channels": perms.manage_channels,
            "Connect / Speak": perms.connect and perms.speak,
        }
        missing = [name for name, held in needed.items() if not held]
        checks.append((
            not missing, "Permissions",
            f"Missing: {', '.join(missing)}." if missing else "All present.",
        ))

        try:
            import yt_dlp  # noqa: F401
            has_ytdlp = True
        except ImportError:
            has_ytdlp = False
        checks.append((has_ytdlp, "Music support", "Install with `pip install yt-dlp`."))

        lines = [
            f"{'PASS' if ok else 'TODO'}  {label}" + ("" if ok else f"\n         {hint}")
            for ok, label, hint in checks
        ]
        embed = discord.Embed(
            title="Setup check",
            description="```\n" + "\n".join(lines) + "\n```",
            colour=0x2B2D31,
        )
        embed.set_footer(text="Run /help for the full command list.")
        await interaction.followup.send(embed=embed)


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(TalkCmds(bot))
