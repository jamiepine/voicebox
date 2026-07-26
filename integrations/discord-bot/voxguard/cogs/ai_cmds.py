"""`/personality-ai`, `/voiceclone`, `/vctalk`, `/roam`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator
from ..voiceclone import CONSENT_PHRASE, ConsentError, clone_voice
from ..voicebox_client import VoiceboxError

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

VOICE_SAMPLE_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


class AICmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    # -- /personality-ai ------------------------------------------------

    @app_commands.command(
        name="personality-ai",
        description="Set how the AI talks, and which cloned voice it speaks with.",
    )
    @app_commands.describe(
        personality="How the AI should talk (tone, attitude, catchphrases, etc.)",
        voice="Name of a voice profile (from /voiceclone) for the AI to speak with",
        model="Override the Ollama model for this server",
        emotion="Let the AI vary vocal delivery (tone/pace) based on context",
    )
    @require_operator()
    async def personality_ai(
        self,
        interaction: discord.Interaction,
        personality: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        emotion: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        ai = config["ai"]

        if personality is None and voice is None and model is None and emotion is None:
            profile_note = ai.get("voice_profile_name") or "(none bound — text only)"
            await interaction.response.send_message(
                f"**Current personality:**\n> {ai['personality']}\n"
                f"**Voice:** {profile_note}\n"
                f"**Model:** {ai.get('model') or self.bot.settings.ollama_model} (default)\n"
                f"**Emotion:** {ai.get('emotion', True)}",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        if voice:
            profile = await runtime.voicebox.find_profile(voice)
            if profile is None:
                await interaction.followup.send(
                    f"No voice profile named '{voice}'. Clone one first with `/voiceclone`."
                )
                return
            if not runtime.store.has_consent(interaction.guild.id, profile.id):
                await interaction.followup.send(
                    "That voice profile wasn't cloned through this bot, so there's no recorded "
                    "consent for it. Only clone and use voices you have the right to use — see "
                    "`/voiceclone`."
                )
                return
            ai["voice_profile_id"] = profile.id
            ai["voice_profile_name"] = profile.name

        if personality:
            ai["personality"] = personality[:2000]
        if model:
            ai["model"] = model
        if emotion is not None:
            ai["emotion"] = emotion

        runtime.save_config(interaction.guild.id, config)
        await interaction.followup.send("Personality updated.")

    # -- /voiceclone ----------------------------------------------------

    @app_commands.command(
        name="voiceclone",
        description="Clone a voice from an audio sample for the AI to speak with.",
    )
    @app_commands.describe(
        name="A name for this voice profile",
        file="An audio sample of the voice (mp3/wav/m4a/ogg, 10-30s of clear speech works best)",
        attestation=f'Type exactly: "{CONSENT_PHRASE}"',
        reference_text="What is said in the clip (auto-transcribed if omitted)",
    )
    @require_operator()
    async def voiceclone(
        self,
        interaction: discord.Interaction,
        name: str,
        file: discord.Attachment,
        attestation: str,
        reference_text: str | None = None,
    ) -> None:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VOICE_SAMPLE_EXTS:
            await interaction.response.send_message(
                f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(VOICE_SAMPLE_EXTS))}",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        try:
            data = await file.read()
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Couldn't download that file: {exc}")
            return

        try:
            profile = await clone_voice(
                self.bot.runtime.voicebox,
                self.bot.runtime.store,
                guild_id=interaction.guild.id,
                uploader_id=interaction.user.id,
                name=name,
                audio=data,
                filename=file.filename or "sample.wav",
                reference_text=reference_text,
                attestation=attestation,
            )
        except ConsentError as exc:
            await interaction.followup.send(str(exc))
            return
        except VoiceboxError as exc:
            await interaction.followup.send(f"Voicebox couldn't process that sample: {exc}")
            return

        await interaction.followup.send(
            f"Cloned voice **{profile.name}**. Bind it with "
            f"`/personality-ai voice:{profile.name}`, then talk to it with `/vctalk join`."
        )

    # -- /vctalk ----------------------------------------------------------

    vctalk = app_commands.Group(name="vctalk", description="Live voice conversation with the AI.")

    @vctalk.command(name="join", description="Join a voice channel and start a live conversation.")
    @require_operator()
    async def vctalk_join(self, interaction: discord.Interaction, channel: discord.VoiceChannel) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        if not config["ai"].get("voice_profile_id"):
            await interaction.response.send_message(
                "No voice is bound yet. Clone one with `/voiceclone`, then run "
                "`/personality-ai voice:<name>` before starting `/vctalk`.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        try:
            await self.bot.vctalk.start(
                interaction.guild, channel, text_log_channel=interaction.channel, language=None
            )
        except discord.ClientException as exc:
            await interaction.followup.send(f"Couldn't join: {exc}")
            return

        runtime.vctalk_active[interaction.guild.id] = channel.id
        await interaction.followup.send(
            f"Joined {channel.mention}. Speaking with **{config['ai']['voice_profile_name']}**'s "
            "voice — just talk normally."
        )

    @vctalk.command(name="stop", description="End the live voice conversation.")
    @require_operator()
    async def vctalk_stop(self, interaction: discord.Interaction) -> None:
        stopped = await self.bot.vctalk.stop(interaction.guild.id)
        self.bot.runtime.vctalk_active.pop(interaction.guild.id, None)
        await interaction.response.send_message(
            "Conversation ended." if stopped else "No active conversation here."
        )

    # -- /roam --------------------------------------------------------------

    roam = app_commands.Group(name="roam", description="The AI's free-standing presence in text channels.")

    @roam.command(name="toggle", description="Turn roam mode on or off.")
    @app_commands.describe(
        manage="Allow channel/role management tools (create/rename/delete channels & roles, server icon)",
        moderate="Allow moderation tools (timeout/kick/ban/purge)",
    )
    @require_operator()
    async def roam_toggle(
        self,
        interaction: discord.Interaction,
        enabled: bool,
        manage: bool | None = None,
        moderate: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["roam"]["enabled"] = enabled

        # Tri-state on purpose: omitting `manage`/`moderate` preserves what
        # the guild already granted. Defaulting them to False would silently
        # revoke tiers every time someone toggled roam off and on again.
        tiers = {t for t in config["roam"].get("tiers") or ["chat"]}
        tiers.add("chat")
        for name, value in (("manage", manage), ("moderate", moderate)):
            if value is True:
                tiers.add(name)
            elif value is False:
                tiers.discard(name)

        ordered = [t for t in ("chat", "manage", "moderate") if t in tiers]
        config["roam"]["tiers"] = ordered
        runtime.save_config(interaction.guild.id, config)

        warning = ""
        if "moderate" in tiers:
            warning = (
                "\n⚠️ Moderation tools are enabled — the AI can time out, kick, and ban based on "
                "its own judgement of the conversation. Destructive actions still require an "
                "admin's approval unless you disable that with `/roam configure`."
            )
        await interaction.response.send_message(
            f"Roam is now **{'on' if enabled else 'off'}**. Tiers: {', '.join(ordered)}.{warning}"
        )

    @roam.command(name="channels", description="Restrict roam to specific channels (empty = everywhere).")
    @require_operator()
    async def roam_channels(
        self, interaction: discord.Interaction, channel: discord.TextChannel, remove: bool = False
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        channels = {str(c) for c in config["roam"]["channels"]}
        if remove:
            channels.discard(str(channel.id))
        else:
            channels.add(str(channel.id))
        config["roam"]["channels"] = list(channels)
        runtime.save_config(interaction.guild.id, config)
        state = "removed from" if remove else "added to"
        await interaction.response.send_message(f"{channel.mention} {state} the roam allowlist.")

    @roam.command(name="configure", description="Tune roam behaviour.")
    @app_commands.describe(
        idle_reply_seconds="Minimum seconds between unprompted replies per channel",
        max_actions_per_hour="Cap on agent tool actions per hour",
        require_confirm_destructive="Require admin approval before bans/kicks/deletions",
        audit_channel="Where approval requests and action logs are posted",
    )
    @require_operator()
    async def roam_configure(
        self,
        interaction: discord.Interaction,
        idle_reply_seconds: app_commands.Range[int, 30, 3600] | None = None,
        max_actions_per_hour: app_commands.Range[int, 1, 100] | None = None,
        require_confirm_destructive: bool | None = None,
        audit_channel: discord.TextChannel | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        roam = config["roam"]
        if idle_reply_seconds:
            roam["idle_reply_seconds"] = idle_reply_seconds
        if max_actions_per_hour:
            roam["max_actions_per_hour"] = max_actions_per_hour
        if require_confirm_destructive is not None:
            roam["require_confirm_destructive"] = require_confirm_destructive
        if audit_channel:
            roam["audit_channel_id"] = audit_channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Roam settings updated.")

    @roam.command(name="status", description="Show current roam configuration.")
    @require_operator()
    async def roam_status(self, interaction: discord.Interaction) -> None:
        roam = self.bot.runtime.config(interaction.guild.id)["roam"]
        embed = discord.Embed(title="Roam mode")
        embed.add_field(name="Enabled", value=str(roam["enabled"]), inline=True)
        embed.add_field(name="Tiers", value=", ".join(roam["tiers"]), inline=True)
        embed.add_field(
            name="Channels",
            value=", ".join(f"<#{c}>" for c in roam["channels"]) or "(everywhere)",
            inline=False,
        )
        embed.add_field(
            name="Approval required for destructive actions",
            value=str(roam["require_confirm_destructive"]),
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(AICmds(bot))
