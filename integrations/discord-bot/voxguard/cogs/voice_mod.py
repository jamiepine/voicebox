"""`/join`, `/leave`, `/catch`, `/blacklist`, `/voicenotes`, `/guard`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator
from ..matching import parse_terms

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

ACTION_CHOICES = [
    app_commands.Choice(name="Log only", value="log"),
    app_commands.Choice(name="Warn", value="warn"),
    app_commands.Choice(name="Timeout", value="timeout"),
    app_commands.Choice(name="Kick", value="kick"),
    app_commands.Choice(name="Ban", value="ban"),
]
ESCALATE_CHOICES = [c for c in ACTION_CHOICES if c.value != "warn"]


HANDLER_KEY = "moderation"


class VoiceMod(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    # -- /join, /leave --------------------------------------------------

    @app_commands.command(name="join", description="Join a voice channel and start listening for blocked language.")
    @app_commands.describe(channel="The voice channel to join")
    @require_operator()
    async def join(self, interaction: discord.Interaction, channel: discord.VoiceChannel) -> None:
        await interaction.response.defer(thinking=True)
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)

        matcher = runtime.matchers.get(interaction.guild.id, "voice")
        if config["voice"]["enabled"] and len(matcher) == 0:
            await interaction.followup.send(
                "⚠️ Voice moderation is enabled but no blacklisted words are configured yet. "
                "I'll join and transcribe, but nothing will be flagged until you run "
                "`/blacklist add`. Use `/guard status` to check settings.",
            )

        try:
            session = await runtime.sessions.join(channel, language=None)
        except discord.ClientException as exc:
            await interaction.followup.send(f"Couldn't join: {exc}")
            return

        # Idempotent: re-running /join to move channels replaces the handler
        # under the same key rather than stacking a second one.
        session.add_handler(HANDLER_KEY, self._make_moderation_handler(interaction.guild.id))

        # If spoken commands are enabled, this channel takes them too — the
        # bot is already transcribing, so it costs nothing extra.
        note = ""
        if config["voice_commands"]["enabled"]:
            self.bot.vctalk.attach_command_listener(interaction.guild.id)
            words = self.bot.voice_router.wake_words(config, interaction.guild.me.display_name)
            note = f" Say \"{words[0]}, …\" to give me spoken instructions."

        await interaction.followup.send(
            f"Joined {channel.mention} and started listening.{note}"
        )

    @app_commands.command(name="leave", description="Leave the current voice channel and stop listening.")
    @require_operator()
    async def leave(self, interaction: discord.Interaction) -> None:
        left = await self.bot.runtime.sessions.leave(interaction.guild.id)
        await self.bot.vctalk.stop(interaction.guild.id)
        self.bot.runtime.vctalk_active.pop(interaction.guild.id, None)
        await interaction.response.send_message(
            "Left the voice channel." if left else "I wasn't in a voice channel here."
        )

    def _make_moderation_handler(self, guild_id: int):
        async def handle(utterance) -> None:  # noqa: ANN001
            guild = self.bot.get_guild(guild_id)
            if guild is None:
                return
            config = self.bot.runtime.config(guild_id)
            if not config["voice"]["enabled"]:
                return

            member = guild.get_member(utterance.user_id)
            channel = guild.get_channel(utterance.channel_id)
            channel_name = getattr(channel, "name", str(utterance.channel_id))

            if config["voice"].get("console_transcript", True):
                who = member.display_name if member else utterance.user_id
                # Logged, not printed: this is other people's speech, and it
                # should honour the same level/format/filtering as everything
                # else rather than going straight to stdout.
                log.info("[voice:%s/%s] %s: %s", guild.name, channel_name, who, utterance.text)

            matcher = self.bot.runtime.matchers.get(guild_id, "voice")
            if not len(matcher) or member is None:
                return

            matches = matcher.scan(utterance.text, min_confidence=config["voice"]["min_confidence"])
            if not matches:
                return

            log.info(
                "[voice-mod] guild=%s user=%s matched %s in %r",
                guild_id, member, [m.term for m in matches], utterance.text,
            )
            await self.bot.runtime.enforcer.enforce(
                member,
                scope="voice",
                config=config,
                matches=matches,
                transcript=utterance.text,
                source=f"voice channel #{channel_name}",
            )

        return handle

    # -- /catch -----------------------------------------------------------

    @app_commands.command(
        name="catch",
        description="Configure what happens when blocked language is caught in voice chat.",
    )
    @app_commands.describe(
        action="What to do on the first offense",
        warn_message="Message DM'd on a warning (use {count} and {limit})",
        warn_limit="How many warnings before escalating",
        escalate_to="What happens once the warning limit is hit",
        timeout_minutes="Timeout duration in minutes, if the action is timeout",
    )
    @app_commands.choices(action=ACTION_CHOICES, escalate_to=ESCALATE_CHOICES)
    @require_operator()
    async def catch(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        warn_message: str | None = None,
        warn_limit: app_commands.Range[int, 1, 20] | None = None,
        escalate_to: app_commands.Choice[str] | None = None,
        timeout_minutes: app_commands.Range[int, 1, 10080] | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        voice = config["voice"]
        voice["action"] = action.value
        if warn_message:
            voice["warn_message"] = warn_message[:500]
        if warn_limit:
            voice["warn_limit"] = warn_limit
        if escalate_to:
            voice["escalate_to"] = escalate_to.value
        if timeout_minutes:
            voice["timeout_minutes"] = timeout_minutes

        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Voice enforcement set: first offense → **{action.value}**"
            + (f", escalating to **{voice['escalate_to']}** after {voice['warn_limit']} warnings." if action.value == "warn" else ".")
        )

    # -- /blacklist ---------------------------------------------------------

    blacklist = app_commands.Group(name="blacklist", description="Manage blocked/allowed word lists.")

    @blacklist.command(name="add", description="Add words to a block or allow list, by text or uploaded file.")
    @app_commands.describe(
        scope="Where this list applies",
        listing="Block (flag these) or allow (never flag these)",
        text="Words/phrases, one per line or comma-separated",
        file="A .txt file with one term per line",
    )
    @app_commands.choices(
        scope=[
            app_commands.Choice(name="Live voice chat", value="voice"),
            app_commands.Choice(name="Voice notes", value="voice_notes"),
        ],
        listing=[
            app_commands.Choice(name="Block", value="block"),
            app_commands.Choice(name="Allow (exception)", value="allow"),
        ],
    )
    @require_operator()
    async def blacklist_add(
        self,
        interaction: discord.Interaction,
        scope: app_commands.Choice[str],
        listing: app_commands.Choice[str] | None = None,
        text: str | None = None,
        file: discord.Attachment | None = None,
    ) -> None:
        if not text and not file:
            await interaction.response.send_message("Provide `text`, a `file`, or both.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        raw_parts = []
        if text:
            raw_parts.append(text)
        if file:
            if file.size > 2 * 1024 * 1024:
                await interaction.followup.send("File too large (max 2 MB).")
                return
            try:
                raw_parts.append((await file.read()).decode("utf-8", errors="replace"))
            except discord.HTTPException as exc:
                await interaction.followup.send(f"Couldn't download the file: {exc}")
                return

        terms = parse_terms("\n".join(raw_parts))
        if not terms:
            await interaction.followup.send("No usable terms found in that input.")
            return

        listing_value = (listing or app_commands.Choice(name="Block", value="block")).value
        added = self.bot.runtime.store.add_terms(
            interaction.guild.id, scope.value, terms, listing=listing_value, added_by=interaction.user.id
        )
        self.bot.runtime.matchers.invalidate(interaction.guild.id)
        await interaction.followup.send(
            f"Added {added} new term(s) to the **{scope.name}** {listing_value} list "
            f"({len(terms) - added} were already present)."
        )

    @blacklist.command(name="remove", description="Remove a single term from a list.")
    @app_commands.choices(
        scope=[
            app_commands.Choice(name="Live voice chat", value="voice"),
            app_commands.Choice(name="Voice notes", value="voice_notes"),
        ]
    )
    @require_operator()
    async def blacklist_remove(
        self, interaction: discord.Interaction, scope: app_commands.Choice[str], term: str
    ) -> None:
        removed = self.bot.runtime.store.remove_term(interaction.guild.id, scope.value, term)
        self.bot.runtime.matchers.invalidate(interaction.guild.id)
        await interaction.response.send_message(
            f"Removed `{term}`." if removed else f"`{term}` wasn't on the list."
        )

    @blacklist.command(name="clear", description="Remove every term from a list. This cannot be undone.")
    @app_commands.choices(
        scope=[
            app_commands.Choice(name="Live voice chat", value="voice"),
            app_commands.Choice(name="Voice notes", value="voice_notes"),
        ]
    )
    @require_operator()
    async def blacklist_clear(
        self, interaction: discord.Interaction, scope: app_commands.Choice[str]
    ) -> None:
        count = self.bot.runtime.store.clear_terms(interaction.guild.id, scope.value)
        self.bot.runtime.matchers.invalidate(interaction.guild.id)
        await interaction.response.send_message(f"Cleared {count} term(s) from **{scope.name}**.")

    @blacklist.command(name="list", description="Show the current word list.")
    @app_commands.choices(
        scope=[
            app_commands.Choice(name="Live voice chat", value="voice"),
            app_commands.Choice(name="Voice notes", value="voice_notes"),
        ]
    )
    @require_operator()
    async def blacklist_list(
        self, interaction: discord.Interaction, scope: app_commands.Choice[str]
    ) -> None:
        blocked = self.bot.runtime.store.list_terms(interaction.guild.id, scope.value, listing="block")
        allowed = self.bot.runtime.store.list_terms(interaction.guild.id, scope.value, listing="allow")

        embed = discord.Embed(title=f"Word list — {scope.name}")
        embed.add_field(
            name=f"Blocked ({len(blocked)})",
            value=", ".join(f"`{r['term']}`" for r in blocked[:60]) or "(empty)",
            inline=False,
        )
        if allowed:
            embed.add_field(
                name=f"Allowed exceptions ({len(allowed)})",
                value=", ".join(f"`{r['term']}`" for r in allowed[:30]),
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- /voicenotes --------------------------------------------------------

    voicenotes = app_commands.Group(name="voicenotes", description="Moderate Discord voice messages.")

    @voicenotes.command(name="toggle", description="Turn voice-note moderation on or off.")
    @require_operator()
    async def voicenotes_toggle(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["voice_notes"]["enabled"] = enabled
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Voice-note moderation is now **{'on' if enabled else 'off'}**."
        )

    @voicenotes.command(name="actions", description="Configure what happens when a voice note is flagged.")
    @app_commands.choices(action=ACTION_CHOICES, escalate_to=ESCALATE_CHOICES)
    @require_operator()
    async def voicenotes_actions(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        warn_message: str | None = None,
        warn_limit: app_commands.Range[int, 1, 20] | None = None,
        escalate_to: app_commands.Choice[str] | None = None,
        delete_message: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        notes = config["voice_notes"]
        notes["action"] = action.value
        if warn_message:
            notes["warn_message"] = warn_message[:500]
        if warn_limit:
            notes["warn_limit"] = warn_limit
        if escalate_to:
            notes["escalate_to"] = escalate_to.value
        if delete_message is not None:
            notes["delete_message"] = delete_message
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(f"Voice-note enforcement set to **{action.value}**.")

    # -- /guard (shared safety controls) ------------------------------------

    guard = app_commands.Group(name="guard", description="Safety controls shared across all enforcement.")

    @guard.command(name="status", description="Show current moderation configuration.")
    @require_operator()
    async def guard_status(self, interaction: discord.Interaction) -> None:
        config = self.bot.runtime.config(interaction.guild.id)
        v, n, r, g = config["voice"], config["voice_notes"], config["raid"], config["guardrails"]
        breaker = "TRIPPED" if self.bot.runtime.limiter.is_tripped(interaction.guild.id) else "ok"

        embed = discord.Embed(title="VoxGuard status")
        embed.add_field(
            name="Live voice",
            value=f"{'on' if v['enabled'] else 'off'} · action={v['action']} · warn limit={v['warn_limit']}",
            inline=False,
        )
        embed.add_field(
            name="Voice notes",
            value=f"{'on' if n['enabled'] else 'off'} · action={n['action']}",
            inline=False,
        )
        embed.add_field(
            name="Raid detection",
            value=f"{'on' if r['enabled'] else 'off'} · action={r['action']} · threshold={r['score_threshold']}",
            inline=False,
        )
        embed.add_field(
            name="Guardrails",
            value=(
                f"dry_run={g['dry_run']} · hourly limit={g['max_actions_per_hour']} · "
                f"circuit breaker: {breaker}"
            ),
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @guard.command(name="dry-run", description="When on, actions are logged but never applied.")
    @require_operator()
    async def guard_dry_run(self, interaction: discord.Interaction, enabled: bool) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config["guardrails"]["dry_run"] = enabled
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(f"Dry run is now **{'on' if enabled else 'off'}**.")

    @guard.command(name="resume", description="Resume automated enforcement after the hourly limit paused it.")
    @require_operator()
    async def guard_resume(self, interaction: discord.Interaction) -> None:
        self.bot.runtime.limiter.reset(interaction.guild.id)
        await interaction.response.send_message("Enforcement resumed.")

    @guard.command(name="log-channel", description="Set where moderation logs are posted.")
    @app_commands.choices(
        scope=[
            app_commands.Choice(name="Live voice chat", value="voice"),
            app_commands.Choice(name="Voice notes", value="voice_notes"),
        ]
    )
    @require_operator()
    async def guard_log_channel(
        self,
        interaction: discord.Interaction,
        scope: app_commands.Choice[str],
        channel: discord.TextChannel,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        config[scope.value]["log_channel_id"] = channel.id
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(f"{scope.name} logs will post in {channel.mention}.")

    @guard.command(name="immune-role", description="Exempt a role from automated enforcement.")
    @require_operator()
    async def guard_immune_role(
        self, interaction: discord.Interaction, role: discord.Role, remove: bool = False
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        roles = {int(r) for r in config["guardrails"]["immune_roles"]}
        if remove:
            roles.discard(role.id)
        else:
            roles.add(role.id)
        config["guardrails"]["immune_roles"] = list(roles)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"{role.mention} is now {'immune' if not remove else 'no longer immune'}."
        )

    @guard.command(name="warnings", description="Show a member's recent moderation history.")
    @require_operator()
    async def guard_warnings(self, interaction: discord.Interaction, member: discord.Member) -> None:
        rows = self.bot.runtime.store.recent_infractions(interaction.guild.id, member.id)
        if not rows:
            await interaction.response.send_message(f"No infractions recorded for {member.display_name}.")
            return
        lines = [f"`{r['action']}` — {r['scope']} — {r['term']} — {r['transcript'][:60]}" for r in rows]
        await interaction.response.send_message(
            f"**{member.display_name}**'s recent infractions:\n" + "\n".join(lines), ephemeral=True
        )

    @guard.command(name="clear-warnings", description="Reset a member's warning count.")
    @require_operator()
    async def guard_clear_warnings(self, interaction: discord.Interaction, member: discord.Member) -> None:
        n = self.bot.runtime.store.clear_warnings(interaction.guild.id, member.id)
        await interaction.response.send_message(f"Cleared {n} recorded infraction(s) for {member.display_name}.")


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(VoiceMod(bot))
