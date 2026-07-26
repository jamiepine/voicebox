"""The VoxGuard Discord client — event wiring only; commands live in cogs."""

from __future__ import annotations

import datetime as dt
import logging

import discord
from discord.ext import commands, tasks

from . import checks, guardrails
from .agent import AgentContext
from .cogs.community_cmds import GiveawayView
from .config import Settings
from .roam import RoamController
from .runtime import Runtime
from .vctalk import VCTalkController
from .voice_notes import is_voice_message

log = logging.getLogger(__name__)

EXTENSIONS = (
    "voxguard.cogs.voice_mod",
    "voxguard.cogs.raid_cmds",
    "voxguard.cogs.ai_cmds",
    "voxguard.cogs.mod_cmds",
    "voxguard.cogs.roles_cmds",
    "voxguard.cogs.levels_cmds",
    "voxguard.cogs.automod_cmds",
    "voxguard.cogs.community_cmds",
    "voxguard.cogs.utility_cmds",
)

INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.members = True
INTENTS.voice_states = True


class VoxGuardBot(commands.Bot):
    def __init__(self, settings: Settings, runtime: Runtime) -> None:
        super().__init__(command_prefix=commands.when_mentioned, intents=INTENTS)
        self.settings = settings
        self.runtime = runtime
        self.vctalk = VCTalkController(runtime.sessions, runtime.speaker, runtime.agent)
        self.vctalk.set_resolver(self._resolve_vctalk_context)
        self.roam = RoamController(runtime.agent)
        self.dashboard = None

    async def setup_hook(self) -> None:
        # Wire the error handler before any command can be invoked, so a
        # failed permission check produces the intended ephemeral reply
        # rather than discord.py's default traceback-to-console behaviour.
        self.tree.on_error = checks.on_app_command_error

        for ext in EXTENSIONS:
            await self.load_extension(ext)

        if self.settings.dev_guild_id:
            guild = discord.Object(id=self.settings.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Synced commands to dev guild %s", self.settings.dev_guild_id)
        else:
            await self.tree.sync()
            log.info("Synced global commands")

        # Re-register persistent views so buttons on old messages keep working
        # across restarts.
        self.add_view(GiveawayView(self))

        for loop in self._loops():
            loop.start()

        if self.settings.dashboard_enabled:
            from .dashboard import DashboardServer

            self.dashboard = DashboardServer(self)
            try:
                await self.dashboard.start()
            except OSError as exc:
                log.error("Dashboard failed to bind: %s", exc)
                self.dashboard = None

    def _loops(self) -> tuple:
        return (
            self._lockdown_watch,
            self._voice_xp_tick,
            self._giveaway_watch,
            self._case_expiry,
            self._retention_sweep,
        )

    async def close(self) -> None:
        for loop in self._loops():
            loop.cancel()
        if self.dashboard is not None:
            await self.dashboard.stop()
        await self.runtime.aclose()
        await super().close()

    async def on_ready(self) -> None:
        log.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "?")
        try:
            up = await self.runtime.voicebox.health()
            log.info("Voicebox at %s: %s", self.settings.voicebox_url, "reachable" if up else "UNREACHABLE")
        except Exception:
            log.warning("Could not reach Voicebox at %s", self.settings.voicebox_url)

    # -- moderation tiers -----------------------------------------------

    def allowed_tiers(self, config: dict) -> tuple[str, ...]:
        tiers = config.get("roam", {}).get("tiers") or ["chat"]
        # "chat" is always available once roam-adjacent features are used —
        # it's the tier with no side effects.
        return tuple(dict.fromkeys(["chat", *tiers]))

    def _resolve_vctalk_context(self, guild_id: int):
        guild = self.get_guild(guild_id)
        if guild is None:
            return None
        voice_client = guild.voice_client
        if voice_client is None:
            return None
        channel_id = self.runtime.vctalk_active.get(guild_id)
        channel = guild.get_channel(channel_id) if channel_id else None
        config = self.runtime.config(guild_id)
        ctx = AgentContext(
            guild=guild,
            channel=channel,
            invoker=None,
            config=config,
            allowed_tiers=self.allowed_tiers(config),
            voice_client=voice_client,
            approval_channel=channel,
        )
        return guild, ctx, voice_client

    # -- events -----------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        await self.process_commands(message)

        config = self.runtime.config(message.guild.id)

        if is_voice_message(message) is not None:
            matcher = self.runtime.matchers.get(message.guild.id, "voice_notes")
            if len(matcher):
                await self.runtime.voice_notes.handle(message, config, matcher)

        # Automod runs before anything that might reply, so a rule-breaking
        # message doesn't get quoted back by the agent before it's removed.
        if await self._run_automod(message, config):
            return

        await self._auto_thread(message, config)

        try:
            await self.runtime.levels.on_message(message, config)
        except Exception:
            self._record_error("levels", message.guild.id)

        if self.user and self.roam.should_reply(message, config, self.user.id):
            try:
                await self.roam.handle_message(message, config, self.allowed_tiers(config))
            except Exception:
                self._record_error("roam", message.guild.id)

    async def _run_automod(self, message: discord.Message, config: dict) -> bool:
        """Returns True if the message was actioned."""
        if not isinstance(message.author, discord.Member):
            return False
        immune = guardrails.may_action(message.author, config)
        if not immune:
            return False

        matcher = self.runtime.matchers.get(message.guild.id, "voice")
        trigger = self.runtime.automod.check(message, config, matcher)
        if trigger is None:
            return False

        cfg = config.get("automod", {})
        dry = guardrails.dry_run(config)
        actions: list[str] = []

        if trigger.action != "log" and cfg.get("delete_on_trigger", True) and not dry:
            try:
                await message.delete()
                actions.append("deleted")
            except discord.HTTPException:
                pass

        if trigger.action in ("timeout", "kick", "ban") and not dry:
            feasible = guardrails.can_action(message.guild, message.author, trigger.action)
            if feasible:
                try:
                    if trigger.action == "timeout":
                        until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10)
                        await message.author.timeout(until, reason=f"Automod: {trigger.rule}")
                    elif trigger.action == "kick":
                        await message.author.kick(reason=f"Automod: {trigger.rule}")
                    else:
                        await message.author.ban(
                            reason=f"Automod: {trigger.rule}", delete_message_seconds=0
                        )
                    actions.append(trigger.action)
                    self.runtime.store.add_case(
                        message.guild.id, message.author.id, self.user.id,
                        trigger.action, f"Automod: {trigger.detail}",
                    )
                except discord.HTTPException:
                    pass

        self.runtime.store.bump_metric(message.guild.id, "automod_triggers")
        self.runtime.store.audit(
            message.guild.id, "automod", trigger.rule, str(message.author.id), trigger.detail
        )

        if channel_id := cfg.get("log_channel_id"):
            channel = message.guild.get_channel(int(channel_id))
            if isinstance(channel, discord.abc.Messageable):
                embed = discord.Embed(title=f"Automod — {trigger.rule}", colour=0xE67E22)
                embed.add_field(name="Member", value=message.author.mention, inline=True)
                embed.add_field(name="Channel", value=message.channel.mention, inline=True)
                embed.add_field(name="Reason", value=trigger.detail, inline=False)
                embed.add_field(
                    name="Action",
                    value=("dry run — nothing applied" if dry else ", ".join(actions) or "logged"),
                    inline=False,
                )
                if message.content:
                    embed.add_field(
                        name="Content", value=f">>> {message.content[:500]}", inline=False
                    )
                try:
                    await channel.send(embed=embed)
                except discord.HTTPException:
                    pass
        return bool(actions)

    async def _auto_thread(self, message: discord.Message, config: dict) -> None:
        channels = config.get("threads", {}).get("auto_thread_channels", [])
        if str(message.channel.id) not in {str(c) for c in channels}:
            return
        if not isinstance(message.channel, discord.TextChannel) or message.thread is not None:
            return
        try:
            await message.create_thread(
                name=(message.content or f"{message.author.display_name}'s thread")[:100],
                auto_archive_duration=config["threads"].get("auto_archive_minutes", 1440),
            )
            self.runtime.store.bump_metric(message.guild.id, "threads_created")
        except discord.HTTPException:
            pass

    async def on_member_join(self, member: discord.Member) -> None:
        config = self.runtime.config(member.guild.id)

        await self.runtime.welcome.on_join(member, config)
        await self.runtime.events.member_joined(member, config)
        self.runtime.store.bump_metric(member.guild.id, "member_joins")

        event = self.runtime.raid.record_join(member, config)
        if event is None:
            return

        log.warning(
            "Raid detected in guild=%s score=%s (%s)", member.guild.id, event.score, event.summary
        )
        report = await self.runtime.raid.respond(member.guild, config, event)

        channel_id = config.get("raid", {}).get("alert_channel_id")
        if not channel_id:
            return
        channel = member.guild.get_channel(int(channel_id))
        if not isinstance(channel, discord.abc.Messageable):
            return

        mention_role = config.get("raid", {}).get("mention_role_id")
        prefix = f"<@&{mention_role}> " if mention_role else ""
        embed = discord.Embed(
            title="🚨 Raid detected",
            description=event.summary,
            colour=0xE74C3C,
        )
        embed.add_field(name="Score", value=f"{event.score}/100", inline=True)
        embed.add_field(name="Accounts in window", value=str(len(event.joiners)), inline=True)
        embed.add_field(name="Response", value="\n".join(report) or "(none)", inline=False)
        try:
            await channel.send(content=prefix or None, embed=embed)
        except discord.HTTPException:
            pass

    async def on_member_remove(self, member: discord.Member) -> None:
        config = self.runtime.config(member.guild.id)
        await self.runtime.welcome.on_leave(member, config)
        await self.runtime.events.member_left(member, config)
        self.runtime.store.bump_metric(member.guild.id, "member_leaves")

    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        await self.runtime.events.member_updated(
            before, after, self.runtime.config(after.guild.id)
        )

    async def on_message_delete(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        await self.runtime.events.message_deleted(
            message, self.runtime.config(message.guild.id)
        )

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        if after.guild is None:
            return
        config = self.runtime.config(after.guild.id)
        await self.runtime.events.message_edited(before, after, config)
        # An edit can smuggle in content that the original didn't have.
        await self._run_automod(after, config)

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        if member.bot:
            return
        config = self.runtime.config(member.guild.id)
        await self.runtime.events.voice_state(member, before, after, config)

        levels = self.runtime.levels
        if before.channel is None and after.channel is not None:
            levels.voice_joined(member.guild.id, member.id)
        elif after.channel is None and before.channel is not None:
            seconds = levels.voice_left(member.guild.id, member.id)
            if seconds:
                await levels.award_voice(member.guild, member, seconds, config)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id is None:
            return
        guild = self.get_guild(payload.guild_id)
        if guild is None:
            return
        await self.runtime.starboard.on_reaction(
            payload, guild, self.runtime.config(payload.guild_id)
        )

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id is None:
            return
        guild = self.get_guild(payload.guild_id)
        if guild is None:
            return
        await self.runtime.starboard.on_reaction(
            payload, guild, self.runtime.config(payload.guild_id)
        )

    # -- anti-nuke sources --------------------------------------------------
    #
    # Discord doesn't say *who* deleted a channel in the gateway event, so
    # the actor comes from the audit log. Failing to resolve one is not a
    # reason to act — an unattributed deletion is ignored rather than blamed
    # on the wrong person.

    async def _audit_actor(
        self, guild: discord.Guild, action: discord.AuditLogAction
    ) -> discord.Member | None:
        if not guild.me.guild_permissions.view_audit_log:
            return None
        try:
            cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=15)
            async for entry in guild.audit_logs(limit=5, action=action, after=cutoff):
                if entry.user and entry.user.id != self.user.id:
                    member = guild.get_member(entry.user.id)
                    if member is not None:
                        return member
        except (discord.Forbidden, discord.HTTPException):
            return None
        return None

    async def _antinuke(
        self, guild: discord.Guild, action: discord.AuditLogAction, kind: str
    ) -> None:
        config = self.runtime.config(guild.id)
        if not config.get("antinuke", {}).get("enabled", False):
            return
        actor = await self._audit_actor(guild, action)
        if actor is None:
            return
        breach = self.runtime.antinuke.record(guild.id, actor.id, kind, config)
        if breach is None:
            return

        count, _ = breach
        note = await self.runtime.antinuke.respond(guild, actor, kind, count, config)
        log.warning("Anti-nuke in guild=%s: %s", guild.id, note)

        channel_id = config["antinuke"].get("alert_channel_id")
        if channel_id:
            channel = guild.get_channel(int(channel_id))
            if isinstance(channel, discord.abc.Messageable):
                try:
                    await channel.send(note)
                except discord.HTTPException:
                    pass

    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel) -> None:
        await self._antinuke(
            channel.guild, discord.AuditLogAction.channel_delete, "channel_delete"
        )

    async def on_guild_role_delete(self, role: discord.Role) -> None:
        await self._antinuke(role.guild, discord.AuditLogAction.role_delete, "role_delete")

    async def on_member_ban(self, guild: discord.Guild, user: discord.abc.User) -> None:
        await self._antinuke(guild, discord.AuditLogAction.ban, "ban")

    # -- guild lifecycle (dashboard metrics) --------------------------------

    async def on_guild_join(self, guild: discord.Guild) -> None:
        self.runtime.store.record_guild_event(
            guild.id, guild.name, "added", guild.member_count
        )
        self.runtime.store.bump_metric(guild.id, "guild_added")
        log.info("Added to guild %s (%s members)", guild.name, guild.member_count)

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        self.runtime.store.record_guild_event(
            guild.id, guild.name, "removed", guild.member_count
        )
        log.info("Removed from guild %s", guild.name)

    def _record_error(self, source: str, guild_id: int | None = None) -> None:
        """Log an exception and persist it for the dashboard's error feed."""
        import traceback

        detail = traceback.format_exc()
        log.exception("%s failed", source)
        try:
            self.runtime.store.log_error(
                source, detail.strip().splitlines()[-1][:500], guild_id=guild_id, detail=detail
            )
        except Exception:
            pass

    async def on_error(self, event_method: str, *args, **kwargs) -> None:  # noqa: ANN002
        self._record_error(f"event:{event_method}")

    # -- background loops ---------------------------------------------------

    @tasks.loop(seconds=60)
    async def _lockdown_watch(self) -> None:
        for guild_id in self.runtime.raid.expired_lockdowns():
            guild = self.get_guild(guild_id)
            if guild is None:
                continue
            _, note = await self.runtime.raid.lockdown(guild, False, reason="lockdown expired")
            log.info("Auto-lifted lockdown in guild=%s: %s", guild_id, note)

    @_lockdown_watch.before_loop
    async def _before_lockdown_watch(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(minutes=5)
    async def _voice_xp_tick(self) -> None:
        """Bank voice XP during long calls instead of only on disconnect."""
        for guild in self.guilds:
            config = self.runtime.config(guild.id)
            if not config.get("levels", {}).get("enabled", False):
                continue
            try:
                for member, seconds in self.runtime.levels.voice_tick(guild, config):
                    await self.runtime.levels.award_voice(guild, member, seconds, config)
            except Exception:
                self._record_error("voice_xp", guild.id)

    @_voice_xp_tick.before_loop
    async def _before_voice_xp(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(seconds=30)
    async def _giveaway_watch(self) -> None:
        for row in self.runtime.store.due_giveaways():
            try:
                await self.runtime.giveaways.draw(self, row)
            except Exception:
                self._record_error("giveaway", int(row["guild_id"]))

    @_giveaway_watch.before_loop
    async def _before_giveaway_watch(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(minutes=1)
    async def _case_expiry(self) -> None:
        """Close out timed cases whose duration has elapsed."""
        for row in self.runtime.store.expired_cases():
            self.runtime.store.deactivate_case(int(row["guild_id"]), int(row["case_number"]))

    @_case_expiry.before_loop
    async def _before_case_expiry(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(hours=6)
    async def _retention_sweep(self) -> None:
        """Apply each guild's retention policy on a schedule."""
        for guild in self.guilds:
            cfg = self.runtime.config(guild.id).get("retention", {})
            try:
                self.runtime.store.purge_expired(cfg)
                self.runtime.store.scrub_transcripts(int(cfg.get("transcript_days", 0)))
            except Exception:
                self._record_error("retention", guild.id)

    @_retention_sweep.before_loop
    async def _before_retention_sweep(self) -> None:
        await self.wait_until_ready()
