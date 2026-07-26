"""The VoxGuard Discord client — event wiring only; commands live in cogs."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands, tasks

from .agent import AgentContext
from .config import Settings
from .roam import RoamController
from .runtime import Runtime
from .vctalk import VCTalkController
from .voice_notes import is_voice_message

log = logging.getLogger(__name__)

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

    async def setup_hook(self) -> None:
        for ext in (
            "voxguard.cogs.voice_mod",
            "voxguard.cogs.raid_cmds",
            "voxguard.cogs.ai_cmds",
        ):
            await self.load_extension(ext)

        if self.settings.dev_guild_id:
            guild = discord.Object(id=self.settings.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Synced commands to dev guild %s", self.settings.dev_guild_id)
        else:
            await self.tree.sync()
            log.info("Synced global commands")

        self._lockdown_watch.start()

    async def close(self) -> None:
        self._lockdown_watch.cancel()
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

        if self.user and self.roam.should_reply(message, config, self.user.id):
            try:
                await self.roam.handle_message(message, config, self.allowed_tiers(config))
            except Exception:
                log.exception("Roam reply failed in guild=%s", message.guild.id)

    async def on_member_join(self, member: discord.Member) -> None:
        config = self.runtime.config(member.guild.id)
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

    @tasks.loop(seconds=60)
    async def _lockdown_watch(self) -> None:
        for guild_id in self.runtime.raid.expired_lockdowns():
            guild = self.get_guild(guild_id)
            if guild is None:
                continue
            ok, note = await self.runtime.raid.lockdown(guild, False, reason="lockdown expired")
            log.info("Auto-lifted lockdown in guild=%s: %s", guild_id, note)

    @_lockdown_watch.before_loop
    async def _before_lockdown_watch(self) -> None:
        await self.wait_until_ready()
