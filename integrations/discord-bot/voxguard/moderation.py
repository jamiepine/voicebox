"""Turning a detection into an action.

One entry point — `enforce` — so live voice, voice notes and any future scope
share the same escalation ladder, the same guardrails and the same audit
trail.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import discord

from . import guardrails
from .matching import Match
from .store import Store

log = logging.getLogger(__name__)

SEVERITY_COLOURS = {1: 0xF1C40F, 2: 0xE67E22, 3: 0xE74C3C}


@dataclass
class Outcome:
    action: str  # what actually happened
    detail: str
    applied: bool


class Enforcer:
    def __init__(self, store: Store, limiter: guardrails.RateLimiter) -> None:
        self.store = store
        self.limiter = limiter

    async def enforce(
        self,
        member: discord.Member,
        *,
        scope: str,
        config: dict,
        matches: list[Match],
        transcript: str,
        source: str,
        message: discord.Message | None = None,
    ) -> Outcome:
        """Apply the configured response to a confirmed detection.

        `scope` is "voice" or "voice_notes" and selects the config block;
        `source` is a human-readable origin for the log ("#general VC").
        """
        guild = member.guild
        scope_cfg = config.get(scope, {})
        top = matches[0]

        # Counted before the guardrail checks: the dashboard should show what
        # the filter caught, not only what it was allowed to act on.
        self.store.bump_metric(guild.id, "voice_flags")

        immune = guardrails.may_action(member, config)
        if not immune:
            outcome = Outcome("skipped", f"No action — {immune.reason}.", False)
            await self._log(guild, config, scope, member, matches, transcript, source, outcome)
            return outcome

        limited = self.limiter.check(guild.id, config, actor=f"{scope}-mod")
        if not limited:
            outcome = Outcome("skipped", limited.reason, False)
            await self._log(guild, config, scope, member, matches, transcript, source, outcome)
            await self._alert(guild, config, scope, limited.reason)
            return outcome

        action = scope_cfg.get("action", "warn")

        # A severity-3 term jumps the ladder straight to the escalation action.
        if top.severity >= 3 and action == "warn":
            action = scope_cfg.get("escalate_to", "timeout")

        if action == "warn":
            count = self.store.warning_count(guild.id, member.id, scope) + 1
            limit = int(scope_cfg.get("warn_limit", 3))
            # Escalate only once the limit has been *exceeded*, so a limit of 3
            # delivers three warnings and escalates on the fourth offense.
            if count > limit:
                action = scope_cfg.get("escalate_to", "timeout")
                detail = f"Warning limit reached ({count}/{limit}) — escalating to {action}."
            else:
                detail = f"Warned ({count}/{limit})."
                applied = await self._warn(member, scope_cfg, count, limit, top, config)
                self.store.record_infraction(
                    guild.id, member.id, scope, top.term, transcript, "warn"
                )
                self.store.audit(
                    guild.id, f"{scope}-mod", "warn", str(member.id), f"term={top.term}"
                )
                outcome = Outcome("warn", detail, applied)
                await self._log(
                    guild, config, scope, member, matches, transcript, source, outcome
                )
                return outcome
        else:
            detail = ""

        outcome = await self._apply(member, action, top, config, scope_cfg, transcript)
        if detail:
            outcome = Outcome(outcome.action, f"{detail} {outcome.detail}".strip(), outcome.applied)

        self.store.record_infraction(
            guild.id, member.id, scope, top.term, transcript, outcome.action
        )
        self.store.audit(
            guild.id, f"{scope}-mod", outcome.action, str(member.id), f"term={top.term}"
        )

        if message is not None and scope_cfg.get("delete_message", False):
            try:
                await message.delete()
            except discord.HTTPException:
                pass

        await self._log(guild, config, scope, member, matches, transcript, source, outcome)
        return outcome

    # -- individual actions -------------------------------------------------

    async def _warn(
        self,
        member: discord.Member,
        scope_cfg: dict,
        count: int,
        limit: int,
        match: Match,
        config: dict,
    ) -> bool:
        if guardrails.dry_run(config):
            return False
        template = scope_cfg.get("warn_message") or "Please watch your language."
        text = template.format(
            user=member.display_name,
            mention=member.mention,
            count=count,
            limit=limit,
            term=match.term,
            server=member.guild.name,
        )
        try:
            await member.send(text)
            return True
        except discord.Forbidden:
            # DMs closed — fall back to the log channel, which _log handles.
            return False

    async def _apply(
        self,
        member: discord.Member,
        action: str,
        match: Match,
        config: dict,
        scope_cfg: dict,
        transcript: str,
    ) -> Outcome:
        reason = f"VoxGuard: blocked term '{match.term}' ({match.method}, {match.confidence:.0%})"

        if action == "log":
            return Outcome("log", "Logged only — no action configured.", True)

        if guardrails.dry_run(config):
            return Outcome(action, f"Dry run — would have applied `{action}`.", False)

        feasible = guardrails.can_action(member.guild, member, action)
        if not feasible:
            return Outcome(action, f"Could not {action}: {feasible.reason}.", False)

        try:
            if action == "timeout":
                minutes = int(scope_cfg.get("timeout_minutes", 10))
                until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)
                await member.timeout(until, reason=reason)
                return Outcome("timeout", f"Timed out for {minutes} minutes.", True)
            if action == "kick":
                await member.kick(reason=reason)
                return Outcome("kick", "Kicked.", True)
            if action == "ban":
                await member.ban(reason=reason, delete_message_seconds=0)
                return Outcome("ban", "Banned.", True)
        except discord.Forbidden:
            return Outcome(action, f"Discord refused the {action} (missing permission).", False)
        except discord.HTTPException as exc:
            return Outcome(action, f"Discord rejected the {action}: {exc}.", False)

        return Outcome("log", f"Unknown action '{action}' — logged instead.", False)

    # -- reporting ----------------------------------------------------------

    async def _log(
        self,
        guild: discord.Guild,
        config: dict,
        scope: str,
        member: discord.Member,
        matches: list[Match],
        transcript: str,
        source: str,
        outcome: Outcome,
    ) -> None:
        channel_id = config.get(scope, {}).get("log_channel_id")
        if not channel_id:
            return
        channel = guild.get_channel(int(channel_id))
        if not isinstance(channel, discord.abc.Messageable):
            return

        top = matches[0]
        embed = discord.Embed(
            title=f"Blocked language detected — {outcome.action}",
            colour=SEVERITY_COLOURS.get(top.severity, 0xF1C40F),
            timestamp=dt.datetime.now(dt.timezone.utc),
        )
        embed.add_field(name="Member", value=f"{member.mention} (`{member.id}`)", inline=True)
        embed.add_field(name="Source", value=source, inline=True)
        embed.add_field(
            name="Match",
            value=f"`{top.term}` via {top.method} ({top.confidence:.0%})",
            inline=True,
        )
        if len(matches) > 1:
            embed.add_field(
                name="Also matched",
                value=", ".join(f"`{m.term}`" for m in matches[1:6]),
                inline=False,
            )
        embed.add_field(name="Transcript", value=f">>> {transcript[:900]}", inline=False)
        embed.add_field(name="Result", value=outcome.detail, inline=False)
        if not outcome.applied and outcome.action == "warn":
            embed.set_footer(text="DM failed — the member has DMs closed.")

        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            log.warning("Could not post moderation log to channel %s", channel_id)

    async def _alert(self, guild: discord.Guild, config: dict, scope: str, text: str) -> None:
        channel_id = config.get(scope, {}).get("log_channel_id")
        if not channel_id:
            return
        channel = guild.get_channel(int(channel_id))
        if isinstance(channel, discord.abc.Messageable):
            try:
                await channel.send(f"⚠️ **VoxGuard**: {text}")
            except discord.HTTPException:
                pass
