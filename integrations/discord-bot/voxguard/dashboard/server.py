"""Dashboard HTTP server.

Runs inside the bot process on aiohttp (already a dependency), so it reads
live gateway state — guild list, member counts, latency — alongside the
SQLite history, with no IPC in between.

Everything except the login page and static assets requires a bearer token.
The page holds it in sessionStorage and sends it as an Authorization header;
there are no cookies, so there's nothing for a cross-site request to ride on.
"""

from __future__ import annotations

import datetime as dt
import hmac
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Metrics surfaced as headline tiles, in display order.
ACTION_METRICS = ("ban", "kick", "timeout", "warn")


def _day_range(days: int) -> list[str]:
    today = dt.datetime.now(dt.timezone.utc).date()
    return [(today - dt.timedelta(days=n)).isoformat() for n in range(days - 1, -1, -1)]


def _series(rows, days: int) -> list[dict]:
    """Fill gaps so the chart has one point per day, not per recorded day."""
    lookup = {r["day"]: int(r["value"]) for r in rows}
    return [{"day": day, "value": lookup.get(day, 0)} for day in _day_range(days)]


class DashboardServer:
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot
        self.settings = bot.settings
        self.store = bot.runtime.store
        self._runner: web.AppRunner | None = None

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        app = web.Application(middlewares=[self._auth_middleware])
        app.add_routes(
            [
                web.get("/", self.page),
                web.get("/api/overview", self.api_overview),
                web.get("/api/guilds", self.api_guilds),
                web.get("/api/guild/{guild_id}", self.api_guild),
                web.get("/api/errors", self.api_errors),
                web.post("/api/auth", self.api_auth),
                web.static("/static", STATIC_DIR),
            ]
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.settings.dashboard_host, self.settings.dashboard_port)
        await site.start()
        log.info("Dashboard listening on %s", self.settings.dashboard_public_url)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # -- auth ---------------------------------------------------------------

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler) -> web.StreamResponse:  # noqa: ANN001
        if not request.path.startswith("/api/") or request.path == "/api/auth":
            return await handler(request)

        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        # compare_digest keeps the check constant-time so a token can't be
        # recovered a character at a time by timing the responses.
        if not hmac.compare_digest(token, self.settings.dashboard_token):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    async def api_auth(self, request: web.Request) -> web.Response:
        body = await request.json()
        supplied = str(body.get("token", ""))
        if hmac.compare_digest(supplied, self.settings.dashboard_token):
            return web.json_response({"ok": True})
        return web.json_response({"error": "invalid token"}, status=401)

    # -- pages --------------------------------------------------------------

    async def page(self, request: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "index.html")

    # -- api ----------------------------------------------------------------

    async def api_overview(self, request: web.Request) -> web.Response:
        bot = self.bot
        days = int(request.query.get("days", 30))

        guilds = list(bot.guilds)
        total_members = sum(g.member_count or 0 for g in guilds)

        totals: dict[str, int] = {}
        for guild in guilds:
            for metric, value in self.store.metric_totals(guild.id).items():
                totals[metric] = totals.get(metric, 0) + value

        cases = self.store.case_counts()
        now = time.time()

        payload: dict[str, Any] = {
            "bot": {
                "name": str(bot.user) if bot.user else "VoxGuard",
                "avatar": bot.user.display_avatar.url if bot.user else None,
                # discord.py reports NaN latency before the first heartbeat,
                # and NaN is truthy — so this needs an explicit isnan check.
                "latency_ms": (
                    0 if (bot.latency is None or math.isnan(bot.latency))
                    else round(bot.latency * 1000)
                ),
                "uptime_seconds": int(now - bot.runtime.started_at),
                "ready": bot.is_ready(),
            },
            "totals": {
                "guilds": len(guilds),
                "members": total_members,
                "bans": cases.get("ban", 0),
                "kicks": cases.get("kick", 0),
                "timeouts": cases.get("timeout", 0),
                "warns": cases.get("warn", 0),
                "voice_flags": totals.get("voice_flags", 0),
                "automod_triggers": totals.get("automod_triggers", 0),
                "ai_moderation_hits": totals.get("ai_moderation_hits", 0),
                "levelups": totals.get("levelups", 0),
                "tickets_opened": totals.get("tickets_opened", 0),
                "errors_24h": self.store.error_count_since(now - 86400),
                "errors_total": self.store.error_count_since(0),
            },
            "charts": {
                "bans": _series(self.store.metric_series("ban", days), days),
                "kicks": _series(self.store.metric_series("kick", days), days),
                "timeouts": _series(self.store.metric_series("timeout", days), days),
                "warns": _series(self.store.metric_series("warn", days), days),
                "joins": _series(self.store.metric_series("member_joins", days), days),
                "automod": _series(self.store.metric_series("automod_triggers", days), days),
                "ai_moderation": _series(
                    self.store.metric_series("ai_moderation_hits", days), days
                ),
            },
            "growth": self._growth(days),
        }
        return web.json_response(payload)

    def _growth(self, days: int) -> list[dict]:
        """Cumulative server count — 'how many people added the bot'."""
        since = time.time() - days * 86400
        events = self.store.guild_events_since(since)

        # Start from today's real count and walk backwards through the
        # recorded adds/removes, so the line ends at the true present value
        # even if the bot was in servers before event tracking existed.
        per_day: dict[str, int] = {}
        for row in events:
            day = dt.datetime.fromtimestamp(row["created_at"], dt.timezone.utc).date().isoformat()
            per_day[day] = per_day.get(day, 0) + (1 if row["event"] == "added" else -1)

        days_list = _day_range(days)
        running = len(self.bot.guilds) - sum(per_day.values())
        out = []
        for day in days_list:
            running += per_day.get(day, 0)
            out.append({"day": day, "value": max(0, running), "added": max(0, per_day.get(day, 0))})
        return out

    async def api_guilds(self, request: web.Request) -> web.Response:
        rows = []
        for guild in sorted(
            self.bot.guilds, key=lambda g: g.member_count or 0, reverse=True
        ):
            cases = self.store.case_counts(guild.id)
            config = self.bot.runtime.config(guild.id)
            rows.append(
                {
                    "id": str(guild.id),
                    "name": guild.name,
                    "icon": guild.icon.url if guild.icon else None,
                    "members": guild.member_count or 0,
                    "owner_id": str(guild.owner_id),
                    "joined_at": guild.me.joined_at.isoformat() if guild.me and guild.me.joined_at else None,
                    "actions": sum(cases.values()),
                    "features_on": self._enabled_features(config),
                }
            )
        return web.json_response({"guilds": rows})

    @staticmethod
    def _enabled_features(config: dict) -> list[str]:
        checks = [
            ("Voice moderation", config.get("voice", {}).get("enabled")),
            ("Voice notes", config.get("voice_notes", {}).get("enabled")),
            ("Raid detection", config.get("raid", {}).get("enabled")),
            ("Text automod", config.get("automod", {}).get("enabled")),
            ("AI moderation", config.get("ai_moderation", {}).get("enabled")),
            ("Spoken commands", config.get("voice_commands", {}).get("enabled")),
            ("Anti-nuke", config.get("antinuke", {}).get("enabled")),
            ("Levelling", config.get("levels", {}).get("enabled")),
            ("Welcome", config.get("welcome", {}).get("enabled")),
            ("Logging", config.get("logging", {}).get("enabled")),
            ("Starboard", config.get("starboard", {}).get("enabled")),
            ("Tickets", config.get("tickets", {}).get("enabled")),
            ("AI roam", config.get("roam", {}).get("enabled")),
        ]
        return [name for name, on in checks if on]

    async def api_guild(self, request: web.Request) -> web.Response:
        guild_id = int(request.match_info["guild_id"])
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return web.json_response({"error": "not in that guild"}, status=404)

        days = int(request.query.get("days", 30))
        config = self.bot.runtime.config(guild_id)
        cases = self.store.case_counts(guild_id)
        totals = self.store.metric_totals(guild_id)

        # Resolve display names from the member cache so the UI shows people,
        # not raw snowflakes. Falls back to the ID for members who left.
        def name_of(user_id: str) -> str | None:
            member = guild.get_member(int(user_id)) if user_id.isdigit() else None
            return member.display_name if member else None

        recent = [
            {
                "case": r["case_number"],
                "user_id": r["user_id"],
                "user_name": name_of(r["user_id"]),
                "action": r["action"],
                "reason": r["reason"],
                "at": r["created_at"],
            }
            for r in self.store.recent_cases(guild_id, 15)
        ]

        top = [
            {
                "user_id": r["user_id"],
                "name": name_of(r["user_id"]),
                "xp": int(r["xp"]),
                "messages": int(r["messages"]),
                "voice_seconds": int(r["voice_seconds"]),
            }
            for r in self.store.leaderboard(guild_id, 10)
        ]

        return web.json_response(
            {
                "guild": {
                    "id": str(guild.id),
                    "name": guild.name,
                    "icon": guild.icon.url if guild.icon else None,
                    "members": guild.member_count or 0,
                    "channels": len(guild.channels),
                    "roles": len(guild.roles),
                    "boosts": guild.premium_subscription_count,
                    "created_at": guild.created_at.isoformat(),
                },
                "features": {
                    "enabled": self._enabled_features(config),
                    "all": self._feature_matrix(config),
                },
                "totals": {
                    "bans": cases.get("ban", 0),
                    "kicks": cases.get("kick", 0),
                    "timeouts": cases.get("timeout", 0),
                    "warns": cases.get("warn", 0),
                    "automod_triggers": totals.get("automod_triggers", 0),
                    "ai_moderation_hits": totals.get("ai_moderation_hits", 0),
                    "voice_flags": totals.get("voice_flags", 0),
                    "levelups": totals.get("levelups", 0),
                    "tickets_opened": totals.get("tickets_opened", 0),
                    "threads_created": totals.get("threads_created", 0),
                    "errors_24h": self.store.error_count_since(time.time() - 86400, guild_id),
                },
                "charts": {
                    "bans": _series(self.store.metric_series("ban", days, guild_id), days),
                    "kicks": _series(self.store.metric_series("kick", days, guild_id), days),
                    "timeouts": _series(self.store.metric_series("timeout", days, guild_id), days),
                    "warns": _series(self.store.metric_series("warn", days, guild_id), days),
                    "joins": _series(
                        self.store.metric_series("member_joins", days, guild_id), days
                    ),
                    "automod": _series(
                        self.store.metric_series("automod_triggers", days, guild_id), days
                    ),
                },
                "recent_cases": recent,
                "leaderboard": top,
                "errors": [
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "level": r["level"],
                        "message": r["message"],
                        "at": r["created_at"],
                    }
                    for r in self.store.recent_errors(10, guild_id)
                ],
            }
        )

    @staticmethod
    def _feature_matrix(config: dict) -> list[dict]:
        """Every toggleable feature and its state, for the per-server view."""
        spec = [
            ("Voice moderation", "voice", "enabled", "Real-time VC transcription + word filter"),
            ("Voice notes", "voice_notes", "enabled", "Moderates Discord voice messages"),
            ("Raid detection", "raid", "enabled", "Join-burst scoring and lockdown"),
            ("Text automod", "automod", "enabled", "Invites, links, spam, caps, mentions"),
            ("AI text moderation", "ai_moderation", "enabled", "Local model classifies message content"),
            ("Spoken commands", "voice_commands", "enabled", "Act on instructions spoken in voice chat"),
            ("Anti-nuke", "antinuke", "enabled", "Mass channel/role/ban protection"),
            ("Levelling", "levels", "enabled", "Text + voice XP and rank roles"),
            ("Welcome", "welcome", "enabled", "Join/leave messages and autorole"),
            ("Event logging", "logging", "enabled", "Message, member and voice audit trail"),
            ("Starboard", "starboard", "enabled", "Highlights popular messages"),
            ("Tickets", "tickets", "enabled", "Private support channels"),
            ("AI roam", "roam", "enabled", "Agent participates in text channels"),
        ]
        out = []
        for label, block, key, description in spec:
            section = config.get(block, {})
            out.append(
                {
                    "name": label,
                    "enabled": bool(section.get(key)),
                    "description": description,
                    "detail": DashboardServer._feature_detail(block, section),
                }
            )
        return out

    @staticmethod
    def _feature_detail(block: str, section: dict) -> str:
        if block in ("voice", "voice_notes"):
            return f"action: {section.get('action', '—')} · warn limit: {section.get('warn_limit', '—')}"
        if block == "raid":
            return f"action: {section.get('action', '—')} · score ≥ {section.get('score_threshold', '—')}"
        if block == "automod":
            active = [k for k, v in (section.get("rules") or {}).items() if v.get("enabled")]
            return f"rules: {', '.join(active) if active else 'none'}"
        if block == "antinuke":
            return f"response: {section.get('response', '—')}"
        if block == "levels":
            return (
                f"{section.get('xp_per_message', 0)} XP/msg · "
                f"{section.get('xp_per_voice_minute', 0)} XP/voice min"
            )
        if block == "roam":
            return f"tiers: {', '.join(section.get('tiers') or [])}"
        if block == "ai_moderation":
            actions = section.get("actions") or {}
            return (
                f"{len(section.get('categories') or [])} categories · "
                f"floor {float(section.get('min_confidence', 0)):.0%} · "
                f"severe: {actions.get('3', '-')}"
            )
        if block == "voice_commands":
            return f"actions: {'allowed' if section.get('allow_actions') else 'reply only'}"
        return ""

    async def api_errors(self, request: web.Request) -> web.Response:
        limit = min(int(request.query.get("limit", 100)), 500)
        rows = self.store.recent_errors(limit)
        guild_names = {str(g.id): g.name for g in self.bot.guilds}
        return web.json_response(
            {
                "errors": [
                    {
                        "id": r["id"],
                        "guild_id": r["guild_id"],
                        "guild_name": guild_names.get(r["guild_id"] or "", "—"),
                        "source": r["source"],
                        "level": r["level"],
                        "message": r["message"],
                        "detail": r["detail"],
                        "at": r["created_at"],
                    }
                    for r in rows
                ],
                "counts": {
                    "last_hour": self.store.error_count_since(time.time() - 3600),
                    "last_24h": self.store.error_count_since(time.time() - 86400),
                    "last_7d": self.store.error_count_since(time.time() - 7 * 86400),
                    "total": self.store.error_count_since(0),
                },
            }
        )
