"""Process-level settings and the per-guild config schema."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path


def _ids(raw: str | None) -> set[int]:
    out: set[int] = set()
    for chunk in (raw or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            out.add(int(chunk))
    return out


@dataclass(frozen=True)
class Settings:
    discord_token: str
    voicebox_url: str
    whisper_model: str
    tts_engine: str
    ollama_host: str
    ollama_model: str
    auto_install_ollama: bool
    data_dir: Path
    dashboard_enabled: bool = False
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8420
    dashboard_token: str = ""
    dashboard_public_url: str = ""
    owner_ids: set[int] = field(default_factory=set)
    dev_guild_id: int | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("DISCORD_TOKEN", "").strip()
        if not token:
            raise SystemExit("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")

        dev_guild = os.environ.get("VOXGUARD_DEV_GUILD_ID", "").strip()
        data_dir = Path(os.environ.get("VOXGUARD_DATA_DIR", "./data")).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        dashboard_on = os.environ.get("VOXGUARD_DASHBOARD", "0") == "1"
        dashboard_token = os.environ.get("VOXGUARD_DASHBOARD_TOKEN", "").strip()
        host = os.environ.get("VOXGUARD_DASHBOARD_HOST", "127.0.0.1").strip()
        port = int(os.environ.get("VOXGUARD_DASHBOARD_PORT", "8420") or 8420)

        # The dashboard exposes moderation history and error logs, so it does
        # not start without a token. Refusing here beats booting an
        # unauthenticated stats page onto a public interface.
        if dashboard_on and not dashboard_token:
            raise SystemExit(
                "VOXGUARD_DASHBOARD=1 requires VOXGUARD_DASHBOARD_TOKEN to be set.\n"
                "Generate one with:  python -c \"import secrets;print(secrets.token_urlsafe(32))\""
            )
        if dashboard_on and host not in ("127.0.0.1", "localhost") and len(dashboard_token) < 32:
            raise SystemExit(
                f"Refusing to bind the dashboard to {host} with a token shorter than 32 "
                "characters. Use a longer token, or bind to 127.0.0.1 and use an SSH tunnel."
            )

        return cls(
            discord_token=token,
            voicebox_url=os.environ.get("VOICEBOX_URL", "http://127.0.0.1:17493").rstrip("/"),
            whisper_model=os.environ.get("VOICEBOX_WHISPER_MODEL", "turbo"),
            tts_engine=os.environ.get("VOICEBOX_TTS_ENGINE", "qwen_custom_voice"),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            auto_install_ollama=os.environ.get("VOXGUARD_AUTO_INSTALL_OLLAMA", "0") == "1",
            data_dir=data_dir,
            dashboard_enabled=dashboard_on,
            dashboard_host=host,
            dashboard_port=port,
            dashboard_token=dashboard_token,
            dashboard_public_url=(
                os.environ.get("VOXGUARD_DASHBOARD_URL", "").strip()
                or f"http://{'localhost' if host == '127.0.0.1' else host}:{port}"
            ),
            owner_ids=_ids(os.environ.get("VOXGUARD_OWNER_IDS")),
            dev_guild_id=int(dev_guild) if dev_guild.isdigit() else None,
            log_level=os.environ.get("VOXGUARD_LOG_LEVEL", "INFO").upper(),
        )


# Actions a detection can trigger. Ordered by severity — `escalate_to` must
# always name something at least as severe as the base action.
ACTIONS = ("log", "warn", "timeout", "kick", "ban")

# Roam capability tiers. `chat` is the only one enabled by default; the other
# two have to be turned on deliberately per guild.
ROAM_TIERS = ("chat", "manage", "moderate")


DEFAULT_GUILD_CONFIG: dict = {
    # Live voice-channel moderation.
    "voice": {
        "enabled": True,
        "action": "warn",
        "warn_message": "Watch your language in voice chat. This is warning {count}/{limit}.",
        "warn_limit": 3,
        "escalate_to": "timeout",
        "timeout_minutes": 10,
        "log_channel_id": None,
        # Print every transcript line to the console, not just flagged ones.
        "console_transcript": True,
        # Post non-flagged transcript lines to the log channel too. Off by
        # default — a full VC transcript in a text channel surprises people.
        "log_all_transcripts": False,
        "min_confidence": 0.55,
        # None = let Whisper auto-detect.
        "language": None,
    },
    # Discord voice-message (voice note) moderation.
    "voice_notes": {
        "enabled": False,
        "action": "warn",
        "warn_message": "Your voice message contained blocked language. Warning {count}/{limit}.",
        "warn_limit": 3,
        "escalate_to": "timeout",
        "timeout_minutes": 10,
        "delete_message": True,
        "log_channel_id": None,
    },
    "raid": {
        "enabled": False,
        # Join-burst thresholds.
        "join_window_seconds": 60,
        "join_threshold": 8,
        # Accounts newer than this contribute to the risk score.
        "new_account_days": 7,
        # Score at which the response fires (see raid.py for the scoring).
        "score_threshold": 60,
        "action": "lockdown",  # alert | lockdown | kick | ban
        "lockdown_minutes": 15,
        "alert_channel_id": None,
        "mention_role_id": None,
    },
    "ai": {
        "personality": (
            "You are a helpful, slightly dry Discord assistant. Keep replies to a "
            "couple of sentences unless asked for detail."
        ),
        "model": None,  # falls back to OLLAMA_MODEL
        "voice_profile_id": None,
        "voice_profile_name": None,
        "emotion": True,
    },
    "roam": {
        "enabled": False,
        "tiers": ["chat"],
        # Empty = every channel the bot can see. Otherwise an allowlist.
        "channels": [],
        # How often roam may speak unprompted, per channel.
        "idle_reply_seconds": 300,
        "max_actions_per_hour": 10,
        # Irreversible / destructive actions post an approval prompt instead of
        # executing. Turning this off is a deliberate choice, not a default.
        "require_confirm_destructive": True,
        "audit_channel_id": None,
    },
    # Text-channel automod, complementing the voice filter.
    "automod": {
        "enabled": False,
        "log_channel_id": None,
        "delete_on_trigger": True,
        "rules": {
            "invites": {"enabled": False, "action": "delete"},
            "links": {"enabled": False, "action": "delete", "allowed_domains": []},
            "mass_mentions": {"enabled": False, "action": "timeout", "limit": 5},
            "spam": {"enabled": False, "action": "timeout", "messages": 5, "seconds": 5},
            "caps": {"enabled": False, "action": "delete", "percent": 70, "min_length": 10},
            "words": {"enabled": False, "action": "delete"},
        },
    },
    # LLM-based text moderation — catches what word lists structurally can't
    # (threats with no slur in them, scams, coordinated harassment).
    "ai_moderation": {
        "enabled": False,
        "log_channel_id": None,
        "model": None,  # falls back to the guild's ai.model, then OLLAMA_MODEL
        "categories": ["harassment", "hate", "threats", "sexual", "self_harm", "scam"],
        # Below this confidence the verdict is logged, never enforced.
        "min_confidence": 0.7,
        "max_checks_per_minute": 20,
        # Action per severity band returned by the classifier.
        "actions": {"1": "log", "2": "delete", "3": "timeout"},
        "timeout_minutes": 30,
    },
    # Spoken commands: talk in a voice channel, the bot acts on it.
    "voice_commands": {
        "enabled": False,
        # Empty = the bot's display name and the bound voice profile name.
        "wake_words": [],
        # When false the agent will answer but never invoke a tool by voice.
        "allow_actions": True,
        "transcript_channel_id": None,
    },
    # Protection against a compromised or rogue account with admin powers.
    "antinuke": {
        "enabled": False,
        "alert_channel_id": None,
        # Actions by a single moderator within the window that trip the alarm.
        "window_seconds": 30,
        "channel_delete_limit": 3,
        "role_delete_limit": 3,
        "ban_limit": 5,
        "kick_limit": 5,
        # strip_roles removes the actor's privileged roles; alert just reports.
        "response": "strip_roles",
        "whitelist": [],
    },
    "levels": {
        "enabled": False,
        "announce_channel_id": None,
        "announce": True,
        "xp_per_message": 15,
        "message_cooldown_seconds": 60,
        # Voice XP is the reason this exists on a voice bot: time spent
        # actually talking counts, not just time parked in a channel.
        "xp_per_voice_minute": 8,
        "voice_requires_unmuted": True,
        "no_xp_channels": [],
        "stack_rewards": False,
    },
    "welcome": {
        "enabled": False,
        "channel_id": None,
        "message": "Welcome {mention} to **{server}**! You're member #{count}.",
        "goodbye_enabled": False,
        "goodbye_channel_id": None,
        "goodbye_message": "**{user}** has left the server.",
        "autorole_ids": [],
        "dm_message": None,
    },
    "logging": {
        "enabled": False,
        "channel_id": None,
        "message_delete": True,
        "message_edit": True,
        "member_join": True,
        "member_leave": True,
        "member_update": False,
        "voice_state": False,
        "moderation": True,
    },
    "starboard": {
        "enabled": False,
        "channel_id": None,
        "emoji": "⭐",
        "threshold": 3,
        "self_star": False,
        "ignore_channels": [],
    },
    "tickets": {
        "enabled": False,
        "category_id": None,
        "support_role_id": None,
        "log_channel_id": None,
        "open_message": "Thanks for opening a ticket. A staff member will be with you shortly.",
    },
    "threads": {
        # Channels where every message spawns a thread automatically.
        "auto_thread_channels": [],
        "auto_archive_minutes": 1440,
    },
    # How long voice-derived and behavioural data is kept. 0 = forever.
    # Defaults are deliberately finite: this bot records people's speech.
    "retention": {
        "infraction_days": 180,
        "transcript_days": 30,
        "conversation_days": 30,
        "audit_days": 365,
        "error_days": 30,
    },
    "guardrails": {
        # Roles that are never actioned by automated enforcement.
        "immune_roles": [],
        "immune_users": [],
        # Members with these permissions are never actioned automatically.
        "immune_permissions": ["administrator", "manage_guild", "moderate_members"],
        # Global circuit breaker: if automated enforcement exceeds this in an
        # hour, enforcement drops to log-only and an alert is posted.
        "max_actions_per_hour": 20,
        # Log-only mode. Everything is detected and reported, nothing is applied.
        "dry_run": False,
    },
}


def merged_config(stored: dict | None) -> dict:
    """Deep-merge stored guild config over the defaults."""
    out = copy.deepcopy(DEFAULT_GUILD_CONFIG)
    if not stored:
        return out

    def merge(base: dict, over: dict) -> None:
        for key, value in over.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(out, stored)
    return out
