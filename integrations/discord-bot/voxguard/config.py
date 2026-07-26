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

        return cls(
            discord_token=token,
            voicebox_url=os.environ.get("VOICEBOX_URL", "http://127.0.0.1:17493").rstrip("/"),
            whisper_model=os.environ.get("VOICEBOX_WHISPER_MODEL", "turbo"),
            tts_engine=os.environ.get("VOICEBOX_TTS_ENGINE", "qwen_custom_voice"),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            auto_install_ollama=os.environ.get("VOXGUARD_AUTO_INSTALL_OLLAMA", "0") == "1",
            data_dir=data_dir,
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
