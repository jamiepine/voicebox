"""SQLite persistence.

Everything the bot needs to survive a restart lives here: per-guild config,
word lists, infraction history, agent memory, voice-clone consent records and
the action audit trail.

Calls are synchronous and guarded by a lock. Every statement here is a
single-row read or write against an indexed table, so the cost is well under
the latency of the Discord API call that follows it; the async wrappers exist
only for the few places that batch.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS guild_config (
    guild_id TEXT PRIMARY KEY,
    data     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS terms (
    guild_id  TEXT NOT NULL,
    scope     TEXT NOT NULL,          -- voice | notes | text
    listing   TEXT NOT NULL,          -- block | allow
    term      TEXT NOT NULL,
    kind      TEXT NOT NULL,          -- word | phrase | regex
    severity  INTEGER NOT NULL DEFAULT 1,
    added_by  TEXT,
    added_at  REAL NOT NULL,
    PRIMARY KEY (guild_id, scope, listing, term)
);

CREATE TABLE IF NOT EXISTS infractions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    scope      TEXT NOT NULL,
    term       TEXT NOT NULL,
    transcript TEXT,
    action     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_infractions_lookup
    ON infractions (guild_id, user_id, scope, created_at);

CREATE TABLE IF NOT EXISTS memory (
    guild_id   TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (guild_id, key)
);

CREATE TABLE IF NOT EXISTS conversation (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    author     TEXT,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_lookup
    ON conversation (guild_id, channel_id, created_at);

CREATE TABLE IF NOT EXISTS voice_consent (
    guild_id     TEXT NOT NULL,
    profile_id   TEXT NOT NULL,
    profile_name TEXT NOT NULL,
    uploader_id  TEXT NOT NULL,
    attestation  TEXT NOT NULL,
    source_file  TEXT,
    created_at   REAL NOT NULL,
    PRIMARY KEY (guild_id, profile_id)
);

CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT NOT NULL,
    actor      TEXT NOT NULL,          -- 'voice-mod' | 'roam' | user id
    action     TEXT NOT NULL,
    target     TEXT,
    detail     TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_lookup ON audit (guild_id, created_at);
"""

# Tables backing the feature set beyond voice moderation. Split out only for
# readability — they're created in the same transaction as SCHEMA.
FEATURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS levels (
    guild_id      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    xp            INTEGER NOT NULL DEFAULT 0,
    messages      INTEGER NOT NULL DEFAULT 0,
    voice_seconds INTEGER NOT NULL DEFAULT 0,
    last_award_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_levels_rank ON levels (guild_id, xp DESC);

CREATE TABLE IF NOT EXISTS level_rewards (
    guild_id TEXT NOT NULL,
    level    INTEGER NOT NULL,
    role_id  TEXT NOT NULL,
    PRIMARY KEY (guild_id, level)
);

CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id     TEXT NOT NULL,
    case_number  INTEGER NOT NULL,
    user_id      TEXT NOT NULL,
    moderator_id TEXT NOT NULL,
    action       TEXT NOT NULL,
    reason       TEXT,
    expires_at   REAL,
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_number ON cases (guild_id, case_number);
CREATE INDEX IF NOT EXISTS idx_cases_user ON cases (guild_id, user_id, created_at);

CREATE TABLE IF NOT EXISTS tags (
    guild_id   TEXT NOT NULL,
    name       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_by TEXT,
    uses       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (guild_id, name)
);

CREATE TABLE IF NOT EXISTS tickets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    subject    TEXT,
    status     TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL,
    closed_at  REAL,
    closed_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_guild ON tickets (guild_id, status);

CREATE TABLE IF NOT EXISTS starboard (
    guild_id      TEXT NOT NULL,
    message_id    TEXT NOT NULL,
    star_msg_id   TEXT,
    stars         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, message_id)
);

CREATE TABLE IF NOT EXISTS giveaways (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    message_id TEXT,
    prize      TEXT NOT NULL,
    winners    INTEGER NOT NULL DEFAULT 1,
    host_id    TEXT NOT NULL,
    ends_at    REAL NOT NULL,
    ended      INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_giveaways_open ON giveaways (ended, ends_at);

CREATE TABLE IF NOT EXISTS giveaway_entries (
    giveaway_id INTEGER NOT NULL,
    user_id     TEXT NOT NULL,
    PRIMARY KEY (giveaway_id, user_id)
);

CREATE TABLE IF NOT EXISTS reaction_roles (
    guild_id   TEXT NOT NULL,
    message_id TEXT NOT NULL,
    key        TEXT NOT NULL,          -- emoji name/id, or button custom_id
    role_id    TEXT NOT NULL,
    label      TEXT,
    PRIMARY KEY (guild_id, message_id, key)
);

-- Dashboard telemetry -------------------------------------------------

CREATE TABLE IF NOT EXISTS errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   TEXT,
    source     TEXT NOT NULL,
    level      TEXT NOT NULL DEFAULT 'error',
    message    TEXT NOT NULL,
    detail     TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_time ON errors (created_at);

CREATE TABLE IF NOT EXISTS guild_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id     TEXT NOT NULL,
    guild_name   TEXT,
    event        TEXT NOT NULL,          -- added | removed
    member_count INTEGER,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_guild_events_time ON guild_events (created_at);

CREATE TABLE IF NOT EXISTS metrics_daily (
    guild_id TEXT NOT NULL,
    day      TEXT NOT NULL,             -- YYYY-MM-DD (UTC)
    metric   TEXT NOT NULL,
    value    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, day, metric)
);
CREATE INDEX IF NOT EXISTS idx_metrics_day ON metrics_daily (day);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.executescript(FEATURE_SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _write(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, tuple(params))
            self._db.commit()
            return cur

    def _read(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    # -- guild config -------------------------------------------------------

    def get_config(self, guild_id: int) -> dict:
        rows = self._read("SELECT data FROM guild_config WHERE guild_id = ?", (str(guild_id),))
        if not rows:
            return {}
        try:
            return json.loads(rows[0]["data"])
        except json.JSONDecodeError:
            return {}

    def save_config(self, guild_id: int, data: dict) -> None:
        self._write(
            "INSERT INTO guild_config (guild_id, data) VALUES (?, ?) "
            "ON CONFLICT(guild_id) DO UPDATE SET data = excluded.data",
            (str(guild_id), json.dumps(data)),
        )

    # -- word lists ---------------------------------------------------------

    def add_terms(
        self,
        guild_id: int,
        scope: str,
        terms: Iterable[tuple[str, str, int]],
        *,
        listing: str = "block",
        added_by: int | None = None,
    ) -> int:
        """Insert (term, kind, severity) triples. Returns the number added."""
        now = time.time()
        added = 0
        with self._lock:
            for term, kind, severity in terms:
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO terms "
                    "(guild_id, scope, listing, term, kind, severity, added_by, added_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(guild_id),
                        scope,
                        listing,
                        term,
                        kind,
                        severity,
                        str(added_by) if added_by else None,
                        now,
                    ),
                )
                added += cur.rowcount or 0
            self._db.commit()
        return added

    def remove_term(self, guild_id: int, scope: str, term: str, *, listing: str = "block") -> bool:
        cur = self._write(
            "DELETE FROM terms WHERE guild_id = ? AND scope = ? AND listing = ? AND term = ?",
            (str(guild_id), scope, listing, term),
        )
        return bool(cur.rowcount)

    def clear_terms(self, guild_id: int, scope: str, *, listing: str = "block") -> int:
        cur = self._write(
            "DELETE FROM terms WHERE guild_id = ? AND scope = ? AND listing = ?",
            (str(guild_id), scope, listing),
        )
        return cur.rowcount or 0

    def list_terms(self, guild_id: int, scope: str, *, listing: str = "block") -> list[sqlite3.Row]:
        return self._read(
            "SELECT term, kind, severity FROM terms "
            "WHERE guild_id = ? AND scope = ? AND listing = ? ORDER BY term",
            (str(guild_id), scope, listing),
        )

    # -- infractions --------------------------------------------------------

    def record_infraction(
        self,
        guild_id: int,
        user_id: int,
        scope: str,
        term: str,
        transcript: str,
        action: str,
    ) -> None:
        self._write(
            "INSERT INTO infractions (guild_id, user_id, scope, term, transcript, action, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(user_id), scope, term, transcript[:2000], action, time.time()),
        )

    def warning_count(self, guild_id: int, user_id: int, scope: str) -> int:
        rows = self._read(
            "SELECT COUNT(*) AS n FROM infractions "
            "WHERE guild_id = ? AND user_id = ? AND scope = ? AND action = 'warn'",
            (str(guild_id), str(user_id), scope),
        )
        return int(rows[0]["n"]) if rows else 0

    def clear_warnings(self, guild_id: int, user_id: int, scope: str | None = None) -> int:
        if scope:
            cur = self._write(
                "DELETE FROM infractions WHERE guild_id = ? AND user_id = ? AND scope = ?",
                (str(guild_id), str(user_id), scope),
            )
        else:
            cur = self._write(
                "DELETE FROM infractions WHERE guild_id = ? AND user_id = ?",
                (str(guild_id), str(user_id)),
            )
        return cur.rowcount or 0

    def recent_infractions(self, guild_id: int, user_id: int, limit: int = 10) -> list[sqlite3.Row]:
        return self._read(
            "SELECT scope, term, action, transcript, created_at FROM infractions "
            "WHERE guild_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT ?",
            (str(guild_id), str(user_id), limit),
        )

    # -- agent memory -------------------------------------------------------

    def remember(self, guild_id: int, key: str, value: str) -> None:
        self._write(
            "INSERT INTO memory (guild_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(guild_id, key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (str(guild_id), key[:200], value[:4000], time.time()),
        )

    def forget(self, guild_id: int, key: str) -> bool:
        cur = self._write("DELETE FROM memory WHERE guild_id = ? AND key = ?", (str(guild_id), key))
        return bool(cur.rowcount)

    def all_memory(self, guild_id: int, limit: int = 60) -> list[sqlite3.Row]:
        return self._read(
            "SELECT key, value FROM memory WHERE guild_id = ? ORDER BY updated_at DESC LIMIT ?",
            (str(guild_id), limit),
        )

    def add_turn(
        self, guild_id: int, channel_id: int, role: str, content: str, author: str | None = None
    ) -> None:
        self._write(
            "INSERT INTO conversation (guild_id, channel_id, role, author, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(channel_id), role, author, content[:4000], time.time()),
        )

    def recent_turns(self, guild_id: int, channel_id: int, limit: int = 20) -> list[sqlite3.Row]:
        rows = self._read(
            "SELECT role, author, content FROM conversation "
            "WHERE guild_id = ? AND channel_id = ? ORDER BY id DESC LIMIT ?",
            (str(guild_id), str(channel_id), limit),
        )
        return list(reversed(rows))

    def trim_conversation(self, guild_id: int, channel_id: int, keep: int = 200) -> None:
        self._write(
            "DELETE FROM conversation WHERE guild_id = ? AND channel_id = ? AND id NOT IN "
            "(SELECT id FROM conversation WHERE guild_id = ? AND channel_id = ? "
            " ORDER BY id DESC LIMIT ?)",
            (str(guild_id), str(channel_id), str(guild_id), str(channel_id), keep),
        )

    # -- voice clone consent ------------------------------------------------

    def record_consent(
        self,
        guild_id: int,
        profile_id: str,
        profile_name: str,
        uploader_id: int,
        attestation: str,
        source_file: str | None,
    ) -> None:
        self._write(
            "INSERT INTO voice_consent "
            "(guild_id, profile_id, profile_name, uploader_id, attestation, source_file, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            # Re-attesting replaces the whole record, not just the phrase —
            # a consent trail that keeps the first uploader's ID next to a
            # later person's attestation would misattribute responsibility.
            "ON CONFLICT(guild_id, profile_id) DO UPDATE SET "
            "profile_name = excluded.profile_name, attestation = excluded.attestation, "
            "uploader_id = excluded.uploader_id, source_file = excluded.source_file, "
            "created_at = excluded.created_at",
            (
                str(guild_id),
                profile_id,
                profile_name,
                str(uploader_id),
                attestation,
                source_file,
                time.time(),
            ),
        )

    def list_consent(self, guild_id: int) -> list[sqlite3.Row]:
        return self._read(
            "SELECT profile_id, profile_name, uploader_id, attestation, created_at "
            "FROM voice_consent WHERE guild_id = ? ORDER BY created_at DESC",
            (str(guild_id),),
        )

    def has_consent(self, guild_id: int, profile_id: str) -> bool:
        return bool(
            self._read(
                "SELECT 1 FROM voice_consent WHERE guild_id = ? AND profile_id = ?",
                (str(guild_id), profile_id),
            )
        )

    # -- audit --------------------------------------------------------------

    def audit(
        self, guild_id: int, actor: str, action: str, target: str | None, detail: str | None
    ) -> None:
        self._write(
            "INSERT INTO audit (guild_id, actor, action, target, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id), actor, action, target, (detail or "")[:2000], time.time()),
        )

    def audit_count_since(self, guild_id: int, since: float, actor: str | None = None) -> int:
        if actor:
            rows = self._read(
                "SELECT COUNT(*) AS n FROM audit WHERE guild_id = ? AND created_at >= ? AND actor = ?",
                (str(guild_id), since, actor),
            )
        else:
            rows = self._read(
                "SELECT COUNT(*) AS n FROM audit WHERE guild_id = ? AND created_at >= ?",
                (str(guild_id), since),
            )
        return int(rows[0]["n"]) if rows else 0

    def recent_audit(self, guild_id: int, limit: int = 20) -> list[sqlite3.Row]:
        return self._read(
            "SELECT actor, action, target, detail, created_at FROM audit "
            "WHERE guild_id = ? ORDER BY id DESC LIMIT ?",
            (str(guild_id), limit),
        )

    # -- retention & erasure ------------------------------------------------
    #
    # This database accumulates a lot of other people's speech: voice
    # transcripts attached to infractions, conversation turns, and consent
    # records tied to a real person's voice. Keeping that forever by default
    # is the wrong posture for a bot that records voice channels, so rows
    # expire on a schedule and an owner can erase a member outright.

    def purge_expired(self, retention: dict) -> dict[str, int]:
        """Delete rows past their retention window. Returns per-table counts.

        A retention of 0 (or missing) means "keep indefinitely" for that
        table, so an operator has to opt into unbounded storage rather than
        getting it silently.
        """
        now = time.time()
        removed: dict[str, int] = {}
        plan = (
            ("infractions", "infraction_days"),
            ("conversation", "conversation_days"),
            ("audit", "audit_days"),
            ("errors", "error_days"),
        )
        for table, key in plan:
            days = int(retention.get(key, 0) or 0)
            if days <= 0:
                continue
            cutoff = now - days * 86400
            cur = self._write(f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
            if cur.rowcount:
                removed[table] = cur.rowcount
        return removed

    def scrub_transcripts(self, older_than_days: int) -> int:
        """Blank stored speech while keeping the infraction counts intact.

        Lets a guild honour a short transcript-retention policy without
        losing the warning history that escalation depends on.
        """
        if older_than_days <= 0:
            return 0
        cutoff = time.time() - older_than_days * 86400
        cur = self._write(
            "UPDATE infractions SET transcript = '[expired]' "
            "WHERE created_at < ? AND transcript IS NOT NULL AND transcript != '[expired]'",
            (cutoff,),
        )
        return cur.rowcount or 0

    def erase_user(self, guild_id: int, user_id: int) -> dict[str, int]:
        """Remove everything this bot stores about one member of one guild."""
        gid, uid = str(guild_id), str(user_id)
        removed: dict[str, int] = {}
        statements = (
            ("infractions", "DELETE FROM infractions WHERE guild_id = ? AND user_id = ?", (gid, uid)),
            ("levels", "DELETE FROM levels WHERE guild_id = ? AND user_id = ?", (gid, uid)),
            ("cases", "DELETE FROM cases WHERE guild_id = ? AND user_id = ?", (gid, uid)),
            ("conversation", "DELETE FROM conversation WHERE guild_id = ? AND author = ?", (gid, uid)),
            ("audit", "DELETE FROM audit WHERE guild_id = ? AND target = ?", (gid, uid)),
            ("voice_consent", "DELETE FROM voice_consent WHERE guild_id = ? AND uploader_id = ?", (gid, uid)),
        )
        for name, sql, params in statements:
            cur = self._write(sql, params)
            if cur.rowcount:
                removed[name] = cur.rowcount
        return removed

    def erase_guild(self, guild_id: int) -> dict[str, int]:
        """Remove all data for a guild — used when the bot is removed from it."""
        gid = str(guild_id)
        tables = (
            "guild_config", "terms", "infractions", "memory", "conversation",
            "voice_consent", "audit", "levels", "level_rewards", "cases",
            "tags", "tickets", "starboard", "giveaways", "reaction_roles",
            "metrics_daily",
        )
        removed: dict[str, int] = {}
        for table in tables:
            cur = self._write(f"DELETE FROM {table} WHERE guild_id = ?", (gid,))
            if cur.rowcount:
                removed[table] = cur.rowcount
        return removed

    # -- telemetry ----------------------------------------------------------

    def log_error(
        self,
        source: str,
        message: str,
        *,
        guild_id: int | None = None,
        level: str = "error",
        detail: str | None = None,
    ) -> None:
        self._write(
            "INSERT INTO errors (guild_id, source, level, message, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(guild_id) if guild_id else None,
                source,
                level,
                message[:500],
                (detail or "")[:4000] or None,
                time.time(),
            ),
        )

    def recent_errors(self, limit: int = 50, guild_id: int | None = None) -> list[sqlite3.Row]:
        if guild_id:
            return self._read(
                "SELECT id, guild_id, source, level, message, detail, created_at FROM errors "
                "WHERE guild_id = ? ORDER BY id DESC LIMIT ?",
                (str(guild_id), limit),
            )
        return self._read(
            "SELECT id, guild_id, source, level, message, detail, created_at FROM errors "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )

    def error_count_since(self, since: float, guild_id: int | None = None) -> int:
        if guild_id:
            rows = self._read(
                "SELECT COUNT(*) AS n FROM errors WHERE created_at >= ? AND guild_id = ?",
                (since, str(guild_id)),
            )
        else:
            rows = self._read("SELECT COUNT(*) AS n FROM errors WHERE created_at >= ?", (since,))
        return int(rows[0]["n"]) if rows else 0

    def record_guild_event(
        self, guild_id: int, guild_name: str, event: str, member_count: int | None
    ) -> None:
        self._write(
            "INSERT INTO guild_events (guild_id, guild_name, event, member_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(guild_id), guild_name[:120], event, member_count, time.time()),
        )

    def guild_events_since(self, since: float) -> list[sqlite3.Row]:
        return self._read(
            "SELECT guild_id, guild_name, event, member_count, created_at FROM guild_events "
            "WHERE created_at >= ? ORDER BY created_at",
            (since,),
        )

    def bump_metric(self, guild_id: int, metric: str, delta: int = 1) -> None:
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        self._write(
            "INSERT INTO metrics_daily (guild_id, day, metric, value) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(guild_id, day, metric) DO UPDATE SET value = value + excluded.value",
            (str(guild_id), day, metric, delta),
        )

    def metric_series(
        self, metric: str, days: int = 30, guild_id: int | None = None
    ) -> list[sqlite3.Row]:
        start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
        if guild_id:
            return self._read(
                "SELECT day, SUM(value) AS value FROM metrics_daily "
                "WHERE metric = ? AND day >= ? AND guild_id = ? GROUP BY day ORDER BY day",
                (metric, start, str(guild_id)),
            )
        return self._read(
            "SELECT day, SUM(value) AS value FROM metrics_daily "
            "WHERE metric = ? AND day >= ? GROUP BY day ORDER BY day",
            (metric, start),
        )

    def metric_totals(self, guild_id: int | None = None) -> dict[str, int]:
        if guild_id:
            rows = self._read(
                "SELECT metric, SUM(value) AS value FROM metrics_daily WHERE guild_id = ? "
                "GROUP BY metric",
                (str(guild_id),),
            )
        else:
            rows = self._read("SELECT metric, SUM(value) AS value FROM metrics_daily GROUP BY metric")
        return {r["metric"]: int(r["value"]) for r in rows}

    # -- levels -------------------------------------------------------------

    def add_xp(
        self, guild_id: int, user_id: int, xp: int, *, messages: int = 0, voice_seconds: int = 0
    ) -> int:
        """Add XP and return the member's new total."""
        self._write(
            "INSERT INTO levels (guild_id, user_id, xp, messages, voice_seconds, last_award_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(guild_id, user_id) DO UPDATE SET "
            "xp = xp + excluded.xp, messages = messages + excluded.messages, "
            "voice_seconds = voice_seconds + excluded.voice_seconds, "
            "last_award_at = excluded.last_award_at",
            (str(guild_id), str(user_id), xp, messages, voice_seconds, time.time()),
        )
        rows = self._read(
            "SELECT xp FROM levels WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        return int(rows[0]["xp"]) if rows else xp

    def get_level_row(self, guild_id: int, user_id: int) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT xp, messages, voice_seconds, last_award_at FROM levels "
            "WHERE guild_id = ? AND user_id = ?",
            (str(guild_id), str(user_id)),
        )
        return rows[0] if rows else None

    def leaderboard(self, guild_id: int, limit: int = 10, offset: int = 0) -> list[sqlite3.Row]:
        return self._read(
            "SELECT user_id, xp, messages, voice_seconds FROM levels WHERE guild_id = ? "
            "ORDER BY xp DESC LIMIT ? OFFSET ?",
            (str(guild_id), limit, offset),
        )

    def rank_of(self, guild_id: int, user_id: int) -> int:
        rows = self._read(
            "SELECT COUNT(*) + 1 AS rank FROM levels WHERE guild_id = ? AND xp > "
            "(SELECT xp FROM levels WHERE guild_id = ? AND user_id = ?)",
            (str(guild_id), str(guild_id), str(user_id)),
        )
        return int(rows[0]["rank"]) if rows else 0

    def set_level_reward(self, guild_id: int, level: int, role_id: int) -> None:
        self._write(
            "INSERT INTO level_rewards (guild_id, level, role_id) VALUES (?, ?, ?) "
            "ON CONFLICT(guild_id, level) DO UPDATE SET role_id = excluded.role_id",
            (str(guild_id), level, str(role_id)),
        )

    def remove_level_reward(self, guild_id: int, level: int) -> bool:
        cur = self._write(
            "DELETE FROM level_rewards WHERE guild_id = ? AND level = ?", (str(guild_id), level)
        )
        return bool(cur.rowcount)

    def level_rewards(self, guild_id: int) -> list[sqlite3.Row]:
        return self._read(
            "SELECT level, role_id FROM level_rewards WHERE guild_id = ? ORDER BY level",
            (str(guild_id),),
        )

    def reset_levels(self, guild_id: int) -> int:
        cur = self._write("DELETE FROM levels WHERE guild_id = ?", (str(guild_id),))
        return cur.rowcount or 0

    # -- moderation cases ---------------------------------------------------

    def next_case_number(self, guild_id: int) -> int:
        rows = self._read(
            "SELECT COALESCE(MAX(case_number), 0) + 1 AS n FROM cases WHERE guild_id = ?",
            (str(guild_id),),
        )
        return int(rows[0]["n"]) if rows else 1

    def add_case(
        self,
        guild_id: int,
        user_id: int,
        moderator_id: int,
        action: str,
        reason: str | None,
        expires_at: float | None = None,
    ) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(MAX(case_number), 0) + 1 AS n FROM cases WHERE guild_id = ?",
                (str(guild_id),),
            ).fetchone()
            number = int(row["n"])
            self._db.execute(
                "INSERT INTO cases "
                "(guild_id, case_number, user_id, moderator_id, action, reason, expires_at, "
                " active, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    str(guild_id), number, str(user_id), str(moderator_id), action,
                    (reason or "")[:1000] or None, expires_at, time.time(),
                ),
            )
            self._db.commit()
        return number

    def get_case(self, guild_id: int, number: int) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT case_number, user_id, moderator_id, action, reason, expires_at, active, "
            "created_at FROM cases WHERE guild_id = ? AND case_number = ?",
            (str(guild_id), number),
        )
        return rows[0] if rows else None

    def user_cases(self, guild_id: int, user_id: int, limit: int = 25) -> list[sqlite3.Row]:
        return self._read(
            "SELECT case_number, moderator_id, action, reason, active, created_at FROM cases "
            "WHERE guild_id = ? AND user_id = ? ORDER BY case_number DESC LIMIT ?",
            (str(guild_id), str(user_id), limit),
        )

    def recent_cases(self, guild_id: int, limit: int = 25) -> list[sqlite3.Row]:
        return self._read(
            "SELECT case_number, user_id, moderator_id, action, reason, active, created_at "
            "FROM cases WHERE guild_id = ? ORDER BY case_number DESC LIMIT ?",
            (str(guild_id), limit),
        )

    def set_case_reason(self, guild_id: int, number: int, reason: str) -> bool:
        cur = self._write(
            "UPDATE cases SET reason = ? WHERE guild_id = ? AND case_number = ?",
            (reason[:1000], str(guild_id), number),
        )
        return bool(cur.rowcount)

    def deactivate_case(self, guild_id: int, number: int) -> bool:
        cur = self._write(
            "UPDATE cases SET active = 0 WHERE guild_id = ? AND case_number = ?",
            (str(guild_id), number),
        )
        return bool(cur.rowcount)

    def expired_cases(self, now: float | None = None) -> list[sqlite3.Row]:
        return self._read(
            "SELECT guild_id, case_number, user_id, action FROM cases "
            "WHERE active = 1 AND expires_at IS NOT NULL AND expires_at <= ?",
            (now or time.time(),),
        )

    def case_counts(self, guild_id: int | None = None) -> dict[str, int]:
        if guild_id:
            rows = self._read(
                "SELECT action, COUNT(*) AS n FROM cases WHERE guild_id = ? GROUP BY action",
                (str(guild_id),),
            )
        else:
            rows = self._read("SELECT action, COUNT(*) AS n FROM cases GROUP BY action")
        return {r["action"]: int(r["n"]) for r in rows}

    # -- tags ---------------------------------------------------------------

    def set_tag(self, guild_id: int, name: str, content: str, created_by: int) -> None:
        self._write(
            "INSERT INTO tags (guild_id, name, content, created_by, uses, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(guild_id, name) DO UPDATE SET content = excluded.content",
            (str(guild_id), name.lower()[:60], content[:2000], str(created_by), time.time()),
        )

    def get_tag(self, guild_id: int, name: str) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT name, content, uses FROM tags WHERE guild_id = ? AND name = ?",
            (str(guild_id), name.lower()),
        )
        if rows:
            self._write(
                "UPDATE tags SET uses = uses + 1 WHERE guild_id = ? AND name = ?",
                (str(guild_id), name.lower()),
            )
        return rows[0] if rows else None

    def delete_tag(self, guild_id: int, name: str) -> bool:
        cur = self._write(
            "DELETE FROM tags WHERE guild_id = ? AND name = ?", (str(guild_id), name.lower())
        )
        return bool(cur.rowcount)

    def list_tags(self, guild_id: int) -> list[sqlite3.Row]:
        return self._read(
            "SELECT name, uses FROM tags WHERE guild_id = ? ORDER BY uses DESC, name",
            (str(guild_id),),
        )

    # -- tickets ------------------------------------------------------------

    def open_ticket(self, guild_id: int, channel_id: int, user_id: int, subject: str | None) -> int:
        cur = self._write(
            "INSERT INTO tickets (guild_id, channel_id, user_id, subject, status, created_at) "
            "VALUES (?, ?, ?, ?, 'open', ?)",
            (str(guild_id), str(channel_id), str(user_id), (subject or "")[:200] or None, time.time()),
        )
        return int(cur.lastrowid or 0)

    def close_ticket(self, guild_id: int, channel_id: int, closed_by: int) -> bool:
        cur = self._write(
            "UPDATE tickets SET status = 'closed', closed_at = ?, closed_by = ? "
            "WHERE guild_id = ? AND channel_id = ? AND status = 'open'",
            (time.time(), str(closed_by), str(guild_id), str(channel_id)),
        )
        return bool(cur.rowcount)

    def open_ticket_for(self, guild_id: int, user_id: int) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT id, channel_id FROM tickets "
            "WHERE guild_id = ? AND user_id = ? AND status = 'open' LIMIT 1",
            (str(guild_id), str(user_id)),
        )
        return rows[0] if rows else None

    def ticket_counts(self, guild_id: int) -> dict[str, int]:
        rows = self._read(
            "SELECT status, COUNT(*) AS n FROM tickets WHERE guild_id = ? GROUP BY status",
            (str(guild_id),),
        )
        return {r["status"]: int(r["n"]) for r in rows}

    # -- starboard ----------------------------------------------------------

    def get_star(self, guild_id: int, message_id: int) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT message_id, star_msg_id, stars FROM starboard "
            "WHERE guild_id = ? AND message_id = ?",
            (str(guild_id), str(message_id)),
        )
        return rows[0] if rows else None

    def upsert_star(
        self, guild_id: int, message_id: int, stars: int, star_msg_id: int | None
    ) -> None:
        self._write(
            "INSERT INTO starboard (guild_id, message_id, star_msg_id, stars) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(guild_id, message_id) DO UPDATE SET stars = excluded.stars, "
            "star_msg_id = COALESCE(excluded.star_msg_id, starboard.star_msg_id)",
            (str(guild_id), str(message_id), str(star_msg_id) if star_msg_id else None, stars),
        )

    # -- giveaways ----------------------------------------------------------

    def create_giveaway(
        self, guild_id: int, channel_id: int, prize: str, winners: int, host_id: int, ends_at: float
    ) -> int:
        cur = self._write(
            "INSERT INTO giveaways (guild_id, channel_id, prize, winners, host_id, ends_at, "
            "ended, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (str(guild_id), str(channel_id), prize[:200], winners, str(host_id), ends_at, time.time()),
        )
        return int(cur.lastrowid or 0)

    def set_giveaway_message(self, giveaway_id: int, message_id: int) -> None:
        self._write(
            "UPDATE giveaways SET message_id = ? WHERE id = ?", (str(message_id), giveaway_id)
        )

    def enter_giveaway(self, giveaway_id: int, user_id: int) -> bool:
        cur = self._write(
            "INSERT OR IGNORE INTO giveaway_entries (giveaway_id, user_id) VALUES (?, ?)",
            (giveaway_id, str(user_id)),
        )
        return bool(cur.rowcount)

    def leave_giveaway(self, giveaway_id: int, user_id: int) -> bool:
        cur = self._write(
            "DELETE FROM giveaway_entries WHERE giveaway_id = ? AND user_id = ?",
            (giveaway_id, str(user_id)),
        )
        return bool(cur.rowcount)

    def giveaway_entries(self, giveaway_id: int) -> list[str]:
        return [
            r["user_id"]
            for r in self._read(
                "SELECT user_id FROM giveaway_entries WHERE giveaway_id = ?", (giveaway_id,)
            )
        ]

    def giveaway_by_message(self, message_id: int) -> sqlite3.Row | None:
        rows = self._read(
            "SELECT id, guild_id, channel_id, prize, winners, host_id, ends_at, ended "
            "FROM giveaways WHERE message_id = ?",
            (str(message_id),),
        )
        return rows[0] if rows else None

    def due_giveaways(self, now: float | None = None) -> list[sqlite3.Row]:
        return self._read(
            "SELECT id, guild_id, channel_id, message_id, prize, winners, host_id FROM giveaways "
            "WHERE ended = 0 AND ends_at <= ?",
            (now or time.time(),),
        )

    def end_giveaway(self, giveaway_id: int) -> None:
        self._write("UPDATE giveaways SET ended = 1 WHERE id = ?", (giveaway_id,))

    # -- reaction / button roles --------------------------------------------

    def add_reaction_role(
        self, guild_id: int, message_id: int, key: str, role_id: int, label: str | None = None
    ) -> None:
        self._write(
            "INSERT INTO reaction_roles (guild_id, message_id, key, role_id, label) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(guild_id, message_id, key) DO UPDATE SET role_id = excluded.role_id, "
            "label = excluded.label",
            (str(guild_id), str(message_id), key, str(role_id), label),
        )

    def reaction_role(self, guild_id: int, message_id: int, key: str) -> str | None:
        rows = self._read(
            "SELECT role_id FROM reaction_roles WHERE guild_id = ? AND message_id = ? AND key = ?",
            (str(guild_id), str(message_id), key),
        )
        return rows[0]["role_id"] if rows else None

    def reaction_roles_for(self, guild_id: int, message_id: int) -> list[sqlite3.Row]:
        return self._read(
            "SELECT key, role_id, label FROM reaction_roles WHERE guild_id = ? AND message_id = ?",
            (str(guild_id), str(message_id)),
        )

    def delete_reaction_roles(self, guild_id: int, message_id: int) -> int:
        cur = self._write(
            "DELETE FROM reaction_roles WHERE guild_id = ? AND message_id = ?",
            (str(guild_id), str(message_id)),
        )
        return cur.rowcount or 0
