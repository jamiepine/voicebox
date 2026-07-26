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
            "ON CONFLICT(guild_id, profile_id) DO UPDATE SET "
            "profile_name = excluded.profile_name, attestation = excluded.attestation",
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
