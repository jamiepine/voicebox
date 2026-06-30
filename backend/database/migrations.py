"""Column-level migrations for the voicebox SQLite database.

Why not Alembic?  voicebox is a single-user desktop app shipping as a
PyInstaller binary.  Every user has exactly one SQLite file.  Alembic's
strengths -- migration tracking across environments, rollback, team
coordination -- don't apply here and would add bundling complexity
(alembic.ini, env.py, versions/ directory all need to survive
PyInstaller).  The column-existence checks below are idempotent, run in
<50 ms on startup, and have worked reliably across 12 schema changes.
If the project ever moves to a server-based deployment or Postgres, this
decision should be revisited.

Adding a new migration:
    1. Append a new ``_migrate_*`` helper at the bottom of this file.
    2. Call it from ``run_migrations()`` in the appropriate spot.
    3. The helper should check column/table existence before acting
       (idempotent) and print a short message when it does real work.
"""

import json
import logging
import sqlite3

from sqlalchemy import inspect, text

from ..utils.capture_chords import (
    default_push_to_talk_chord,
    default_toggle_to_talk_chord,
)

logger = logging.getLogger(__name__)


def run_migrations(engine) -> None:
    """Run all schema migrations.  Safe to call on every startup."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    _migrate_story_items(engine, inspector, tables)
    _migrate_profiles(engine, inspector, tables)
    _migrate_generations(engine, inspector, tables)
    _migrate_effect_presets(engine, inspector, tables)
    _migrate_generation_versions(engine, inspector, tables)
    _migrate_capture_settings(engine, inspector, tables)
    _migrate_mcp_bindings(engine, inspector, tables)
    _normalize_storage_paths(engine, tables)


# -- helpers ---------------------------------------------------------------

def _get_columns(inspector, table: str) -> set[str]:
    return {col["name"] for col in inspector.get_columns(table)}


def _add_column(engine, table: str, column_sql: str, label: str) -> None:
    """Add a column if it doesn't already exist."""
    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_sql}"))
        conn.commit()
    logger.info("Added %s column to %s", label, table)


# -- per-table migrations --------------------------------------------------

def _migrate_story_items(engine, inspector, tables: set[str]) -> None:
    if "story_items" not in tables:
        return

    columns = _get_columns(inspector, "story_items")

    # Replace position-based ordering with absolute timecodes
    if "position" in columns:
        logger.info("Migrating story_items: removing position column, using start_time_ms")
        with engine.connect() as conn:
            if "start_time_ms" not in columns:
                conn.execute(text(
                    "ALTER TABLE story_items ADD COLUMN start_time_ms INTEGER DEFAULT 0"
                ))
                result = conn.execute(text("""
                    SELECT si.id, si.story_id, si.position, g.duration
                    FROM story_items si
                    JOIN generations g ON si.generation_id = g.id
                    ORDER BY si.story_id, si.position
                """))
                current_story_id = None
                current_time_ms = 0
                for item_id, story_id, _position, duration in result.fetchall():
                    if story_id != current_story_id:
                        current_story_id = story_id
                        current_time_ms = 0
                    conn.execute(
                        text("UPDATE story_items SET start_time_ms = :time WHERE id = :id"),
                        {"time": current_time_ms, "id": item_id},
                    )
                    current_time_ms += int((duration or 0) * 1000) + 200
                conn.commit()

            # Recreate table without the position column (SQLite lacks DROP COLUMN)
            conn.execute(text("""
                CREATE TABLE story_items_new (
                    id VARCHAR PRIMARY KEY,
                    story_id VARCHAR NOT NULL,
                    generation_id VARCHAR NOT NULL,
                    start_time_ms INTEGER NOT NULL DEFAULT 0,
                    track INTEGER NOT NULL DEFAULT 0,
                    trim_start_ms INTEGER NOT NULL DEFAULT 0,
                    trim_end_ms INTEGER NOT NULL DEFAULT 0,
                    version_id VARCHAR,
                    created_at DATETIME,
                    FOREIGN KEY (story_id) REFERENCES stories(id),
                    FOREIGN KEY (generation_id) REFERENCES generations(id)
                )
            """))
            conn.execute(text("""
                INSERT INTO story_items_new (id, story_id, generation_id, start_time_ms, track, trim_start_ms, trim_end_ms, version_id, created_at)
                SELECT id, story_id, generation_id, start_time_ms,
                    COALESCE(track, 0), COALESCE(trim_start_ms, 0), COALESCE(trim_end_ms, 0), version_id, created_at
                FROM story_items
            """))
            conn.execute(text("DROP TABLE story_items"))
            conn.execute(text("ALTER TABLE story_items_new RENAME TO story_items"))
            conn.commit()

        # Re-read after table recreation
        columns = _get_columns(inspector, "story_items")

    if "track" not in columns:
        _add_column(engine, "story_items", "track INTEGER NOT NULL DEFAULT 0", "track")
    # Re-read so subsequent checks see new columns
    columns = _get_columns(inspector, "story_items")
    if "trim_start_ms" not in columns:
        _add_column(engine, "story_items", "trim_start_ms INTEGER NOT NULL DEFAULT 0", "trim_start_ms")
    if "trim_end_ms" not in columns:
        _add_column(engine, "story_items", "trim_end_ms INTEGER NOT NULL DEFAULT 0", "trim_end_ms")
    if "version_id" not in columns:
        _add_column(engine, "story_items", "version_id VARCHAR", "version_id")
    if "volume" not in columns:
        _add_column(engine, "story_items", "volume FLOAT NOT NULL DEFAULT 1.0", "volume")


def _migrate_profiles(engine, inspector, tables: set[str]) -> None:
    if "profiles" not in tables:
        return
    columns = _get_columns(inspector, "profiles")
    if "avatar_path" not in columns:
        _add_column(engine, "profiles", "avatar_path VARCHAR", "avatar_path")
    if "effects_chain" not in columns:
        _add_column(engine, "profiles", "effects_chain TEXT", "effects_chain")
    # Voice type system — v0.3.x
    if "voice_type" not in columns:
        _add_column(engine, "profiles", "voice_type VARCHAR DEFAULT 'cloned'", "voice_type")
    if "preset_engine" not in columns:
        _add_column(engine, "profiles", "preset_engine VARCHAR", "preset_engine")
    if "preset_voice_id" not in columns:
        _add_column(engine, "profiles", "preset_voice_id VARCHAR", "preset_voice_id")
    if "design_prompt" not in columns:
        _add_column(engine, "profiles", "design_prompt TEXT", "design_prompt")
    if "default_engine" not in columns:
        _add_column(engine, "profiles", "default_engine VARCHAR", "default_engine")
    if "personality" not in columns:
        _add_column(engine, "profiles", "personality TEXT", "personality")


def _migrate_generations(engine, inspector, tables: set[str]) -> None:
    if "generations" not in tables:
        return
    columns = _get_columns(inspector, "generations")
    if "status" not in columns:
        _add_column(engine, "generations", "status VARCHAR DEFAULT 'completed'", "status")
    if "error" not in columns:
        _add_column(engine, "generations", "error TEXT", "error")
    if "engine" not in columns:
        _add_column(engine, "generations", "engine VARCHAR DEFAULT 'qwen'", "engine")
    # Re-read after engine column (variable name shadows outer scope in old code)
    columns = _get_columns(inspector, "generations")
    if "model_size" not in columns:
        _add_column(engine, "generations", "model_size VARCHAR", "model_size")
    if "is_favorited" not in columns:
        _add_column(engine, "generations", "is_favorited BOOLEAN DEFAULT 0", "is_favorited")
    if "source" not in columns:
        _add_column(
            engine,
            "generations",
            "source VARCHAR NOT NULL DEFAULT 'manual'",
            "source",
        )


def _migrate_effect_presets(engine, inspector, tables: set[str]) -> None:
    if "effect_presets" not in tables:
        return
    columns = _get_columns(inspector, "effect_presets")
    if "sort_order" not in columns:
        _add_column(engine, "effect_presets", "sort_order INTEGER DEFAULT 100", "sort_order")


def _migrate_generation_versions(engine, inspector, tables: set[str]) -> None:
    if "generation_versions" not in tables:
        return
    columns = _get_columns(inspector, "generation_versions")
    if "source_version_id" not in columns:
        _add_column(engine, "generation_versions", "source_version_id VARCHAR", "source_version_id")


def _migrate_capture_settings(engine, inspector, tables: set[str]) -> None:
    if "capture_settings" not in tables:
        return
    columns = _get_columns(inspector, "capture_settings")
    push_default = json.dumps(default_push_to_talk_chord())
    toggle_default = json.dumps(default_toggle_to_talk_chord())
    if "allow_auto_paste" not in columns:
        _add_column(
            engine,
            "capture_settings",
            "allow_auto_paste BOOLEAN NOT NULL DEFAULT 1",
            "allow_auto_paste",
        )
    if "default_playback_voice_id" not in columns:
        _add_column(
            engine,
            "capture_settings",
            "default_playback_voice_id VARCHAR",
            "default_playback_voice_id",
        )
    if "chord_push_to_talk_keys" not in columns:
        _add_column(
            engine,
            "capture_settings",
            f"chord_push_to_talk_keys TEXT NOT NULL DEFAULT '{push_default}'",
            "chord_push_to_talk_keys",
        )
    if "chord_toggle_to_talk_keys" not in columns:
        _add_column(
            engine,
            "capture_settings",
            f"chord_toggle_to_talk_keys TEXT NOT NULL DEFAULT '{toggle_default}'",
            "chord_toggle_to_talk_keys",
        )
    if "hotkey_enabled" not in columns:
        _add_column(
            engine,
            "capture_settings",
            "hotkey_enabled BOOLEAN NOT NULL DEFAULT 0",
            "hotkey_enabled",
        )


def _migrate_mcp_bindings(engine, inspector, tables: set[str]) -> None:
    """Drop the legacy ``default_intent`` column and add ``default_personality``.

    The intent tri-state (respond / rewrite / compose) has been collapsed
    to a boolean: when true, ``voicebox.speak`` rewrites input through the
    profile's personality LLM before TTS.
    """
    if "mcp_client_bindings" not in tables:
        return
    columns = _get_columns(inspector, "mcp_client_bindings")
    if "default_personality" not in columns:
        _add_column(
            engine,
            "mcp_client_bindings",
            "default_personality BOOLEAN NOT NULL DEFAULT 0",
            "default_personality",
        )
    if "default_intent" in columns:
        if _supports_drop_column(engine):
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE mcp_client_bindings DROP COLUMN default_intent"))
                conn.commit()
            logger.info("Dropped legacy default_intent column from mcp_client_bindings")
        else:
            # ALTER TABLE … DROP COLUMN on SQLite requires 3.35+ (Mar
            # 2021). Production PyInstaller builds bundle Python 3.12
            # which links to SQLite 3.40+; this branch only fires for
            # dev environments running the backend directly against an
            # old system SQLite (Ubuntu 20.04 = 3.31, Debian 11 = 3.34).
            # Leaving the unused column in place is harmless — the ORM
            # only maps declared columns, so a stray one does no work
            # and gets no reads or writes.
            logger.warning(
                "SQLite %s too old to DROP COLUMN (need 3.35+); leaving unused default_intent column on mcp_client_bindings in place.",
                sqlite3.sqlite_version,
            )


def _supports_drop_column(engine) -> bool:
    """Whether ``ALTER TABLE … DROP COLUMN`` is supported by the dialect +
    runtime. Non-SQLite dialects (Postgres, MySQL) have supported it for
    decades; SQLite only gained the feature in 3.35."""
    if engine.dialect.name != "sqlite":
        return True
    return tuple(int(p) for p in sqlite3.sqlite_version.split(".")[:3]) >= (3, 35, 0)


def _normalize_storage_paths(engine, tables: set[str]) -> None:
    """Normalize stored file paths to be relative to the configured data dir."""
    from pathlib import Path

    from ..config import get_data_dir, to_storage_path, resolve_storage_path

    data_dir = get_data_dir()

    path_columns = [
        ("generations", "audio_path"),
        ("generation_versions", "audio_path"),
        ("profile_samples", "audio_path"),
        ("profiles", "avatar_path"),
    ]

    total_fixed = 0
    with engine.connect() as conn:
        for table, column in path_columns:
            if table not in tables:
                continue
            rows = conn.execute(
                text(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL")
            ).fetchall()
            for row_id, path_val in rows:
                if not path_val:
                    continue
                p = Path(path_val)
                resolved = resolve_storage_path(p)
                if resolved is None:
                    continue

                normalized = to_storage_path(resolved)

                if normalized != path_val:
                    conn.execute(
                        text(f"UPDATE {table} SET {column} = :path WHERE id = :id"),
                        {"path": normalized, "id": row_id},
                    )
                    total_fixed += 1
        if total_fixed > 0:
            conn.commit()
            logger.info("Normalized %d stored file paths", total_fixed)
