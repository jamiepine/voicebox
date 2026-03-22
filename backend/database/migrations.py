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

import logging

from sqlalchemy import inspect, text

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
