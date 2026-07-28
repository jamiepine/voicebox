"""Engine creation, initialization, and session management."""

import logging
import sqlite3 as _sqlite3
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .. import config
from .models import (
    Base,
    AudioChannel,
    EffectPreset,
    Generation,
    GenerationVersion,
    ProfileChannelMapping,
    VoiceProfile,
)
from .migrations import run_migrations
from .seed import backfill_generation_versions, seed_builtin_presets

logger = logging.getLogger(__name__)


def _make_connection(db_path: str) -> _sqlite3.Connection:
    """Open a SQLite connection with WAL journal mode and a 5-second busy timeout.

    WAL allows concurrent readers while a write is in progress (the default
    DELETE journal blocks all readers).  This matters for voicebox because SSE
    status polls and history queries run concurrently with the generation worker
    writing to the same database.  The busy timeout prevents "database is
    locked" errors when two writers briefly contend on the same write slot.
    """
    conn = _sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# Initialized by init_db()
engine = None
SessionLocal = None
_db_path = None


def init_db() -> None:
    """Initialize the database engine, run migrations, create tables, and seed data."""
    global engine, SessionLocal, _db_path

    _db_path = config.get_db_path()
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False},
        # Each connection enables WAL journal mode and sets a 5-second busy
        # timeout.  WAL allows concurrent readers during a write (the default
        # DELETE/ROLLBACK journal blocks all readers), which matters for
        # voicebox because SSE status polls and history queries run
        # concurrently with the generation worker writing to the same db.
        # busy_timeout prevents "database is locked" errors when two
        # connections briefly contend on the same write slot.
        creator=lambda: _make_connection(str(_db_path)),
    )

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    run_migrations(engine)
    Base.metadata.create_all(bind=engine)

    # Create default audio channel if it doesn't exist
    db = SessionLocal()
    try:
        default_channel = db.query(AudioChannel).filter(AudioChannel.is_default == True).first()
        if not default_channel:
            default_channel = AudioChannel(
                id=str(uuid.uuid4()),
                name="Default",
                is_default=True,
            )
            db.add(default_channel)

            for profile in db.query(VoiceProfile).all():
                db.add(ProfileChannelMapping(
                    profile_id=profile.id,
                    channel_id=default_channel.id,
                ))
            db.commit()
    finally:
        db.close()

    backfill_generation_versions(SessionLocal, Generation, GenerationVersion)
    seed_builtin_presets(SessionLocal, EffectPreset)


def get_db():
    """Yield a database session (FastAPI dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
