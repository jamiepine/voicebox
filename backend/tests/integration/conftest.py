"""Shared fixtures for integration tests.

Uses an in-memory SQLite database.  Tests call the service layer directly
(e.g. history.list_generations) rather than mounting the full FastAPI app,
so that torch / soundfile / MCP are never imported at test-collection time.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.database import Base, VoiceProfile as DBVoiceProfile


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def seeded_db(db_session):
    """Session pre-loaded with a profile and a handful of generation rows."""
    from backend.database import Generation as DBGeneration

    profile = DBVoiceProfile(id="prof-1", name="Alice")
    db_session.add(profile)
    db_session.add_all(
        [
            DBGeneration(id="gen-1", profile_id="prof-1", text="Hello world", status="completed"),
            DBGeneration(id="gen-2", profile_id="prof-1", text="50% off sale", status="completed"),
            DBGeneration(id="gen-3", profile_id="prof-1", text="path_to_file.wav", status="completed"),
            DBGeneration(id="gen-4", profile_id="prof-1", text="Say 100%", status="completed"),
            DBGeneration(id="gen-5", profile_id="prof-1", text="Goodbye world", status="completed"),
        ]
    )
    db_session.commit()
    return db_session


def run(coro):
    """Run an async coroutine synchronously (asyncio.run wrapper)."""
    return asyncio.run(coro)
