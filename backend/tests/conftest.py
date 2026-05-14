"""
Shared pytest fixtures for the voicebox backend test suite.

All fixtures here produce lightweight in-memory databases seeded with
minimal data so individual tests don't need to repeat boilerplate.
No ML dependencies (torch, transformers, …) are imported in this file.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure the repo root is on sys.path so "from backend.xxx import …" works
# whether pytest is run from the repo root or from backend/.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from backend.database import Base, VoiceProfile as DBVoiceProfile


@pytest.fixture
def db_engine():
    """Create a fresh in-memory SQLite engine for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Provide a transactional database session, rolled back after each test."""
    make_session = sessionmaker(bind=db_engine)
    session = make_session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def db_with_profile(db_session):
    """Session pre-seeded with a single voice profile (id='default-profile')."""
    profile = DBVoiceProfile(id="default-profile", name="Default Voice")
    db_session.add(profile)
    db_session.commit()
    return db_session
