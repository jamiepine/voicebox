"""Unit tests for the LIKE-escape logic in backend/services/history.py.

These tests verify that percent signs, underscores, and backslashes in a
search query are treated as literal characters rather than LIKE wildcards.
They run against an in-memory SQLite database so no ML dependencies are
needed.
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

from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from backend.models import HistoryQuery

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """In-memory SQLite session pre-loaded with a profile and test rows."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    session.add(DBVoiceProfile(id="p1", name="Test Voice"))
    session.add_all(
        [
            DBGeneration(id="g1", profile_id="p1", text="Hello world", status="completed"),
            DBGeneration(id="g2", profile_id="p1", text="50% off sale", status="completed"),
            DBGeneration(id="g3", profile_id="p1", text="path_to_file.wav", status="completed"),
            DBGeneration(id="g4", profile_id="p1", text="Say 100%", status="completed"),
            DBGeneration(id="g5", profile_id="p1", text="under_score test", status="completed"),
        ]
    )
    session.commit()
    yield session
    session.close()


def _run(query, db):
    from backend.services.history import list_generations

    return asyncio.run(list_generations(query, db))


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------


class TestBasicSearch:
    def test_no_filter_returns_all(self, db):
        result = _run(HistoryQuery(), db)
        assert result.total == 5

    def test_plain_text_match(self, db):
        result = _run(HistoryQuery(search="Hello"), db)
        assert result.total == 1
        assert result.items[0].id == "g1"

    def test_no_match_returns_empty(self, db):
        result = _run(HistoryQuery(search="zzznomatch"), db)
        assert result.total == 0


# ---------------------------------------------------------------------------
# Metacharacter escaping — regression tests
# ---------------------------------------------------------------------------


class TestLIKEEscaping:
    """LIKE metacharacters in the search string must be treated literally."""

    def test_percent_matches_literally(self, db):
        """'50%' must only match the row that literally contains '50%'."""
        result = _run(HistoryQuery(search="50%"), db)
        ids = {item.id for item in result.items}
        assert "g2" in ids  # "50% off sale"
        assert "g4" not in ids  # "Say 100%" does not contain "50%"

    def test_bare_percent_not_a_wildcard(self, db):
        """A bare '%' must not match every row (as unescaped LIKE '%' would)."""
        result = _run(HistoryQuery(search="%"), db)
        for item in result.items:
            assert "%" in item.text, f"'{item.text}' matched but lacks a literal '%'"

    def test_underscore_matches_literally(self, db):
        """'path_to' must not use '_' as a single-char wildcard."""
        result = _run(HistoryQuery(search="path_to"), db)
        ids = {item.id for item in result.items}
        assert "g3" in ids  # "path_to_file.wav"
        assert "g1" not in ids  # "Hello world" must not match via wildcard

    def test_bare_underscore_not_a_wildcard(self, db):
        """A bare '_' must not match every single-character position in all rows."""
        result = _run(HistoryQuery(search="_"), db)
        for item in result.items:
            assert "_" in item.text, f"'{item.text}' matched but lacks a literal '_'"

    def test_combined_metacharacters_escaped(self, db):
        """A query with both '%' and '_' still matches only the exact literal string."""
        result = _run(HistoryQuery(search="path_to_file.wav"), db)
        assert result.total == 1
        assert result.items[0].id == "g3"
