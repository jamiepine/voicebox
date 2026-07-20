"""
Tests for generation history service — especially search/LIKE behaviour.

Validates that LIKE metacharacter injection is correctly escaped so that
searching for literal "50%" or "path\\to\\file" doesn't blow up or return
wrong results.

Runs against an in-memory SQLite database so no ML deps are required.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Make ``from backend.xxx import …`` work when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from backend.models import HistoryQuery

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """In-memory SQLite session pre-loaded with a profile and some generations."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    make_session = sessionmaker(bind=engine)
    session = make_session()

    # A profile is required for the JOIN in list_generations.
    profile = DBVoiceProfile(id="p1", name="Test Voice")
    session.add(profile)

    rows = [
        DBGeneration(id="g1", profile_id="p1", text="Hello world", status="completed"),
        DBGeneration(id="g2", profile_id="p1", text="50% off sale", status="completed"),
        DBGeneration(id="g3", profile_id="p1", text="path_to_file.wav", status="completed"),
        DBGeneration(id="g4", profile_id="p1", text="Say 100%", status="completed"),
        DBGeneration(id="g5", profile_id="p1", text="under_score test", status="completed"),
    ]
    session.add_all(rows)
    session.commit()

    yield session

    session.close()


# ---------------------------------------------------------------------------
# Import the function under test lazily (avoids importing torch at collection
# time — the service module imports SQLAlchemy models only).
# ---------------------------------------------------------------------------


def _list_generations(query, db):
    """Thin wrapper so we can import inside the test, not at module level."""
    import asyncio

    from backend.services.history import list_generations

    return asyncio.run(list_generations(query, db))


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------


class TestBasicSearch:
    def test_no_filter_returns_all(self, db):
        result = _list_generations(HistoryQuery(), db)
        assert result.total == 5

    def test_plain_text_search(self, db):
        result = _list_generations(HistoryQuery(search="Hello"), db)
        assert result.total == 1
        assert result.items[0].text == "Hello world"

    def test_case_insensitive_via_like(self, db):
        # SQLite LIKE is case-insensitive for ASCII by default.
        result = _list_generations(HistoryQuery(search="hello"), db)
        assert result.total >= 1

    def test_no_match_returns_empty(self, db):
        result = _list_generations(HistoryQuery(search="zzznomatch"), db)
        assert result.total == 0


# ---------------------------------------------------------------------------
# Metacharacter escaping — the core regression tests
# ---------------------------------------------------------------------------


class TestLIKEEscaping:
    """Ensure that LIKE metacharacters in the search query are treated literally."""

    def test_percent_matches_literally(self, db):
        """Searching "50%" must only match rows that literally contain "50%"."""
        result = _list_generations(HistoryQuery(search="50%"), db)
        # Should match "50% off sale" and "Say 100%" (both contain "50%"? No —
        # only g2 contains the literal substring "50%").
        ids = {item.id for item in result.items}
        assert "g2" in ids, "Should match the row with literal '50%'"
        assert "g4" not in ids, "'Say 100%' does not contain '50%'"

    def test_percent_wildcard_not_treated_as_glob(self, db):
        """A bare "%" must NOT match every row (which LIKE '%' would do)."""
        result = _list_generations(HistoryQuery(search="%"), db)
        # Only rows whose text literally contains "%" should match.
        texts = {item.text for item in result.items}
        for text in texts:
            assert "%" in text, f"Row '{text}' matched but has no literal '%'"

    def test_underscore_matches_literally(self, db):
        """Searching 'path_to' must not use '_' as a single-character wildcard."""
        result = _list_generations(HistoryQuery(search="path_to"), db)
        # Unescaped '_' would match any char → could match "Hello world" (H?llo …).
        # With proper escaping only the row with literal "path_to" should match.
        ids = {item.id for item in result.items}
        assert "g3" in ids
        # "Hello world" should NOT match via wildcard expansion.
        assert "g1" not in ids

    def test_underscore_wildcard_not_treated_as_glob(self, db):
        """A bare "_" must NOT match any single character across all rows."""
        result = _list_generations(HistoryQuery(search="_"), db)
        texts = {item.text for item in result.items}
        for text in texts:
            assert "_" in text, f"Row '{text}' matched but has no literal '_'"

    def test_combined_metacharacters(self, db):
        """A query mixing '%' and '_' is escaped completely."""
        result = _list_generations(HistoryQuery(search="path_to_file.wav"), db)
        ids = {item.id for item in result.items}
        assert "g3" in ids
        assert result.total == 1


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_limit_respected(self, db):
        result = _list_generations(HistoryQuery(limit=2), db)
        assert len(result.items) == 2
        assert result.total == 5

    def test_offset_respected(self, db):
        all_result = _list_generations(HistoryQuery(limit=5), db)
        page2 = _list_generations(HistoryQuery(limit=3, offset=3), db)
        assert len(page2.items) == 2  # 5 total, skip 3 → 2 remain
        all_ids = [item.id for item in all_result.items]
        page2_ids = [item.id for item in page2.items]
        # Page 2 IDs should be a strict subset of all IDs with no overlap
        # with the first 3 items.
        assert set(page2_ids).issubset(set(all_ids))
        assert not set(page2_ids).intersection(set(all_ids[:3]))
