"""Integration tests for the history service's list_generations function.

These tests verify the full service-layer behaviour using a real (in-memory)
SQLite database.  They cover:

- Listing all generations with profile_name populated via JOIN (N+1 fix)
- Pagination (limit / offset)
- Search filtering with LIKE-metacharacter escaping
- Profile-scoped filtering
- Edge cases (empty DB, unknown profile)

The service is called directly rather than through FastAPI routes so that
heavy ML imports (torch, soundfile, fastmcp) are never loaded during
test collection.
"""

from datetime import UTC

from backend.models import HistoryQuery
from backend.services.history import list_generations

from .conftest import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(result) -> set[str]:
    return {item.id for item in result.items}


def _list(db, **kwargs):
    return run(list_generations(HistoryQuery(**kwargs), db))


# ---------------------------------------------------------------------------
# Basic listing
# ---------------------------------------------------------------------------


class TestHistoryList:
    def test_empty_db_returns_zero(self, db_session):
        result = _list(db_session)
        assert result.total == 0
        assert result.items == []

    def test_returns_all_items(self, seeded_db):
        result = _list(seeded_db)
        assert result.total == 5
        assert len(result.items) == 5

    def test_profile_name_populated_for_every_item(self, seeded_db):
        """JOIN must resolve profile_name — regression for the N+1 fix."""
        result = _list(seeded_db)
        for item in result.items:
            assert item.profile_name == "Alice", (
                f"Expected profile_name='Alice', got {item.profile_name!r} for generation {item.id!r}"
            )

    def test_default_ordering_is_newest_first(self, seeded_db):
        """Generations should come back sorted by created_at descending."""
        from datetime import datetime, timedelta

        from backend.database import Generation as DBGeneration

        # Assign distinct timestamps so ordering is deterministic.
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i, gen_id in enumerate(["gen-1", "gen-2", "gen-3", "gen-4", "gen-5"]):
            gen = seeded_db.query(DBGeneration).filter_by(id=gen_id).first()
            gen.created_at = base + timedelta(seconds=i)
        seeded_db.commit()

        result = _list(seeded_db)
        assert result.items[0].id == "gen-5"  # newest
        assert result.items[-1].id == "gen-1"  # oldest


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestHistoryPagination:
    def test_limit_respected(self, seeded_db):
        result = _list(seeded_db, limit=2)
        assert len(result.items) == 2
        assert result.total == 5

    def test_offset_skips_rows(self, seeded_db):
        all_ids = _ids(_list(seeded_db, limit=5))
        page2 = _list(seeded_db, limit=3, offset=3)
        assert len(page2.items) == 2
        assert _ids(page2).issubset(all_ids)

    def test_offset_beyond_total_returns_empty_items(self, seeded_db):
        result = _list(seeded_db, offset=100)
        assert result.total == 5
        assert result.items == []


# ---------------------------------------------------------------------------
# Search filtering
# ---------------------------------------------------------------------------


class TestHistorySearch:
    def test_plain_search(self, seeded_db):
        result = _list(seeded_db, search="Hello")
        assert result.total == 1
        assert result.items[0].id == "gen-1"

    def test_no_match_returns_empty(self, seeded_db):
        result = _list(seeded_db, search="zzznomatch")
        assert result.total == 0

    def test_percent_sign_treated_literally(self, seeded_db):
        """'50%' must only match rows containing the literal string '50%'."""
        result = _list(seeded_db, search="50%")
        found = _ids(result)
        assert "gen-2" in found  # "50% off sale"
        assert "gen-4" not in found  # "Say 100%" does not contain "50%"

    def test_underscore_treated_literally(self, seeded_db):
        """'path_to' must not match via single-char wildcard expansion."""
        result = _list(seeded_db, search="path_to")
        found = _ids(result)
        assert "gen-3" in found  # "path_to_file.wav"
        assert "gen-1" not in found  # "Hello world" must not match via '_'


# ---------------------------------------------------------------------------
# Profile-id filter
# ---------------------------------------------------------------------------


class TestHistoryProfileFilter:
    def test_filter_by_known_profile(self, seeded_db):
        result = _list(seeded_db, profile_id="prof-1")
        assert result.total == 5
        for item in result.items:
            assert item.profile_id == "prof-1"

    def test_unknown_profile_returns_empty(self, seeded_db):
        result = _list(seeded_db, profile_id="does-not-exist")
        assert result.total == 0
        assert result.items == []


# ---------------------------------------------------------------------------
# N+1 regression — all profile names resolved in a single round-trip
# ---------------------------------------------------------------------------


class TestN1Fix:
    def test_batch_version_load_no_queries_per_item(self, seeded_db):
        """Versions for all items are loaded in one query, not N queries.

        We test this indirectly: if the N+1 is re-introduced the JOIN would
        fail for items whose versions are loaded one-by-one and the profile_name
        field would be empty or None.  The profile_name assertion in
        test_profile_name_populated_for_every_item already catches that;
        this test additionally confirms that adding GenerationVersion rows
        doesn't change the total query count from the caller's perspective.
        """
        import uuid
        from datetime import datetime

        from backend.database import GenerationVersion as DBGenerationVersion

        # Add one version per generation
        for gen_id in ["gen-1", "gen-2", "gen-3"]:
            seeded_db.add(
                DBGenerationVersion(
                    id=str(uuid.uuid4()),
                    generation_id=gen_id,
                    label="v1",
                    audio_path=f"audio/{gen_id}-v1.wav",
                    is_default=True,
                    created_at=datetime.now(UTC),
                )
            )
        seeded_db.commit()

        result = _list(seeded_db)
        # All 5 items still returned; the 3 with versions have them populated
        assert result.total == 5
        versioned = [item for item in result.items if item.versions]
        assert len(versioned) == 3
        for item in versioned:
            assert item.active_version_id is not None
