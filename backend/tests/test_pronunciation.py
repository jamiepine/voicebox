"""
Tests for the pronunciation dictionary (#827).

The rules that matter are the ones that are easy to get wrong: replacements
must not cascade into each other, capitalisation has to survive, scope has to
resolve in a defined order, and the rewrite must never reach the stored text.

Usage:
    python -m pytest backend/tests/test_pronunciation.py -v
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-pronunciation-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import PronunciationEntry, get_db  # noqa: E402
from backend.services.pronunciation import apply_pronunciations, build_pattern  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db(client):
    # Depends on `client` so the app lifespan has run: SessionLocal is None
    # until init_db(), and a test that only asked for `db` would get it unset.
    session = next(get_db())
    try:
        yield session
    finally:
        session.query(PronunciationEntry).delete()
        session.commit()
        session.close()


@pytest.fixture
def profile(client):
    """A throwaway voice profile.

    Unique name and explicit cleanup because ``VOICEBOX_DATA_DIR`` is not
    honoured on this branch (see #981/#1004) — the suite writes to the repo's
    ./data, so a fixed name collides with the previous run's leftovers.
    """
    name = f"Pronunciation Test Voice {uuid.uuid4().hex[:8]}"
    r = client.post("/profiles", json={"name": name, "language": "en"})
    assert r.status_code == 200, r.text
    created = r.json()
    yield created
    client.delete(f"/profiles/{created['id']}")


def add(db, term, replacement, language=None, profile_id=None, enabled=True):
    entry = PronunciationEntry(
        term=term,
        replacement=replacement,
        language=language,
        profile_id=profile_id,
        enabled=enabled,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


# ── Matching ─────────────────────────────────────────────────────────


def test_replaces_a_term(db):
    add(db, "bandeja", "ban-DEH-ha")
    out, applied = apply_pronunciations("He plays a bandeja here.", "en", db)
    assert out == "He plays a ban-DEH-ha here."
    assert len(applied) == 1


def test_matching_is_case_insensitive_but_capitalisation_survives(db):
    """One lowercase entry has to cover the term at the start of a sentence."""
    add(db, "bandeja", "ban-DEH-ha")
    out, _ = apply_pronunciations("Bandeja is the shot.", "en", db)
    assert out == "Ban-DEH-ha is the shot."


def test_all_caps_stays_all_caps(db):
    add(db, "wcag", "W C A G")
    out, _ = apply_pronunciations("Follow WCAG rules.", "en", db)
    assert out == "Follow W C A G rules."


def test_only_whole_words_match(db):
    """Substring matching would turn 'brandeja' into nonsense."""
    add(db, "bandeja", "ban-DEH-ha")
    out, applied = apply_pronunciations("A brandejapalooza appeared.", "en", db)
    assert out == "A brandejapalooza appeared."
    assert applied == []


def test_replacements_do_not_cascade(db):
    """The single-pass alternation exists for this: a loop of per-term
    substitutions would rewrite the output of an earlier rule."""
    add(db, "bandeja", "ban-DEH-ha")
    add(db, "ha", "HAH")
    out, applied = apply_pronunciations("The bandeja.", "en", db)
    assert out == "The ban-DEH-ha."
    assert len(applied) == 1


def test_longer_terms_win(db):
    add(db, "bandeja", "ban-DEH-ha")
    add(db, "bandeja alta", "ban-DEH-ha AL-ta")
    out, _ = apply_pronunciations("A bandeja alta lands deep.", "en", db)
    assert out == "A ban-DEH-ha AL-ta lands deep."


def test_paralinguistic_tags_are_left_alone(db):
    """[laugh] is engine syntax, not speech."""
    add(db, "laugh", "LAFF")
    out, _ = apply_pronunciations("[laugh] I did laugh.", "en", db)
    assert out == "[laugh] I did LAFF."


def test_accented_terms_match(db):
    add(db, "víbora", "VEE-bo-ra")
    out, applied = apply_pronunciations("Then the víbora.", "en", db)
    assert out == "Then the VEE-bo-ra."
    assert len(applied) == 1


def test_reports_what_it_changed(db):
    """A silent rewrite of someone's script is worse than no rewrite."""
    entry = add(db, "bandeja", "ban-DEH-ha")
    _, applied = apply_pronunciations("bandeja and bandeja", "en", db)
    assert len(applied) == 2
    assert all(a["entry_id"] == entry.id for a in applied)


# ── Scope ────────────────────────────────────────────────────────────


def test_language_scoped_entry_only_applies_to_that_language(db):
    """The Spanish word needs respelling when the engine reads English, and
    must be left alone when it is already reading Spanish."""
    add(db, "bandeja", "ban-DEH-ha", language="en")
    assert apply_pronunciations("una bandeja", "es", db)[0] == "una bandeja"
    assert apply_pronunciations("a bandeja", "en", db)[0] == "a ban-DEH-ha"


def test_wildcard_language_applies_everywhere(db):
    add(db, "bandeja", "ban-DEH-ha")
    assert "ban-DEH-ha" in apply_pronunciations("una bandeja", "es", db)[0]


def test_disabled_entries_are_skipped(db):
    add(db, "bandeja", "ban-DEH-ha", enabled=False)
    assert apply_pronunciations("a bandeja", "en", db)[0] == "a bandeja"


def test_profile_entry_beats_global(db, profile):
    add(db, "bandeja", "GLOBAL")
    add(db, "bandeja", "PER-VOICE", profile_id=profile["id"])

    out, _ = apply_pronunciations("a bandeja", "en", db, profile_id=profile["id"])
    assert out == "a PER-VOICE"

    # A different voice still gets the global entry.
    assert apply_pronunciations("a bandeja", "en", db)[0] == "a GLOBAL"


def test_other_profiles_entries_do_not_leak(db, profile):
    add(db, "bandeja", "PER-VOICE", profile_id=profile["id"])
    assert apply_pronunciations("a bandeja", "en", db)[0] == "a bandeja"


# ── Degenerate input ─────────────────────────────────────────────────


def test_no_entries_is_a_no_op(db):
    out, applied = apply_pronunciations("nothing to do", "en", db)
    assert out == "nothing to do"
    assert applied == []


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_text_is_returned_unchanged(db, text):
    add(db, "bandeja", "ban-DEH-ha")
    assert apply_pronunciations(text, "en", db) == (text, [])


def test_build_pattern_handles_no_usable_terms():
    assert build_pattern([]) is None
    assert build_pattern(["", "  "]) is None


def test_regex_metacharacters_in_a_term_are_literal(db):
    """A term is text, not a pattern -- an unescaped '.' would match anything."""
    add(db, "C++", "C plus plus")
    out, _ = apply_pronunciations("I write C++ daily.", "en", db)
    assert out == "I write C plus plus daily."


# ── API ──────────────────────────────────────────────────────────────


def test_crud_roundtrip(client, db):
    created = client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "ban-DEH-ha", "language": "en"}
    )
    assert created.status_code == 200, created.text
    entry_id = created.json()["id"]

    listed = client.get("/pronunciations").json()
    assert any(e["id"] == entry_id for e in listed)

    updated = client.put(f"/pronunciations/{entry_id}", json={"replacement": "ban-DAY-ha"})
    assert updated.status_code == 200
    assert updated.json()["replacement"] == "ban-DAY-ha"
    assert updated.json()["term"] == "bandeja", "omitted fields must be left alone"

    assert client.delete(f"/pronunciations/{entry_id}").status_code == 200
    assert client.get("/pronunciations").json() == []


def test_duplicate_in_the_same_scope_is_rejected(client, db):
    body = {"term": "bandeja", "replacement": "ban-DEH-ha", "language": "en"}
    assert client.post("/pronunciations", json=body).status_code == 200
    clash = client.post("/pronunciations", json=body)
    assert clash.status_code == 409
    assert "already exists" in clash.json()["detail"]


def test_same_term_in_a_different_language_is_allowed(client, db):
    assert client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "A", "language": "en"}
    ).status_code == 200
    assert client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "B", "language": "it"}
    ).status_code == 200


def test_unknown_profile_is_rejected(client, db):
    r = client.post(
        "/pronunciations",
        json={"term": "x", "replacement": "y", "profile_id": "no-such-profile"},
    )
    assert r.status_code == 404


def test_preview_shows_the_rewrite(client, db):
    client.post("/pronunciations", json={"term": "bandeja", "replacement": "ban-DEH-ha"})
    r = client.post(
        "/pronunciations/preview", json={"text": "a bandeja", "language": "en"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original"] == "a bandeja"
    assert body["result"] == "a ban-DEH-ha"
    assert body["applied"][0]["term"] == "bandeja"


def test_missing_entry_404s(client, db):
    assert client.put("/pronunciations/nope", json={"replacement": "x"}).status_code == 404
    assert client.delete("/pronunciations/nope").status_code == 404


# ── The property the design rests on ─────────────────────────────────


@pytest.mark.asyncio
async def test_engine_gets_the_respelling_but_the_row_keeps_the_original(
    client, db, profile, monkeypatch
):
    """The property the whole design rests on.

    If the respelling were stored, History would show a reader ``ban-DEH-ha``
    and editing an entry could never change anything already generated. So the
    engine must see the respelling and the database must not.

    The model itself is mocked -- this is about which string goes where, and
    loading 3.5 GB of weights would not make the assertion any truer.
    """
    import numpy as np

    from backend.services import history as history_service
    from backend.services.generation import run_generation

    client.post("/pronunciations", json={"term": "bandeja", "replacement": "ban-DEH-ha"})

    original = "He plays a bandeja."
    generation = await history_service.create_generation(
        profile_id=profile["id"],
        text=original,
        language="en",
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        status="generating",
        engine="qwen",
    )

    seen = {}

    class FakeBackend:
        def is_loaded(self):
            return True

    async def fake_generate_chunked(_model, text, _voice_prompt, **_kwargs):
        seen["text"] = text
        return np.zeros(2400, dtype=np.float32), 24000

    async def fake_load(*_a, **_k):
        return None

    async def fake_voice_prompt(*_a, **_k):
        return {}

    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _e: FakeBackend())
    monkeypatch.setattr("backend.backends.load_engine_model", fake_load)
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", fake_generate_chunked)
    monkeypatch.setattr(
        "backend.services.profiles.create_voice_prompt_for_profile", fake_voice_prompt
    )

    await run_generation(
        generation_id=generation.id,
        profile_id=profile["id"],
        text=original,
        language="en",
        engine="qwen",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    assert seen["text"] == "He plays a ban-DEH-ha.", "engine should receive the respelling"

    db.expire_all()
    stored = client.get(f"/history/{generation.id}").json()
    assert stored["text"] == original, "the stored row must keep the author's text"
    assert "ban-DEH-ha" not in stored["text"]


def test_preview_is_the_only_way_to_see_the_rewrite(client, db):
    """Since the rewrite is never stored, preview is what makes it inspectable
    rather than a black box between the text and the audio."""
    client.post("/pronunciations", json={"term": "bandeja", "replacement": "ban-DEH-ha"})
    body = client.post(
        "/pronunciations/preview", json={"text": "a bandeja", "language": "en"}
    ).json()
    assert body["result"] != body["original"]
    assert body["applied"]


# ── Review findings (CodeRabbit on #1025) ────────────────────────────


def test_sql_wildcards_in_a_term_are_literal(client, db):
    """`ilike` read `%` and `_` in a term as wildcards, so `band_ja` collided
    with `bandeja` and blocked a legitimate entry as a duplicate."""
    assert client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "ban-DEH-ha"}
    ).status_code == 200
    # Distinct term that ilike would have matched against the one above.
    assert client.post(
        "/pronunciations", json={"term": "band_ja", "replacement": "BAND-ja"}
    ).status_code == 200
    assert client.post(
        "/pronunciations", json={"term": "band%", "replacement": "BAND"}
    ).status_code == 200


def test_untrimmed_term_still_finds_its_duplicate(client, db):
    """The stored value is trimmed, so an untrimmed lookup must normalise too or
    it misses the duplicate it just created."""
    assert client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "A"}
    ).status_code == 200
    clash = client.post("/pronunciations", json={"term": "  bandeja  ", "replacement": "B"})
    assert clash.status_code == 409


@pytest.mark.parametrize("field", ["term", "replacement"])
def test_whitespace_only_values_are_rejected(client, db, field):
    """`min_length=1` accepted "   ", which then stored as empty — a no-op entry,
    or a replacement that deletes the matched speech."""
    body = {"term": "bandeja", "replacement": "ban-DEH-ha"}
    body[field] = "   "
    assert client.post("/pronunciations", json=body).status_code == 422


def test_values_are_stored_trimmed(client, db):
    created = client.post(
        "/pronunciations", json={"term": "  bandeja  ", "replacement": "  ban-DEH-ha  "}
    ).json()
    assert created["term"] == "bandeja"
    assert created["replacement"] == "ban-DEH-ha"


def test_preview_rejects_an_unknown_profile(client, db):
    """Generation 404s on an unknown profile; preview silently fell back to
    global scope, so it reported a different result than it would produce."""
    r = client.post(
        "/pronunciations/preview",
        json={"text": "a bandeja", "profile_id": "no-such-profile"},
    )
    assert r.status_code == 404


def test_the_database_enforces_scope_uniqueness(client, db):
    """find_duplicate is check-then-act; two concurrent creates can both pass
    it. The constraint is what actually holds — including for the NULL scopes,
    which a plain UNIQUE would treat as distinct."""
    from sqlalchemy.exc import IntegrityError

    add(db, "bandeja", "A")
    # Differs only by case, and both are global scope — the pair a plain
    # UNIQUE(term, language, profile_id) would have let through.
    db.add(PronunciationEntry(term="Bandeja", replacement="B"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_terms_are_not_logged_at_info(client, db, caplog):
    """Terms are user-supplied and often names, so the values belong at DEBUG."""
    import logging

    add(db, "Alicia Fernandez", "ah-LEE-see-ah")
    with caplog.at_level(logging.INFO, logger="backend.services.pronunciation"):
        apply_pronunciations("Ask Alicia Fernandez.", "en", db)

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "should still report that a rewrite happened"
    assert all("Alicia" not in r.getMessage() for r in info)
