"""
Tests for regenerating a generation with overrides (#870).

Correcting a typo used to mean creating a whole new generation, which loses the
timeline placement, fades and track the old one had. Regenerate already saved
its output as a new *version*; it just could not be told to change anything.

The property that matters: the generation row keeps the text as first written,
and each take records what produced it. Otherwise a corrected take and the take
it replaced become indistinguishable audio under one row that describes only
one of them.

Usage:
    python -m pytest backend/tests/test_regenerate_overrides.py -v
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-regen-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import get_db  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db(client):
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def profile(client):
    name = f"Regen Test Voice {uuid.uuid4().hex[:8]}"
    r = client.post("/profiles", json={"name": name, "language": "en"})
    assert r.status_code == 200, r.text
    created = r.json()
    yield created
    client.delete(f"/profiles/{created['id']}")


@pytest.fixture
def mock_tts(monkeypatch):
    """Replace the model. What is under test is which strings go where."""
    seen = []

    class FakeBackend:
        def is_loaded(self):
            return True

    async def fake_generate_chunked(_model, text, _vp, **kwargs):
        seen.append({"text": text, "language": kwargs.get("language"),
                     "instruct": kwargs.get("instruct"), "seed": kwargs.get("seed")})
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
    return seen


async def _make_generation(db, profile_id, text="He plays a bandeja.", language="en"):
    from backend.services import history as history_service

    return await history_service.create_generation(
        profile_id=profile_id,
        text=text,
        language=language,
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        status="completed",
        engine="qwen",
    )


async def _regenerate(db, generation, overrides=None):
    from backend.services.generation import run_generation

    await run_generation(
        generation_id=generation.id,
        profile_id=generation.profile_id,
        text=(overrides or {}).get("text", generation.text),
        language=(overrides or {}).get("language", generation.language),
        engine="qwen",
        model_size="1.7B",
        seed=(overrides or {}).get("seed"),
        instruct=(overrides or {}).get("instruct"),
        mode="regenerate",
        version_id=str(uuid.uuid4()),
        version_overrides=overrides or {},
    )


# ── The property the design rests on ─────────────────────────────────


@pytest.mark.asyncio
async def test_corrected_text_reaches_the_engine(client, db, profile, mock_tts):
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen, {"text": "He plays a smash."})
    assert mock_tts[-1]["text"] == "He plays a smash."


@pytest.mark.asyncio
async def test_the_generation_row_keeps_the_original_text(client, db, profile, mock_tts):
    """Rewriting it would leave earlier takes describing audio they did not
    produce, and would make the correction unattributable."""
    gen = await _make_generation(db, profile["id"], text="He plays a bandeja.")
    await _regenerate(db, gen, {"text": "He plays a smash."})

    db.expire_all()
    stored = client.get(f"/history/{gen.id}").json()
    assert stored["text"] == "He plays a bandeja."


@pytest.mark.asyncio
async def test_the_take_records_what_produced_it(client, db, profile, mock_tts):
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen, {"text": "He plays a smash.", "language": "es"})

    versions = client.get(f"/generations/{gen.id}/versions").json()
    latest = versions[-1]
    assert latest["text"] == "He plays a smash."
    assert latest["language"] == "es"


@pytest.mark.asyncio
async def test_an_unoverridden_field_stays_null(client, db, profile, mock_tts):
    """NULL means "same as the generation". Recording the inherited value would
    make "unchanged" and "explicitly set to the same thing" indistinguishable,
    and would need a backfill for every existing row."""
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen, {"text": "Only the text changed."})

    latest = client.get(f"/generations/{gen.id}/versions").json()[-1]
    assert latest["text"] == "Only the text changed."
    assert latest["language"] is None
    assert latest["instruct"] is None


@pytest.mark.asyncio
async def test_a_plain_regenerate_records_nothing(client, db, profile, mock_tts):
    """Unchanged reruns must keep behaving exactly as before."""
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen)

    latest = client.get(f"/generations/{gen.id}/versions").json()[-1]
    assert latest["text"] is None
    assert latest["language"] is None


@pytest.mark.asyncio
async def test_takes_accumulate_rather_than_replace(client, db, profile, mock_tts):
    """The point of correcting in place: the take with the typo survives, so it
    stays available to A/B against."""
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen)
    await _regenerate(db, gen, {"text": "Corrected."})

    versions = client.get(f"/generations/{gen.id}/versions").json()
    assert len(versions) >= 2
    assert versions[-1]["text"] == "Corrected."


# ── Seed handling ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_plain_regenerate_drops_the_seed(client, db, profile, mock_tts):
    """Reusing the seed would reproduce the same take, defeating the point."""
    gen = await _make_generation(db, profile["id"])
    gen.seed = 1234
    db.commit()
    await _regenerate(db, gen)
    assert mock_tts[-1]["seed"] is None


@pytest.mark.asyncio
async def test_an_explicit_seed_survives(client, db, profile, mock_tts):
    """An explicitly requested seed is the caller asking for a specific take,
    usually to reproduce one they liked."""
    gen = await _make_generation(db, profile["id"])
    await _regenerate(db, gen, {"seed": 4242})
    assert mock_tts[-1]["seed"] == 4242


# ── API surface ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regenerate_still_accepts_no_body(client, db, profile, mock_tts):
    """Existing callers post nothing; that must keep working."""
    gen = await _make_generation(db, profile["id"])
    r = client.post(f"/generate/{gen.id}/regenerate")
    assert r.status_code in (200, 400), r.text


def test_unknown_generation_404s(client):
    assert client.post("/generate/no-such-id/regenerate", json={"text": "x"}).status_code == 404


@pytest.mark.asyncio
async def test_an_invalid_language_is_rejected(client, db, profile):
    gen = await _make_generation(db, profile["id"])
    r = client.post(f"/generate/{gen.id}/regenerate", json={"language": "klingon"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_an_empty_text_override_is_rejected(client, db, profile):
    """Omitting text means "reuse it". An empty string is a different request,
    and not a coherent one."""
    gen = await _make_generation(db, profile["id"])
    r = client.post(f"/generate/{gen.id}/regenerate", json={"text": ""})
    assert r.status_code == 422
