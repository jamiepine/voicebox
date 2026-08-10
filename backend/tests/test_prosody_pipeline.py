"""
Tests for the transformer wired into generation.

This is the phase that changes what `/generate` does, so the property under
most scrutiny is the one about *not* changing it: a script with no markup and
no dictionary hits must take the same single-shot call it always did, with the
same arguments.

The model is stubbed. What is being checked is which calls are made with what,
not what an engine does with them.

Usage:
    python -m pytest backend/tests/test_prosody_pipeline.py -v
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-prosody-pipeline-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import PronunciationEntry, get_db  # noqa: E402
from backend.services.prosody.pipeline import build_plan, generate_with_prosody  # noqa: E402

SR = 24000


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
        session.query(PronunciationEntry).delete()
        session.commit()
        session.close()


@pytest.fixture
def spy():
    """Stand in for generate_chunked, recording every call."""
    calls: list[dict] = []

    async def generate_chunked_fn(_model, text, _voice_prompt, **kwargs):
        calls.append({"text": text, **kwargs})
        return np.zeros(SR, dtype=np.float32), SR

    generate_chunked_fn.calls = calls
    return generate_chunked_fn


def add(db, term, replacement, **kwargs):
    entry = PronunciationEntry(term=term, replacement=replacement, **kwargs)
    db.add(entry)
    db.commit()
    return entry


BASE = dict(
    engine="qwen",
    language="en",
    tts_model=object(),
    voice_prompt={},
)


# ── The property that must not change ────────────────────────────────


@pytest.mark.asyncio
async def test_plain_text_takes_the_single_shot_path(spy, db):
    """No markup, no dictionary hits: one call, same text, same arguments."""
    kwargs = dict(language="en", seed=7, instruct=None, max_chunk_chars=800)
    await generate_with_prosody(
        "Just a plain sentence.",
        generate_chunked_fn=spy,
        gen_kwargs=dict(kwargs),
        db=db,
        **BASE,
    )

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["text"] == "Just a plain sentence."
    assert call["seed"] == 7
    assert call["max_chunk_chars"] == 800


@pytest.mark.asyncio
async def test_prose_that_looks_like_markup_is_untouched(spy, db):
    """`5 < 6` must not become a parse error for someone who never used this."""
    await generate_with_prosody(
        "If 5 < 6 then x > y.", generate_chunked_fn=spy, gen_kwargs={}, db=db, **BASE
    )
    assert len(spy.calls) == 1
    assert spy.calls[0]["text"] == "If 5 < 6 then x > y."


@pytest.mark.asyncio
async def test_malformed_markup_falls_back_to_literal_text(spy, db):
    """A stray tag must not be able to fail a generation. Before this feature
    existed the text was literal, so that is what it degrades to."""
    text = 'He said <lang xml:lang="es">hola and left.'
    await generate_with_prosody(
        text, generate_chunked_fn=spy, gen_kwargs={}, db=db, **BASE
    )
    assert len(spy.calls) == 1
    assert spy.calls[0]["text"] == text


@pytest.mark.asyncio
async def test_the_opt_out_speaks_the_text_literally(spy, db):
    """For a script that genuinely contains something shaped like a tag."""
    text = 'Write <break time="700ms"/> to insert a pause.'
    await generate_with_prosody(
        text, generate_chunked_fn=spy, gen_kwargs={}, db=db, enabled=False, **BASE
    )
    assert len(spy.calls) == 1
    assert spy.calls[0]["text"] == text


# ── When it does have work ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_language_span_becomes_separate_calls(spy, db):
    await generate_with_prosody(
        'He plays a <lang xml:lang="es">bandeja alta</lang> here.',
        generate_chunked_fn=spy,
        gen_kwargs=dict(language="en"),
        db=db,
        engine_languages=["en", "es"],
        **{k: v for k, v in BASE.items() if k != "language"},
        language="en",
    )

    assert len(spy.calls) > 1
    languages = {c["language"] for c in spy.calls}
    assert languages == {"en", "es"}


@pytest.mark.asyncio
async def test_a_break_costs_no_generation_call(spy, db):
    """Silence is assembly. The engine never sees a pause."""
    await generate_with_prosody(
        'One.<break time="700ms"/>Two.',
        generate_chunked_fn=spy,
        gen_kwargs={},
        db=db,
        **BASE,
    )
    assert len(spy.calls) == 2
    assert all("break" not in c["text"] for c in spy.calls)


@pytest.mark.asyncio
async def test_chunking_arguments_survive_into_every_run(spy, db):
    """Prosody splits by directive, chunking splits by length. A directive run
    that is still long has to go through both."""
    await generate_with_prosody(
        'a <lang xml:lang="es">uno dos tres</lang> b',
        generate_chunked_fn=spy,
        gen_kwargs=dict(language="en", max_chunk_chars=250, crossfade_ms=30),
        db=db,
        engine_languages=["en", "es"],
        **BASE,
    )
    assert all(c["max_chunk_chars"] == 250 for c in spy.calls)


@pytest.mark.asyncio
async def test_seeds_are_deterministic_across_runs(spy, db):
    """Varied per run so neighbours do not share artefacts, but derived from
    the request seed so the same script renders the same way."""
    markup = 'a <lang xml:lang="es">uno dos tres</lang> b'
    call = dict(
        generate_chunked_fn=spy, gen_kwargs=dict(language="en"), db=db,
        engine_languages=["en", "es"], seed=100, **BASE,
    )
    await generate_with_prosody(markup, **call)
    first = [c["seed"] for c in spy.calls]

    spy.calls.clear()
    await generate_with_prosody(markup, **call)
    assert [c["seed"] for c in spy.calls] == first
    assert len(set(first)) == len(first), "each run should get its own seed"


# ── Dictionary integration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_dictionary_respelling_reaches_the_engine(spy, db):
    add(db, "bandeja", "bandeha")
    await generate_with_prosody(
        "He plays a bandeja.", generate_chunked_fn=spy, gen_kwargs={}, db=db, **BASE
    )
    assert len(spy.calls) == 1, "a respelling must not cut the sentence"
    assert spy.calls[0]["text"] == "He plays a bandeha."


@pytest.mark.asyncio
async def test_a_dictionary_language_entry_cuts_a_run(spy, db):
    add(db, "víbora", "víbora", strategy="language", spoken_language="es")
    await generate_with_prosody(
        "Then the víbora lands.",
        generate_chunked_fn=spy,
        gen_kwargs=dict(language="en"),
        db=db,
        engine_languages=["en", "es"],
        **BASE,
    )
    assert any(c["language"] == "es" for c in spy.calls)


@pytest.mark.asyncio
async def test_no_dictionary_and_no_markup_needs_no_database_work(spy):
    """A caller without a session must still work -- the story path builds
    plans outside a request."""
    await generate_with_prosody(
        "Plain text.", generate_chunked_fn=spy, gen_kwargs={}, db=None, **BASE
    )
    assert spy.calls[0]["text"] == "Plain text."


# ── Plan building ────────────────────────────────────────────────────


def test_build_plan_reports_the_markup_it_used(db):
    add(db, "bandeja", "bandeha")
    plan, markup = build_plan(
        "He plays a bandeja.", engine="qwen", language="en", db=db
    )
    assert 'alias="bandeha"' in markup
    assert plan.is_trivial, "a respelling alone should not need the harness"


def test_build_plan_survives_malformed_markup(db):
    plan, markup = build_plan(
        '<lang xml:lang="es">unclosed', engine="qwen", language="en", db=db
    )
    assert plan.is_trivial
    assert markup == '<lang xml:lang="es">unclosed'


# ── End to end through the API ───────────────────────────────────────


def test_generate_accepts_the_prosody_flag(client):
    """Rejecting the field would break the opt-out before it is used."""
    name = f"Prosody Flag Voice {uuid.uuid4().hex[:8]}"
    profile = client.post("/profiles", json={"name": name, "language": "en"}).json()
    try:
        r = client.post(
            "/generate",
            json={"profile_id": profile["id"], "text": "hello", "prosody": False},
        )
        assert r.status_code == 200, r.text
        client.post(f"/generate/{r.json()['id']}/cancel")
    finally:
        client.delete(f"/profiles/{profile['id']}")
