"""
Tests for LLM-drafted prosody markup and the plan preview.

The model is stubbed throughout. What matters here is not what an LLM says but
what happens to what it says: a suggestion that changed the words must be
thrown away, malformed markup must be thrown away, and a missing model must
degrade to "no help offered" rather than to a broken feature.

Usage:
    python -m pytest backend/tests/test_prosody_llm.py -v
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-prosody-llm-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import PronunciationEntry, get_db  # noqa: E402
from backend.services.prosody import llm_annotate  # noqa: E402
from backend.services.prosody.llm_annotate import (  # noqa: E402
    LLMUnavailableError,
    annotate_with_llm,
    validate_annotation,
)

ORIGINAL = "He plays a bandeja, not a smash."
GOOD = 'He plays a <lang xml:lang="es">bandeja</lang>, not a smash.'


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
def stub_llm(monkeypatch):
    """Replace the model with a scripted sequence of replies."""

    def install(*replies: str):
        calls = {"prompts": []}

        class FakeBackend:
            def is_loaded(self):
                return True

            async def generate(self, prompt, **_kwargs):
                calls["prompts"].append(prompt)
                index = min(len(calls["prompts"]) - 1, len(replies) - 1)
                return replies[index]

        monkeypatch.setattr(llm_annotate, "is_llm_available", lambda *_a, **_k: True)
        monkeypatch.setattr(
            "backend.services.llm.get_llm_model", lambda: FakeBackend()
        )
        return calls

    return install


# ── The invariant ────────────────────────────────────────────────────


def test_a_faithful_annotation_is_accepted():
    assert validate_annotation(ORIGINAL, GOOD) is None


def test_a_rewritten_script_is_rejected():
    """The whole reason the check exists: an LLM quietly rephrasing a line would
    otherwise be discovered only by listening to the audio."""
    tampered = 'He plays a <lang xml:lang="es">bandeja</lang>, obviously not a smash.'
    assert "changed the words" in validate_annotation(ORIGINAL, tampered)


def test_a_dropped_word_is_rejected():
    assert validate_annotation(ORIGINAL, "He plays a bandeja.") is not None


def test_malformed_markup_is_rejected():
    """Passing it through would have the engine read the tag aloud."""
    assert "malformed" in validate_annotation(ORIGINAL, f'<lang xml:lang="es">{ORIGINAL}')


@pytest.mark.parametrize("candidate", ["", "   "])
def test_an_empty_answer_is_rejected(candidate):
    assert validate_annotation(ORIGINAL, candidate) is not None


def test_reflowed_whitespace_is_accepted():
    """Putting a tag on its own line is formatting, not content."""
    assert validate_annotation("a b", 'a\n<break time="1s"/>\n b') is None


# ── Wrappers small models add ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fenced_answer_is_unwrapped(stub_llm):
    stub_llm(f"```xml\n{GOOD}\n```")
    result = await annotate_with_llm(ORIGINAL)
    assert result.accepted
    assert result.markup == GOOD


@pytest.mark.asyncio
async def test_a_quoted_answer_is_unwrapped(stub_llm):
    stub_llm(f'"{GOOD}"')
    result = await annotate_with_llm(ORIGINAL)
    assert result.accepted
    assert result.markup == GOOD


@pytest.mark.asyncio
async def test_cleaning_never_repairs_a_word_change(stub_llm):
    """Only decoration is stripped. Anything that would alter the words has to
    reach the invariant rather than being quietly fixed up."""
    stub_llm("```\nHe plays a totally different sentence.\n```")
    result = await annotate_with_llm(ORIGINAL)
    assert not result.accepted


# ── Retry and rejection ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_bad_first_answer_is_retried(stub_llm):
    """The usual failure is prose around the answer rather than a
    misunderstanding, so a second attempt with the complaint fed back lands."""
    calls = stub_llm("Sure! Here is your annotated script.", GOOD)
    result = await annotate_with_llm(ORIGINAL)

    assert result.accepted
    assert result.attempts == 2
    assert "rejected" in calls["prompts"][1], "the complaint should be fed back"


@pytest.mark.asyncio
async def test_persistent_failure_returns_the_original(stub_llm):
    """`markup` is always safe to use, so a caller can apply it unconditionally
    and read the reason only to explain why nothing changed."""
    stub_llm("nonsense", "still nonsense")
    result = await annotate_with_llm(ORIGINAL)

    assert not result.accepted
    assert result.markup == ORIGINAL
    assert result.rejected_reason
    assert not result.changed


@pytest.mark.asyncio
async def test_an_llm_error_is_not_fatal(stub_llm, monkeypatch):
    class Exploding:
        def is_loaded(self):
            return True

        async def generate(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(llm_annotate, "is_llm_available", lambda *_a, **_k: True)
    monkeypatch.setattr("backend.services.llm.get_llm_model", lambda: Exploding())

    result = await annotate_with_llm(ORIGINAL)
    assert not result.accepted
    assert result.markup == ORIGINAL


@pytest.mark.asyncio
async def test_an_unchanged_script_is_accepted(stub_llm):
    """Nothing to annotate is a valid answer, not a failure."""
    stub_llm(ORIGINAL)
    result = await annotate_with_llm(ORIGINAL)
    assert result.accepted
    assert not result.changed


# ── Without an LLM ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_model_raises_rather_than_downloading(monkeypatch):
    """Annotation is optional help. A feature that silently pulls gigabytes the
    first time it is used is not optional."""
    monkeypatch.setattr(llm_annotate, "is_llm_available", lambda *_a, **_k: False)
    with pytest.raises(LLMUnavailableError):
        await annotate_with_llm(ORIGINAL)


def test_availability_is_reported(client, monkeypatch):
    monkeypatch.setattr(llm_annotate, "is_llm_available", lambda *_a, **_k: False)
    body = client.get("/prosody/annotate/availability").json()
    assert body["available"] is False


def test_the_endpoint_409s_without_a_model(client, monkeypatch):
    monkeypatch.setattr(llm_annotate, "is_llm_available", lambda *_a, **_k: False)
    r = client.post("/prosody/annotate", json={"text": ORIGINAL})
    assert r.status_code == 409
    assert "not downloaded" in r.json()["detail"]


@pytest.mark.asyncio
async def test_blank_text_needs_no_model():
    result = await annotate_with_llm("   ")
    assert result.accepted


# ── Preview ──────────────────────────────────────────────────────────


def test_preview_shows_the_compiled_plan(client, db):
    r = client.post(
        "/prosody/preview",
        json={
            "text": 'One.<break time="700ms"/><lang xml:lang="es">Dos tres cuatro.</lang>',
            "engine": "qwen",
            "language": "en",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    kinds = [n["kind"] for n in body["nodes"]]
    assert kinds == ["speech", "silence", "speech"]
    assert body["nodes"][1]["ms"] == 700
    assert body["nodes"][2]["language"] == "es"
    assert body["run_count"] == 2
    assert body["is_trivial"] is False


def test_preview_resolves_dictionary_entries_into_markup(client, db):
    """The dictionary emits the same directives an author would type, so the
    preview can show them rather than only their effect."""
    client.post(
        "/pronunciations", json={"term": "bandeja", "replacement": "bandeha"}
    )
    body = client.post(
        "/prosody/preview", json={"text": "He plays a bandeja.", "language": "en"}
    ).json()

    assert 'alias="bandeha"' in body["markup"]
    assert body["dictionary_terms"] == ["bandeja"]
    assert body["nodes"][0]["source_text"] == "He plays a bandeja."


def test_preview_reports_what_the_engine_cannot_do(client, db):
    """Base Qwen ignores delivery instructions, and saying so is the difference
    between a limitation and a model that looks like it is refusing."""
    body = client.post(
        "/prosody/preview",
        json={"text": "<emphasis>wow</emphasis> there", "engine": "qwen"},
    ).json()
    assert any(w["code"] == "emphasis_unsupported" for w in body["warnings"])


def test_preview_marks_plain_text_as_trivial(client, db):
    body = client.post(
        "/prosody/preview", json={"text": "Just a plain sentence."}
    ).json()
    assert body["is_trivial"] is True
    assert body["run_count"] == 1


def test_preview_rejects_malformed_markup(client, db):
    r = client.post(
        "/prosody/preview", json={"text": '<lang xml:lang="es">unclosed'}
    )
    assert r.status_code == 400


def test_preview_scopes_the_dictionary_to_the_profile(client, db):
    """A per-voice entry must not leak into a preview for another voice."""
    name = f"Prosody Preview Voice {uuid.uuid4().hex[:8]}"
    profile = client.post("/profiles", json={"name": name, "language": "en"}).json()
    try:
        client.post(
            "/pronunciations",
            json={
                "term": "bandeja",
                "replacement": "PER-VOICE",
                "profile_id": profile["id"],
            },
        )
        scoped = client.post(
            "/prosody/preview",
            json={"text": "a bandeja", "profile_id": profile["id"]},
        ).json()
        assert "PER-VOICE" in scoped["markup"]

        unscoped = client.post("/prosody/preview", json={"text": "a bandeja"}).json()
        assert "PER-VOICE" not in unscoped["markup"]
    finally:
        client.delete(f"/profiles/{profile['id']}")
