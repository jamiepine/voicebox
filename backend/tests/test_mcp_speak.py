"""Tests for the voicebox.speak MCP tool's ``model_size`` plumbing (issue #884).

The MCP speak path used to build its ``GenerationRequest`` without a
``model_size``, so every agent-triggered generation silently fell back to the
schema default ("1.7B") — there was no way to reach 0.6B (or TADA's 1B/3B)
through MCP. These tests pin the fix: ``_speak`` now forwards ``model_size``
straight into the request, matching the REST ``/generate`` surface.
"""

import pytest
from pydantic import ValidationError

import backend.routes.generations as generations
from backend.mcp_server import tools


class _FakeGeneration:
    """Minimal stand-in for GenerationResponse consumed by ``_speak_response``."""

    def model_dump(self, mode="json"):
        return {"id": "gen-test", "status": "generating"}


@pytest.fixture
def captured_request(monkeypatch):
    """Replace the real (torch-backed) generate_speech with a capturing stub.

    ``_speak`` imports ``generate_speech`` lazily from ``routes.generations``,
    so patching the attribute on that module intercepts the call and lets us
    inspect the ``GenerationRequest`` it would have run.
    """
    captured = {}

    async def fake_generate_speech(req, db):
        captured["req"] = req
        return _FakeGeneration()

    monkeypatch.setattr(generations, "generate_speech", fake_generate_speech)
    # Isolate the unit from the MCP event bus — _speak_response fires a
    # speak-start event we don't care about here.
    monkeypatch.setattr(tools.mcp_events, "publish", lambda *a, **k: None)
    return captured


@pytest.mark.asyncio
async def test_speak_forwards_explicit_model_size(captured_request):
    await tools._speak(
        profile_id="p1",
        profile_name="Morgan",
        text="hello",
        engine="qwen",
        language="en",
        personality=False,
        model_size="0.6B",
        db=None,
    )
    assert captured_request["req"].model_size == "0.6B"


@pytest.mark.asyncio
async def test_speak_omitted_model_size_is_none(captured_request):
    # Omitted → None; generate_speech normalizes None to the engine default,
    # so this reproduces the pre-fix behaviour for callers that don't ask.
    await tools._speak(
        profile_id="p1",
        profile_name="Morgan",
        text="hello",
        engine="qwen",
        language="en",
        personality=False,
        db=None,
    )
    assert captured_request["req"].model_size is None


@pytest.mark.asyncio
async def test_speak_rejects_invalid_model_size(captured_request):
    # The GenerationRequest schema pattern is the single source of truth for
    # valid sizes; a bad value is rejected before any generation runs.
    with pytest.raises(ValidationError):
        await tools._speak(
            profile_id="p1",
            profile_name="Morgan",
            text="hello",
            engine="qwen",
            language="en",
            personality=False,
            model_size="9B",
            db=None,
        )
    assert "req" not in captured_request
