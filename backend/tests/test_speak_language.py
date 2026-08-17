"""Tests for voice-profile language fallback on the two speak surfaces.

Both speak paths built their ``GenerationRequest`` with a hardcoded ``"en"``
fallback and never consulted the resolved profile, so a profile created with
``language="fr"`` was still synthesised as English unless every caller passed
``language=`` explicitly. Agents going through MCP had no way to know the
profile's language, so they couldn't pass it either.

These tests pin the fix: the fallback chain is now explicit argument →
resolved profile's language → ``"en"``, matching how ``engine`` and
``personality`` already consult the resolved binding.
"""

import pytest

import backend.routes.generations as generations
import backend.routes.speak as speak_route
from backend import models
from backend.mcp_server import tools


class _FakeGeneration:
    """Minimal stand-in for GenerationResponse consumed by the speak paths."""

    id = "gen-test"
    status = "generating"

    def model_dump(self, mode="json"):
        return {"id": self.id, "status": self.status}


class _FakeProfile:
    def __init__(self, language):
        self.id = "p1"
        self.name = "Siwis"
        self.language = language
        self.personality = None


class _FakeQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        # No per-client binding — engine/personality fall through to their
        # own defaults, leaving language as the only variable under test.
        return None


class _FakeDB:
    def query(self, *args, **kwargs):
        return _FakeQuery()

    def close(self):
        pass


class _FakeRequest:
    """Stands in for starlette's Request — only headers are read."""

    def __init__(self, client_id=None):
        self.headers = {"X-Voicebox-Client-Id": client_id} if client_id else {}


@pytest.fixture
def captured_request(monkeypatch):
    """Capture the GenerationRequest instead of running a real generation.

    Both speak paths import ``generate_speech`` lazily from
    ``routes.generations``, so patching the attribute on that module
    intercepts the call on either surface.
    """
    captured = {}

    async def fake_generate_speech(req, db):
        captured["req"] = req
        return _FakeGeneration()

    monkeypatch.setattr(generations, "generate_speech", fake_generate_speech)
    monkeypatch.setattr(speak_route.mcp_events, "publish", lambda *a, **k: None)
    monkeypatch.setattr(tools.mcp_events, "publish", lambda *a, **k: None)
    return captured


# ─── REST: POST /speak ────────────────────────────────────────────────────


async def _call_rest(monkeypatch, profile_language, requested_language=None):
    monkeypatch.setattr(
        speak_route,
        "resolve_profile",
        lambda profile, client_id, db: _FakeProfile(profile_language),
    )
    await speak_route.speak(
        models.SpeakRequest(text="Bonjour", language=requested_language),
        _FakeRequest(client_id="claude-code"),
        _FakeDB(),
    )


async def test_rest_speak_falls_back_to_profile_language(
    captured_request, monkeypatch
):
    await _call_rest(monkeypatch, profile_language="fr")
    assert captured_request["req"].language == "fr"


async def test_rest_speak_explicit_language_wins(captured_request, monkeypatch):
    # An explicit argument still overrides the profile — a French profile can
    # be asked to read an English string.
    await _call_rest(monkeypatch, profile_language="fr", requested_language="en")
    assert captured_request["req"].language == "en"


async def test_rest_speak_defaults_to_en_without_profile_language(
    captured_request, monkeypatch
):
    # Profiles predating the language column resolve to None; the "en"
    # backstop keeps their behaviour unchanged.
    await _call_rest(monkeypatch, profile_language=None)
    assert captured_request["req"].language == "en"


# ─── MCP: voicebox.speak ──────────────────────────────────────────────────


async def _call_mcp(monkeypatch, profile_language, requested_language=None):
    from mcp.server.fastmcp import FastMCP

    monkeypatch.setattr(
        tools,
        "resolve_profile",
        lambda profile, client_id, db: _FakeProfile(profile_language),
    )
    monkeypatch.setattr(tools, "get_db", lambda: iter([_FakeDB()]))

    mcp = FastMCP("test")
    tools.register_tools(mcp)
    args = {"text": "Bonjour"}
    if requested_language is not None:
        args["language"] = requested_language
    await mcp.call_tool("voicebox.speak", args)


async def test_mcp_speak_falls_back_to_profile_language(
    captured_request, monkeypatch
):
    # The agent-facing path matters most: an MCP client can't know the
    # profile's language, so omitting it must not silently mean English.
    await _call_mcp(monkeypatch, profile_language="fr")
    assert captured_request["req"].language == "fr"


async def test_mcp_speak_explicit_language_wins(captured_request, monkeypatch):
    await _call_mcp(monkeypatch, profile_language="fr", requested_language="en")
    assert captured_request["req"].language == "en"


async def test_mcp_speak_defaults_to_en_without_profile_language(
    captured_request, monkeypatch
):
    await _call_mcp(monkeypatch, profile_language=None)
    assert captured_request["req"].language == "en"
