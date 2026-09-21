"""Shared fixtures for the voice-agent test modules.

An in-memory SQLite database wired into every ``get_db`` the services
captured at import time, plus fake LLM / TTS stand-ins so the
conversation machinery runs without torch or any model download.
"""

from __future__ import annotations

import sys
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import models
from backend.database.models import Base, VoiceProfile
from backend.services import voice_agent as core


@pytest.fixture
def db(monkeypatch):
    """In-memory SQLite session, also wired into ``backend.database.get_db``
    so the routes / runner see the same tables."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _get_db():
        s = session_local()
        try:
            yield s
        finally:
            s.close()

    import backend.database as database_pkg
    import backend.database.session as session_mod

    monkeypatch.setattr(session_mod, "SessionLocal", session_local)
    monkeypatch.setattr(database_pkg, "get_db", _get_db)
    monkeypatch.setattr(session_mod, "get_db", _get_db)
    from backend.services import voice_agent_runner as runner_mod, voice_agent_webhooks as webhooks_mod

    monkeypatch.setattr(runner_mod, "get_db", _get_db)
    monkeypatch.setattr(webhooks_mod, "get_db", _get_db)
    session = session_local()
    session.add(VoiceProfile(id="prof-1", name="Morgan", language="en", voice_type="cloned"))
    session.commit()
    yield session
    session.close()


class FakeLLM:
    """Scripted LLM: returns the next canned reply, records prompts."""

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[dict] = []
        self.model_size = "0.6B"

    def is_loaded(self):
        return True

    def _is_model_cached(self, size):
        return True

    async def generate(self, prompt, system=None, max_tokens=256, temperature=0.7, model_size=None, examples=None):
        self.calls.append({"prompt": prompt, "system": system, "examples": examples, "model_size": model_size})
        if self.replies:
            return self.replies.pop(0)
        return "Sure, happy to help with that. What would work best for you?"


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    fake_module = types.SimpleNamespace(get_llm_model=lambda: llm, unload_llm_model=lambda: None)
    monkeypatch.setitem(sys.modules, "backend.services.llm", fake_module)
    return llm


@pytest.fixture
def fake_tts(monkeypatch):
    """Replace TTS with something that records text and hands back fake
    generation ids (one per piece), so chunking and events can be asserted."""
    spoken: list[str] = []
    counter = {"n": 0}

    async def _generate(agent, text, db, *, source, instruct=None, language=None):
        from backend.database import Generation

        spoken.append(text)
        counter["n"] += 1
        gen_id = f"gen-{counter['n']}"
        # The real path leaves a Generation row behind; keep that contract.
        db.add(
            Generation(
                id=gen_id, profile_id=agent.profile_id, text=text, audio_path="", status="generating", source=source
            )
        )
        db.commit()
        return gen_id

    monkeypatch.setattr(core, "_generate", _generate)
    return spoken


def _agent_payload(**overrides) -> models.VoiceAgentCreate:
    base = dict(
        name="Acme Outbound",
        mode="outbound_sales",
        profile="Morgan",
        agent_name="Sam",
        company_name="Acme Solar",
        brief="Acme installs residential solar panels. 25-year warranty. Free survey. Typical saving 40% on bills.",
        goal="Book a free home survey.",
        objection_notes="Too expensive → mention 0% finance.",
        timezone="UTC",
        calling_window_start=0,
        calling_window_end=24,
        calling_days=[0, 1, 2, 3, 4, 5, 6],
    )
    base.update(overrides)
    return models.VoiceAgentCreate(**base)


@pytest.fixture
def agent(db):
    return core.create_agent(_agent_payload(), db)


@pytest.fixture
def support_agent(db):
    return core.create_agent(
        _agent_payload(
            name="Acme Support",
            mode="support",
            goal="Fix the customer's router issue or open a ticket.",
            brief="Acme provides home broadband.",
            escalation_promise="an engineer will call you back within one business day",
        ),
        db,
    )


def _contact(db, agent, phone="+447700900001", name="Jane Doe", **kw):
    return core.create_contact(agent.id, models.ContactCreate(name=name, phone=phone, **kw), db)
