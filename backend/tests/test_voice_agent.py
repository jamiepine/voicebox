"""Unit + API tests for the voice AI agent (services/voice_agent*.py,
routes/voice_agent.py).

The LLM and TTS are replaced with fakes — the point is to pin the
conversation state machine, the compliance guard-rails (DNC, calling
window, consent, opt-out), knowledge retrieval, and the REST surface.
Runs on an in-memory SQLite database; no models, no torch.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import models
from backend.database.models import Base, VoiceProfile
from backend.services import voice_agent as core, voice_agent_knowledge as kb, voice_agent_prompts as prompts

# ── Fixtures ───────────────────────────────────────────────────────────


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
    from backend.services import voice_agent_runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_db", _get_db)
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
    """Replace generate_speech with something that just records text."""
    spoken: list[str] = []

    async def _voice(agent, call, text, db):
        spoken.append(text)
        return

    monkeypatch.setattr(core, "_voice", _voice)
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


# ── Phone normalisation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+44 (0)20 7946 0958", "+442079460958"),
        ("020 7946 0958", "02079460958"),
        ("+1 (555) 010-9999", "+15550109999"),
        ("555.010.9999", "5550109999"),
    ],
)
def test_normalize_phone(raw, expected):
    assert core.normalize_phone(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "12", "+"])
def test_normalize_phone_rejects_garbage(raw):
    with pytest.raises(ValueError, match="Phone number"):
        core.normalize_phone(raw)


# ── Prompts / NLU ──────────────────────────────────────────────────────


def test_opening_line_always_carries_disclosure():
    line = prompts.build_opening_line(
        mode="outbound_sales",
        agent_name="Sam",
        company_name="Acme",
        disclosure="I'm an AI assistant.",
        contact_name="Jane Doe",
        custom_opening="Quick question about your energy bill?",
    )
    assert line.startswith("Hi Jane, this is Sam calling from Acme.")
    assert "I'm an AI assistant." in line
    assert line.endswith("Quick question about your energy bill?")


def test_system_prompt_contains_guardrails_and_sections():
    p = prompts.build_system_prompt(
        mode="support",
        agent_name="Sam",
        company_name="Acme",
        brief="BRIEF-TEXT",
        goal="GOAL-TEXT",
        knowledge=[("Router reboot", "Hold the button for 10 seconds.")],
        contact_memory="Called last week.",
    )
    assert "You are an AI assistant" in p
    assert "Only state facts that appear in the BRIEF or the KNOWLEDGE" in p
    assert "## Router reboot" in p
    assert "Previous conversations: Called last week." in p
    assert "[TICKET: short subject]" in p
    with pytest.raises(ValueError, match="Unknown agent mode"):
        prompts.build_system_prompt(mode="nope", agent_name="a", company_name="b", brief="c", goal="d")


@pytest.mark.parametrize(
    ("raw", "mode", "text", "outcome", "ticket", "handoff"),
    [
        (
            "Great, I'll book that in. [OUTCOME: interested]",
            "outbound_sales",
            "Great, I'll book that in.",
            "interested",
            None,
            False,
        ),
        (
            '<think>plan</think>Agent: "No problem." [OUTCOME: callback]',
            "outbound_sales",
            "No problem.",
            "callback",
            None,
            False,
        ),
        (
            "I'll log that. [TICKET: router drops wifi nightly]",
            "support",
            "I'll log that.",
            None,
            "router drops wifi nightly",
            False,
        ),
        ("Let me get someone. [HANDOFF]", "customer_service", "Let me get someone.", None, None, True),
        ("Ok. [OUTCOME: interested]", "support", "Ok.", None, None, False),  # not allowed in support → ignored
        ("*smiles* Sure thing (leans in)", "support", "Sure thing", None, None, False),
    ],
)
def test_parse_agent_reply(raw, mode, text, outcome, ticket, handoff):
    r = prompts.parse_agent_reply(raw, mode)
    assert r.text == text
    assert r.outcome == outcome
    assert r.ticket_subject == ticket
    assert r.handoff is handoff


@pytest.mark.parametrize(
    "text",
    [
        "Please stop calling me",
        "take me off your list",
        "put me on the do not call list",
        "don't call again",
        "unsubscribe me",
    ],
)
def test_detect_opt_out(text):
    assert prompts.detect_opt_out(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to speak to a real person", True),
        ("get me a manager now", True),
        ("are you a bot?", False),
        ("thanks, that's helpful", False),
    ],
)
def test_detect_human_request(text, expected):
    assert prompts.detect_human_request(text) is expected


def test_sentiment_direction():
    assert prompts.score_sentiment("this is ridiculous, absolutely useless, I'm furious") < -0.3
    assert prompts.score_sentiment("that worked perfectly, thanks so much") > 0.3
    assert prompts.score_sentiment("not interested thanks") <= 0.1
    assert prompts.score_sentiment("") == 0.0


# ── Knowledge retrieval ────────────────────────────────────────────────


def test_knowledge_ranking_prefers_title_and_tag_hits(db, support_agent):
    core.create_article(
        support_agent.id,
        models.KnowledgeArticleCreate(
            title="Password reset",
            content="Go to settings and choose reset. A link is emailed within 5 minutes.",
            tags=["account"],
        ),
        db,
    )
    core.create_article(
        support_agent.id,
        models.KnowledgeArticleCreate(
            title="Router keeps rebooting",
            content="Check the power supply, then hold reset for 10 seconds.",
            tags=["wifi", "router"],
        ),
        db,
    )
    core.create_article(
        support_agent.id,
        models.KnowledgeArticleCreate(
            title="Billing dates", content="Invoices go out on the 1st. Direct debit on the 5th.", tags=["billing"]
        ),
        db,
    )

    hits = kb.retrieve_for_turn(db, support_agent.id, ["my router reboots every night"])
    assert hits
    assert hits[0][0] == "Router keeps rebooting"
    hits = kb.retrieve_for_turn(db, support_agent.id, ["I forgot my password"])
    assert hits
    assert hits[0][0] == "Password reset"
    assert kb.retrieve_for_turn(db, support_agent.id, []) == []
    assert kb.retrieve_for_turn(db, support_agent.id, ["zzz qqq"]) == []


# ── Scheduling guard-rails ─────────────────────────────────────────────


def test_pick_next_contact_respects_status_window_dnc_consent(db, agent):
    c1 = _contact(db, agent, "+447700900001", "Alpha")
    c2 = _contact(db, agent, "+447700900002", "Bravo")
    core.add_to_dnc("+44 7700 900002", db, reason="asked")
    db.refresh(c2)
    assert c2.status == "do_not_call"

    # Draft agent → nothing.
    assert core.pick_next_contact(agent, db) is None
    core.set_agent_status(agent.id, "active", db)
    assert core.pick_next_contact(agent, db).id == c1.id

    # Consent required → c1 has none.
    agent.require_consent = True
    db.commit()
    assert core.pick_next_contact(agent, db) is None
    c1.consent = True
    db.commit()
    assert core.pick_next_contact(agent, db).id == c1.id

    # Outside the calling window (Sunday 03:00 in the contact's zone).
    agent.calling_window_start, agent.calling_window_end, agent.calling_days = 9, 17, [0, 1, 2, 3, 4]
    c1.timezone = "Europe/London"
    db.commit()
    sunday_night = datetime(2026, 9, 6, 2, 0, tzinfo=UTC)  # Sunday 03:00 BST
    assert core.pick_next_contact(agent, db, sunday_night) is None
    monday_morning = datetime(2026, 9, 7, 9, 0, tzinfo=UTC)  # Monday 10:00 BST
    assert core.pick_next_contact(agent, db, monday_morning).id == c1.id


def test_pick_next_contact_attempts_cap_retry_timing_and_daily_cap(db, agent):
    core.set_agent_status(agent.id, "active", db)
    c = _contact(db, agent, "+447700900010", "Cap")
    c.attempts = agent.max_attempts
    db.commit()
    assert core.pick_next_contact(agent, db) is None

    c.attempts = 0
    c.next_attempt_at = core.utcnow() + timedelta(hours=1)
    db.commit()
    assert core.pick_next_contact(agent, db) is None
    c.next_attempt_at = core.utcnow() - timedelta(minutes=1)
    db.commit()
    assert core.pick_next_contact(agent, db).id == c.id

    agent.daily_call_cap = 1
    db.commit()
    from backend.database import Call

    db.add(Call(agent_id=agent.id, contact_id=c.id))
    db.commit()
    assert core.pick_next_contact(agent, db) is None


def test_bulk_import_and_csv_parsing(db, agent):
    csv_text = "Full Name,Phone Number,Company,Opted In\nAda,+44 7700 900100,Lovelace Ltd,yes\nBob,not a number,,no\nAda again,+447700900100,,\n"
    items = core.parse_contacts_csv(csv_text)
    assert len(items) == 3
    result = core.bulk_create_contacts(agent.id, items, db)
    assert result.imported == 1
    assert result.skipped_reasons == {"invalid_phone": 1, "duplicate": 1}
    rows, total = core.list_contacts(agent.id, db)
    assert total == 1
    assert rows[0].consent is True
    assert rows[0].company == "Lovelace Ltd"
    with pytest.raises(ValueError, match="phone column"):
        core.parse_contacts_csv("name,email\nx,y\n")


# ── Conversation state machine ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_outbound_call_happy_path_books_interest(db, agent, fake_llm, fake_tts):
    fake_llm.replies = [
        "Of course. We install panels with a 25-year warranty and the survey is free. Would a weekday morning suit you?",
        "Perfect, I'll book the survey for Tuesday morning. [OUTCOME: interested]",
    ]
    contact = _contact(db, agent, "+447700900020", "Jane Doe")
    core.set_agent_status(agent.id, "active", db)

    start = await core.start_outbound_call(agent, contact, db)
    assert start.ended is False
    assert "Hi Jane, this is Sam calling from Acme Solar." in start.agent_turn.text
    assert "automated AI assistant" in start.agent_turn.text
    assert fake_tts == [start.agent_turn.text]
    db.refresh(contact)
    assert contact.status == "calling"
    assert contact.attempts == 1

    r1 = await core.handle_customer_turn(start.call, "Sure, what's this about?", db)
    assert r1.ended is False
    assert r1.agent_turn.text.startswith("Of course.")
    # The LLM saw the brief, the contact, and the opening line as history.
    assert "25-year warranty" in fake_llm.calls[0]["system"]
    assert fake_llm.calls[0]["examples"][0][0] == "(call connected)"
    assert fake_llm.calls[0]["prompt"] == "Sure, what's this about?"

    r2 = await core.handle_customer_turn(start.call, "Tuesday morning works, sign me up", db)
    assert r2.ended is True
    assert r2.outcome == "interested"
    assert r2.agent_turn.text == "Perfect, I'll book the survey for Tuesday morning."
    db.refresh(contact)
    assert contact.status == "interested"
    tickets, _ = core.list_tickets(db, agent_id=agent.id)
    assert tickets
    assert tickets[0].kind == "sales_lead"
    call = core.get_call(start.call.id, db)
    assert call.status == "completed"
    assert call.turn_count == 5
    # Summary written into contact memory for next time.
    assert call.summary
    assert contact.memory
    assert call.summary in contact.memory


@pytest.mark.asyncio
async def test_opt_out_ends_call_and_blocks_number_everywhere(db, agent, fake_llm, fake_tts):
    contact = _contact(db, agent, "+447700900030", "Opt Out")
    other_agent = core.create_agent(_agent_payload(name="Other"), db)
    twin = _contact(db, other_agent, "+447700900030", "Same person")
    start = await core.start_outbound_call(agent, contact, db)
    r = await core.handle_customer_turn(start.call, "No. Take me off your list and stop calling.", db)
    assert r.ended
    assert r.outcome == "opt_out"
    assert "do-not-call list" in r.agent_turn.text
    # The model was never asked for a reply — only for the post-call summary.
    assert all(c["system"] == prompts.SUMMARY_SYSTEM for c in fake_llm.calls)
    assert core.is_blocked("+447700900030", db)
    db.refresh(contact)
    db.refresh(twin)
    assert contact.status == "do_not_call"
    assert twin.status == "do_not_call"
    with pytest.raises(ValueError, match="do-not-call"):
        await core.start_outbound_call(other_agent, twin, db)


@pytest.mark.asyncio
async def test_support_call_creates_ticket_from_llm_tag(db, support_agent, fake_llm, fake_tts):
    fake_llm.replies = ["I've logged this for an engineer. [TICKET: router reboots nightly]"]
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900040", "Caller", db)
    start = await core.start_inbound_call(support_agent, contact, db)
    assert "you've reached Acme Solar, this is Sam" in start.agent_turn.text
    r = await core.handle_customer_turn(start.call, "My router reboots every night around 2am", db)
    assert r.ended
    assert r.outcome == "ticket_created"
    assert r.ticket is not None
    assert r.ticket.subject == "router reboots nightly"
    assert r.ticket.kind == "support"
    assert "customer: My router reboots" in r.ticket.description
    db.refresh(contact)
    assert contact.status == "unresolved"
    assert contact.consent is True


@pytest.mark.asyncio
async def test_human_request_hands_off_without_model(db, support_agent, fake_llm, fake_tts):
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900050", None, db)
    start = await core.start_inbound_call(support_agent, contact, db)
    r = await core.handle_customer_turn(start.call, "I want to speak to a real person please", db)
    assert r.ended
    assert r.outcome == "handoff"
    assert "engineer will call you back" in r.agent_turn.text
    assert r.ticket.kind == "handoff"
    assert all(c["system"] == prompts.SUMMARY_SYSTEM for c in fake_llm.calls)


@pytest.mark.asyncio
async def test_negative_streak_escalates(db, support_agent, fake_llm, fake_tts):
    support_agent.handoff_after_negative_turns = 2
    db.commit()
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900060", None, db)
    start = await core.start_inbound_call(support_agent, contact, db)
    r1 = await core.handle_customer_turn(start.call, "This is absolutely ridiculous and useless", db)
    assert not r1.ended
    r2 = await core.handle_customer_turn(start.call, "I'm furious, this is a terrible awful joke", db)
    assert r2.ended
    assert r2.outcome == "handoff"
    assert r2.ticket.priority == "high"


@pytest.mark.asyncio
async def test_goodbye_resolves_service_call(db, support_agent, fake_llm, fake_tts):
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900070", None, db)
    start = await core.start_inbound_call(support_agent, contact, db)
    await core.handle_customer_turn(start.call, "How do I reset my router?", db)
    r = await core.handle_customer_turn(start.call, "Thanks, that's all", db)
    assert r.ended
    assert r.outcome == "resolved"
    db.refresh(contact)
    assert contact.status == "resolved"


@pytest.mark.asyncio
async def test_callback_reschedules_and_voicemail_retries(db, agent, fake_llm, fake_tts):
    fake_llm.replies = ["No problem, I'll call back tomorrow. [OUTCOME: callback]"]
    contact = _contact(db, agent, "+447700900080", "Busy")
    start = await core.start_outbound_call(agent, contact, db)
    r = await core.handle_customer_turn(start.call, "Bad time, can you call me back tomorrow?", db)
    assert r.ended
    assert r.outcome == "callback"
    db.refresh(contact)
    assert contact.status == "callback"
    assert contact.next_attempt_at > core.utcnow() + timedelta(hours=agent.callback_delay_hours - 1)

    vm = _contact(db, agent, "+447700900081", "Machine")
    start = await core.start_outbound_call(agent, vm, db)
    r = await core.handle_customer_turn(start.call, "Hi, you've reached Bob. Leave a message after the tone.", db)
    assert r.ended
    assert r.outcome == "voicemail"
    db.refresh(vm)
    assert vm.status == "new"
    assert vm.next_attempt_at is not None
    assert vm.attempts == 1


@pytest.mark.asyncio
async def test_max_turns_closes_with_ticket(db, support_agent, fake_llm, fake_tts):
    support_agent.max_turns = 4
    db.commit()
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900090", None, db)
    start = await core.start_inbound_call(support_agent, contact, db)
    await core.handle_customer_turn(start.call, "it's broken", db)  # turns: 3
    r = await core.handle_customer_turn(start.call, "still broken", db)  # turn 4 → cap
    assert r.ended
    assert r.outcome == "max_turns"
    assert r.ticket is not None


@pytest.mark.asyncio
async def test_llm_failure_fails_safe(db, support_agent, fake_tts, monkeypatch):
    class Boom(FakeLLM):
        async def generate(self, *a, **k):
            raise RuntimeError("gpu fell over")

    monkeypatch.setitem(sys.modules, "backend.services.llm", types.SimpleNamespace(get_llm_model=lambda: Boom()))
    contact = core.get_or_create_inbound_contact(support_agent.id, "+447700900095", None, db)
    start = await core.start_inbound_call(support_agent, contact, db)
    r = await core.handle_customer_turn(start.call, "hello?", db)
    assert r.ended
    assert r.outcome == "error"
    assert r.ticket.priority == "high"
    call = core.get_call(start.call.id, db)
    assert call.status == "failed"


@pytest.mark.asyncio
async def test_stats(db, agent, fake_llm, fake_tts):
    fake_llm.replies = ["Booked. [OUTCOME: interested]"]
    c = _contact(db, agent, "+447700900100", "Stat")
    start = await core.start_outbound_call(agent, c, db)
    await core.handle_customer_turn(start.call, "yes please", db)
    s = core.agent_stats(agent, db)
    assert s.calls_total == 1
    assert s.calls_by_outcome == {"interested": 1}
    assert s.contacts_by_status == {"interested": 1}
    assert s.resolution_rate == 1.0
    assert s.open_tickets == 1
    assert s.calls_today == 1


# ── Runner ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runner_dials_then_completes_when_list_exhausted(db, agent, fake_llm, fake_tts, monkeypatch):
    from backend.services import voice_agent_runner as runner

    monkeypatch.setattr(runner, "CALL_POLL_S", 0.01)
    _contact(db, agent, "+447700900200", "Only")
    core.set_agent_status(agent.id, "active", db)

    # Close every call as soon as it opens, as an API caller would.
    async def _close_open_calls():
        from backend.database import Call

        for _ in range(200):
            import asyncio

            await asyncio.sleep(0.01)
            s = next(core_get_db())
            try:
                for call in s.query(Call).filter(Call.status == "in_progress").all():
                    await core.end_call(call, "not_interested", s)
            finally:
                s.close()

    import asyncio

    import backend.database as database_pkg

    core_get_db = database_pkg.get_db
    closer = asyncio.create_task(_close_open_calls())
    runner.start(agent.id, idle_timeout_s=5)
    assert runner.is_running(agent.id)
    for _ in range(300):
        await asyncio.sleep(0.01)
        if not runner.is_running(agent.id):
            break
    closer.cancel()
    assert not runner.is_running(agent.id)
    db.expire_all()
    assert core.get_agent(agent.id, db).status == "completed"
    calls, total = core.list_calls(agent.id, db)
    assert total == 1
    assert calls[0].outcome == "not_interested"


@pytest.mark.asyncio
async def test_runner_closes_idle_call_as_no_answer(db, agent, fake_llm, fake_tts, monkeypatch):
    from backend.services import voice_agent_runner as runner

    monkeypatch.setattr(runner, "CALL_POLL_S", 0.01)
    monkeypatch.setattr(runner, "POLL_INTERVAL_S", 0.01)
    c = _contact(db, agent, "+447700900300", "Nobody")
    agent.max_attempts = 1
    core.set_agent_status(agent.id, "active", db)
    import asyncio

    runner.start(agent.id, idle_timeout_s=0.05)
    for _ in range(300):
        await asyncio.sleep(0.01)
        if not runner.is_running(agent.id):
            break
    assert not runner.is_running(agent.id)
    db.expire_all()
    calls, _ = core.list_calls(agent.id, db)
    assert calls
    assert calls[0].outcome == "no_answer"
    db.refresh(c)
    assert c.status == "exhausted"


# ── REST surface ───────────────────────────────────────────────────────


@pytest.fixture
def client(db, fake_llm, fake_tts):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.database import get_db as get_db_dep
    from backend.routes.voice_agent import router

    app = FastAPI()
    app.include_router(router)
    # The router's Depends(get_db) captured the original function object at
    # import time; override it with the test session factory.
    import backend.routes.voice_agent as route_mod

    app.dependency_overrides[route_mod.get_db] = get_db_dep
    return TestClient(app)


def test_rest_end_to_end(client, agent):
    r = client.post(
        "/agents",
        json={
            "name": "Support Desk",
            "mode": "support",
            "profile": "morgan",
            "agent_name": "Sam",
            "company_name": "Acme",
            "brief": "Acme home broadband support.",
            "goal": "Resolve or ticket.",
        },
    )
    assert r.status_code == 200, r.text
    support_id = r.json()["id"]
    assert r.json()["profile_id"] == "prof-1"

    # Duplicate name and bad profile are 400s.
    assert (
        client.post(
            "/agents",
            json={
                "name": "Support Desk",
                "profile": "morgan",
                "agent_name": "S",
                "company_name": "A",
                "brief": "b",
                "goal": "g",
            },
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/agents",
            json={"name": "X", "profile": "nobody", "agent_name": "S", "company_name": "A", "brief": "b", "goal": "g"},
        ).status_code
        == 400
    )

    r = client.post(
        f"/agents/{support_id}/knowledge", json={"title": "Reboot", "content": "Hold reset 10s.", "tags": ["router"]}
    )
    assert r.status_code == 200
    assert r.json()["tags"] == ["router"]

    r = client.post(f"/agents/{support_id}/calls/inbound", json={"phone": "+44 7700 900 400", "name": "Caller"})
    assert r.status_code == 200, r.text
    call_id = r.json()["call_id"]
    assert r.json()["ended"] is False
    assert "this is Sam" in r.json()["text"]

    r = client.post(f"/calls/{call_id}/turn", json={"text": "my router is dead"})
    assert r.status_code == 200
    assert r.json()["ended"] is False

    r = client.post(f"/calls/{call_id}/turn", json={"text": "get me a human now"})
    assert r.status_code == 200
    assert r.json()["ended"] is True
    assert r.json()["outcome"] == "handoff"
    ticket_id = r.json()["ticket_id"]

    assert client.post(f"/calls/{call_id}/turn", json={"text": "hello?"}).status_code == 400  # ended

    r = client.get(f"/calls/{call_id}")
    assert r.status_code == 200
    assert len(r.json()["turns"]) == 5
    assert r.json()["outcome"] == "handoff"

    r = client.get("/tickets", params={"agent_id": support_id})
    assert r.json()["total"] == 1
    assert r.json()["tickets"][0]["id"] == ticket_id
    r = client.put(f"/tickets/{ticket_id}", json={"status": "resolved"})
    assert r.json()["status"] == "resolved"

    r = client.get(f"/agents/{support_id}/stats")
    assert r.status_code == 200
    assert r.json()["calls_by_outcome"] == {"handoff": 1}

    # Outbound: contacts, DNC, next call.
    r = client.post(
        f"/agents/{agent.id}/contacts/bulk",
        json={"contacts": [{"name": "A", "phone": "+447700900500"}, {"name": "B", "phone": "+447700900501"}]},
    )
    assert r.json()["imported"] == 2
    r = client.post("/dnc", json={"phone": "+447700900501", "reason": "asked"})
    assert r.status_code == 200
    r = client.get(f"/agents/{agent.id}/contacts")
    statuses = {c["phone"]: c["status"] for c in r.json()["contacts"]}
    assert statuses == {"+447700900500": "new", "+447700900501": "do_not_call"}

    r = client.post(f"/agents/{agent.id}/calls/next")
    assert r.status_code == 200, r.text
    assert "Hi A, this is Sam calling from Acme Solar." in r.json()["text"]
    call_id = r.json()["call_id"]
    r = client.post(f"/calls/{call_id}/end", json={"outcome": "no_answer"})
    assert r.json()["status"] == "completed"
    assert r.json()["outcome"] == "no_answer"

    r = client.post(f"/agents/{agent.id}/calls/next")
    assert r.status_code == 409  # only contact is on retry delay

    r = client.post(
        f"/agents/{agent.id}/contacts/import", files={"file": ("c.csv", b"name,phone\nZed,+447700900600\n", "text/csv")}
    )
    assert r.json()["imported"] == 1
    assert client.delete("/dnc/+447700900501").status_code == 200
    assert client.delete("/dnc/+447700900501").status_code == 404
    assert client.delete(f"/agents/{support_id}").status_code == 200
    assert client.get(f"/agents/{support_id}").status_code == 404


def test_twilio_signature_and_twiml():
    from backend.services.telephony import TwilioProvider

    p = TwilioProvider(account_sid="AC1", auth_token="12345", public_url="https://example.com/")
    url = "https://example.com/webhooks/twilio/abc/answer"
    params = {"CallSid": "CA1", "From": "+15550001111"}
    import base64
    import hashlib
    import hmac

    sig = base64.b64encode(
        hmac.new(b"12345", (url + "CallSidCA1From+15550001111").encode(), hashlib.sha1).digest()
    ).decode()
    assert p.validate_signature(url, params, sig)
    assert not p.validate_signature(url, params, "nope")
    assert not p.validate_signature(url, params, None)

    xml = p.twiml_play_and_record("abc", "t1")
    assert "<Play>https://example.com/calls/abc/turns/t1/audio</Play>" in xml
    assert 'action="https://example.com/webhooks/twilio/abc/recording"' in xml
    assert "<Hangup/>" in p.twiml_play_and_hangup("abc", "t1")
    assert "<Redirect" in p.twiml_pause_and_retry("abc")
