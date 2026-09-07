"""Voice AI agent — outbound sales, customer service, and support calls.

The agent is a loop over three local models that Voicebox already ships:

    customer audio ──Whisper──▶ text ──Qwen3 LLM──▶ reply ──TTS──▶ agent voice

This module owns the state around that loop: agent configuration,
contacts, the do-not-call list, outbound scheduling (calling windows,
attempt caps), the per-call conversation state machine, tickets, and
post-call summaries. Prompting and the customer-side heuristics live in
:mod:`voice_agent_prompts`; knowledge retrieval in
:mod:`voice_agent_knowledge`; phone transport in :mod:`telephony`; the
auto-dialer loop in :mod:`voice_agent_runner`.

Everything that voices text goes through ``routes.generations.generate_speech``
— the same path as ``voicebox.speak`` — so agent turns land in History,
surface the speaking pill, and respect the profile's engine and effects.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models
from ..database import (
    Call,
    CallTurn,
    Contact,
    DoNotCallEntry,
    KnowledgeArticle,
    Ticket,
    VoiceAgent,
    VoiceProfile,
)
from . import voice_agent_knowledge as knowledge, voice_agent_prompts as prompts

logger = logging.getLogger(__name__)

# Generation.source markers so History can filter agent speech and the
# frontend can skip autoplay for audio that goes down a phone line.
SOURCE_LOCAL = "voice_agent"
SOURCE_REMOTE = "voice_agent_remote"

# Chat history handed to the LLM is capped so a long support call doesn't
# blow past a small model's context. The system prompt carries the
# contact's memory, so early turns aren't lost entirely.
MAX_HISTORY_PAIRS = 12
REPLY_MAX_TOKENS = 200
REPLY_TEMPERATURE = 0.4

# Idle calls (no turn in this long) are closed by the runner as no-answer.
DEFAULT_IDLE_TIMEOUT_S = 600


def utcnow() -> datetime:
    return datetime.utcnow()


# ── Phone numbers ──────────────────────────────────────────────────────


def normalize_phone(raw: str) -> str:
    """Canonical form for matching: digits only, keeping a leading ``+``.

    "+44 (0)20 7946 0958" → "+442079460958", "020 7946 0958" → "02079460958".
    Not a full E.164 validator — the operator's dialler decides what is
    routable; this only guarantees the DNC list matches exactly.
    """
    s = (raw or "").strip()
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if plus and len(digits) > 4:
        # "+44 (0)20 …" — the bracketed trunk zero after a country code is
        # not dialled internationally, so drop it.
        m = re.match(r"^(44|33|49|39|34|31|32|41|43|61|64|27|91|81|86)0", digits)
        if m:
            digits = m.group(1) + digits[len(m.group(1)) + 1 :]
    if not digits:
        raise ValueError("Phone number has no digits.")
    if len(digits) < 5 or len(digits) > 16:
        raise ValueError(f"Phone number '{raw}' does not look valid.")
    return ("+" if plus else "") + digits


# ── Agents ─────────────────────────────────────────────────────────────


def _resolve_profile(profile_ref: str, db: Session) -> VoiceProfile:
    """Id first, then case-insensitive name — same rule as ``voicebox.speak``.
    Inlined rather than imported from services.profiles so this module
    stays importable without the audio stack (numpy / librosa)."""
    ref = (profile_ref or "").strip()
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == ref).first() if ref else None
    if profile is None and ref:
        profile = db.query(VoiceProfile).filter(func.lower(VoiceProfile.name) == ref.lower()).first()
    if profile is None:
        raise ValueError(f"Voice profile '{profile_ref}' not found.")
    return profile


def _validate_window(start: int, end: int, days: list[int]) -> None:
    if not (0 <= start < end <= 24):
        raise ValueError("calling_window_start must be before calling_window_end (hours 0-24).")
    if not days or any(d < 0 or d > 6 for d in days):
        raise ValueError("calling_days must be a non-empty list of weekday numbers (Mon=0 … Sun=6).")


def _validate_timezone(name: str) -> None:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise ValueError(f"Unknown timezone '{name}'. Use an IANA name like 'Europe/London'.") from exc


def _is_running(agent_id: str) -> bool:
    from . import voice_agent_runner  # local: runner imports this module

    return voice_agent_runner.is_running(agent_id)


def agent_to_response(agent: VoiceAgent) -> models.VoiceAgentResponse:
    resp = models.VoiceAgentResponse.model_validate(agent)
    resp.running = _is_running(agent.id)
    return resp


def create_agent(data: models.VoiceAgentCreate, db: Session) -> VoiceAgent:
    if db.query(VoiceAgent).filter(VoiceAgent.name == data.name).first():
        raise ValueError(f"An agent named '{data.name}' already exists.")
    profile = _resolve_profile(data.profile, db)
    _validate_window(data.calling_window_start, data.calling_window_end, data.calling_days)
    _validate_timezone(data.timezone)

    payload = data.model_dump(exclude={"profile"})
    agent = VoiceAgent(id=str(uuid.uuid4()), profile_id=profile.id, **payload)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def update_agent(agent_id: str, data: models.VoiceAgentUpdate, db: Session) -> VoiceAgent | None:
    agent = db.query(VoiceAgent).filter(VoiceAgent.id == agent_id).first()
    if agent is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    if "name" in changes:
        clash = db.query(VoiceAgent).filter(VoiceAgent.name == changes["name"], VoiceAgent.id != agent_id).first()
        if clash:
            raise ValueError(f"An agent named '{changes['name']}' already exists.")
    if "profile" in changes:
        ref = changes.pop("profile")
        if ref:
            agent.profile_id = _resolve_profile(ref, db).id
    start = changes.get("calling_window_start", agent.calling_window_start)
    end = changes.get("calling_window_end", agent.calling_window_end)
    days = changes.get("calling_days", agent.calling_days)
    _validate_window(start, end, days)
    if "timezone" in changes:
        _validate_timezone(changes["timezone"])
    for key, value in changes.items():
        setattr(agent, key, value)
    db.commit()
    db.refresh(agent)
    return agent


def set_agent_status(agent_id: str, status: str, db: Session) -> VoiceAgent | None:
    agent = db.query(VoiceAgent).filter(VoiceAgent.id == agent_id).first()
    if agent is None:
        return None
    agent.status = status
    db.commit()
    db.refresh(agent)
    return agent


def list_agents(db: Session) -> list[VoiceAgent]:
    return db.query(VoiceAgent).order_by(VoiceAgent.created_at.desc()).all()


def get_agent(agent_id: str, db: Session) -> VoiceAgent | None:
    return db.query(VoiceAgent).filter(VoiceAgent.id == agent_id).first()


def delete_agent(agent_id: str, db: Session) -> bool:
    agent = get_agent(agent_id, db)
    if agent is None:
        return False
    if _is_running(agent_id):
        raise ValueError("Stop the agent before deleting it.")
    call_ids = [c.id for c in db.query(Call.id).filter(Call.agent_id == agent_id).all()]
    if call_ids:
        db.query(CallTurn).filter(CallTurn.call_id.in_(call_ids)).delete(synchronize_session=False)
    db.query(Ticket).filter(Ticket.agent_id == agent_id).delete(synchronize_session=False)
    db.query(Call).filter(Call.agent_id == agent_id).delete(synchronize_session=False)
    db.query(Contact).filter(Contact.agent_id == agent_id).delete(synchronize_session=False)
    db.query(KnowledgeArticle).filter(KnowledgeArticle.agent_id == agent_id).delete(synchronize_session=False)
    db.delete(agent)
    db.commit()
    return True


# ── Contacts ───────────────────────────────────────────────────────────


def create_contact(agent_id: str, data: models.ContactCreate, db: Session) -> Contact:
    phone = normalize_phone(data.phone)
    contact = Contact(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        name=data.name.strip(),
        phone=phone,
        company=data.company,
        notes=data.notes,
        timezone=data.timezone,
        consent=data.consent,
    )
    if data.timezone:
        _validate_timezone(data.timezone)
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return contact


def bulk_create_contacts(agent_id: str, items: list[models.ContactCreate], db: Session) -> models.ContactImportResult:
    """Insert many contacts, skipping duplicates (same phone on this agent)
    and bad numbers. Returns counts rather than raising so a 5,000-row
    import with three typos still lands 4,997 rows."""
    existing = {row.phone for row in db.query(Contact.phone).filter(Contact.agent_id == agent_id).all()}
    imported = 0
    reasons: dict[str, int] = {}
    for item in items:
        try:
            phone = normalize_phone(item.phone)
        except ValueError:
            reasons["invalid_phone"] = reasons.get("invalid_phone", 0) + 1
            continue
        if phone in existing:
            reasons["duplicate"] = reasons.get("duplicate", 0) + 1
            continue
        if item.timezone:
            try:
                _validate_timezone(item.timezone)
            except ValueError:
                reasons["invalid_timezone"] = reasons.get("invalid_timezone", 0) + 1
                continue
        db.add(
            Contact(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                name=item.name.strip() or phone,
                phone=phone,
                company=item.company,
                notes=item.notes,
                timezone=item.timezone,
                consent=item.consent,
            )
        )
        existing.add(phone)
        imported += 1
    db.commit()
    return models.ContactImportResult(imported=imported, skipped=sum(reasons.values()), skipped_reasons=reasons)


_CSV_ALIASES = {
    "name": ("name", "full name", "full_name", "contact", "contact name", "first name"),
    "phone": ("phone", "phone number", "phone_number", "number", "mobile", "tel", "telephone"),
    "company": ("company", "organisation", "organization", "business"),
    "notes": ("notes", "note", "comments"),
    "timezone": ("timezone", "time zone", "tz"),
    "consent": ("consent", "opted in", "opt_in", "opt-in", "consented"),
}


def parse_contacts_csv(text: str) -> list[models.ContactCreate]:
    """Read a CSV export with forgiving header names. Requires at least a
    phone column; a missing name falls back to the phone number."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")
    lower = {h.strip().lower(): h for h in reader.fieldnames if h}
    mapping: dict[str, str] = {}
    for field, aliases in _CSV_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[field] = lower[alias]
                break
    if "phone" not in mapping:
        raise ValueError("CSV needs a phone column (accepted headers: phone, phone number, number, mobile, tel).")

    out: list[models.ContactCreate] = []
    for row in reader:
        phone = (row.get(mapping["phone"]) or "").strip()
        if not phone:
            continue
        name = (row.get(mapping["name"]) or "").strip() if "name" in mapping else ""
        consent_raw = (row.get(mapping["consent"]) or "").strip().lower() if "consent" in mapping else ""
        out.append(
            models.ContactCreate(
                name=name or phone,
                phone=phone,
                company=(row.get(mapping["company"]) or None) if "company" in mapping else None,
                notes=(row.get(mapping["notes"]) or None) if "notes" in mapping else None,
                timezone=(row.get(mapping["timezone"]) or None) if "timezone" in mapping else None,
                consent=consent_raw in {"1", "true", "yes", "y"},
            )
        )
    return out


def list_contacts(agent_id: str, db: Session, *, status: str | None = None, limit: int = 200, offset: int = 0):
    q = db.query(Contact).filter(Contact.agent_id == agent_id)
    if status:
        q = q.filter(Contact.status == status)
    total = q.count()
    rows = q.order_by(Contact.created_at.asc()).offset(offset).limit(limit).all()
    return rows, total


def get_contact(contact_id: str, db: Session) -> Contact | None:
    return db.query(Contact).filter(Contact.id == contact_id).first()


def update_contact(contact_id: str, data: models.ContactUpdate, db: Session) -> Contact | None:
    contact = get_contact(contact_id, db)
    if contact is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    if "phone" in changes:
        changes["phone"] = normalize_phone(changes["phone"])
    if changes.get("timezone"):
        _validate_timezone(changes["timezone"])
    for key, value in changes.items():
        setattr(contact, key, value)
    db.commit()
    db.refresh(contact)
    return contact


def delete_contact(contact_id: str, db: Session) -> bool:
    contact = get_contact(contact_id, db)
    if contact is None:
        return False
    call_ids = [c.id for c in db.query(Call.id).filter(Call.contact_id == contact_id).all()]
    if call_ids:
        db.query(CallTurn).filter(CallTurn.call_id.in_(call_ids)).delete(synchronize_session=False)
    db.query(Ticket).filter(Ticket.contact_id == contact_id).delete(synchronize_session=False)
    db.query(Call).filter(Call.contact_id == contact_id).delete(synchronize_session=False)
    db.delete(contact)
    db.commit()
    return True


def get_or_create_inbound_contact(agent_id: str, phone_raw: str, name: str | None, db: Session) -> Contact:
    """Inbound callers are matched on phone across the agent's contacts so
    a repeat caller gets their memory back."""
    phone = normalize_phone(phone_raw)
    contact = db.query(Contact).filter(Contact.agent_id == agent_id, Contact.phone == phone).first()
    if contact is None:
        contact = Contact(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            name=(name or "").strip() or phone,
            phone=phone,
            consent=True,  # they called us
        )
        db.add(contact)
        db.commit()
        db.refresh(contact)
    elif name and contact.name == contact.phone:
        contact.name = name.strip()
        db.commit()
    return contact


# ── Knowledge ──────────────────────────────────────────────────────────


def _tags_to_str(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    cleaned = [t.strip() for t in tags if t and t.strip()]
    return ",".join(cleaned) or None


def _tags_to_list(tags: str | None) -> list[str]:
    return [t.strip() for t in (tags or "").split(",") if t.strip()]


def article_to_response(article: KnowledgeArticle) -> models.KnowledgeArticleResponse:
    return models.KnowledgeArticleResponse(
        id=article.id,
        agent_id=article.agent_id,
        title=article.title,
        content=article.content,
        tags=_tags_to_list(article.tags),
        created_at=article.created_at,
        updated_at=article.updated_at,
    )


def create_article(agent_id: str, data: models.KnowledgeArticleCreate, db: Session) -> KnowledgeArticle:
    article = KnowledgeArticle(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        title=data.title.strip(),
        content=data.content.strip(),
        tags=_tags_to_str(data.tags),
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


def list_articles(agent_id: str, db: Session) -> list[KnowledgeArticle]:
    return (
        db.query(KnowledgeArticle)
        .filter(KnowledgeArticle.agent_id == agent_id)
        .order_by(KnowledgeArticle.created_at.asc())
        .all()
    )


def update_article(article_id: str, data: models.KnowledgeArticleUpdate, db: Session) -> KnowledgeArticle | None:
    article = db.query(KnowledgeArticle).filter(KnowledgeArticle.id == article_id).first()
    if article is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    if "tags" in changes:
        changes["tags"] = _tags_to_str(changes["tags"])
    for key, value in changes.items():
        setattr(article, key, value.strip() if isinstance(value, str) else value)
    db.commit()
    db.refresh(article)
    return article


def delete_article(article_id: str, db: Session) -> bool:
    article = db.query(KnowledgeArticle).filter(KnowledgeArticle.id == article_id).first()
    if article is None:
        return False
    db.delete(article)
    db.commit()
    return True


# ── Do-not-call ────────────────────────────────────────────────────────


def is_blocked(phone: str, db: Session) -> bool:
    return db.query(DoNotCallEntry).filter(DoNotCallEntry.phone == phone).first() is not None


def add_to_dnc(phone_raw: str, db: Session, *, reason: str | None = None, source: str = "manual") -> DoNotCallEntry:
    phone = normalize_phone(phone_raw)
    entry = db.query(DoNotCallEntry).filter(DoNotCallEntry.phone == phone).first()
    if entry is None:
        entry = DoNotCallEntry(phone=phone, reason=reason, source=source)
        db.add(entry)
    elif reason and not entry.reason:
        entry.reason = reason
    # Any contact with this number, on any agent, stops being dialable.
    db.query(Contact).filter(Contact.phone == phone).update(
        {Contact.status: "do_not_call", Contact.next_attempt_at: None}, synchronize_session=False
    )
    db.commit()
    db.refresh(entry)
    return entry


def remove_from_dnc(phone_raw: str, db: Session) -> bool:
    phone = normalize_phone(phone_raw)
    entry = db.query(DoNotCallEntry).filter(DoNotCallEntry.phone == phone).first()
    if entry is None:
        return False
    db.delete(entry)
    db.commit()
    return True


def list_dnc(db: Session) -> list[DoNotCallEntry]:
    return db.query(DoNotCallEntry).order_by(DoNotCallEntry.created_at.desc()).all()


# ── Outbound scheduling ────────────────────────────────────────────────


def _zone(name: str | None, fallback: str) -> ZoneInfo:
    for candidate in (name, fallback, "UTC"):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            continue
    return ZoneInfo("UTC")


def within_calling_window(agent: VoiceAgent, contact: Contact, now_utc: datetime | None = None) -> bool:
    """Is it an acceptable time to ring this contact right now?"""
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    local = now.astimezone(_zone(contact.timezone, agent.timezone))
    days = agent.calling_days or [0, 1, 2, 3, 4]
    if local.weekday() not in days:
        return False
    return agent.calling_window_start <= local.hour < agent.calling_window_end


def calls_started_today(agent_id: str, db: Session, now: datetime | None = None) -> int:
    now = now or utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(Call).filter(Call.agent_id == agent_id, Call.started_at >= start).count()


def _dialable_query(agent: VoiceAgent, db: Session, now: datetime):
    q = db.query(Contact).filter(
        Contact.agent_id == agent.id,
        Contact.status.in_(["new", "callback"]),
        Contact.attempts < agent.max_attempts,
        (Contact.next_attempt_at.is_(None)) | (Contact.next_attempt_at <= now),
    )
    if agent.require_consent:
        q = q.filter(Contact.consent.is_(True))
    return q


def count_dialable(agent: VoiceAgent, db: Session, now: datetime | None = None) -> int:
    now = now or utcnow()
    blocked = {row.phone for row in db.query(DoNotCallEntry.phone).all()}
    return sum(1 for c in _dialable_query(agent, db, now).all() if c.phone not in blocked)


def has_pending_contacts(agent: VoiceAgent, db: Session) -> bool:
    """Anything left that could become dialable later (waiting on a
    callback time, outside the window right now)?"""
    q = db.query(Contact).filter(
        Contact.agent_id == agent.id,
        Contact.status.in_(["new", "callback"]),
        Contact.attempts < agent.max_attempts,
    )
    if agent.require_consent:
        q = q.filter(Contact.consent.is_(True))
    return db.query(q.exists()).scalar()


def pick_next_contact(agent: VoiceAgent, db: Session, now_utc: datetime | None = None) -> Contact | None:
    """The next contact the dialler should ring, or None if nothing is
    dialable right now. Applies, in order: agent status, daily cap,
    attempt cap + retry timing, DNC list, consent, calling window."""
    now = now_utc or datetime.now(UTC)
    now_naive = now.astimezone(UTC).replace(tzinfo=None)
    if agent.status != "active":
        return None
    if calls_started_today(agent.id, db, now_naive) >= agent.daily_call_cap:
        return None
    blocked = {row.phone for row in db.query(DoNotCallEntry.phone).all()}
    candidates = (
        _dialable_query(agent, db, now_naive)
        .order_by(Contact.next_attempt_at.asc().nulls_first(), Contact.attempts.asc(), Contact.created_at.asc())
        .all()
    )
    for contact in candidates:
        if contact.phone in blocked:
            contact.status = "do_not_call"
            continue
        if within_calling_window(agent, contact, now):
            db.commit()
            return contact
    db.commit()
    return None


# ── Calls ──────────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """What one exchange produced."""

    call: Call
    agent_turn: CallTurn
    customer_turn: CallTurn | None
    ended: bool
    outcome: str | None
    ticket: Ticket | None = None
    generation_id: str | None = None


def get_call(call_id: str, db: Session) -> Call | None:
    return db.query(Call).filter(Call.id == call_id).first()


def get_turns(call_id: str, db: Session) -> list[CallTurn]:
    return db.query(CallTurn).filter(CallTurn.call_id == call_id).order_by(CallTurn.created_at.asc()).all()


def call_to_response(call: Call, db: Session, *, include_turns: bool = True) -> models.CallResponse:
    resp = models.CallResponse.model_validate(call)
    if include_turns:
        resp.turns = [models.CallTurnResponse.model_validate(t) for t in get_turns(call.id, db)]
    return resp


def list_calls(agent_id: str, db: Session, *, limit: int = 100, offset: int = 0, status: str | None = None):
    q = db.query(Call).filter(Call.agent_id == agent_id)
    if status:
        q = q.filter(Call.status == status)
    total = q.count()
    rows = q.order_by(Call.started_at.desc()).offset(offset).limit(limit).all()
    return rows, total


def active_call_for_contact(contact_id: str, db: Session) -> Call | None:
    return db.query(Call).filter(Call.contact_id == contact_id, Call.status == "in_progress").first()


def _add_turn(
    call: Call,
    role: str,
    text: str,
    db: Session,
    *,
    sentiment: float | None = None,
    generation_id: str | None = None,
    capture_id: str | None = None,
) -> CallTurn:
    turn = CallTurn(
        id=str(uuid.uuid4()),
        call_id=call.id,
        role=role,
        text=text,
        sentiment=sentiment,
        generation_id=generation_id,
        capture_id=capture_id,
    )
    db.add(turn)
    call.turn_count = (call.turn_count or 0) + 1
    call.last_activity_at = utcnow()
    db.commit()
    db.refresh(turn)
    return turn


async def _voice(agent: VoiceAgent, call: Call, text: str, db: Session) -> str | None:
    """Voice ``text`` through the agent's profile. Returns the generation
    id, or None when TTS could not be queued (the call continues as text
    so a missing model never strands a live conversation)."""
    from ..mcp_server import events as mcp_events  # local: avoid import cycle at module load
    from ..routes.generations import generate_speech  # lazy: heavy import chain

    remote = call.provider != "local"
    req = models.GenerationRequest(
        profile_id=agent.profile_id,
        text=text,
        language=agent.language or "en",
        engine=agent.engine,
        model_size=None,
    )
    try:
        generation = await generate_speech(req, db)
    except Exception:
        logger.exception("Voice agent TTS failed for call %s", call.id)
        return None
    generation_id = getattr(generation, "id", None)
    # Stamp the origin so History can filter and the UI knows whether to
    # autoplay. generate_speech has already committed the row.
    from ..database import Generation as DBGeneration

    row = db.query(DBGeneration).filter(DBGeneration.id == generation_id).first()
    if row is not None:
        row.source = SOURCE_REMOTE if remote else SOURCE_LOCAL
        db.commit()
    if not remote:
        profile = db.query(VoiceProfile).filter(VoiceProfile.id == agent.profile_id).first()
        mcp_events.publish(
            "speak-start",
            {
                "generation_id": generation_id,
                "profile_name": profile.name if profile else agent.agent_name,
                "source": "voice_agent",
                "client_id": None,
            },
        )
    return generation_id


async def start_outbound_call(agent: VoiceAgent, contact: Contact, db: Session, *, voice: bool = True) -> TurnResult:
    """Open a call to ``contact`` and produce the opening line.

    Provider dialling happens in the runner / route after this returns so
    the call row exists (Twilio needs its id for the webhook URL).
    """
    if agent.status not in ("active", "draft", "paused"):
        raise ValueError("Agent is completed; reactivate it to place calls.")
    if is_blocked(contact.phone, db):
        contact.status = "do_not_call"
        db.commit()
        raise ValueError("Contact is on the do-not-call list.")
    if agent.require_consent and not contact.consent:
        raise ValueError("Agent requires consent and this contact has not opted in.")
    if active_call_for_contact(contact.id, db):
        raise ValueError("This contact already has a call in progress.")

    now = utcnow()
    contact.attempts = (contact.attempts or 0) + 1
    contact.last_attempt_at = now
    contact.status = "calling"
    call = Call(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        contact_id=contact.id,
        direction="outbound",
        provider=agent.provider or "local",
        stage="opening",
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return await _open(agent, contact, call, db, voice=voice)


async def start_inbound_call(
    agent: VoiceAgent, contact: Contact, db: Session, *, voice: bool = True, provider_call_id: str | None = None
) -> TurnResult:
    """Answer an inbound call from ``contact``. No window / DNC checks —
    they rang us."""
    if active_call_for_contact(contact.id, db):
        raise ValueError("This contact already has a call in progress.")
    call = Call(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        contact_id=contact.id,
        direction="inbound",
        provider=agent.provider or "local",
        provider_call_id=provider_call_id,
        stage="opening",
    )
    contact.status = "calling"
    contact.last_attempt_at = utcnow()
    db.add(call)
    db.commit()
    db.refresh(call)
    return await _open(agent, contact, call, db, voice=voice)


async def _open(agent: VoiceAgent, contact: Contact, call: Call, db: Session, *, voice: bool) -> TurnResult:
    opening = prompts.build_opening_line(
        mode=agent.mode,
        agent_name=agent.agent_name,
        company_name=agent.company_name,
        disclosure=agent.disclosure,
        contact_name=contact.name if contact.name != contact.phone else None,
        custom_opening=agent.opening_line,
    )
    generation_id = await _voice(agent, call, opening, db) if voice else None
    turn = _add_turn(call, "agent", opening, db, generation_id=generation_id)
    call.stage = "conversation"
    db.commit()
    return TurnResult(
        call=call, agent_turn=turn, customer_turn=None, ended=False, outcome=None, generation_id=generation_id
    )


def _history_pairs(turns: list[CallTurn]) -> list[tuple[str, str]]:
    """Fold the transcript into (customer, agent) pairs for the chat
    template. The opening line has no customer turn before it, so it is
    paired with a synthetic "(call connected)" user message."""
    pairs: list[tuple[str, str]] = []
    pending_customer: str | None = None
    for t in turns:
        if t.role == "customer":
            pending_customer = (pending_customer + " " + t.text) if pending_customer else t.text
        else:
            pairs.append((pending_customer or "(call connected)", t.text))
            pending_customer = None
    return pairs[-MAX_HISTORY_PAIRS:]


async def _llm_reply(agent: VoiceAgent, system: str, history: list[tuple[str, str]], customer_text: str) -> str:
    from . import llm as llm_service  # local: keeps this module importable without torch

    backend = llm_service.get_llm_model()
    size = agent.llm_model_size or backend.model_size
    loaded = backend.is_loaded() and backend.model_size == size
    if not loaded and hasattr(backend, "_is_model_cached") and not backend._is_model_cached(size):
        raise ValueError(f"Qwen3 {size} is not downloaded yet. Open Voicebox → Settings → Models to download it.")
    return await backend.generate(
        prompt=customer_text,
        system=system,
        max_tokens=REPLY_MAX_TOKENS,
        temperature=REPLY_TEMPERATURE,
        model_size=size,
        examples=history or None,
    )


async def handle_customer_turn(
    call: Call,
    customer_text: str,
    db: Session,
    *,
    voice: bool = True,
    capture_id: str | None = None,
) -> TurnResult:
    """Record what the customer said, decide what the agent says next, and
    close the call when the conversation has reached an end."""
    if call.status != "in_progress":
        raise ValueError("Call has already ended.")
    agent = get_agent(call.agent_id, db)
    contact = get_contact(call.contact_id, db)
    if agent is None or contact is None:
        raise ValueError("Call refers to a missing agent or contact.")

    customer_text = (customer_text or "").strip()
    if not customer_text:
        raise ValueError("Customer turn is empty.")

    sentiment = prompts.score_sentiment(customer_text)
    customer_turn = _add_turn(call, "customer", customer_text, db, sentiment=sentiment, capture_id=capture_id)
    if sentiment <= -0.3:
        call.negative_streak = (call.negative_streak or 0) + 1
    elif sentiment >= 0.0:
        call.negative_streak = 0

    # ── Hard stops that never go through the model ──
    intent = prompts.classify_customer_intent(customer_text, agent.mode)
    if intent == "opt_out":
        add_to_dnc(contact.phone, db, reason="Asked not to be called", source="opt_out")
        text = prompts.opt_out_closing(agent.company_name)
        return await _close_with(agent, contact, call, text, "opt_out", db, voice=voice, customer_turn=customer_turn)

    if call.direction == "outbound" and call.turn_count <= 2 and prompts.detect_voicemail(customer_text):
        # First thing we heard back was a voicemail greeting — don't pitch
        # to a machine; schedule a retry instead.
        return await _close_with(
            agent, contact, call, "", "voicemail", db, voice=False, customer_turn=customer_turn, speak=False
        )

    wants_human = intent == "handoff"
    too_negative = (call.negative_streak or 0) >= agent.handoff_after_negative_turns
    if wants_human or (too_negative and agent.mode != prompts.MODE_OUTBOUND_SALES):
        ticket = _create_ticket(
            agent,
            contact,
            call,
            db,
            kind="handoff",
            priority="high" if too_negative else "normal",
            subject=("Caller asked for a person" if wants_human else "Escalated: caller is upset"),
            description=_transcript_text(call, db),
        )
        text = prompts.handoff_closing(agent.escalation_promise)
        return await _close_with(
            agent, contact, call, text, "handoff", db, voice=voice, customer_turn=customer_turn, ticket=ticket
        )

    if call.turn_count >= agent.max_turns:
        ticket = _create_ticket(
            agent,
            contact,
            call,
            db,
            kind="callback",
            priority="normal",
            subject="Call hit the turn limit",
            description=_transcript_text(call, db),
        )
        text = prompts.max_turns_closing(agent.escalation_promise)
        return await _close_with(
            agent, contact, call, text, "max_turns", db, voice=voice, customer_turn=customer_turn, ticket=ticket
        )

    if prompts.detect_goodbye(customer_text) and call.turn_count > 2:
        outcome = "resolved" if agent.mode != prompts.MODE_OUTBOUND_SALES else (intent or "contacted")
        if outcome == "contacted":
            outcome = "not_interested" if call.turn_count <= 4 else "callback"
        text = prompts.goodbye_closing(agent.mode)
        return await _close_with(agent, contact, call, text, outcome, db, voice=voice, customer_turn=customer_turn)

    # ── The model's turn ──
    turns = get_turns(call.id, db)
    recent_customer = [t.text for t in turns if t.role == "customer"]
    kb = knowledge.retrieve_for_turn(db, agent.id, recent_customer)
    system = prompts.build_system_prompt(
        mode=agent.mode,
        agent_name=agent.agent_name,
        company_name=agent.company_name,
        brief=agent.brief,
        goal=agent.goal,
        objection_notes=agent.objection_notes,
        persona=agent.persona,
        escalation_promise=agent.escalation_promise,
        contact_name=contact.name if contact.name != contact.phone else None,
        contact_company=contact.company,
        contact_notes=contact.notes,
        contact_memory=contact.memory,
        knowledge=kb,
    )
    history = _history_pairs(turns[:-1])  # everything before the turn we're answering
    try:
        raw = await _llm_reply(agent, system, history, customer_text)
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Voice agent LLM failed on call %s", call.id)
        text = prompts.max_turns_closing(agent.escalation_promise)
        ticket = _create_ticket(
            agent,
            contact,
            call,
            db,
            kind="callback",
            priority="high",
            subject="Agent error mid-call",
            description=f"{exc}\n\n{_transcript_text(call, db)}",
        )
        return await _close_with(
            agent, contact, call, text, "error", db, voice=voice, customer_turn=customer_turn, ticket=ticket
        )

    parsed = prompts.parse_agent_reply(raw, agent.mode)
    reply_text = parsed.text or "Sorry, could you say that again?"

    ticket: Ticket | None = None
    outcome = parsed.outcome
    if parsed.handoff:
        outcome = "handoff"
        ticket = _create_ticket(
            agent,
            contact,
            call,
            db,
            kind="handoff",
            priority="normal",
            subject="Agent handed off to a person",
            description=_transcript_text(call, db),
        )
    elif parsed.ticket_subject:
        outcome = "ticket_created"
        ticket = _create_ticket(
            agent,
            contact,
            call,
            db,
            kind="support" if agent.mode == prompts.MODE_SUPPORT else "callback",
            priority="normal",
            subject=parsed.ticket_subject,
            description=_transcript_text(call, db),
        )
    elif outcome is None and intent in ("not_interested", "callback") and agent.mode == prompts.MODE_OUTBOUND_SALES:
        # The model didn't tag it but the customer was explicit.
        outcome = intent

    if outcome in prompts.TERMINAL_OUTCOMES:
        return await _close_with(
            agent, contact, call, reply_text, outcome, db, voice=voice, customer_turn=customer_turn, ticket=ticket
        )

    generation_id = await _voice(agent, call, reply_text, db) if voice else None
    agent_turn = _add_turn(call, "agent", reply_text, db, generation_id=generation_id)
    return TurnResult(
        call=call,
        agent_turn=agent_turn,
        customer_turn=customer_turn,
        ended=False,
        outcome=None,
        generation_id=generation_id,
    )


async def _close_with(
    agent: VoiceAgent,
    contact: Contact,
    call: Call,
    text: str,
    outcome: str,
    db: Session,
    *,
    voice: bool,
    customer_turn: CallTurn | None,
    ticket: Ticket | None = None,
    speak: bool = True,
) -> TurnResult:
    call.stage = "closing"
    generation_id = await _voice(agent, call, text, db) if (voice and speak and text) else None
    agent_turn = _add_turn(call, "agent", text or "(hung up)", db, generation_id=generation_id)
    await end_call(call, outcome, db)
    return TurnResult(
        call=call,
        agent_turn=agent_turn,
        customer_turn=customer_turn,
        ended=True,
        outcome=outcome,
        ticket=ticket,
        generation_id=generation_id,
    )


# Outcome → what happens to the contact afterwards.
_OUTCOME_CONTACT_STATUS: dict[str, str] = {
    "interested": "interested",
    "not_interested": "not_interested",
    "callback": "callback",
    "opt_out": "do_not_call",
    "resolved": "resolved",
    "unresolved": "unresolved",
    "ticket_created": "unresolved",
    "handoff": "unresolved",
    "no_answer": "new",
    "voicemail": "new",
    "max_turns": "contacted",
    "error": "new",
}
_RETRY_OUTCOMES = frozenset({"no_answer", "voicemail", "error"})


async def end_call(call: Call, outcome: str, db: Session, *, summary: str | None = None) -> Call:
    """Finish a call: set outcome, update the contact, schedule retries or
    callbacks, and (best effort) summarise the transcript into the
    contact's memory."""
    if call.status != "in_progress":
        return call
    agent = get_agent(call.agent_id, db)
    contact = get_contact(call.contact_id, db)
    now = utcnow()
    call.status = "failed" if outcome == "error" else "completed"
    call.stage = "ended"
    call.outcome = outcome
    call.ended_at = now

    if agent is not None and contact is not None:
        contact.last_outcome = outcome
        status = _OUTCOME_CONTACT_STATUS.get(outcome, "contacted")
        if outcome in _RETRY_OUTCOMES:
            if contact.attempts >= agent.max_attempts:
                status = "exhausted"
                contact.next_attempt_at = None
            else:
                contact.next_attempt_at = now + timedelta(hours=agent.retry_delay_hours)
        elif outcome == "callback":
            contact.next_attempt_at = now + timedelta(hours=agent.callback_delay_hours)
            if contact.attempts >= agent.max_attempts:
                # Honour the callback request even at the attempt cap —
                # they asked for it — by granting one more attempt.
                contact.attempts = agent.max_attempts - 1
        else:
            contact.next_attempt_at = None
        if is_blocked(contact.phone, db):
            status = "do_not_call"
        contact.status = status
        if outcome == "interested":
            _create_ticket(
                agent,
                contact,
                call,
                db,
                kind="sales_lead",
                priority="high",
                subject=f"Interested: {contact.name}",
                description=_transcript_text(call, db),
            )
    db.commit()

    if summary:
        call.summary = summary
    elif agent is not None and call.turn_count >= 3:
        call.summary = await _summarize(agent, contact, call, db)
    if call.summary and contact is not None:
        stamp = now.strftime("%Y-%m-%d")
        entry = f"[{stamp}] {call.summary.strip()}"
        contact.memory = (contact.memory + "\n" + entry) if contact.memory else entry
        # Keep memory bounded so the system prompt stays small.
        contact.memory = contact.memory[-4000:]
    db.commit()
    db.refresh(call)
    return call


async def _summarize(agent: VoiceAgent, contact: Contact | None, call: Call, db: Session) -> str | None:
    """LLM summary of the call. Never raises — a missing model just means
    no summary."""
    turns = [(t.role, t.text) for t in get_turns(call.id, db) if t.text and t.text != "(hung up)"]
    if len(turns) < 2:
        return None
    try:
        from . import llm as llm_service  # local

        backend = llm_service.get_llm_model()
        size = agent.llm_model_size or backend.model_size
        loaded = backend.is_loaded() and backend.model_size == size
        if not loaded and hasattr(backend, "_is_model_cached") and not backend._is_model_cached(size):
            return None
        raw = await backend.generate(
            prompt=prompts.build_summary_prompt(turns, contact.name if contact else None),
            system=prompts.SUMMARY_SYSTEM,
            max_tokens=200,
            temperature=0.2,
            model_size=size,
        )
        text = prompts.parse_agent_reply(raw, agent.mode).text
        return text[:1500] or None
    except Exception:
        logger.debug("Call summary failed for %s", call.id, exc_info=True)
        return None


def _transcript_text(call: Call, db: Session) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in get_turns(call.id, db))


# ── Tickets ────────────────────────────────────────────────────────────


def _create_ticket(
    agent: VoiceAgent,
    contact: Contact,
    call: Call | None,
    db: Session,
    *,
    kind: str,
    priority: str,
    subject: str,
    description: str | None,
) -> Ticket:
    ticket = Ticket(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        contact_id=contact.id,
        call_id=call.id if call else None,
        kind=kind,
        priority=priority,
        subject=subject[:200],
        description=description,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def list_tickets(
    db: Session, *, agent_id: str | None = None, status: str | None = None, limit: int = 100, offset: int = 0
):
    q = db.query(Ticket)
    if agent_id:
        q = q.filter(Ticket.agent_id == agent_id)
    if status:
        q = q.filter(Ticket.status == status)
    total = q.count()
    rows = q.order_by(Ticket.created_at.desc()).offset(offset).limit(limit).all()
    return rows, total


def get_ticket(ticket_id: str, db: Session) -> Ticket | None:
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()


def update_ticket(ticket_id: str, data: models.TicketUpdate, db: Session) -> Ticket | None:
    ticket = get_ticket(ticket_id, db)
    if ticket is None:
        return None
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(ticket, key, value)
    db.commit()
    db.refresh(ticket)
    return ticket


# ── Stats ──────────────────────────────────────────────────────────────


def agent_stats(agent: VoiceAgent, db: Session) -> models.VoiceAgentStats:
    contacts_by_status = {
        status: n
        for status, n in db.query(Contact.status, func.count(Contact.id))
        .filter(Contact.agent_id == agent.id)
        .group_by(Contact.status)
        .all()
    }
    calls_by_outcome = {
        (outcome or "in_progress"): n
        for outcome, n in db.query(Call.outcome, func.count(Call.id))
        .filter(Call.agent_id == agent.id)
        .group_by(Call.outcome)
        .all()
    }
    calls_total = sum(calls_by_outcome.values())
    avg_turns = (
        db.query(func.avg(Call.turn_count)).filter(Call.agent_id == agent.id, Call.status != "in_progress").scalar()
        or 0.0
    )
    if agent.mode == prompts.MODE_OUTBOUND_SALES:
        good = calls_by_outcome.get("interested", 0)
    else:
        good = calls_by_outcome.get("resolved", 0)
    finished = calls_total - calls_by_outcome.get("in_progress", 0)
    resolution_rate = (good / finished) if finished else 0.0
    open_tickets = (
        db.query(Ticket).filter(Ticket.agent_id == agent.id, Ticket.status.in_(["open", "in_progress"])).count()
    )
    return models.VoiceAgentStats(
        agent_id=agent.id,
        mode=agent.mode,
        status=agent.status,
        running=_is_running(agent.id),
        contacts_total=sum(contacts_by_status.values()),
        contacts_by_status=contacts_by_status,
        calls_total=calls_total,
        calls_today=calls_started_today(agent.id, db),
        calls_by_outcome=calls_by_outcome,
        avg_turns=round(float(avg_turns), 2),
        resolution_rate=round(resolution_rate, 3),
        open_tickets=open_tickets,
        next_dialable=count_dialable(agent, db),
    )
