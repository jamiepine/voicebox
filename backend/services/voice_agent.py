"""Voice AI agent — outbound sales, customer service, and support calls.

The agent is a loop over three local models that Voicebox already ships:

    customer audio ──Whisper──▶ text ──Qwen3 LLM──▶ reply ──TTS──▶ agent voice

This module owns the state around that loop: agent configuration and
versions, contacts, the do-not-call list, outbound scheduling (calling
windows, attempt caps), the per-call conversation state machine (with
tool calls, supervisor take-over and interruptions), tickets,
appointments, post-call summaries / analysis / webhooks, simulations,
analytics and exports. Prompting and the customer-side heuristics live in
:mod:`voice_agent_prompts`; knowledge retrieval in
:mod:`voice_agent_knowledge`; tools in :mod:`voice_agent_tools`; phone
transport in :mod:`telephony`; the auto-dialer loop in
:mod:`voice_agent_runner`; live-console events in :mod:`voice_agent_events`.

Everything that voices text goes through ``routes.generations.generate_speech``
— the same path as ``voicebox.speak`` — so agent turns land in History,
surface the speaking pill, and respect the profile's engine and effects.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import random
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models
from ..database import (
    AgentTool,
    Appointment,
    Call,
    CallTurn,
    Contact,
    DoNotCallEntry,
    KnowledgeArticle,
    Message,
    Ticket,
    VoiceAgent,
    VoiceAgentVersion,
    VoiceProfile,
    WebhookDelivery,
)
from . import (
    telephony,
    voice_agent_events as events,
    voice_agent_knowledge as knowledge,
    voice_agent_prompts as prompts,
    voice_agent_tools as agent_tools,
    voice_agent_webhooks as webhooks,
)

logger = logging.getLogger(__name__)

# Generation.source markers so History can filter agent speech and the
# frontend can skip autoplay for audio that goes down a phone line.
SOURCE_LOCAL = "voice_agent"
SOURCE_REMOTE = "voice_agent_remote"
SOURCE_FILLER = "voice_agent_filler"

# Chat history handed to the LLM is capped so a long support call doesn't
# blow past a small model's context. The system prompt carries the
# contact's memory, so early turns aren't lost entirely.
MAX_HISTORY_PAIRS = 12
REPLY_MAX_TOKENS = 200
REPLY_TEMPERATURE = 0.4
MAX_TOOL_CALLS_PER_TURN = 2

# Idle calls (no turn in this long) are closed by the runner as no-answer.
DEFAULT_IDLE_TIMEOUT_S = 600

TEST_CONTACT_PHONE = "+19999999999"
SIMULATION = "simulation"

# Operator-configured fields captured in every version snapshot.
VERSIONED_FIELDS: tuple[str, ...] = (
    "name",
    "mode",
    "profile_id",
    "engine",
    "language",
    "llm_model_size",
    "voice_style",
    "empathetic_voice_style",
    "agent_name",
    "company_name",
    "brief",
    "goal",
    "objection_notes",
    "persona",
    "opening_line",
    "disclosure",
    "escalation_promise",
    "variants",
    "filler_phrases",
    "fast_first_audio",
    "tools_enabled",
    "booking_instructions",
    "appointment_duration_min",
    "analysis_schema",
    "webhook_url",
    "webhook_secret",
    "timezone",
    "calling_window_start",
    "calling_window_end",
    "calling_days",
    "max_attempts",
    "daily_call_cap",
    "retry_delay_hours",
    "callback_delay_hours",
    "require_consent",
    "max_turns",
    "handoff_after_negative_turns",
    "redact_pii",
    "max_concurrent_calls",
    "schedule_start_at",
    "schedule_end_at",
    "provider",
    "from_number",
    "transfer_number",
    "voicemail_message",
    "sms_followup_template",
    "sms_followup_outcomes",
)
# Changing any of these makes the cached filler audio stale.
_VOICE_FIELDS = {"profile_id", "engine", "language", "voice_style", "filler_phrases"}


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


def _validate_variants(variants: list[dict] | None) -> None:
    names = [v.get("name", "") for v in (variants or [])]
    if len(names) != len({n.lower() for n in names}):
        raise ValueError("Variant names must be unique.")


def _validate_schedule(start: datetime | None, end: datetime | None) -> None:
    if start and end and end <= start:
        raise ValueError("schedule_end_at must be after schedule_start_at.")


def _naive_utc(value: datetime | None) -> datetime | None:
    """SQLite keeps no offset, so store schedule times as naive UTC."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _is_running(agent_id: str) -> bool:
    from . import voice_agent_runner  # local: runner imports this module

    return voice_agent_runner.is_running(agent_id)


def agent_to_response(agent: VoiceAgent) -> models.VoiceAgentResponse:
    resp = models.VoiceAgentResponse.model_validate(agent)
    resp.running = _is_running(agent.id)
    return resp


def snapshot_agent(agent: VoiceAgent) -> dict:
    out: dict = {}
    for key in VERSIONED_FIELDS:
        value = getattr(agent, key, None)
        out[key] = value.isoformat() if isinstance(value, datetime) else value
    return out


def _record_version(agent: VoiceAgent, db: Session, note: str | None) -> VoiceAgentVersion:
    row = VoiceAgentVersion(
        id=str(uuid.uuid4()), agent_id=agent.id, version=agent.version, snapshot=snapshot_agent(agent), note=note
    )
    db.add(row)
    db.commit()
    return row


def create_agent(data: models.VoiceAgentCreate, db: Session) -> VoiceAgent:
    if db.query(VoiceAgent).filter(VoiceAgent.name == data.name).first():
        raise ValueError(f"An agent named '{data.name}' already exists.")
    profile = _resolve_profile(data.profile, db)
    _validate_window(data.calling_window_start, data.calling_window_end, data.calling_days)
    _validate_timezone(data.timezone)
    _validate_schedule(data.schedule_start_at, data.schedule_end_at)
    payload = data.model_dump(exclude={"profile"})
    for key in ("schedule_start_at", "schedule_end_at"):
        payload[key] = _naive_utc(payload.get(key))
    _validate_variants(payload.get("variants"))
    agent = VoiceAgent(id=str(uuid.uuid4()), profile_id=profile.id, version=1, **payload)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    _record_version(agent, db, "created")
    return agent


def update_agent(
    agent_id: str, data: models.VoiceAgentUpdate, db: Session, *, note: str | None = None
) -> VoiceAgent | None:
    agent = db.query(VoiceAgent).filter(VoiceAgent.id == agent_id).first()
    if agent is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    if not changes:
        return agent
    if "name" in changes:
        clash = db.query(VoiceAgent).filter(VoiceAgent.name == changes["name"], VoiceAgent.id != agent_id).first()
        if clash:
            raise ValueError(f"An agent named '{changes['name']}' already exists.")
    if "profile" in changes:
        ref = changes.pop("profile")
        if ref:
            changes["profile_id"] = _resolve_profile(ref, db).id
    start = changes.get("calling_window_start", agent.calling_window_start)
    end = changes.get("calling_window_end", agent.calling_window_end)
    days = changes.get("calling_days", agent.calling_days)
    _validate_window(start, end, days)
    if "timezone" in changes:
        _validate_timezone(changes["timezone"])
    _validate_schedule(
        changes.get("schedule_start_at", agent.schedule_start_at), changes.get("schedule_end_at", agent.schedule_end_at)
    )
    if "variants" in changes:
        _validate_variants(changes["variants"])
    for key in ("schedule_start_at", "schedule_end_at"):
        if key in changes:
            changes[key] = _naive_utc(changes[key])
    if any(k in _VOICE_FIELDS and getattr(agent, k) != v for k, v in changes.items()):
        agent.filler_audio = None
    for key, value in changes.items():
        setattr(agent, key, value)
    agent.version = (agent.version or 1) + 1
    db.commit()
    db.refresh(agent)
    _record_version(agent, db, note or "updated")
    return agent


def list_versions(agent_id: str, db: Session) -> list[VoiceAgentVersion]:
    return (
        db.query(VoiceAgentVersion)
        .filter(VoiceAgentVersion.agent_id == agent_id)
        .order_by(VoiceAgentVersion.version.desc())
        .all()
    )


def restore_version(agent_id: str, version_id: str, db: Session) -> VoiceAgent | None:
    agent = get_agent(agent_id, db)
    row = (
        db.query(VoiceAgentVersion)
        .filter(VoiceAgentVersion.id == version_id, VoiceAgentVersion.agent_id == agent_id)
        .first()
    )
    if agent is None or row is None:
        return None
    snap = dict(row.snapshot or {})
    for key in VERSIONED_FIELDS:
        if key not in snap:
            continue
        value = snap[key]
        if key in {"schedule_start_at", "schedule_end_at"} and isinstance(value, str):
            value = datetime.fromisoformat(value)
        if key == "name":
            clash = db.query(VoiceAgent).filter(VoiceAgent.name == value, VoiceAgent.id != agent_id).first()
            if clash:
                continue
        setattr(agent, key, value)
    agent.filler_audio = None
    agent.version = (agent.version or 1) + 1
    db.commit()
    db.refresh(agent)
    _record_version(agent, db, f"restored from v{row.version}")
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
    for model in (Ticket, Appointment, Message, WebhookDelivery, AgentTool, VoiceAgentVersion, KnowledgeArticle):
        db.query(model).filter(model.agent_id == agent_id).delete(synchronize_session=False)
    db.query(Call).filter(Call.agent_id == agent_id).delete(synchronize_session=False)
    db.query(Contact).filter(Contact.agent_id == agent_id).delete(synchronize_session=False)
    db.delete(agent)
    db.commit()
    return True


# ── Filler audio (latency masking) ─────────────────────────────────────


async def ensure_filler_audio(agent: VoiceAgent, db: Session) -> dict[str, str]:
    """Pre-generate the agent's filler phrases in its voice. Idempotent;
    re-generates entries whose generation went missing or failed. Never
    raises — a missing TTS model just means no fillers for now."""
    from ..database import Generation as DBGeneration

    cache: dict[str, str] = dict(agent.filler_audio or {})
    phrases = [p for p in (agent.filler_phrases or []) if p and p.strip()]
    changed = False
    for phrase in phrases:
        gen_id = cache.get(phrase)
        if gen_id:
            row = db.query(DBGeneration).filter(DBGeneration.id == gen_id).first()
            if row is not None and row.status != "failed":
                continue
        new_id = await _generate(
            agent, phrase, db, source=SOURCE_FILLER, instruct=agent.voice_style, language=agent.language
        )
        if new_id:
            cache[phrase] = new_id
            changed = True
    for phrase in list(cache):
        if phrase not in phrases:
            cache.pop(phrase)
            changed = True
    if changed:
        agent.filler_audio = cache
        db.commit()
    return cache


def _pick_filler(agent: VoiceAgent, db: Session) -> tuple[str, str] | None:
    from ..database import Generation as DBGeneration

    cache = agent.filler_audio or {}
    ready = []
    for phrase, gen_id in cache.items():
        row = db.query(DBGeneration).filter(DBGeneration.id == gen_id).first()
        if row is not None and row.status == "completed":
            ready.append((phrase, gen_id))
    return random.choice(ready) if ready else None


# ── Contacts ───────────────────────────────────────────────────────────


def _clean_custom_fields(fields: dict | None) -> dict[str, str] | None:
    if not fields:
        return None
    out = {}
    for k, v in fields.items():
        key = re.sub(r"[^a-z0-9_]+", "_", str(k).strip().lower()).strip("_")
        if key and v not in (None, ""):
            out[key[:40]] = str(v)[:500]
    return out or None


def create_contact(agent_id: str, data: models.ContactCreate, db: Session) -> Contact:
    phone = normalize_phone(data.phone)
    if data.timezone:
        _validate_timezone(data.timezone)
    contact = Contact(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        name=data.name.strip(),
        phone=phone,
        company=data.company,
        notes=data.notes,
        timezone=data.timezone,
        language=data.language,
        custom_fields=_clean_custom_fields(data.custom_fields),
        consent=data.consent,
    )
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
                language=item.language,
                custom_fields=_clean_custom_fields(item.custom_fields),
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
    "language": ("language", "lang"),
    "consent": ("consent", "opted in", "opt_in", "opt-in", "consented"),
}
_LANG_CODES = set(prompts.LANGUAGE_NAMES)


def parse_contacts_csv(text: str) -> list[models.ContactCreate]:
    """Read a CSV export with forgiving header names. Requires at least a
    phone column; a missing name falls back to the phone number. Any
    other column becomes a custom field usable as {{contact.custom.<col>}}."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")
    lower = {h.strip().lower(): h for h in reader.fieldnames if h}
    mapping: dict[str, str] = {}
    for field_name, aliases in _CSV_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[field_name] = lower[alias]
                break
    if "phone" not in mapping:
        raise ValueError("CSV needs a phone column (accepted headers: phone, phone number, number, mobile, tel).")
    known = set(mapping.values())

    out: list[models.ContactCreate] = []
    for row in reader:
        phone = (row.get(mapping["phone"]) or "").strip()
        if not phone:
            continue
        name = (row.get(mapping["name"]) or "").strip() if "name" in mapping else ""
        consent_raw = (row.get(mapping["consent"]) or "").strip().lower() if "consent" in mapping else ""
        language = (row.get(mapping["language"]) or "").strip().lower() if "language" in mapping else ""
        custom = {h: v for h, v in row.items() if h and h not in known and v not in (None, "")}
        out.append(
            models.ContactCreate(
                name=name or phone,
                phone=phone,
                company=(row.get(mapping["company"]) or None) if "company" in mapping else None,
                notes=(row.get(mapping["notes"]) or None) if "notes" in mapping else None,
                timezone=(row.get(mapping["timezone"]) or None) if "timezone" in mapping else None,
                language=language if language in _LANG_CODES else None,
                custom_fields=custom or None,
                consent=consent_raw in {"1", "true", "yes", "y"},
            )
        )
    return out


def list_contacts(agent_id: str, db: Session, *, status: str | None = None, limit: int = 200, offset: int = 0):
    q = db.query(Contact).filter(Contact.agent_id == agent_id, Contact.is_test.is_(False))
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
    if "custom_fields" in changes:
        changes["custom_fields"] = _clean_custom_fields(changes["custom_fields"])
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
    for model in (Ticket, Appointment, Message):
        db.query(model).filter(model.contact_id == contact_id).delete(synchronize_session=False)
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


def get_or_create_test_contact(agent_id: str, db: Session) -> Contact:
    """The stand-in a simulation talks to. Never dialled, never counted."""
    contact = db.query(Contact).filter(Contact.agent_id == agent_id, Contact.is_test.is_(True)).first()
    if contact is None:
        contact = Contact(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            name="Test customer",
            phone=TEST_CONTACT_PHONE,
            is_test=True,
            consent=True,
            status="new",
        )
        db.add(contact)
        db.commit()
        db.refresh(contact)
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
        source=article.source,
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


# ── Tools (custom HTTP) ────────────────────────────────────────────────

_RESERVED_TOOL_NAMES = {"book_appointment", "schedule_callback", "transfer_to_human", "send_sms"}


def create_tool(agent_id: str, data: models.AgentToolCreate, db: Session) -> AgentTool:
    if data.name in _RESERVED_TOOL_NAMES:
        raise ValueError(f"'{data.name}' is a built-in tool name.")
    if db.query(AgentTool).filter(AgentTool.agent_id == agent_id, AgentTool.name == data.name).first():
        raise ValueError(f"A tool named '{data.name}' already exists on this agent.")
    payload = data.model_dump()
    tool = AgentTool(id=str(uuid.uuid4()), agent_id=agent_id, **payload)
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return tool


def list_tools(agent_id: str, db: Session) -> list[AgentTool]:
    return db.query(AgentTool).filter(AgentTool.agent_id == agent_id).order_by(AgentTool.created_at.asc()).all()


def update_tool(tool_id: str, data: models.AgentToolUpdate, db: Session) -> AgentTool | None:
    tool = db.query(AgentTool).filter(AgentTool.id == tool_id).first()
    if tool is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    if "name" in changes:
        if changes["name"] in _RESERVED_TOOL_NAMES:
            raise ValueError(f"'{changes['name']}' is a built-in tool name.")
        clash = (
            db.query(AgentTool)
            .filter(AgentTool.agent_id == tool.agent_id, AgentTool.name == changes["name"], AgentTool.id != tool_id)
            .first()
        )
        if clash:
            raise ValueError(f"A tool named '{changes['name']}' already exists on this agent.")
    for key, value in changes.items():
        setattr(tool, key, value)
    db.commit()
    db.refresh(tool)
    return tool


def delete_tool(tool_id: str, db: Session) -> bool:
    tool = db.query(AgentTool).filter(AgentTool.id == tool_id).first()
    if tool is None:
        return False
    db.delete(tool)
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


def parse_dnc_csv(text: str) -> list[str]:
    """Phone numbers from a suppression file: a ``phone``-ish column if
    there is a header row, otherwise the first column."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    col = 0
    has_header = False
    for i, h in enumerate(header):
        if h in _CSV_ALIASES["phone"]:
            col, has_header = i, True
            break
    if not has_header and header and not re.search(r"\d{5,}", header[0]):
        has_header = True
    out: list[str] = []
    for row in rows[1:] if has_header else rows:
        if len(row) > col and row[col].strip():
            out.append(row[col].strip())
    return out


def bulk_add_dnc(phones: list[str], db: Session, *, source: str = "import") -> models.DoNotCallImportResult:
    imported = skipped = 0
    for raw in phones:
        try:
            phone = normalize_phone(raw)
        except ValueError:
            skipped += 1
            continue
        if db.query(DoNotCallEntry).filter(DoNotCallEntry.phone == phone).first():
            skipped += 1
            continue
        db.add(DoNotCallEntry(phone=phone, source=source))
        db.query(Contact).filter(Contact.phone == phone).update(
            {Contact.status: "do_not_call", Contact.next_attempt_at: None}, synchronize_session=False
        )
        imported += 1
    db.commit()
    return models.DoNotCallImportResult(imported=imported, skipped=skipped)


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


def within_schedule(agent: VoiceAgent, now: datetime | None = None) -> str:
    """ "before" / "open" / "after" relative to the campaign window."""
    now = now or utcnow()
    if agent.schedule_start_at and now < agent.schedule_start_at:
        return "before"
    if agent.schedule_end_at and now >= agent.schedule_end_at:
        return "after"
    return "open"


def calls_started_today(agent_id: str, db: Session, now: datetime | None = None) -> int:
    now = now or utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (
        db.query(Call).filter(Call.agent_id == agent_id, Call.started_at >= start, Call.direction != SIMULATION).count()
    )


def count_in_progress_calls(agent_id: str, db: Session) -> int:
    return (
        db.query(Call)
        .filter(Call.agent_id == agent_id, Call.status == "in_progress", Call.direction != SIMULATION)
        .count()
    )


def _dialable_query(agent: VoiceAgent, db: Session, now: datetime):
    q = db.query(Contact).filter(
        Contact.agent_id == agent.id,
        Contact.is_test.is_(False),
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
        Contact.is_test.is_(False),
        Contact.status.in_(["new", "callback"]),
        Contact.attempts < agent.max_attempts,
    )
    if agent.require_consent:
        q = q.filter(Contact.consent.is_(True))
    return db.query(q.exists()).scalar()


def pick_next_contact(agent: VoiceAgent, db: Session, now_utc: datetime | None = None) -> Contact | None:
    """The next contact the dialler should ring, or None if nothing is
    dialable right now. Applies, in order: agent status, schedule window,
    daily cap, attempt cap + retry timing, DNC list, consent, calling window."""
    now = now_utc or datetime.now(UTC)
    now_naive = now.astimezone(UTC).replace(tzinfo=None)
    if agent.status != "active":
        return None
    if within_schedule(agent, now_naive) != "open":
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
        if active_call_for_contact(contact.id, db):
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
    agent_turn: CallTurn | None
    customer_turn: CallTurn | None
    ended: bool
    outcome: str | None
    ticket: Ticket | None = None
    generation_ids: list[str] = field(default_factory=list)
    appointment_id: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    awaiting_operator: bool = False
    stt_ms: int | None = None
    llm_ms: int | None = None

    @property
    def generation_id(self) -> str | None:
        return self.generation_ids[0] if self.generation_ids else None

    @property
    def text(self) -> str:
        return self.agent_turn.text if self.agent_turn is not None else ""


def get_call(call_id: str, db: Session) -> Call | None:
    return db.query(Call).filter(Call.id == call_id).first()


def get_turns(call_id: str, db: Session) -> list[CallTurn]:
    return db.query(CallTurn).filter(CallTurn.call_id == call_id).order_by(CallTurn.created_at.asc()).all()


def call_to_response(call: Call, db: Session, *, include_turns: bool = True) -> models.CallResponse:
    resp = models.CallResponse.model_validate(call)
    if include_turns:
        resp.turns = [models.CallTurnResponse.model_validate(t) for t in get_turns(call.id, db)]
    return resp


def list_calls(
    agent_id: str,
    db: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    status: str | None = None,
    include_simulations: bool = True,
):
    q = db.query(Call).filter(Call.agent_id == agent_id)
    if status:
        q = q.filter(Call.status == status)
    if not include_simulations:
        q = q.filter(Call.direction != SIMULATION)
    total = q.count()
    rows = q.order_by(Call.started_at.desc()).offset(offset).limit(limit).all()
    return rows, total


def active_call_for_contact(contact_id: str, db: Session) -> Call | None:
    return db.query(Call).filter(Call.contact_id == contact_id, Call.status == "in_progress").first()


def _add_flag(call: Call, flag: str) -> None:
    flags = list(call.flags or [])
    if flag not in flags:
        flags.append(flag)
        call.flags = flags


def _add_turn(
    call: Call,
    role: str,
    text: str,
    db: Session,
    *,
    source: str = "llm",
    sentiment: float | None = None,
    generation_ids: list[str] | None = None,
    capture_id: str | None = None,
    stt_ms: int | None = None,
    llm_ms: int | None = None,
    tool_name: str | None = None,
    meta: dict | None = None,
) -> CallTurn:
    turn = CallTurn(
        id=str(uuid.uuid4()),
        call_id=call.id,
        role=role,
        text=text,
        source=source,
        sentiment=sentiment,
        generation_id=generation_ids[0] if generation_ids else None,
        generation_ids=list(generation_ids) if generation_ids else None,
        capture_id=capture_id,
        stt_ms=stt_ms,
        llm_ms=llm_ms,
        tool_name=tool_name,
        meta=meta,
    )
    db.add(turn)
    call.turn_count = (call.turn_count or 0) + 1
    call.last_activity_at = utcnow()
    db.commit()
    db.refresh(turn)
    return turn


def _turn_payload(turn: CallTurn) -> dict:
    return {
        "turn_id": turn.id,
        "role": turn.role,
        "text": turn.text,
        "source": turn.source,
        "generation_id": turn.generation_id,
        "generation_ids": list(turn.generation_ids or []),
        "sentiment": turn.sentiment,
        "stt_ms": turn.stt_ms,
        "llm_ms": turn.llm_ms,
        "tool_name": turn.tool_name,
        "meta": turn.meta,
    }


async def _generate(
    agent: VoiceAgent,
    text: str,
    db: Session,
    *,
    source: str,
    instruct: str | None = None,
    language: str | None = None,
) -> str | None:
    """Queue one TTS generation of ``text`` in the agent's voice and return
    its id, or None when TTS could not be queued (the call continues as
    text so a missing model never strands a live conversation)."""
    from ..routes.generations import generate_speech  # lazy: heavy import chain

    req = models.GenerationRequest(
        profile_id=agent.profile_id,
        text=text,
        language=language or agent.language or "en",
        engine=agent.engine,
        model_size=None,
        instruct=(instruct or None) and str(instruct)[:500],
    )
    try:
        generation = await generate_speech(req, db)
    except Exception:
        logger.exception("Voice agent TTS failed for agent %s", agent.id)
        return None
    generation_id = getattr(generation, "id", None)
    if not generation_id:
        return None
    from ..database import Generation as DBGeneration

    row = db.query(DBGeneration).filter(DBGeneration.id == generation_id).first()
    if row is not None:
        row.source = source
        db.commit()
    return generation_id


async def _voice(
    agent: VoiceAgent,
    call: Call,
    text: str,
    db: Session,
    *,
    instruct: str | None = None,
    language: str | None = None,
    client_plays: bool = False,
) -> list[str]:
    """Voice ``text`` for a call. Returns the generation ids in playback
    order (two when fast-first-audio split the reply).

    ``client_plays`` means the live console will play the audio itself:
    the reply may be split for lower first-audio latency and the speaking
    pill is not signalled (it would double-play on desktop).
    """
    if call.direction == SIMULATION:
        return []
    remote = call.provider not in ("local", SIMULATION)
    source = SOURCE_REMOTE if remote else SOURCE_LOCAL
    pieces = [text]
    if not remote and client_plays and agent.fast_first_audio:
        first, rest = prompts.split_first_sentence(text)
        pieces = [first, rest] if rest else [first]
    ids: list[str] = []
    for piece in pieces:
        gen_id = await _generate(agent, piece, db, source=source, instruct=instruct, language=language)
        if gen_id:
            ids.append(gen_id)
    if ids and not remote and not client_plays:
        _notify_pill(agent, ids[0], db)
    return ids


def _notify_pill(agent: VoiceAgent, generation_id: str, db: Session) -> None:
    """Surface the desktop speaking pill for local agent speech. Best
    effort — never lets a missing MCP stack break a call."""
    try:
        from ..mcp_server import events as mcp_events  # local: avoid import cycle at module load

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
    except Exception:
        logger.debug("speak-start publish failed", exc_info=True)


def _variables(agent: VoiceAgent, contact: Contact) -> dict[str, str]:
    now_local = datetime.now(UTC).astimezone(_zone(contact.timezone, agent.timezone))
    return prompts.template_variables(
        agent_name=agent.agent_name,
        company_name=agent.company_name,
        contact_name=contact.name if contact.name != contact.phone else None,
        contact_phone=contact.phone,
        contact_company=contact.company,
        custom_fields=contact.custom_fields,
        now_local=now_local,
    )


def pick_variant(agent: VoiceAgent) -> str | None:
    variants = [v for v in (agent.variants or []) if v.get("name")]
    if not variants:
        return None
    weights = [max(1, int(v.get("weight") or 1)) for v in variants]
    return random.choices(variants, weights=weights, k=1)[0]["name"]


def _effective_script(agent: VoiceAgent, variant_name: str | None) -> tuple[str, str, str | None]:
    """(brief, goal, opening_line) with the variant's overrides applied."""
    brief, goal, opening = agent.brief, agent.goal, agent.opening_line
    if variant_name:
        for v in agent.variants or []:
            if v.get("name") == variant_name:
                brief = v.get("brief") or brief
                goal = v.get("goal") or goal
                opening = v.get("opening_line") or opening
                break
    return brief, goal, opening


def _reply_language(agent: VoiceAgent, contact: Contact) -> str:
    return contact.language or agent.language or "en"


def _style_for(agent: VoiceAgent, sentiment: float | None) -> str | None:
    if sentiment is not None and sentiment <= -0.3 and agent.empathetic_voice_style:
        return agent.empathetic_voice_style
    return agent.voice_style or None


async def start_outbound_call(
    agent: VoiceAgent,
    contact: Contact,
    db: Session,
    *,
    voice: bool = True,
    client_plays: bool = False,
    variant: str | None = None,
) -> TurnResult:
    """Open a call to ``contact`` and produce the opening line.

    Provider dialling happens in the runner / route after this returns so
    the call row exists (Twilio needs its id for the webhook URL).
    """
    if agent.status not in ("active", "draft", "paused"):
        raise ValueError("Agent is completed; reactivate it to place calls.")
    if contact.is_test:
        raise ValueError("Test contacts can only be used in simulations.")
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
        variant=variant or pick_variant(agent),
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return await _open(agent, contact, call, db, voice=voice, client_plays=client_plays)


async def start_inbound_call(
    agent: VoiceAgent,
    contact: Contact,
    db: Session,
    *,
    voice: bool = True,
    client_plays: bool = False,
    provider_call_id: str | None = None,
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
        variant=pick_variant(agent),
    )
    contact.status = "calling"
    contact.last_attempt_at = utcnow()
    db.add(call)
    db.commit()
    db.refresh(call)
    return await _open(agent, contact, call, db, voice=voice, client_plays=client_plays)


async def _open(
    agent: VoiceAgent, contact: Contact, call: Call, db: Session, *, voice: bool, client_plays: bool
) -> TurnResult:
    _, _, opening_hook = _effective_script(agent, call.variant)
    variables = _variables(agent, contact)
    opening = prompts.build_opening_line(
        mode=agent.mode,
        agent_name=agent.agent_name,
        company_name=agent.company_name,
        disclosure=agent.disclosure,
        contact_name=contact.name if contact.name != contact.phone else None,
        custom_opening=prompts.render_template(opening_hook, variables) if opening_hook else None,
    )
    ids = (
        await _voice(
            agent,
            call,
            opening,
            db,
            instruct=agent.voice_style,
            language=_reply_language(agent, contact),
            client_plays=client_plays,
        )
        if voice
        else []
    )
    turn = _add_turn(call, "agent", opening, db, source="system", generation_ids=ids)
    call.stage = "conversation"
    db.commit()
    events.publish(call.id, "agent_turn", {**_turn_payload(turn), "ended": False})
    return TurnResult(call=call, agent_turn=turn, customer_turn=None, ended=False, outcome=None, generation_ids=ids)


def _history_pairs(turns: list[CallTurn]) -> list[tuple[str, str]]:
    """Fold the transcript into (customer, agent) pairs for the chat
    template. The opening line has no customer turn before it, so it is
    paired with a synthetic "(call connected)" user message. Tool turns
    stay out; interrupted agent turns are marked so the model knows the
    person didn't hear the end."""
    pairs: list[tuple[str, str]] = []
    pending_customer: str | None = None
    pending_agent: str | None = None
    for t in turns:
        if t.role == "tool":
            continue
        if t.role == "customer":
            if pending_agent is not None:
                pairs.append((pending_customer or "(call connected)", pending_agent))
                pending_customer, pending_agent = None, None
            pending_customer = (pending_customer + " " + t.text) if pending_customer else t.text
        else:
            text = t.text + (" [customer interrupted]" if t.interrupted else "")
            pending_agent = (pending_agent + " " + text) if pending_agent else text
    if pending_agent is not None:
        pairs.append((pending_customer or "(call connected)", pending_agent))
    return pairs[-MAX_HISTORY_PAIRS:]


async def _llm(
    agent: VoiceAgent,
    prompt: str,
    system: str,
    *,
    examples: list[tuple[str, str]] | None = None,
    max_tokens: int = REPLY_MAX_TOKENS,
    temperature: float = REPLY_TEMPERATURE,
) -> str:
    from . import llm as llm_service  # local: keeps this module importable without torch

    backend = llm_service.get_llm_model()
    size = agent.llm_model_size or backend.model_size
    loaded = backend.is_loaded() and backend.model_size == size
    if not loaded and hasattr(backend, "_is_model_cached") and not backend._is_model_cached(size):
        raise ValueError(f"Qwen3 {size} is not downloaded yet. Open Voicebox → Settings → Models to download it.")
    return await backend.generate(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model_size=size,
        examples=examples or None,
    )


def _build_system(agent: VoiceAgent, contact: Contact, call: Call, db: Session, *, extra_notes: str = "") -> str:
    turns = get_turns(call.id, db)
    recent_customer = [t.text for t in turns if t.role == "customer"]
    kb = knowledge.retrieve_for_turn(db, agent.id, recent_customer)
    brief, goal, _ = _effective_script(agent, call.variant)
    variables = _variables(agent, contact)
    tools_section = ""
    if agent.tools_enabled:
        tools_section = prompts.build_tools_section(agent_tools.prompt_lines(agent, list_tools(agent.id, db)))
    return prompts.build_system_prompt(
        mode=agent.mode,
        agent_name=agent.agent_name,
        company_name=agent.company_name,
        brief=prompts.render_template(brief, variables),
        goal=prompts.render_template(goal, variables),
        objection_notes=prompts.render_template(agent.objection_notes, variables) or None,
        persona=prompts.render_template(agent.persona, variables) or None,
        escalation_promise=agent.escalation_promise,
        contact_name=contact.name if contact.name != contact.phone else None,
        contact_company=contact.company,
        contact_notes=contact.notes,
        contact_memory=contact.memory,
        knowledge=kb,
        reply_language=_reply_language(agent, contact),
        tools_section=tools_section,
        extra_notes=extra_notes,
    )


async def handle_customer_turn(
    call: Call,
    customer_text: str,
    db: Session,
    *,
    voice: bool = True,
    capture_id: str | None = None,
    stt_ms: int | None = None,
    client_plays: bool = False,
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
    if agent.redact_pii:
        customer_text, kinds = prompts.redact_pii(customer_text)
        if kinds:
            _add_flag(call, "pii_redacted")

    sentiment = prompts.score_sentiment(customer_text)
    customer_turn = _add_turn(call, "customer", customer_text, db, sentiment=sentiment, capture_id=capture_id)
    if sentiment <= -0.3:
        call.negative_streak = (call.negative_streak or 0) + 1
    elif sentiment >= 0.0:
        call.negative_streak = 0
    db.commit()
    events.publish(call.id, "customer_turn", _turn_payload(customer_turn))

    language = _reply_language(agent, contact)
    style = _style_for(agent, sentiment)

    # ── Hard stops that never go through the model ──
    intent = prompts.classify_customer_intent(customer_text, agent.mode)
    if intent == "opt_out":
        add_to_dnc(contact.phone, db, reason="Asked not to be called", source="opt_out")
        text = prompts.opt_out_closing(agent.company_name)
        return await _close_with(
            agent,
            contact,
            call,
            text,
            "opt_out",
            db,
            voice=voice,
            customer_turn=customer_turn,
            client_plays=client_plays,
            language=language,
        )

    if call.direction == "outbound" and call.turn_count <= 2 and prompts.detect_voicemail(customer_text):
        # First thing we heard back was a voicemail greeting — don't pitch
        # to a machine. Leave the configured message, or retry later.
        if agent.voicemail_message:
            text = prompts.render_template(agent.voicemail_message, _variables(agent, contact))
            return await _close_with(
                agent,
                contact,
                call,
                text,
                "voicemail_left",
                db,
                voice=voice,
                customer_turn=customer_turn,
                client_plays=client_plays,
                language=language,
                source="system",
            )
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
            agent,
            contact,
            call,
            text,
            "handoff",
            db,
            voice=voice,
            customer_turn=customer_turn,
            ticket=ticket,
            client_plays=client_plays,
            language=language,
            instruct=style,
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
            agent,
            contact,
            call,
            text,
            "max_turns",
            db,
            voice=voice,
            customer_turn=customer_turn,
            ticket=ticket,
            client_plays=client_plays,
            language=language,
        )

    if prompts.detect_goodbye(customer_text) and call.turn_count > 2:
        flags = set(call.flags or [])
        if agent.mode != prompts.MODE_OUTBOUND_SALES:
            outcome = "resolved"
        elif "appointment_booked" in flags:
            outcome = "interested"
        elif "callback_scheduled" in flags:
            outcome = "callback"
        else:
            outcome = intent or ("not_interested" if call.turn_count <= 4 else "callback")
        text = prompts.goodbye_closing(agent.mode)
        return await _close_with(
            agent,
            contact,
            call,
            text,
            outcome,
            db,
            voice=voice,
            customer_turn=customer_turn,
            client_plays=client_plays,
            language=language,
            instruct=style,
        )

    # ── Supervisor take-over: the model stays silent ──
    if call.ai_paused:
        events.publish(call.id, "awaiting_operator", {"customer_turn_id": customer_turn.id})
        return TurnResult(
            call=call,
            agent_turn=None,
            customer_turn=customer_turn,
            ended=False,
            outcome=None,
            awaiting_operator=True,
            stt_ms=stt_ms,
        )

    # ── Filler while the model thinks ──
    if voice and client_plays and call.provider == "local":
        filler = _pick_filler(agent, db)
        if filler:
            events.publish(call.id, "filler", {"text": filler[0], "generation_id": filler[1]})

    # ── The model's turn ──
    notes = ""
    if prompts.detect_injection(customer_text):
        _add_flag(call, "injection_attempt")
        db.commit()
        notes = prompts.INJECTION_NOTE
    system = _build_system(agent, contact, call, db, extra_notes=notes)
    history = _history_pairs(get_turns(call.id, db)[:-1])  # everything before the turn we're answering

    llm_ms = 0
    generation_ids: list[str] = []
    tool_calls: list[dict] = []
    appointment_id: str | None = None
    ticket: Ticket | None = None
    prompt = customer_text
    examples = list(history)
    parsed: prompts.ParsedReply | None = None
    try:
        for _ in range(MAX_TOOL_CALLS_PER_TURN + 1):
            t0 = time.perf_counter()
            raw = await _llm(agent, prompt, system, examples=examples)
            llm_ms += int((time.perf_counter() - t0) * 1000)
            parsed = prompts.parse_agent_reply(raw, agent.mode)
            if not (parsed.tool_name and agent.tools_enabled) or len(tool_calls) >= MAX_TOOL_CALLS_PER_TURN:
                break
            # Voice the holding line right away, then run the tool.
            if parsed.text:
                ids = (
                    await _voice(
                        agent, call, parsed.text, db, instruct=style, language=language, client_plays=client_plays
                    )
                    if voice
                    else []
                )
                holding = _add_turn(call, "agent", parsed.text, db, generation_ids=ids, meta={"holding": True})
                generation_ids.extend(ids)
                events.publish(call.id, "agent_turn", {**_turn_payload(holding), "ended": False, "holding": True})
            custom = list_tools(agent.id, db)
            t1 = time.perf_counter()
            result = await agent_tools.execute(
                parsed.tool_name, parsed.tool_args, agent=agent, contact=contact, call=call, db=db, custom_tools=custom
            )
            tool_ms = int((time.perf_counter() - t1) * 1000)
            record = {
                "name": parsed.tool_name,
                "args": parsed.tool_args,
                "ok": result.ok,
                "result": result.text,
                "ms": tool_ms,
            }
            tool_calls.append(record)
            _add_turn(call, "tool", result.text, db, source="tool", tool_name=parsed.tool_name, meta=record)
            events.publish(call.id, "tool_call", record)
            for flag in result.flags:
                _add_flag(call, flag)
            db.commit()
            if result.appointment_id:
                appointment_id = result.appointment_id
            if result.end_outcome == "handoff":
                ticket = _create_ticket(
                    agent,
                    contact,
                    call,
                    db,
                    kind="handoff",
                    priority="normal",
                    subject=f"Transfer requested: {result.data.get('reason') or 'no reason given'}",
                    description=_transcript_text(call, db),
                )
                text = prompts.handoff_closing(agent.escalation_promise)
                res = await _close_with(
                    agent,
                    contact,
                    call,
                    text,
                    "handoff",
                    db,
                    voice=voice,
                    customer_turn=customer_turn,
                    ticket=ticket,
                    client_plays=client_plays,
                    language=language,
                    instruct=style,
                )
                res.generation_ids = generation_ids + res.generation_ids
                res.tool_calls = tool_calls
                res.stt_ms, res.llm_ms = stt_ms, llm_ms
                return res
            examples = [
                *examples,
                (prompt, f"{parsed.text} [TOOL: {parsed.tool_name} {json.dumps(parsed.tool_args)}]".strip()),
            ]
            prompt = prompts.TOOL_RESULT_PROMPT.format(name=parsed.tool_name, result=result.text)
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
            agent,
            contact,
            call,
            text,
            "error",
            db,
            voice=voice,
            customer_turn=customer_turn,
            ticket=ticket,
            client_plays=client_plays,
            language=language,
        )

    assert parsed is not None
    reply_text = parsed.text or "Sorry, could you say that again?"

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
        res = await _close_with(
            agent,
            contact,
            call,
            reply_text,
            outcome,
            db,
            voice=voice,
            customer_turn=customer_turn,
            ticket=ticket,
            client_plays=client_plays,
            language=language,
            instruct=style,
            llm_ms=llm_ms,
            stt_ms=stt_ms,
        )
        res.generation_ids = generation_ids + res.generation_ids
        res.tool_calls, res.appointment_id = tool_calls, appointment_id
        return res

    ids = (
        await _voice(agent, call, reply_text, db, instruct=style, language=language, client_plays=client_plays)
        if voice
        else []
    )
    agent_turn = _add_turn(call, "agent", reply_text, db, generation_ids=ids, stt_ms=stt_ms, llm_ms=llm_ms)
    events.publish(call.id, "agent_turn", {**_turn_payload(agent_turn), "ended": False})
    return TurnResult(
        call=call,
        agent_turn=agent_turn,
        customer_turn=customer_turn,
        ended=False,
        outcome=None,
        generation_ids=generation_ids + ids,
        tool_calls=tool_calls,
        appointment_id=appointment_id,
        stt_ms=stt_ms,
        llm_ms=llm_ms,
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
    client_plays: bool = False,
    language: str | None = None,
    instruct: str | None = None,
    source: str = "llm",
    llm_ms: int | None = None,
    stt_ms: int | None = None,
) -> TurnResult:
    call.stage = "closing"
    ids = (
        await _voice(agent, call, text, db, instruct=instruct, language=language, client_plays=client_plays)
        if (voice and speak and text)
        else []
    )
    agent_turn = _add_turn(
        call, "agent", text or "(hung up)", db, source=source, generation_ids=ids, llm_ms=llm_ms, stt_ms=stt_ms
    )
    events.publish(call.id, "agent_turn", {**_turn_payload(agent_turn), "ended": True, "outcome": outcome})
    await end_call(call, outcome, db)
    return TurnResult(
        call=call,
        agent_turn=agent_turn,
        customer_turn=customer_turn,
        ended=True,
        outcome=outcome,
        ticket=ticket,
        generation_ids=ids,
        stt_ms=stt_ms,
        llm_ms=llm_ms,
    )


async def interrupt(call: Call, db: Session, *, turn_id: str | None = None) -> CallTurn | None:
    """The caller spoke over the agent: cancel any audio still generating
    and mark the turn so the model knows it wasn't heard to the end."""
    turns = [t for t in get_turns(call.id, db) if t.role == "agent"]
    if not turns:
        return None
    turn = next((t for t in turns if t.id == turn_id), turns[-1]) if turn_id else turns[-1]
    try:
        from .task_queue import cancel_generation

        for gen_id in turn.generation_ids or ([turn.generation_id] if turn.generation_id else []):
            cancel_generation(gen_id)
    except Exception:
        logger.debug("cancel_generation failed for call %s", call.id, exc_info=True)
    turn.interrupted = True
    call.last_activity_at = utcnow()
    db.commit()
    events.publish(call.id, "interrupted", {"turn_id": turn.id})
    return turn


async def operator_say(
    call: Call, text: str, db: Session, *, voice: bool = True, client_plays: bool = False
) -> TurnResult:
    """A supervisor speaks as the agent (voiced in the agent's voice)."""
    if call.status != "in_progress":
        raise ValueError("Call has already ended.")
    agent = get_agent(call.agent_id, db)
    contact = get_contact(call.contact_id, db)
    if agent is None or contact is None:
        raise ValueError("Call refers to a missing agent or contact.")
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to say.")
    ids = (
        await _voice(
            agent,
            call,
            text,
            db,
            instruct=agent.voice_style,
            language=_reply_language(agent, contact),
            client_plays=client_plays,
        )
        if voice
        else []
    )
    turn = _add_turn(call, "agent", text, db, source="operator", generation_ids=ids)
    events.publish(call.id, "agent_turn", {**_turn_payload(turn), "ended": False})
    return TurnResult(call=call, agent_turn=turn, customer_turn=None, ended=False, outcome=None, generation_ids=ids)


def set_ai_paused(call: Call, paused: bool, db: Session) -> Call:
    if call.status != "in_progress":
        raise ValueError("Call has already ended.")
    call.ai_paused = bool(paused)
    call.last_activity_at = utcnow()
    db.commit()
    events.publish(call.id, "ai_paused", {"paused": call.ai_paused})
    return call


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
    "voicemail_left": "new",
    "max_turns": "contacted",
    "error": "new",
}
_RETRY_OUTCOMES = frozenset({"no_answer", "voicemail", "voicemail_left", "error"})


async def end_call(call: Call, outcome: str, db: Session, *, summary: str | None = None) -> Call:
    """Finish a call: set outcome, update the contact, schedule retries or
    callbacks, and (best effort) summarise / analyse the transcript, text
    a follow-up, and fire the webhook."""
    if call.status != "in_progress":
        return call
    agent = get_agent(call.agent_id, db)
    contact = get_contact(call.contact_id, db)
    now = utcnow()
    call.status = "failed" if outcome == "error" else "completed"
    call.stage = "ended"
    call.outcome = outcome
    call.ended_at = now
    call.ai_paused = False
    simulation = call.direction == SIMULATION

    if agent is not None and contact is not None and not simulation:
        contact.last_outcome = outcome
        status = _OUTCOME_CONTACT_STATUS.get(outcome, "contacted")
        if outcome in _RETRY_OUTCOMES:
            if contact.attempts >= agent.max_attempts:
                status = "exhausted"
                contact.next_attempt_at = None
            else:
                contact.next_attempt_at = now + timedelta(hours=agent.retry_delay_hours)
        elif outcome == "callback":
            if not contact.next_attempt_at or contact.next_attempt_at <= now:
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
    if call.summary and contact is not None and not simulation:
        stamp = now.strftime("%Y-%m-%d")
        entry = f"[{stamp}] {call.summary.strip()}"
        contact.memory = (contact.memory + "\n" + entry) if contact.memory else entry
        # Keep memory bounded so the system prompt stays small.
        contact.memory = contact.memory[-4000:]
    db.commit()

    if agent is not None and call.turn_count >= 3:
        await _analyze(agent, contact, call, db)

    if agent is not None and contact is not None and not simulation:
        await _send_followup_sms(agent, contact, call, db)
        _dispatch_webhook(agent, contact, call, db)

    db.commit()
    db.refresh(call)
    events.publish(call.id, "ended", {"outcome": outcome, "summary": call.summary, "score": call.score})
    return call


async def _summarize(agent: VoiceAgent, contact: Contact | None, call: Call, db: Session) -> str | None:
    """LLM summary of the call. Never raises — a missing model just means
    no summary."""
    turns = [(t.role, t.text) for t in get_turns(call.id, db) if t.role != "tool" and t.text and t.text != "(hung up)"]
    if len(turns) < 2:
        return None
    try:
        raw = await _llm(
            agent,
            prompts.build_summary_prompt(turns, contact.name if contact else None),
            prompts.SUMMARY_SYSTEM,
            max_tokens=200,
            temperature=0.2,
        )
        text = prompts.parse_agent_reply(raw, agent.mode).text
        return text[:1500] or None
    except Exception:
        logger.debug("Call summary failed for %s", call.id, exc_info=True)
        return None


async def _analyze(agent: VoiceAgent, contact: Contact | None, call: Call, db: Session) -> None:
    """Structured post-call extraction + a 0-100 goal score. Best effort."""
    schema = [dict(f) for f in (agent.analysis_schema or [])]
    transcript = _transcript_text(call, db)
    if not transcript.strip():
        return
    _, goal, _ = _effective_script(agent, call.variant)
    try:
        raw = await _llm(
            agent,
            prompts.build_analysis_prompt(schema, goal, transcript, contact.name if contact else None),
            prompts.ANALYSIS_SYSTEM,
            max_tokens=400,
            temperature=0.1,
        )
    except Exception:
        logger.debug("Call analysis failed for %s", call.id, exc_info=True)
        return
    answers, score, reason = prompts.parse_analysis(raw, schema)
    call.analysis = answers or None
    call.score = score
    call.score_reason = reason
    db.commit()


async def _send_followup_sms(agent: VoiceAgent, contact: Contact, call: Call, db: Session) -> None:
    if not agent.sms_followup_template or call.outcome not in (agent.sms_followup_outcomes or []):
        return
    if (agent.provider or "local") != "twilio":
        return
    body = prompts.render_template(agent.sms_followup_template, _variables(agent, contact)).strip()[:640]
    if not body:
        return
    row = Message(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        contact_id=contact.id,
        call_id=call.id,
        to_number=contact.phone,
        body=body,
    )
    try:
        sid = await telephony.get_provider(agent.provider).send_sms(
            to_number=contact.phone, from_number=agent.from_number, body=body
        )
        row.status, row.provider_message_id = "sent", sid
    except Exception as exc:
        row.status, row.error = "failed", str(exc)[:500]
    db.add(row)
    db.commit()


def _dispatch_webhook(agent: VoiceAgent, contact: Contact, call: Call, db: Session) -> None:
    if not agent.webhook_url:
        return
    payload = build_webhook_payload(agent, contact, call, db)
    try:
        webhooks.dispatch(agent.id, call.id, agent.webhook_url, agent.webhook_secret, "call.ended", payload, db)
        call.webhook_status = "pending"
    except Exception:
        logger.debug("webhook dispatch failed for %s", call.id, exc_info=True)
        call.webhook_status = "failed"


def build_webhook_payload(agent: VoiceAgent, contact: Contact, call: Call, db: Session) -> dict:
    turns = get_turns(call.id, db)
    tickets = db.query(Ticket).filter(Ticket.call_id == call.id).all()
    appts = db.query(Appointment).filter(Appointment.call_id == call.id).all()
    return {
        "event": "call.ended",
        "sent_at": datetime.now(UTC).isoformat(),
        "agent": {"id": agent.id, "name": agent.name, "mode": agent.mode, "version": agent.version},
        "call": {
            "id": call.id,
            "direction": call.direction,
            "status": call.status,
            "outcome": call.outcome,
            "variant": call.variant,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            "turn_count": call.turn_count,
            "summary": call.summary,
            "analysis": call.analysis,
            "score": call.score,
            "score_reason": call.score_reason,
            "flags": call.flags or [],
        },
        "contact": {
            "id": contact.id,
            "name": contact.name,
            "phone": contact.phone,
            "company": contact.company,
            "status": contact.status,
            "custom_fields": contact.custom_fields or {},
        },
        "transcript": [
            {
                "role": t.role,
                "text": t.text,
                "source": t.source,
                "sentiment": t.sentiment,
                "at": t.created_at.isoformat(),
            }
            for t in turns
        ],
        "tickets": [{"id": t.id, "kind": t.kind, "priority": t.priority, "subject": t.subject} for t in tickets],
        "appointments": [
            {
                "id": a.id,
                "starts_at": a.starts_at.isoformat(),
                "ends_at": a.ends_at.isoformat(),
                "timezone": a.timezone,
                "notes": a.notes,
            }
            for a in appts
        ],
    }


def _transcript_text(call: Call, db: Session) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in get_turns(call.id, db) if t.role != "tool")


# ── Simulation ─────────────────────────────────────────────────────────


async def simulate_call(
    agent: VoiceAgent,
    persona: str,
    db: Session,
    *,
    max_turns: int = 12,
    variant: str | None = None,
) -> Call:
    """Run a test conversation: the LLM plays a customer with ``persona``
    against the agent's real prompt, tools and knowledge (no audio)."""
    contact = get_or_create_test_contact(agent.id, db)
    existing = active_call_for_contact(contact.id, db)
    if existing is not None:
        await end_call(existing, "unresolved", db)
    if variant and not any(v.get("name") == variant for v in (agent.variants or [])):
        raise ValueError(f"Unknown variant '{variant}'.")
    call = Call(
        id=str(uuid.uuid4()),
        agent_id=agent.id,
        contact_id=contact.id,
        direction=SIMULATION,
        provider=SIMULATION,
        stage="opening",
        variant=variant or pick_variant(agent),
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    result = await _open(agent, contact, call, db, voice=False, client_plays=False)
    system = prompts.build_customer_sim_prompt(persona, agent.company_name)
    customer_lines = 0
    while not result.ended and customer_lines < max_turns:
        # The customer model sees the agent's lines as the "user" side.
        turns = [t for t in get_turns(call.id, db) if t.role != "tool"]
        examples: list[tuple[str, str]] = []
        pending_agent: str | None = None
        for t in turns[:-1]:
            if t.role == "agent":
                pending_agent = (pending_agent + " " + t.text) if pending_agent else t.text
            elif pending_agent is not None:
                examples.append((pending_agent, t.text))
                pending_agent = None
        last_agent = turns[-1].text if turns and turns[-1].role == "agent" else (pending_agent or "(silence)")
        try:
            raw = await _llm(
                agent, last_agent, system, examples=examples[-MAX_HISTORY_PAIRS:], max_tokens=120, temperature=0.8
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.exception("simulated customer failed on %s", call.id)
            await end_call(call, "error", db, summary=f"Simulation error: {exc}")
            break
        line = prompts.clean_customer_sim_line(raw) or "Sorry, say that again?"
        customer_lines += 1
        result = await handle_customer_turn(call, line, db, voice=False)
    if not result.ended:
        await end_call(call, "max_turns", db)
    db.refresh(call)
    return call


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


# ── Appointments, messages, deliveries ─────────────────────────────────


def list_appointments(agent_id: str, db: Session, *, upcoming_only: bool = False, limit: int = 200):
    q = db.query(Appointment).filter(Appointment.agent_id == agent_id)
    if upcoming_only:
        q = q.filter(Appointment.starts_at >= utcnow(), Appointment.status.in_(["booked", "confirmed"]))
    return q.order_by(Appointment.starts_at.asc()).limit(limit).all()


def update_appointment(appointment_id: str, data: models.AppointmentUpdate, db: Session) -> Appointment | None:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt is None:
        return None
    changes = data.model_dump(exclude_unset=True)
    for key in ("starts_at", "ends_at"):
        if changes.get(key) is not None and changes[key].tzinfo is not None:
            changes[key] = changes[key].astimezone(UTC).replace(tzinfo=None)
    for key, value in changes.items():
        setattr(appt, key, value)
    if appt.ends_at <= appt.starts_at:
        raise ValueError("ends_at must be after starts_at.")
    db.commit()
    db.refresh(appt)
    return appt


def appointment_ics(appt: Appointment, agent: VoiceAgent, contact: Contact | None) -> str:
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%SZ")

    who = contact.name if contact else "Customer"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Voicebox//Voice Agent//EN",
        "BEGIN:VEVENT",
        f"UID:{appt.id}@voicebox",
        f"DTSTAMP:{_fmt(appt.created_at or utcnow())}",
        f"DTSTART:{_fmt(appt.starts_at)}",
        f"DTEND:{_fmt(appt.ends_at)}",
        f"SUMMARY:{agent.company_name} — {who}",
        f"DESCRIPTION:{(appt.notes or '').replace(chr(10), ' ')} Booked by {agent.agent_name} (voice agent).",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def list_messages(agent_id: str, db: Session, *, limit: int = 200) -> list[Message]:
    return db.query(Message).filter(Message.agent_id == agent_id).order_by(Message.created_at.desc()).limit(limit).all()


def list_deliveries(agent_id: str, db: Session, *, limit: int = 200) -> list[WebhookDelivery]:
    return (
        db.query(WebhookDelivery)
        .filter(WebhookDelivery.agent_id == agent_id)
        .order_by(WebhookDelivery.created_at.desc())
        .limit(limit)
        .all()
    )


# ── Stats & analytics ──────────────────────────────────────────────────


def _goal_outcome(agent: VoiceAgent) -> str:
    return "interested" if agent.mode == prompts.MODE_OUTBOUND_SALES else "resolved"


def agent_stats(agent: VoiceAgent, db: Session) -> models.VoiceAgentStats:
    contacts_by_status = {
        status: n
        for status, n in db.query(Contact.status, func.count(Contact.id))
        .filter(Contact.agent_id == agent.id, Contact.is_test.is_(False))
        .group_by(Contact.status)
        .all()
    }
    real_calls = db.query(Call).filter(Call.agent_id == agent.id, Call.direction != SIMULATION)
    calls_by_outcome = {
        (outcome or "in_progress"): n
        for outcome, n in db.query(Call.outcome, func.count(Call.id))
        .filter(Call.agent_id == agent.id, Call.direction != SIMULATION)
        .group_by(Call.outcome)
        .all()
    }
    calls_total = sum(calls_by_outcome.values())
    avg_turns = real_calls.filter(Call.status != "in_progress").with_entities(func.avg(Call.turn_count)).scalar() or 0.0
    avg_score = real_calls.filter(Call.score.isnot(None)).with_entities(func.avg(Call.score)).scalar()
    good = calls_by_outcome.get(_goal_outcome(agent), 0)
    finished = calls_total - calls_by_outcome.get("in_progress", 0)
    resolution_rate = (good / finished) if finished else 0.0
    open_tickets = (
        db.query(Ticket).filter(Ticket.agent_id == agent.id, Ticket.status.in_(["open", "in_progress"])).count()
    )
    upcoming = len(list_appointments(agent.id, db, upcoming_only=True))
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
        appointments_upcoming=upcoming,
        avg_score=round(float(avg_score), 1) if avg_score is not None else None,
    )


def agent_analytics(agent: VoiceAgent, db: Session, *, days: int = 30) -> models.AnalyticsResponse:
    days = max(1, min(days, 365))
    since = utcnow() - timedelta(days=days)
    calls = (
        db.query(Call)
        .filter(Call.agent_id == agent.id, Call.started_at >= since, Call.direction != SIMULATION)
        .order_by(Call.started_at.asc())
        .all()
    )
    simulations = (
        db.query(Call).filter(Call.agent_id == agent.id, Call.started_at >= since, Call.direction == SIMULATION).count()
    )
    goal = _goal_outcome(agent)
    by_day: dict[str, dict] = defaultdict(lambda: {"calls": 0, "goal": 0, "by_outcome": Counter()})
    outcomes: Counter = Counter()
    durations: list[float] = []
    turns_list: list[int] = []
    scores: list[int] = []
    variants: dict[str, dict] = defaultdict(lambda: {"calls": 0, "goal": 0})
    analysis: dict[str, Counter] = defaultdict(Counter)
    connected = 0
    for c in calls:
        key = c.started_at.strftime("%Y-%m-%d")
        by_day[key]["calls"] += 1
        outcome = c.outcome or "in_progress"
        by_day[key]["by_outcome"][outcome] += 1
        outcomes[outcome] += 1
        if outcome == goal:
            by_day[key]["goal"] += 1
        if c.turn_count and c.turn_count > 2:
            connected += 1
        if c.ended_at and c.started_at:
            durations.append((c.ended_at - c.started_at).total_seconds())
        if c.status != "in_progress":
            turns_list.append(c.turn_count or 0)
        if c.score is not None:
            scores.append(c.score)
        if c.variant:
            variants[c.variant]["calls"] += 1
            if outcome == goal:
                variants[c.variant]["goal"] += 1
        for k, v in (c.analysis or {}).items():
            if isinstance(v, (bool, str)) or v is None:
                analysis[k][str(v).lower() if v is not None else "null"] += 1
    call_ids = [c.id for c in calls]
    avg_sent = avg_stt = avg_llm = None
    if call_ids:
        avg_sent = (
            db.query(func.avg(CallTurn.sentiment))
            .filter(CallTurn.call_id.in_(call_ids), CallTurn.role == "customer", CallTurn.sentiment.isnot(None))
            .scalar()
        )
        avg_stt = (
            db.query(func.avg(CallTurn.stt_ms))
            .filter(CallTurn.call_id.in_(call_ids), CallTurn.stt_ms.isnot(None))
            .scalar()
        )
        avg_llm = (
            db.query(func.avg(CallTurn.llm_ms))
            .filter(CallTurn.call_id.in_(call_ids), CallTurn.llm_ms.isnot(None))
            .scalar()
        )
    contacts_total = db.query(Contact).filter(Contact.agent_id == agent.id, Contact.is_test.is_(False)).count()
    attempted = len({c.contact_id for c in calls})
    series = [
        models.AnalyticsSeriesPoint(date=d, calls=v["calls"], goal=v["goal"], by_outcome=dict(v["by_outcome"]))
        for d, v in sorted(by_day.items())
    ]
    appointments = (
        db.query(Appointment).filter(Appointment.agent_id == agent.id, Appointment.created_at >= since).count()
    )
    return models.AnalyticsResponse(
        agent_id=agent.id,
        days=days,
        series=series,
        funnel={
            "contacts": contacts_total,
            "attempted": attempted,
            "connected": connected,
            "goal": outcomes.get(goal, 0),
        },
        outcomes=dict(outcomes),
        avg_turns=round(sum(turns_list) / len(turns_list), 2) if turns_list else 0.0,
        avg_duration_s=round(sum(durations) / len(durations), 1) if durations else 0.0,
        avg_sentiment=round(float(avg_sent), 3) if avg_sent is not None else None,
        avg_stt_ms=round(float(avg_stt), 1) if avg_stt is not None else None,
        avg_llm_ms=round(float(avg_llm), 1) if avg_llm is not None else None,
        avg_score=round(sum(scores) / len(scores), 1) if scores else None,
        variants=[
            models.AnalyticsVariant(
                name=n,
                calls=v["calls"],
                goal=v["goal"],
                goal_rate=round(v["goal"] / v["calls"], 3) if v["calls"] else 0.0,
            )
            for n, v in sorted(variants.items())
        ],
        analysis={k: dict(v) for k, v in analysis.items()},
        simulations=simulations,
        appointments=appointments,
    )


# ── Exports ────────────────────────────────────────────────────────────


def calls_csv(agent: VoiceAgent, db: Session, *, include_simulations: bool = False) -> str:
    keys = [f.get("key") for f in (agent.analysis_schema or []) if f.get("key")]
    contacts = {c.id: c for c in db.query(Contact).filter(Contact.agent_id == agent.id).all()}
    rows, _ = list_calls(agent.id, db, limit=100000, include_simulations=include_simulations)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "call_id",
            "direction",
            "started_at",
            "ended_at",
            "duration_s",
            "contact_name",
            "phone",
            "company",
            "variant",
            "outcome",
            "score",
            "score_reason",
            "turns",
            "summary",
            "flags",
            *[f"analysis.{k}" for k in keys],
        ]
    )
    for c in rows:
        contact = contacts.get(c.contact_id)
        duration = (c.ended_at - c.started_at).total_seconds() if c.ended_at and c.started_at else ""
        analysis = c.analysis or {}
        writer.writerow(
            [
                c.id,
                c.direction,
                c.started_at.isoformat() if c.started_at else "",
                c.ended_at.isoformat() if c.ended_at else "",
                round(duration) if duration != "" else "",
                contact.name if contact else "",
                contact.phone if contact else "",
                contact.company if contact else "",
                c.variant or "",
                c.outcome or "",
                c.score if c.score is not None else "",
                c.score_reason or "",
                c.turn_count,
                c.summary or "",
                ";".join(c.flags or []),
                *["" if analysis.get(k) is None else analysis.get(k) for k in keys],
            ]
        )
    return buf.getvalue()


def contacts_csv(agent_id: str, db: Session) -> str:
    rows, _ = list_contacts(agent_id, db, limit=100000)
    custom_keys = sorted({k for c in rows for k in (c.custom_fields or {})})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "name",
            "phone",
            "company",
            "status",
            "attempts",
            "last_outcome",
            "next_attempt_at",
            "consent",
            "timezone",
            "language",
            "notes",
            "memory",
            *custom_keys,
        ]
    )
    for c in rows:
        writer.writerow(
            [
                c.name,
                c.phone,
                c.company or "",
                c.status,
                c.attempts,
                c.last_outcome or "",
                c.next_attempt_at.isoformat() if c.next_attempt_at else "",
                "yes" if c.consent else "no",
                c.timezone or "",
                c.language or "",
                c.notes or "",
                c.memory or "",
                *[(c.custom_fields or {}).get(k, "") for k in custom_keys],
            ]
        )
    return buf.getvalue()


def transcript_text(call: Call, db: Session) -> str:
    agent = get_agent(call.agent_id, db)
    contact = get_contact(call.contact_id, db)
    who = {
        "agent": agent.agent_name if agent else "Agent",
        "customer": contact.name if contact else "Customer",
        "tool": "tool",
    }
    lines = [
        f"Call {call.id} — {call.direction}, {call.started_at.isoformat() if call.started_at else ''} — outcome: {call.outcome or 'in progress'}",
    ]
    if call.summary:
        lines += ["", f"Summary: {call.summary}"]
    if call.score is not None:
        lines.append(f"Score: {call.score}/100 — {call.score_reason or ''}")
    lines.append("")
    for t in get_turns(call.id, db):
        stamp = t.created_at.strftime("%H:%M:%S") if t.created_at else ""
        marker = " (interrupted)" if t.interrupted else ""
        lines.append(f"[{stamp}] {who.get(t.role, t.role)}: {t.text}{marker}")
    return "\n".join(lines) + "\n"
