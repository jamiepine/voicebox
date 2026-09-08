"""Tools the voice agent can call mid-conversation.

Two kinds:

- **Built-ins** — ``book_appointment``, ``schedule_callback``,
  ``transfer_to_human`` and (on Twilio agents) ``send_sms``. They write
  to Voicebox's own tables and never leave the machine except for SMS.
- **Custom HTTP tools** — ``va_tools`` rows the operator defines: look up
  an order, check stock, create a CRM lead. The model supplies arguments,
  we validate them against the declared params, fill ``{placeholders}``
  in the URL, and hand the response text back to the model.

The model invokes a tool with ``[TOOL: name {json}]`` (see
``voice_agent_prompts.parse_tool_call``); ``execute`` returns a
:class:`ToolResult` whose ``text`` goes straight back into the prompt.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from ..database import AgentTool, Appointment, Call, Contact, Message, VoiceAgent
from . import telephony
from .voice_agent_knowledge import check_fetch_url

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 1500


@dataclass
class ToolSpec:
    name: str
    description: str
    params: list[dict]  # {"name", "type", "description", "required"}
    builtin: bool = True

    def prompt_line(self) -> str:
        sig = ", ".join(
            f"{p['name']}: {p.get('type', 'string')}" + ("" if p.get("required", True) else " (optional)")
            for p in self.params
        )
        return f"{self.name}({sig}) — {self.description}"


@dataclass
class ToolResult:
    ok: bool
    text: str
    data: dict = field(default_factory=dict)
    # Set when the tool decides the call's fate (transfer → handoff).
    end_outcome: str | None = None
    appointment_id: str | None = None
    message_id: str | None = None
    flags: list[str] = field(default_factory=list)


# ── Specs ──────────────────────────────────────────────────────────────


def builtin_specs(agent: VoiceAgent) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    if agent.mode == "outbound_sales" or agent.booking_instructions:
        rules = f" Booking rules: {agent.booking_instructions.strip()}" if agent.booking_instructions else ""
        specs.append(
            ToolSpec(
                "book_appointment",
                "Book the person's appointment once they have agreed a day and time. Pass exactly what they said "
                f'for "when" (for example "next Tuesday at 10am").{rules}',
                [
                    {"name": "when", "type": "string", "required": True},
                    {"name": "notes", "type": "string", "required": False},
                ],
            )
        )
    specs.append(
        ToolSpec(
            "schedule_callback",
            'Arrange to call the person back at the time they asked for. Pass what they said for "when".',
            [{"name": "when", "type": "string", "required": True}],
        )
    )
    specs.append(
        ToolSpec(
            "transfer_to_human",
            "Hand the call to a person when the caller insists on one or the request is beyond what you may do. "
            "Say a short line first explaining what happens next.",
            [{"name": "reason", "type": "string", "required": True}],
        )
    )
    if (agent.provider or "local") == "twilio":
        specs.append(
            ToolSpec(
                "send_sms",
                "Text the person a short message (a link, address or confirmation) when they ask for it in writing.",
                [{"name": "message", "type": "string", "required": True}],
            )
        )
    return specs


def custom_specs(tools: list[AgentTool]) -> list[ToolSpec]:
    return [ToolSpec(t.name, t.description, list(t.params or []), builtin=False) for t in tools if t.enabled]


def prompt_lines(agent: VoiceAgent, tools: list[AgentTool]) -> list[str]:
    return [s.prompt_line() for s in builtin_specs(agent) + custom_specs(tools)]


def validate_args(spec: ToolSpec, args: dict) -> tuple[dict, str | None]:
    """Coerce and check the model's arguments. Returns (clean_args, error)."""
    clean: dict = {}
    for p in spec.params:
        name = p["name"]
        value = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if p.get("required", True):
                return clean, f"missing required argument '{name}'"
            continue
        ptype = p.get("type", "string")
        if ptype == "number":
            try:
                clean[name] = float(value) if not isinstance(value, (int, float)) else value
            except (TypeError, ValueError):
                return clean, f"argument '{name}' must be a number"
        elif ptype == "boolean":
            if isinstance(value, bool):
                clean[name] = value
            else:
                s = str(value).strip().lower()
                if s not in {"true", "false", "yes", "no"}:
                    return clean, f"argument '{name}' must be true or false"
                clean[name] = s in {"true", "yes"}
        else:
            clean[name] = str(value).strip()[:1000]
    return clean, None


# ── Natural-language "when" parsing ────────────────────────────────────

_WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
_PERIODS = {
    "morning": 10,
    "noon": 12,
    "midday": 12,
    "lunch": 12,
    "lunchtime": 12,
    "afternoon": 14,
    "evening": 18,
    "tonight": 18,
}
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b(?!\s*(st|nd|rd|th)\b)", re.IGNORECASE)


def _zone(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo("UTC")


def parse_when(text: str, tz_name: str | None, now: datetime | None = None) -> datetime | None:
    """Turn "next Tuesday at 10am", "tomorrow afternoon", "3 Oct 2pm",
    "in 2 hours" or an ISO timestamp into an aware datetime in ``tz``.

    Returns None when the phrase is too vague; the tool then asks the
    model to get a concrete day and time from the caller.
    """
    tz = _zone(tz_name)
    now_local = (now or datetime.now(UTC)).astimezone(tz)
    raw = (text or "").strip()
    if not raw:
        return None

    # ISO 8601 first.
    iso = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
    except ValueError:
        pass

    s = raw.lower().replace(",", " ")
    s = re.sub(
        r"\b(at|on|the|next|this|coming|around|about|o'?clock)\b", lambda m: "next" if m.group(1) == "next" else " ", s
    )
    s = re.sub(r"\s+", " ", s).strip()

    # Relative offsets.
    m = re.search(r"\bin (\d+) (minute|minutes|hour|hours|day|days|week|weeks)\b", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
        delta = {
            "minute": timedelta(minutes=n),
            "hour": timedelta(hours=n),
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
        }[unit]
        return (now_local + delta).replace(second=0, microsecond=0)

    day: datetime | None = None
    if "tomorrow" in s:
        day = now_local + timedelta(days=1)
    elif "today" in s:
        day = now_local
    else:
        for name, idx in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", s):
                ahead = (idx - now_local.weekday()) % 7
                if ahead == 0 and "next" not in s:
                    ahead = 7  # "monday" said on a Monday means next week
                elif "next" in s and ahead == 0:
                    ahead = 7
                day = now_local + timedelta(days=ahead)
                break
        if day is None:
            # "3 october", "october 3rd", "3/10" is ambiguous → not handled.
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\b", s) or re.search(
                r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", s
            )
            if m:
                a, b = m.group(1), m.group(2)
                dnum, mname = (a, b) if a.isdigit() else (b, a)
                if mname in _MONTHS:
                    month = _MONTHS[mname]
                    year = now_local.year
                    try:
                        candidate = now_local.replace(
                            month=month, day=int(dnum), hour=0, minute=0, second=0, microsecond=0
                        )
                    except ValueError:
                        return None
                    if candidate.date() < now_local.date():
                        candidate = candidate.replace(year=year + 1)
                    day = candidate
                    # Drop the date words so the day number isn't read as a time.
                    s = (s[: m.start()] + " " + s[m.end() :]).strip()
    if day is None:
        return None

    hour: int | None = None
    minute = 0
    tm = _TIME_RE.search(s)
    if tm:
        hour = int(tm.group(1))
        minute = int(tm.group(2) or 0)
        suffix = (tm.group(3) or "").replace(".", "")
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        elif not suffix and hour <= 7:
            hour += 12  # "at 3" on a phone call means 3 pm
        if hour > 23 or minute > 59:
            return None
    if hour is None:
        for word, h in _PERIODS.items():
            if re.search(rf"\b{word}\b", s):
                hour = h
                break
    if hour is None:
        return None
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ── Execution ──────────────────────────────────────────────────────────


async def execute(
    name: str,
    args: dict,
    *,
    agent: VoiceAgent,
    contact: Contact,
    call: Call,
    db: Session,
    custom_tools: list[AgentTool],
) -> ToolResult:
    specs = {s.name: s for s in builtin_specs(agent) + custom_specs(custom_tools)}
    spec = specs.get(name)
    if spec is None:
        return ToolResult(False, f"unknown tool '{name}'. Available: {', '.join(specs) or 'none'}.")
    clean, error = validate_args(spec, args or {})
    if error:
        return ToolResult(False, f"{error}. Ask the person for it, then try again.")
    try:
        if spec.builtin:
            return await _run_builtin(name, clean, agent=agent, contact=contact, call=call, db=db)
        tool = next(t for t in custom_tools if t.name == name)
        return await _run_http(tool, clean)
    except Exception as exc:
        logger.exception("tool %s failed on call %s", name, call.id)
        return ToolResult(False, f"the tool failed ({str(exc)[:120]}). Apologise and offer to follow up another way.")


async def _run_builtin(
    name: str, args: dict, *, agent: VoiceAgent, contact: Contact, call: Call, db: Session
) -> ToolResult:
    tz_name = contact.timezone or agent.timezone
    if name == "book_appointment":
        when = parse_when(args["when"], tz_name)
        if when is None:
            return ToolResult(
                False, "could not understand the day and time. Ask for a specific day and time, then try again."
            )
        if when < datetime.now(UTC):
            return ToolResult(False, "that time is in the past. Ask for a future day and time.")
        duration = timedelta(minutes=agent.appointment_duration_min or 30)
        start_utc = when.astimezone(UTC).replace(tzinfo=None)
        end_utc = start_utc + duration
        clash = (
            db.query(Appointment)
            .filter(
                Appointment.agent_id == agent.id,
                Appointment.status.in_(["booked", "confirmed"]),
                Appointment.starts_at < end_utc,
                Appointment.ends_at > start_utc,
            )
            .first()
        )
        if clash is not None:
            return ToolResult(False, "that slot is already taken. Offer a different time.")
        appt = Appointment(
            id=str(uuid.uuid4()),
            agent_id=agent.id,
            contact_id=contact.id,
            call_id=call.id,
            starts_at=start_utc,
            ends_at=end_utc,
            timezone=tz_name,
            notes=args.get("notes"),
        )
        db.add(appt)
        db.commit()
        pretty = when.strftime("%A %d %B at %H:%M")
        return ToolResult(
            True,
            f"appointment booked for {pretty} ({tz_name}), {agent.appointment_duration_min} minutes. Confirm it back to the person.",
            data={"appointment_id": appt.id, "starts_at": start_utc.isoformat()},
            appointment_id=appt.id,
            flags=["appointment_booked"],
        )
    if name == "schedule_callback":
        when = parse_when(args["when"], tz_name)
        if when is None:
            return ToolResult(False, "could not understand the time. Ask for a specific day and time, then try again.")
        contact.next_attempt_at = when.astimezone(UTC).replace(tzinfo=None)
        db.commit()
        return ToolResult(
            True,
            f"callback scheduled for {when.strftime('%A %d %B at %H:%M')} ({tz_name}). Confirm it and wrap up politely.",
            data={"next_attempt_at": contact.next_attempt_at.isoformat()},
            flags=["callback_scheduled"],
        )
    if name == "transfer_to_human":
        return ToolResult(
            True,
            "transfer arranged. Say goodbye politely; the hand-off happens now.",
            data={"reason": args.get("reason")},
            end_outcome="handoff",
        )
    if name == "send_sms":
        body = str(args["message"]).strip()[:640]
        provider = telephony.get_provider(agent.provider)
        row = Message(
            id=str(uuid.uuid4()),
            agent_id=agent.id,
            contact_id=contact.id,
            call_id=call.id,
            to_number=contact.phone,
            body=body,
        )
        try:
            sid = await provider.send_sms(to_number=contact.phone, from_number=agent.from_number, body=body)
            row.status = "sent"
            row.provider_message_id = sid
        except telephony.ProviderError as exc:
            row.status = "unsent_no_provider"
            row.error = str(exc)
        db.add(row)
        db.commit()
        if row.status == "sent":
            return ToolResult(True, "text message sent. Tell the person it's on its way.", message_id=row.id)
        return ToolResult(
            False,
            "text messages are not available on this line. Offer to read the details out instead.",
            message_id=row.id,
        )
    return ToolResult(False, f"unknown built-in '{name}'")


async def _run_http(tool: AgentTool, args: dict) -> ToolResult:
    url = tool.url
    remaining = dict(args)
    for key in list(remaining):
        placeholder = "{" + key + "}"
        if placeholder in url:
            from urllib.parse import quote

            url = url.replace(placeholder, quote(str(remaining.pop(key)), safe=""))
    check_fetch_url(url)
    import httpx  # lazy

    method = (tool.method or "GET").upper()
    headers = {"User-Agent": "voicebox-voice-agent", **(tool.headers or {})}
    kwargs: dict = {"headers": headers}
    if method in {"GET", "DELETE"}:
        kwargs["params"] = remaining
    else:
        kwargs["json"] = remaining
    async with httpx.AsyncClient(timeout=float(tool.timeout_s or 10), follow_redirects=False) as client:
        resp = await client.request(method, url, **kwargs)
    body = resp.text or ""
    with contextlib.suppress(Exception):
        body = json.dumps(resp.json(), ensure_ascii=False)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > MAX_RESULT_CHARS:
        body = body[:MAX_RESULT_CHARS] + "…"
    ok = 200 <= resp.status_code < 300
    return ToolResult(ok, f"HTTP {resp.status_code}: {body or '(empty)'}", data={"status": resp.status_code})
