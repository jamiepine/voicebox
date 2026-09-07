"""Voice AI agent endpoints — agents, contacts, knowledge, calls, tickets,
do-not-call list, and the Twilio webhooks.

Conversation flow for a local (speakers + API) call::

    POST /agents/{id}/calls/next          → opening line spoken, call id returned
    POST /calls/{call_id}/turn            → {"text": "..."} what the customer said
    POST /calls/{call_id}/turn/audio      → multipart audio, transcribed first
    …repeat until the response has ended=true…
    GET  /calls/{call_id}                 → transcript, outcome, summary
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from .. import config, models
from ..database import Generation as DBGeneration, get_db
from ..services import telephony, voice_agent as core, voice_agent_runner as runner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice-agent"])

UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}
MAX_AUDIO_BYTES = 50 * 1024 * 1024
# How long a Twilio webhook waits for TTS before telling Twilio to hold
# the line and come back. Twilio itself times out at 15 s.
TWIML_TTS_WAIT_S = 9.0


def _agent_or_404(agent_id: str, db: Session):
    agent = core.get_agent(agent_id, db)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _call_or_404(call_id: str, db: Session):
    call = core.get_call(call_id, db)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


def _turn_response(result: core.TurnResult) -> models.AgentTurnResponse:
    return models.AgentTurnResponse(
        call_id=result.call.id,
        text=result.agent_turn.text,
        generation_id=result.generation_id,
        poll_url=f"/generate/{result.generation_id}/status" if result.generation_id else None,
        ended=result.ended,
        outcome=result.outcome,
        ticket_id=result.ticket.id if result.ticket else None,
        customer_text=result.customer_turn.text if result.customer_turn else None,
        sentiment=result.customer_turn.sentiment if result.customer_turn else None,
    )


# ── Agents ─────────────────────────────────────────────────────────────


@router.get("/agents", response_model=list[models.VoiceAgentResponse])
async def list_agents(db: Session = Depends(get_db)):
    return [core.agent_to_response(a) for a in core.list_agents(db)]


@router.post("/agents", response_model=models.VoiceAgentResponse)
async def create_agent(data: models.VoiceAgentCreate, db: Session = Depends(get_db)):
    try:
        return core.agent_to_response(core.create_agent(data, db))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/agents/{agent_id}", response_model=models.VoiceAgentResponse)
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    return core.agent_to_response(_agent_or_404(agent_id, db))


@router.put("/agents/{agent_id}", response_model=models.VoiceAgentResponse)
async def update_agent(agent_id: str, data: models.VoiceAgentUpdate, db: Session = Depends(get_db)):
    try:
        agent = core.update_agent(agent_id, data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return core.agent_to_response(agent)


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    try:
        if not core.delete_agent(agent_id, db):
            raise HTTPException(status_code=404, detail="Agent not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"message": "Agent deleted"}


@router.get("/agents/{agent_id}/stats", response_model=models.VoiceAgentStats)
async def agent_stats(agent_id: str, db: Session = Depends(get_db)):
    return core.agent_stats(_agent_or_404(agent_id, db), db)


@router.post("/agents/{agent_id}/start", response_model=models.VoiceAgentResponse)
async def start_agent(agent_id: str, db: Session = Depends(get_db)):
    """Activate the agent. Outbound agents also start their auto-dialer;
    inbound agents just become answerable."""
    agent = _agent_or_404(agent_id, db)
    if agent.mode == "outbound_sales" and agent.provider == "twilio":
        try:
            telephony.get_provider("twilio")._require_config()  # type: ignore[attr-defined]
        except telephony.ProviderError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    agent = core.set_agent_status(agent_id, "active", db)
    if agent.mode == "outbound_sales":
        runner.start(agent_id)
    return core.agent_to_response(agent)


@router.post("/agents/{agent_id}/pause", response_model=models.VoiceAgentResponse)
async def pause_agent(agent_id: str, db: Session = Depends(get_db)):
    """Stop dialling. In-progress calls finish naturally."""
    _agent_or_404(agent_id, db)
    await runner.stop(agent_id)
    return core.agent_to_response(core.set_agent_status(agent_id, "paused", db))


# ── Contacts ───────────────────────────────────────────────────────────


@router.get("/agents/{agent_id}/contacts")
async def list_contacts(
    agent_id: str,
    status: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    _agent_or_404(agent_id, db)
    rows, total = core.list_contacts(agent_id, db, status=status, limit=min(max(limit, 1), 1000), offset=max(offset, 0))
    return {"contacts": [models.ContactResponse.model_validate(r) for r in rows], "total": total}


@router.post("/agents/{agent_id}/contacts", response_model=models.ContactResponse)
async def create_contact(agent_id: str, data: models.ContactCreate, db: Session = Depends(get_db)):
    _agent_or_404(agent_id, db)
    try:
        return models.ContactResponse.model_validate(core.create_contact(agent_id, data, db))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/agents/{agent_id}/contacts/bulk", response_model=models.ContactImportResult)
async def bulk_contacts(agent_id: str, data: models.ContactBulkCreate, db: Session = Depends(get_db)):
    _agent_or_404(agent_id, db)
    return core.bulk_create_contacts(agent_id, data.contacts, db)


@router.post("/agents/{agent_id}/contacts/import", response_model=models.ContactImportResult)
async def import_contacts_csv(agent_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV (headers: name, phone[, company, notes, timezone, consent])."""
    _agent_or_404(agent_id, db)
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV larger than 10 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        items = core.parse_contacts_csv(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not items:
        return models.ContactImportResult(imported=0, skipped=0, skipped_reasons={})
    return core.bulk_create_contacts(agent_id, items, db)


@router.get("/contacts/{contact_id}", response_model=models.ContactResponse)
async def get_contact(contact_id: str, db: Session = Depends(get_db)):
    contact = core.get_contact(contact_id, db)
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return models.ContactResponse.model_validate(contact)


@router.put("/contacts/{contact_id}", response_model=models.ContactResponse)
async def update_contact(contact_id: str, data: models.ContactUpdate, db: Session = Depends(get_db)):
    try:
        contact = core.update_contact(contact_id, data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return models.ContactResponse.model_validate(contact)


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, db: Session = Depends(get_db)):
    if not core.delete_contact(contact_id, db):
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"message": "Contact deleted"}


# ── Knowledge ──────────────────────────────────────────────────────────


@router.get("/agents/{agent_id}/knowledge", response_model=list[models.KnowledgeArticleResponse])
async def list_knowledge(agent_id: str, db: Session = Depends(get_db)):
    _agent_or_404(agent_id, db)
    return [core.article_to_response(a) for a in core.list_articles(agent_id, db)]


@router.post("/agents/{agent_id}/knowledge", response_model=models.KnowledgeArticleResponse)
async def create_knowledge(agent_id: str, data: models.KnowledgeArticleCreate, db: Session = Depends(get_db)):
    _agent_or_404(agent_id, db)
    return core.article_to_response(core.create_article(agent_id, data, db))


@router.put("/knowledge/{article_id}", response_model=models.KnowledgeArticleResponse)
async def update_knowledge(article_id: str, data: models.KnowledgeArticleUpdate, db: Session = Depends(get_db)):
    article = core.update_article(article_id, data, db)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return core.article_to_response(article)


@router.delete("/knowledge/{article_id}")
async def delete_knowledge(article_id: str, db: Session = Depends(get_db)):
    if not core.delete_article(article_id, db):
        raise HTTPException(status_code=404, detail="Article not found")
    return {"message": "Article deleted"}


# ── Calls ──────────────────────────────────────────────────────────────


@router.get("/agents/{agent_id}/calls")
async def list_calls(
    agent_id: str,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    _agent_or_404(agent_id, db)
    rows, total = core.list_calls(agent_id, db, status=status, limit=min(max(limit, 1), 500), offset=max(offset, 0))
    return {"calls": [core.call_to_response(c, db, include_turns=False) for c in rows], "total": total}


@router.post("/agents/{agent_id}/calls/next", response_model=models.AgentTurnResponse)
async def next_call(agent_id: str, contact_id: str | None = None, db: Session = Depends(get_db)):
    """Manually place the next outbound call (or a specific contact's).

    Returns the opening line. For ``local`` agents the audio plays on the
    speakers; feed the customer's replies to ``/calls/{id}/turn``.
    """
    agent = _agent_or_404(agent_id, db)
    if contact_id:
        contact = core.get_contact(contact_id, db)
        if contact is None or contact.agent_id != agent_id:
            raise HTTPException(status_code=404, detail="Contact not found on this agent")
    else:
        contact = core.pick_next_contact(agent, db) if agent.status == "active" else None
        if contact is None:
            # Manual calls ignore the "active" gate but keep every other rule.
            was = agent.status
            agent.status = "active"
            contact = core.pick_next_contact(agent, db)
            agent.status = was
            db.commit()
        if contact is None:
            raise HTTPException(
                status_code=409, detail="No contact is dialable right now (window, caps, DNC, or consent)."
            )
    try:
        result = await core.start_outbound_call(agent, contact, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    provider = telephony.get_provider(agent.provider)
    try:
        dial = await provider.dial(call_id=result.call.id, to_number=contact.phone, from_number=agent.from_number)
    except telephony.ProviderError as e:
        await core.end_call(result.call, "error", db, summary=f"Dial failed: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    if dial.provider_call_id:
        result.call.provider_call_id = dial.provider_call_id
        db.commit()
    return _turn_response(result)


@router.post("/agents/{agent_id}/calls/inbound", response_model=models.AgentTurnResponse)
async def inbound_call(agent_id: str, data: models.InboundCallRequest, db: Session = Depends(get_db)):
    """Start an inbound conversation (customer service / support). The
    caller is matched or created by phone number."""
    agent = _agent_or_404(agent_id, db)
    try:
        contact = core.get_or_create_inbound_contact(agent_id, data.phone, data.name, db)
        result = await core.start_inbound_call(agent, contact, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _turn_response(result)


@router.get("/calls/{call_id}", response_model=models.CallResponse)
async def get_call(call_id: str, db: Session = Depends(get_db)):
    return core.call_to_response(_call_or_404(call_id, db), db)


@router.post("/calls/{call_id}/turn", response_model=models.AgentTurnResponse)
async def customer_turn(call_id: str, data: models.CustomerTurnRequest, db: Session = Depends(get_db)):
    """Submit what the customer said (text) and get the agent's reply."""
    call = _call_or_404(call_id, db)
    try:
        result = await core.handle_customer_turn(call, data.text, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _turn_response(result)


@router.post("/calls/{call_id}/turn/audio", response_model=models.AgentTurnResponse)
async def customer_turn_audio(
    call_id: str,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Submit the customer's audio; it is transcribed with Whisper first."""
    call = _call_or_404(call_id, db)
    agent = _agent_or_404(call.agent_id, db)
    text = await _transcribe_upload(file, language or agent.language)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not hear anything in that audio.")
    try:
        result = await core.handle_customer_turn(call, text, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _turn_response(result)


@router.post("/calls/{call_id}/end", response_model=models.CallResponse)
async def end_call(call_id: str, data: models.EndCallRequest, db: Session = Depends(get_db)):
    """Force-close a call (operator hung up, wrong number, …)."""
    call = _call_or_404(call_id, db)
    agent = core.get_agent(call.agent_id, db)
    if agent is not None and call.provider_call_id:
        try:
            await telephony.get_provider(agent.provider).hangup(call.provider_call_id)
        except Exception:
            logger.debug("hangup failed for %s", call_id, exc_info=True)
    call = await core.end_call(call, data.outcome, db, summary=data.summary)
    return core.call_to_response(call, db)


@router.get("/calls/{call_id}/turns/{turn_id}/audio")
async def turn_audio(call_id: str, turn_id: str, db: Session = Depends(get_db)):
    """Serve the audio for one agent turn (what Twilio ``<Play>``s)."""
    turn = next((t for t in core.get_turns(call_id, db) if t.id == turn_id), None)
    if turn is None or not turn.generation_id:
        raise HTTPException(status_code=404, detail="Turn has no audio")
    gen = db.query(DBGeneration).filter(DBGeneration.id == turn.generation_id).first()
    if gen is None or gen.status != "completed":
        raise HTTPException(status_code=409, detail="Audio not ready yet")
    path = config.resolve_storage_path(gen.audio_path)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, media_type=media_type or "audio/wav")


# ── Tickets ────────────────────────────────────────────────────────────


@router.get("/tickets")
async def list_tickets(
    agent_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    rows, total = core.list_tickets(
        db, agent_id=agent_id, status=status, limit=min(max(limit, 1), 500), offset=max(offset, 0)
    )
    return {"tickets": [models.TicketResponse.model_validate(t) for t in rows], "total": total}


@router.get("/tickets/{ticket_id}", response_model=models.TicketResponse)
async def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    ticket = core.get_ticket(ticket_id, db)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return models.TicketResponse.model_validate(ticket)


@router.put("/tickets/{ticket_id}", response_model=models.TicketResponse)
async def update_ticket(ticket_id: str, data: models.TicketUpdate, db: Session = Depends(get_db)):
    ticket = core.update_ticket(ticket_id, data, db)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return models.TicketResponse.model_validate(ticket)


# ── Do-not-call ────────────────────────────────────────────────────────


@router.get("/dnc", response_model=list[models.DoNotCallResponse])
async def list_dnc(db: Session = Depends(get_db)):
    return [models.DoNotCallResponse.model_validate(e) for e in core.list_dnc(db)]


@router.post("/dnc", response_model=models.DoNotCallResponse)
async def add_dnc(data: models.DoNotCallCreate, db: Session = Depends(get_db)):
    try:
        return models.DoNotCallResponse.model_validate(core.add_to_dnc(data.phone, db, reason=data.reason))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/dnc/{phone}")
async def remove_dnc(phone: str, db: Session = Depends(get_db)):
    try:
        if not core.remove_from_dnc(phone, db):
            raise HTTPException(status_code=404, detail="Number not on the list")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"message": "Removed"}


# ── Twilio webhooks ────────────────────────────────────────────────────


async def _twilio_form(request: Request) -> dict[str, str]:
    form = await request.form()
    return {k: str(v) for k, v in form.items()}


def _twilio_provider() -> telephony.TwilioProvider:
    provider = telephony.get_provider("twilio")
    if not isinstance(provider, telephony.TwilioProvider):
        raise HTTPException(status_code=500, detail="Twilio provider unavailable")
    return provider


def _verify_twilio(request: Request, params: dict[str, str], provider: telephony.TwilioProvider) -> None:
    signature = request.headers.get("X-Twilio-Signature")
    # Twilio signs the URL it was given (the public one), so rebuild it
    # from the configured public base rather than the proxied request.
    url = f"{provider.public_url}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    if not provider.validate_signature(url, params, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


def _twiml(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")


async def _wait_for_audio(generation_id: str | None, db: Session, timeout: float) -> bool:
    if not generation_id:
        return False
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        db.expire_all()
        gen = db.query(DBGeneration).filter(DBGeneration.id == generation_id).first()
        if gen is not None and gen.status == "completed":
            return True
        if gen is not None and gen.status == "failed":
            return False
        await asyncio.sleep(0.4)
    return False


def _latest_agent_turn(call_id: str, db: Session):
    turns = [t for t in core.get_turns(call_id, db) if t.role == "agent"]
    return turns[-1] if turns else None


async def _twiml_for_state(call, provider: telephony.TwilioProvider, db: Session) -> Response:
    """Emit the TwiML that plays the agent's latest turn and records the
    reply — or hangs up if the call has ended."""
    turn = _latest_agent_turn(call.id, db)
    if turn is None:
        return _twiml(provider.twiml_hangup())
    ready = await _wait_for_audio(turn.generation_id, db, TWIML_TTS_WAIT_S)
    if not ready:
        if call.status != "in_progress":
            return _twiml(provider.twiml_hangup())
        return _twiml(provider.twiml_pause_and_retry(call.id))
    if call.status != "in_progress":
        return _twiml(provider.twiml_play_and_hangup(call.id, turn.id))
    return _twiml(provider.twiml_play_and_record(call.id, turn.id))


@router.post("/webhooks/twilio/{call_id}/answer")
async def twilio_answer(call_id: str, request: Request, db: Session = Depends(get_db)):
    """Twilio fetches this when the callee picks up (and on every
    ``<Redirect>`` while waiting for TTS)."""
    provider = _twilio_provider()
    params = await _twilio_form(request)
    _verify_twilio(request, params, provider)
    call = _call_or_404(call_id, db)
    if params.get("AnsweredBy", "").startswith("machine"):
        await core.end_call(call, "voicemail", db)
        return _twiml(provider.twiml_hangup())
    return await _twiml_for_state(call, provider, db)


@router.post("/webhooks/twilio/{call_id}/recording")
async def twilio_recording(call_id: str, request: Request, db: Session = Depends(get_db)):
    """After ``<Record>``: fetch the clip, transcribe, run a turn, and
    return the next TwiML."""
    provider = _twilio_provider()
    params = await _twilio_form(request)
    _verify_twilio(request, params, provider)
    call = _call_or_404(call_id, db)
    if call.status != "in_progress":
        return _twiml(provider.twiml_hangup())
    agent = _agent_or_404(call.agent_id, db)

    recording_url = params.get("RecordingUrl")
    text = ""
    if recording_url and float(params.get("RecordingDuration", "0") or 0) > 0.3:
        try:
            audio = await provider.fetch_recording(recording_url)
            text = await _transcribe_bytes(audio, ".wav", agent.language)
        except Exception:
            logger.exception("Twilio recording transcription failed for %s", call_id)
    if not text.strip():
        # Silence: prompt once more rather than ending the call.
        turn = _latest_agent_turn(call.id, db)
        if turn is not None:
            return _twiml(provider.twiml_play_and_record(call.id, turn.id, max_seconds=20))
        return _twiml(provider.twiml_hangup())

    try:
        await core.handle_customer_turn(call, text, db)
    except ValueError as e:
        logger.warning("Twilio turn rejected for %s: %s", call_id, e)
        return _twiml(provider.twiml_hangup())
    db.refresh(call)
    return await _twiml_for_state(call, provider, db)


@router.post("/webhooks/twilio/{call_id}/status")
async def twilio_status(call_id: str, request: Request, db: Session = Depends(get_db)):
    """Call-status callback: close the call if the far end hung up."""
    provider = _twilio_provider()
    params = await _twilio_form(request)
    _verify_twilio(request, params, provider)
    call = core.get_call(call_id, db)
    if call is not None and call.status == "in_progress":
        status = params.get("CallStatus", "")
        outcome = {"no-answer": "no_answer", "busy": "no_answer", "failed": "error", "canceled": "no_answer"}.get(
            status, "unresolved" if call.turn_count > 1 else "no_answer"
        )
        await core.end_call(call, outcome, db)
    return Response(status_code=204)


@router.post("/webhooks/twilio/inbound/{agent_id}")
async def twilio_inbound(agent_id: str, request: Request, db: Session = Depends(get_db)):
    """Point a Twilio number's "A call comes in" webhook here to have the
    agent answer. Creates the call and returns the greeting TwiML."""
    provider = _twilio_provider()
    params = await _twilio_form(request)
    _verify_twilio(request, params, provider)
    agent = _agent_or_404(agent_id, db)
    if agent.status != "active":
        return _twiml(provider.twiml_hangup())
    caller = params.get("From") or params.get("Caller") or ""
    if not caller:
        return _twiml(provider.twiml_hangup())
    try:
        contact = core.get_or_create_inbound_contact(agent_id, caller, params.get("CallerName"), db)
        result = await core.start_inbound_call(agent, contact, db, provider_call_id=params.get("CallSid"))
    except ValueError as e:
        logger.warning("Twilio inbound rejected: %s", e)
        return _twiml(provider.twiml_hangup())
    return await _twiml_for_state(result.call, provider, db)


# ── Transcription helpers ─────────────────────────────────────────────


async def _transcribe_upload(file: UploadFile, language: str | None) -> str:
    ext = Path(file.filename or "").suffix.lower()
    suffix = ext if ext in ALLOWED_AUDIO_EXTS else ".wav"
    size = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_AUDIO_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Audio larger than 50 MB")
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        return await _transcribe_path(tmp_path, suffix, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _transcribe_bytes(raw: bytes, suffix: str, language: str | None) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return await _transcribe_path(tmp_path, suffix, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _transcribe_path(path: str, suffix: str, language: str | None) -> str:
    from ..services import transcribe  # lazy: heavy import chain
    from ..utils.audio import load_audio, save_audio

    stt_path = path
    try:
        if suffix != ".wav":
            # Same dance as /transcribe: decode with librosa, hand Whisper a WAV.
            audio, sr = await asyncio.to_thread(load_audio, path)
            stt_path = f"{path}.stt.wav"
            await asyncio.to_thread(save_audio, audio, stt_path, sr)
        whisper = transcribe.get_whisper_model()
        size = whisper.model_size
        if not whisper.is_loaded() and not whisper._is_model_cached(size):
            raise HTTPException(
                status_code=409,
                detail=f"Whisper '{size}' is not downloaded. Open Voicebox → Settings → Models to download it.",
            )
        lang = None if (language in (None, "", "auto")) else language
        return await whisper.transcribe(stt_path, lang, size)
    finally:
        if stt_path != path:
            Path(stt_path).unlink(missing_ok=True)
