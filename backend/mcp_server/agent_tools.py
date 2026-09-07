"""MCP tools for driving the voice agent from Claude Code, Cursor, etc.

These let an MCP client act as the "phone line" for a local agent: place
the next call, relay what the customer said, and read the outcome. An
agent harness can therefore run a whole outbound list — or answer a
support conversation — while the human watches the transcript in the
Agents tab.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .. import models
from ..database import get_db
from ..services import telephony, voice_agent as core


def _turn_payload(result: core.TurnResult) -> dict[str, Any]:
    return {
        "call_id": result.call.id,
        "agent_said": result.agent_turn.text,
        "generation_id": result.generation_id,
        "poll_url": f"/generate/{result.generation_id}/status" if result.generation_id else None,
        "ended": result.ended,
        "outcome": result.outcome,
        "ticket_id": result.ticket.id if result.ticket else None,
        "sentiment": result.customer_turn.sentiment if result.customer_turn else None,
    }


def register_agent_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        name="voicebox.agent.list",
        description=(
            "List configured voice agents (outbound sales, customer service, support) "
            "with their mode, status, and whether the auto-dialer is running."
        ),
    )
    async def agent_list() -> dict[str, Any]:
        db = next(get_db())
        try:
            return {
                "agents": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "mode": a.mode,
                        "status": a.status,
                        "provider": a.provider,
                        "running": core._is_running(a.id),
                    }
                    for a in core.list_agents(db)
                ]
            }
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.next_call",
        description=(
            "Place the next outbound call for an agent (or call a specific contact_id). "
            "Speaks the opening line in the agent's voice and returns the call_id to "
            "continue with voicebox.agent.reply. Honours the do-not-call list, calling "
            "window, attempt caps and consent rules."
        ),
    )
    async def agent_next_call(agent: str, contact_id: str | None = None) -> dict[str, Any]:
        db = next(get_db())
        try:
            row = _resolve_agent(agent, db)
            if contact_id:
                contact = core.get_contact(contact_id, db)
                if contact is None or contact.agent_id != row.id:
                    raise ValueError("Contact not found on this agent.")
            else:
                was = row.status
                row.status = "active"
                contact = core.pick_next_contact(row, db)
                row.status = was
                db.commit()
                if contact is None:
                    raise ValueError("No contact is dialable right now (calling window, caps, DNC, or consent).")
            result = await core.start_outbound_call(row, contact, db)
            provider = telephony.get_provider(row.provider)
            try:
                dial = await provider.dial(call_id=result.call.id, to_number=contact.phone, from_number=row.from_number)
            except telephony.ProviderError as exc:
                await core.end_call(result.call, "error", db, summary=f"Dial failed: {exc}")
                raise ValueError(str(exc)) from exc
            if dial.provider_call_id:
                result.call.provider_call_id = dial.provider_call_id
                db.commit()
            payload = _turn_payload(result)
            payload["contact"] = {"id": contact.id, "name": contact.name, "phone": contact.phone}
            return payload
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.inbound_call",
        description=(
            "Start an inbound customer-service / support conversation with an agent for the "
            "given caller phone number. Returns the greeting and a call_id."
        ),
    )
    async def agent_inbound_call(agent: str, phone: str, name: str | None = None) -> dict[str, Any]:
        db = next(get_db())
        try:
            row = _resolve_agent(agent, db)
            contact = core.get_or_create_inbound_contact(row.id, phone, name, db)
            result = await core.start_inbound_call(row, contact, db)
            return _turn_payload(result)
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.reply",
        description=(
            "Relay what the customer said on a call and get the agent's spoken reply. "
            "The response says whether the call ended and with which outcome."
        ),
    )
    async def agent_reply(call_id: str, customer_text: str) -> dict[str, Any]:
        db = next(get_db())
        try:
            call = core.get_call(call_id, db)
            if call is None:
                raise ValueError("Call not found.")
            result = await core.handle_customer_turn(call, customer_text, db)
            return _turn_payload(result)
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.end_call",
        description="Force-close a call with an outcome (no_answer, callback, not_interested, resolved, …).",
    )
    async def agent_end_call(call_id: str, outcome: str, summary: str | None = None) -> dict[str, Any]:
        models.EndCallRequest(outcome=outcome, summary=summary)  # validates the outcome
        db = next(get_db())
        try:
            call = core.get_call(call_id, db)
            if call is None:
                raise ValueError("Call not found.")
            call = await core.end_call(call, outcome, db, summary=summary)
            return core.call_to_response(call, db).model_dump(mode="json")
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.get_call",
        description="Fetch a call's transcript, outcome, and summary.",
    )
    async def agent_get_call(call_id: str) -> dict[str, Any]:
        db = next(get_db())
        try:
            call = core.get_call(call_id, db)
            if call is None:
                raise ValueError("Call not found.")
            return core.call_to_response(call, db).model_dump(mode="json")
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.agent.stats",
        description="Campaign / queue statistics for an agent: contacts by status, calls by outcome, open tickets.",
    )
    async def agent_stats(agent: str) -> dict[str, Any]:
        db = next(get_db())
        try:
            row = _resolve_agent(agent, db)
            return core.agent_stats(row, db).model_dump(mode="json")
        finally:
            db.close()


def _resolve_agent(name_or_id: str, db):
    row = core.get_agent(name_or_id, db)
    if row is None:
        from sqlalchemy import func

        from ..database import VoiceAgent

        row = db.query(VoiceAgent).filter(func.lower(VoiceAgent.name) == (name_or_id or "").lower()).first()
    if row is None:
        raise ValueError(f"Voice agent '{name_or_id}' not found.")
    return row
