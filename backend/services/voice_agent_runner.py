"""Auto-dialer: one background loop per active outbound agent.

The loop picks the next dialable contact (see
``voice_agent.pick_next_contact``), opens a call, hands the dial to the
telephony provider, and then waits for the conversation to finish before
moving on. Conversations are driven elsewhere — by the Twilio webhooks
for real calls, or by ``POST /calls/{id}/turn`` / the MCP tools for local
ones — so the runner's job is just pacing and hygiene:

- idle calls (no turn within ``idle_timeout``) are closed as ``no_answer``
  so a contact who never picked up gets retried instead of blocking the
  queue forever;
- when nothing is dialable the loop sleeps and re-checks, and when nothing
  is *pending* either (every contact finished / exhausted) the agent is
  marked ``completed`` and the loop exits.

State is in-memory: a restart drops running loops, and the agent's
``status`` row stays ``active`` so the UI offers "Resume".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import timedelta

from ..database import get_db
from . import telephony, voice_agent as core

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30.0
CALL_POLL_S = 2.0

_tasks: dict[str, asyncio.Task] = {}
_idle_timeout_s: float = float(core.DEFAULT_IDLE_TIMEOUT_S)


def is_running(agent_id: str) -> bool:
    task = _tasks.get(agent_id)
    return task is not None and not task.done()


def running_agent_ids() -> list[str]:
    return [aid for aid, t in _tasks.items() if not t.done()]


def start(agent_id: str, *, idle_timeout_s: float | None = None) -> None:
    """Start the loop for ``agent_id``. No-op if already running."""
    global _idle_timeout_s
    if idle_timeout_s is not None:
        _idle_timeout_s = idle_timeout_s
    if is_running(agent_id):
        return
    loop = asyncio.get_running_loop()
    task = loop.create_task(_run(agent_id), name=f"voice-agent-runner:{agent_id}")
    _tasks[agent_id] = task
    task.add_done_callback(lambda t: _tasks.pop(agent_id, None) if _tasks.get(agent_id) is t else None)
    logger.info("Voice agent %s: dialer started", agent_id)


async def stop(agent_id: str) -> None:
    task = _tasks.pop(agent_id, None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    logger.info("Voice agent %s: dialer stopped", agent_id)


async def stop_all() -> None:
    for agent_id in list(_tasks):
        await stop(agent_id)


async def _run(agent_id: str) -> None:
    try:
        while True:
            outcome = await _tick(agent_id)
            if outcome == "done":
                return
            if outcome == "idle":
                await asyncio.sleep(POLL_INTERVAL_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Voice agent %s: dialer crashed", agent_id)


async def _tick(agent_id: str) -> str:
    """One pass: dial one contact and see the call through.

    Returns "called" (go again immediately), "idle" (sleep first) or
    "done" (exit the loop).
    """
    db = next(get_db())
    try:
        agent = core.get_agent(agent_id, db)
        if agent is None or agent.status != "active":
            return "done"
        if agent.mode != "outbound_sales":
            # Inbound agents don't dial; nothing for the loop to do.
            return "done"

        await _reap_idle_calls(agent_id, db)

        contact = core.pick_next_contact(agent, db)
        if contact is None:
            if not core.has_pending_contacts(agent, db):
                core.set_agent_status(agent_id, "completed", db)
                logger.info("Voice agent %s: no contacts left, marking completed", agent_id)
                return "done"
            return "idle"

        try:
            result = await core.start_outbound_call(agent, contact, db)
        except ValueError as exc:
            logger.info("Voice agent %s: skipping %s (%s)", agent_id, contact.phone, exc)
            return "called"
        call = result.call

        provider = telephony.get_provider(agent.provider)
        try:
            dial = await provider.dial(call_id=call.id, to_number=contact.phone, from_number=agent.from_number)
        except telephony.ProviderError as exc:
            logger.error("Voice agent %s: dial failed for %s: %s", agent_id, contact.phone, exc)
            await core.end_call(call, "error", db, summary=f"Dial failed: {exc}")
            # A misconfigured provider will fail every dial; back off rather
            # than burning through the whole list.
            return "idle"
        if dial.provider_call_id:
            call.provider_call_id = dial.provider_call_id
            db.commit()
        call_id = call.id
    finally:
        db.close()

    await _wait_for_call(call_id)
    return "called"


async def _wait_for_call(call_id: str) -> None:
    """Block until the call leaves ``in_progress`` or goes idle too long."""
    while True:
        await asyncio.sleep(CALL_POLL_S)
        db = next(get_db())
        try:
            call = core.get_call(call_id, db)
            if call is None or call.status != "in_progress":
                return
            idle_for = core.utcnow() - (call.last_activity_at or call.started_at)
            if idle_for > timedelta(seconds=_idle_timeout_s):
                logger.info(
                    "Voice agent call %s idle for %ss, closing as no_answer", call_id, int(idle_for.total_seconds())
                )
                agent = core.get_agent(call.agent_id, db)
                if agent is not None:
                    try:
                        await telephony.get_provider(agent.provider).hangup(call.provider_call_id)
                    except Exception:
                        logger.debug("hangup failed for %s", call_id, exc_info=True)
                await core.end_call(call, "no_answer", db)
                return
        finally:
            db.close()


async def _reap_idle_calls(agent_id: str, db) -> None:
    """Close stale in-progress calls left over from a crash / restart."""
    from ..database import Call

    cutoff = core.utcnow() - timedelta(seconds=_idle_timeout_s)
    stale = (
        db.query(Call)
        .filter(Call.agent_id == agent_id, Call.status == "in_progress", Call.last_activity_at < cutoff)
        .all()
    )
    for call in stale:
        await core.end_call(call, "no_answer", db)
