"""Auto-dialer: one background loop per active outbound agent.

Each tick the loop reaps idle calls, counts the lines in use, and — while
the agent is active, inside its schedule window, and under
``max_concurrent_calls`` — dials the next contact (see
``voice_agent.pick_next_contact``). Conversations are driven elsewhere
(Twilio webhooks for real calls, ``POST /calls/{id}/turn`` / the MCP tools
for local ones), so the runner's job is pacing and hygiene:

- idle calls (no turn within ``idle_timeout``) are closed as ``no_answer``
  so a contact who never picked up gets retried instead of blocking the
  queue forever;
- when nothing is dialable the loop sleeps and re-checks; when nothing is
  *pending* either (every contact finished / exhausted) or the schedule
  window has closed, the agent is marked ``completed`` and the loop exits.

The local provider is one line — a person can only hold one handset —
so concurrency only applies to Twilio agents.

State is in-memory: a restart drops running loops, and the agent's
``status`` row stays ``active`` so the UI offers "Resume".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import timedelta

from ..database import Call, get_db
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


async def warm_up(agent_id: str) -> None:
    """Pre-generate filler audio so the first live turn has it ready."""
    db = next(get_db())
    try:
        agent = core.get_agent(agent_id, db)
        if agent is not None:
            await core.ensure_filler_audio(agent, db)
    except Exception:
        logger.debug("filler warm-up failed for %s", agent_id, exc_info=True)
    finally:
        db.close()


async def _run(agent_id: str) -> None:
    try:
        await warm_up(agent_id)
        while True:
            state, sleep_s = await _tick(agent_id)
            if state == "done":
                return
            await asyncio.sleep(sleep_s)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Voice agent %s: dialer crashed", agent_id)


async def _tick(agent_id: str) -> tuple[str, float]:
    """One scheduling pass. Returns ("done" | "busy" | "idle", seconds to sleep)."""
    db = next(get_db())
    try:
        agent = core.get_agent(agent_id, db)
        if agent is None or agent.status != "active":
            return "done", 0
        if agent.mode != "outbound_sales":
            # Inbound agents don't dial; nothing for the loop to do.
            return "done", 0

        window = core.within_schedule(agent)
        if window == "after":
            core.set_agent_status(agent_id, "completed", db)
            logger.info("Voice agent %s: schedule window closed, marking completed", agent_id)
            return "done", 0

        await _reap_idle_calls(agent, db)
        in_progress = core.count_in_progress_calls(agent_id, db)
        if window == "before":
            return ("busy", CALL_POLL_S) if in_progress else ("idle", POLL_INTERVAL_S)

        capacity = (agent.max_concurrent_calls if (agent.provider or "local") == "twilio" else 1) - in_progress
        dialed = 0
        while capacity > 0:
            contact = core.pick_next_contact(agent, db)
            if contact is None:
                break
            if not await _dial(agent, contact, db):
                # A misconfigured provider fails every dial; back off rather
                # than burning through the whole list.
                return "idle", POLL_INTERVAL_S
            dialed += 1
            capacity -= 1

        in_progress = core.count_in_progress_calls(agent_id, db)
        if in_progress:
            return "busy", CALL_POLL_S
        if dialed == 0 and not core.has_pending_contacts(agent, db):
            core.set_agent_status(agent_id, "completed", db)
            logger.info("Voice agent %s: no contacts left, marking completed", agent_id)
            return "done", 0
        return "idle", POLL_INTERVAL_S
    finally:
        db.close()


async def _dial(agent, contact, db) -> bool:
    """Open the call and hand it to the provider. False on provider failure."""
    try:
        result = await core.start_outbound_call(agent, contact, db)
    except ValueError as exc:
        logger.info("Voice agent %s: skipping %s (%s)", agent.id, contact.phone, exc)
        return True
    call = result.call
    provider = telephony.get_provider(agent.provider)
    try:
        dial = await provider.dial(
            call_id=call.id,
            to_number=contact.phone,
            from_number=agent.from_number,
            machine_detection="DetectMessageEnd" if agent.voicemail_message else "Enable",
        )
    except telephony.ProviderError as exc:
        logger.error("Voice agent %s: dial failed for %s: %s", agent.id, contact.phone, exc)
        await core.end_call(call, "error", db, summary=f"Dial failed: {exc}")
        return False
    if dial.provider_call_id:
        call.provider_call_id = dial.provider_call_id
        db.commit()
    return True


async def _reap_idle_calls(agent, db) -> None:
    """Close in-progress calls nobody has touched for ``idle_timeout``
    (never answered, or left over from a crash / restart)."""
    cutoff = core.utcnow() - timedelta(seconds=_idle_timeout_s)
    stale = (
        db.query(Call)
        .filter(
            Call.agent_id == agent.id,
            Call.status == "in_progress",
            Call.direction != core.SIMULATION,
            Call.last_activity_at < cutoff,
        )
        .all()
    )
    for call in stale:
        logger.info("Voice agent call %s idle too long, closing as no_answer", call.id)
        try:
            await telephony.get_provider(agent.provider).hangup(call.provider_call_id)
        except Exception:
            logger.debug("hangup failed for %s", call.id, exc_info=True)
        await core.end_call(call, "no_answer", db)
