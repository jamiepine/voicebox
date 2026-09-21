"""Post-call webhooks: signed, retried, logged.

When an agent has a ``webhook_url``, every finished call is POSTed there as
JSON (call, contact, transcript, analysis, tickets, appointments). The
body is signed with HMAC-SHA256 over the raw bytes using
``webhook_secret`` and sent as ``X-Voicebox-Signature: sha256=<hex>`` so
the receiver can verify origin. Delivery runs in the background with
three attempts and is recorded in ``va_webhook_deliveries``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from ..database import WebhookDelivery, get_db

logger = logging.getLogger(__name__)

RETRY_DELAYS_S: tuple[float, ...] = (0.0, 5.0, 30.0)
SIGNATURE_HEADER = "X-Voicebox-Signature"
EVENT_HEADER = "X-Voicebox-Event"
DELIVERY_HEADER = "X-Voicebox-Delivery"

# Tests swap this for a client built on httpx.MockTransport.
_client_factory: Callable[[], Any] | None = None
_pending: set[asyncio.Task] = set()


def sign(secret: str | None, body: bytes) -> str:
    digest = hmac.new((secret or "").encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(secret: str | None, body: bytes, signature: str | None) -> bool:
    if not signature:
        return False
    return hmac.compare_digest(sign(secret, body), signature)


def _client():
    if _client_factory is not None:
        return _client_factory()
    import httpx  # lazy

    return httpx.AsyncClient(timeout=15.0, follow_redirects=False)


def create_delivery(agent_id: str, call_id: str | None, url: str, event: str, db) -> WebhookDelivery:
    row = WebhookDelivery(id=str(uuid.uuid4()), agent_id=agent_id, call_id=call_id, url=url, event=event)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def dispatch(
    agent_id: str, call_id: str | None, url: str, secret: str | None, event: str, payload: dict, db
) -> WebhookDelivery:
    """Record the delivery and start sending it in the background."""
    row = create_delivery(agent_id, call_id, url, event, db)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        task = loop.create_task(deliver(row.id, url, secret, event, payload))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    return row


async def deliver(delivery_id: str, url: str, secret: str | None, event: str, payload: dict) -> bool:
    """POST with retries; updates the delivery row after each attempt."""
    body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: sign(secret, body),
        EVENT_HEADER: event,
        DELIVERY_HEADER: delivery_id,
        "User-Agent": "voicebox-voice-agent",
    }
    ok = False
    for attempt, delay in enumerate(RETRY_DELAYS_S, start=1):
        if delay:
            await asyncio.sleep(delay)
        code: int | None = None
        error: str | None = None
        try:
            async with _client() as client:
                resp = await client.post(url, content=body, headers=headers)
            code = resp.status_code
            ok = 200 <= code < 300
            if not ok:
                error = f"HTTP {code}"
        except Exception as exc:  # network errors, timeouts
            error = str(exc)[:500]
        _record(delivery_id, attempt, code, error, ok)
        if ok:
            return True
    return False


def _record(delivery_id: str, attempts: int, code: int | None, error: str | None, ok: bool) -> None:
    try:
        db = next(get_db())
    except Exception:
        return
    try:
        row = db.query(WebhookDelivery).filter(WebhookDelivery.id == delivery_id).first()
        if row is None:
            return
        row.attempts = attempts
        row.response_code = code
        row.last_error = error
        row.status = "delivered" if ok else ("failed" if attempts >= len(RETRY_DELAYS_S) else "pending")
        db.commit()
        # Mirror the final state onto the call for the UI.
        if row.call_id and row.status != "pending":
            from ..database import Call

            call = db.query(Call).filter(Call.id == row.call_id).first()
            if call is not None:
                call.webhook_status = row.status
                db.commit()
    except Exception:
        logger.debug("webhook delivery record failed", exc_info=True)
        db.rollback()
    finally:
        db.close()


async def wait_for_pending() -> None:
    """Test helper: block until in-flight deliveries settle."""
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
