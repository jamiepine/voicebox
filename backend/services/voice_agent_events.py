"""Per-call event bus for the live console.

``GET /calls/{id}/events`` streams these as SSE so a client can react the
moment something happens on a call: the customer's transcript landed, a
filler phrase should play now, the agent's reply (or its first sentence)
is being voiced, a tool ran, the caller interrupted, the call ended.
Same shape as ``mcp_server.events`` but keyed by call id.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

_subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}


def subscribe(call_id: str) -> asyncio.Queue[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
    _subscribers.setdefault(call_id, set()).add(queue)
    return queue


def unsubscribe(call_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    bucket = _subscribers.get(call_id)
    if not bucket:
        return
    bucket.discard(queue)
    if not bucket:
        _subscribers.pop(call_id, None)


def publish(call_id: str, kind: str, payload: dict[str, Any] | None = None) -> None:
    """Fan out to the call's subscribers. Non-blocking; drops on a full queue."""
    for queue in list(_subscribers.get(call_id, ())):
        event = {"kind": kind, "call_id": call_id, **(payload or {})}
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)


def subscriber_count(call_id: str) -> int:
    return len(_subscribers.get(call_id, ()))
