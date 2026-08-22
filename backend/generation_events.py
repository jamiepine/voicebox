"""In-process publish/subscribe bus for generation status events.

Used by /events/generations to broadcast one SSE stream containing
updates for every generation, so the browser only needs one
EventSource instead of N. Solves the HTTP/1.1 per-origin 6-connection
cap that blocked button clicks when many generations ran in parallel.

Design:
- Single module-level EventBus (cheap; no app.state plumbing needed).
- Subscribers are asyncio.Queue instances; publisher calls put_nowait
  on each one. If a subscriber's queue is full we drop the oldest entry
  rather than block the publisher (keeps the generator coroutine
  unblocked).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _Bus:
    def __init__(self, max_queue: int = 256):
        self._subscribers: list[asyncio.Queue] = []
        self._max_queue = max_queue

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.append(q)
        logger.debug("generation_events: subscribe, total=%d", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
            logger.debug("generation_events: unsubscribe, total=%d", len(self._subscribers))
        except ValueError:
            pass

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        msg = {"topic": topic, "data": data}
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except Exception:
                    pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


bus = _Bus()