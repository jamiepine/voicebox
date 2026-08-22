"""Server-Sent-Event streams the frontend subscribes to.

``GET /events/speak`` -- broadcasts ``speak-start`` / ``speak-end``
events whenever an agent-initiated speak (MCP tool or POST /speak)
runs. The DictateWindow uses them to show the floating pill in a
``speaking`` state.

``GET /events/generations`` -- single multiplexed stream of every
generation status update. The browser opens exactly one EventSource
for this endpoint, avoiding the HTTP/1.1 per-origin 6-connection cap
that previously made the "Send" button unresponsive once 6 generations
were in flight.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..mcp_server import events as mcp_events
from .. import generation_events


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/events/speak")
async def speak_events(request: Request):
    """SSE stream of speak-start / speak-end events."""

    async def event_stream():
        queue = mcp_events.subscribe()
        try:
            # Immediate hello so EventSource knows the connection is live.
            yield {"event": "ready", "data": "{}"}
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # Heartbeat so proxies don't reap idle streams.
                    yield {"event": "ping", "data": "{}"}
                    continue
                kind = event.pop("kind", "message")
                yield {"event": kind, "data": json.dumps(event)}
        finally:
            mcp_events.unsubscribe(queue)

    return EventSourceResponse(event_stream())


@router.get("/events/generations")
async def generation_events_stream(request: Request):
    """Single SSE stream that multiplexes every generation status update.

    The frontend opens exactly one EventSource for this endpoint. Each
    event published to the in-process generation_events.bus is fanned
    out to every subscriber here, regardless of how many generations
    are in flight.
    """
    queue = generation_events.bus.subscribe()

    async def event_stream():
        try:
            # Immediate hello so EventSource knows the connection is live.
            yield {"event": "ready", "data": "{}"}
            while True:
                if await request.is_disconnected():
                    return
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                topic = msg.get("topic", "message")
                data = msg.get("data", {})
                yield {"event": topic, "data": json.dumps(data)}
        finally:
            generation_events.bus.unsubscribe(queue)

    return EventSourceResponse(event_stream())