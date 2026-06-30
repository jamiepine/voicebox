"""Server-Sent-Event streams the frontend subscribes to.

``GET /events/speak`` — broadcasts ``speak-start`` / ``speak-end`` events
whenever an agent-initiated speak (MCP tool or POST /speak) runs. The
DictateWindow uses them to show the floating pill in a `speaking` state.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..mcp_server import events as mcp_events


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
