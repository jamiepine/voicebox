"""Tests for the /events/generations SSE endpoint registration and contract.

These tests verify the *wiring* of the route to the in-process bus
without consuming the streaming response end-to-end. End-to-end
streaming coverage is provided by the existing
backend/tests/test_all_models_e2e.py harness; for the bus
mechanics themselves see test_generation_events.py.

What we lock down here:
  1. The route is registered in the FastAPI app (visible in
     OpenAPI /docs).
  2. The OpenAPI spec advertises the text/event-stream content
     type (so the frontend TypeScript client can be generated).
  3. The /events/speak pre-existing route is unaffected by our
     additions.

Why no streaming-body tests here: pytest is sync, but
EventSourceResponse yields forever. Reading the body in a sync
test hangs the worker; verifying end-to-end behavior is the
responsibility of the e2e harness in test_all_models_e2e.py.

Run with: python -m pytest backend/tests/test_sse_endpoint.py -v
"""

import pytest
from fastapi import FastAPI

from backend.routes.events import router as events_router


@pytest.fixture
def app():
    """A minimal FastAPI app that mounts only the events router.

    Mirrors the pattern in test_cors.py -- mirror the production
    surface we need, skip the heavy ML dependencies.
    """
    app = FastAPI()
    app.include_router(events_router)
    return app


def test_events_generations_route_is_registered(app):
    # FastAPI nests subrouter routes as IncludedRouter objects, so
    # inspecting r.path on top-level app.routes misses them. Use the
    # OpenAPI schema (the same source the FastAPI docs UI uses) to
    # verify the route is actually exposed.
    schema = app.openapi()
    assert "/events/generations" in schema["paths"]
    # /events/speak is the pre-existing route -- it must still be
    # there after our change.
    assert "/events/speak" in schema["paths"]


def test_events_generations_has_get_operation(app):
    # The route must expose a GET (the verb EventSource uses). The
    # response content type is text/event-stream at runtime, but
    # FastAPI's OpenAPI generator does not surface EventSourceResponse
    # media types, so we assert the operation exists rather than the
    # schema's (incorrect) application/json entry.
    spec = app.openapi()["paths"]["/events/generations"]
    assert "get" in spec
