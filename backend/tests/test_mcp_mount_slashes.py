"""Regression tests for the MCP mount's optional trailing slash.

Starlette strips a bare ``/mcp`` mount to an empty child path, while
FastMCP's HTTP app routes the endpoint at ``/``. The wrapper must normalize
that empty path without redirecting a POST request.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.mcp_server.server import (
    MountRootSlashRewrite,
    build_mcp_server,
    compose_lifespan,
)

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "voicebox-regression", "version": "1.0"},
    },
}
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _build_mcp_only_app() -> FastAPI:
    """Build only the MCP surface so this test avoids Voicebox model startup."""
    mcp_app = build_mcp_server().http_app(path="/", transport="http")
    app = FastAPI(lifespan=compose_lifespan(mcp_app.router.lifespan_context))
    app.mount("/mcp", MountRootSlashRewrite(mcp_app))
    return app


def _response_payload(response) -> dict:
    """Decode either FastMCP's JSON or SSE response representation."""
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()

    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"initialize response was not JSON/SSE: {response.text!r}")


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_mcp_initialize_accepts_both_slashes(path: str):
    with TestClient(_build_mcp_only_app()) as client:
        response = client.post(path, json=INITIALIZE, headers=HEADERS)

    assert response.status_code == 200, response.text
    payload = _response_payload(response)
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 1
    assert payload["result"]["serverInfo"]["name"] == "voicebox"
