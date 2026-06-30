"""Unit tests for the ClientIdMiddleware path predicate.

Locks down which endpoints advance ``last_seen_at`` on the
``MCPClientBinding`` row. Getting this wrong is silent: the Settings UI
just shows a stale "last heard from" timestamp and bindings never get
auto-created for new REST callers.
"""

import pytest

from backend.mcp_server.context import _is_stamped_path


@pytest.mark.parametrize(
    "path",
    [
        "/mcp",
        "/mcp/",
        "/mcp/tools/call",
        "/mcp/bindings",  # admin REST; benign — frontend never sets the header
        "/speak",
        "/speak/",
    ],
)
def test_mcp_semantic_paths_are_stamped(path: str) -> None:
    assert _is_stamped_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/health",
        "/generate",
        "/captures",
        "/profiles",
        "/profiles/abc/compose",
        "/events/speak",
        "/tasks/active",
        "/llm/generate",
        # Prefix overlap should not match — /speakers is a hypothetical
        # future endpoint that shouldn't leak the stamp.
        "/speakers",
        # Same for anything starting with /mcpfoo.
        "/mcpfoo",
    ],
)
def test_other_paths_are_not_stamped(path: str) -> None:
    assert _is_stamped_path(path) is False
