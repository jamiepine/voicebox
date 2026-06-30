"""Construct the FastMCP server and mount it on the FastAPI app.

The MCP endpoint lives at ``/mcp`` (Streamable HTTP transport). Modern MCP
clients (Claude Code, Cursor, Windsurf, VS Code MCP extensions) connect
directly via URL; older stdio-only clients use the ``voicebox-mcp`` shim
binary bundled with the desktop app.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from collections.abc import Callable

from fastapi import FastAPI
from fastmcp import FastMCP

from .context import ClientIdMiddleware
from .tools import register_tools


logger = logging.getLogger(__name__)


def build_mcp_server() -> FastMCP:
    """Create the FastMCP instance with Voicebox tools registered."""
    mcp = FastMCP(
        name="voicebox",
        instructions=(
            "Voicebox is a local voice I/O layer. Use `voicebox.speak` to "
            "play text in a voice profile, `voicebox.transcribe` for "
            "audio→text, and the `list_*` tools to discover profiles and "
            "captures."
        ),
    )
    register_tools(mcp)
    return mcp


def mount_into(
    app: FastAPI,
    *,
    extra_startup: Callable[[], None] | None = None,
) -> None:
    """Attach the MCP app to ``app`` at ``/mcp`` and install the client-id middleware.

    ``extra_startup`` — if provided, runs during the FastAPI lifespan. This
    is the hook that lets ``app.py`` keep its existing startup/shutdown
    bodies while also driving FastMCP's session manager.
    """
    mcp = build_mcp_server()
    mcp_app = mcp.http_app(path="/", transport="http")

    # ClientIdMiddleware must run before FastMCP so the ContextVar is set
    # by the time tool handlers execute. Starlette composes middlewares
    # outermost-first, so adding here on the parent app is correct.
    app.add_middleware(ClientIdMiddleware)
    app.mount("/mcp", mcp_app)
    app.state.mcp_lifespan = mcp_app.router.lifespan_context
    logger.info("MCP: mounted at /mcp (FastMCP %s)", getattr(mcp, "version", ""))


def compose_lifespan(*lifespans):
    """Combine multiple async context managers into a single FastAPI lifespan.

    Used by ``create_app`` to run the existing Voicebox startup/shutdown
    together with FastMCP's session manager (which MUST run in the
    ASGI lifespan for Streamable HTTP to work).
    """

    @asynccontextmanager
    async def _combined(app):
        async with AsyncExitStack() as stack:
            for cm_factory in lifespans:
                cm = cm_factory(app) if callable(cm_factory) else cm_factory
                await stack.enter_async_context(cm)
            yield

    return _combined
