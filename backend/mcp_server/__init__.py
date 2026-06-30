"""Model Context Protocol server — exposes Voicebox tools to local AI agents.

Mounts a FastMCP instance at /mcp on the main FastAPI app (Streamable HTTP).
A bundled stdio shim (backend/mcp_shim) forwards JSON-RPC into the same
endpoint for MCP clients that only speak stdio.
"""

from .server import mount_into

__all__ = ["mount_into"]
