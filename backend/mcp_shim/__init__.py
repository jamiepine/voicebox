"""Stdio → Streamable HTTP bridge for the Voicebox MCP server.

Some MCP clients only know how to spawn a subprocess and talk to it over
stdin/stdout (the "stdio" transport). This package is a ~150-line adapter:
the client spawns us as ``voicebox-mcp``; we proxy every JSON-RPC frame
to http://127.0.0.1:17493/mcp/ and stream responses back out.

All the real work (tools, models, inference) lives in the Voicebox server
process — this package contains no business logic.
"""
