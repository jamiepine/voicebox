# Voicebox MCP server

Local **Model Context Protocol** server — lets any MCP-aware agent
(Claude Code, Cursor, Windsurf, VS Code MCP extensions, etc.) speak text
in your cloned voices, transcribe audio, and browse captures.

The server runs inside the same `uvicorn` process as the rest of Voicebox
and is mounted at `/mcp` (Streamable HTTP transport).

## Install into your agent

Preferred — direct HTTP:

```json
{
  "mcpServers": {
    "voicebox": {
      "url": "http://127.0.0.1:17493/mcp",
      "headers": { "X-Voicebox-Client-Id": "claude-code" }
    }
  }
}
```

Fallback — stdio shim (when the client doesn't speak HTTP MCP). The
`voicebox-mcp` binary ships inside the Voicebox.app bundle:

```json
{
  "mcpServers": {
    "voicebox": {
      "command": "/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp",
      "env": { "VOICEBOX_CLIENT_ID": "claude-code" }
    }
  }
}
```

Claude Code one-liner:

```
claude mcp add voicebox \
  --transport http \
  --url http://127.0.0.1:17493/mcp \
  --header "X-Voicebox-Client-Id: claude-code"
```

## Tools

| Name | Purpose |
|---|---|
| `voicebox.speak`          | Speak text in a voice profile. Returns a generation id you can poll. |
| `voicebox.transcribe`     | Whisper transcription of a base64 blob or an absolute local path. |
| `voicebox.list_captures`  | Recent captures (dictation / recording / file) with transcripts. |
| `voicebox.list_profiles`  | Available voice profiles (cloned + preset). |

All tools resolve voice profiles in this precedence:

1. Explicit `profile` arg (name or id — case-insensitive)
2. Per-client binding keyed by `X-Voicebox-Client-Id`
3. `capture_settings.default_playback_voice_id` (global default)

Bindings are managed via `GET|PUT /mcp/bindings` or in the app under
Settings → MCP.

## Debug with MCP Inspector

```
npx @modelcontextprotocol/inspector http://127.0.0.1:17493/mcp
```

Point it at the URL, hit "List tools," call `voicebox.list_profiles`
first to confirm wiring, then `voicebox.speak` for end-to-end.

## Non-MCP REST surface

`POST /speak` is a thin wrapper on the same code path for callers that
don't speak MCP (shell scripts, ACP, A2A):

```
curl -X POST http://127.0.0.1:17493/speak \
  -H 'Content-Type: application/json' \
  -H 'X-Voicebox-Client-Id: claude-code' \
  -d '{"text":"Build complete.","profile":"Morgan"}'
```

## Code layout

```
backend/mcp_server/
├── __init__.py      # re-export mount_into
├── server.py        # build_mcp_server() + mount_into(app)
├── tools.py         # @mcp.tool() implementations
├── context.py       # ClientIdMiddleware + current_client_id ContextVar
├── resolve.py       # profile resolution precedence
├── events.py        # pub/sub queue for /events/speak pill SSE
└── README.md        # you are here

backend/mcp_shim/    # stdio ↔ Streamable-HTTP proxy (see its README)
```

The package is **`mcp_server`**, not `mcp`, to avoid shadowing the
installed `mcp` PyPI package that FastMCP imports internally.
