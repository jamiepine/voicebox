# MCP Server — Voicebox Speed Run

**Status:** v1 shipped — HTTP transport, all 4 tools, per-client bindings, `POST /speak`, stdio shim (binary built, bundled into Tauri sidecar), Settings UI, speak-pill via SSE with Rust-side `dictate:show` handler so agent-initiated speech surfaces the pill on screen. `cargo check` clean, `tsc` clean, full Inspector round-trip verified.
**Last reviewed:** 2026-04-23

## Status

### Shipped (backend)
- **`fastmcp` + `sse-starlette`** pinned in `backend/requirements.txt`.
- **`backend/mcp_server/`** package with `server.py`, `tools.py`, `context.py`, `resolve.py`, `events.py`, `README.md`. Named `mcp_server` (not `mcp`) to sidestep a shadowing conflict with the installed `mcp` PyPI package that FastMCP imports internally.
- **Streamable HTTP mount at `/mcp`** via FastMCP's `http_app(transport='http')`. Sub-app lifespan composed with Voicebox's own startup/shutdown through an `@asynccontextmanager lifespan=` in `backend/app.py` (migrated away from the deprecated `@app.on_event` handlers).
- **Four MCP tools**, dot-named to match the landing and ecosystem convention:
  - `voicebox.speak(text, profile?, engine?, personality?, language?)`
  - `voicebox.transcribe(audio_base64?, audio_path?, language?, model?)`
  - `voicebox.list_captures(limit, offset)`
  - `voicebox.list_profiles()`
- **`ClientIdMiddleware`** pulls `X-Voicebox-Client-Id` into a `ContextVar` on every `/mcp*` request; auto-stamps `MCPClientBinding.last_seen_at`, auto-creating the row if the client is new.
- **Profile resolution precedence** `explicit → per-client binding → capture_settings.default_playback_voice_id → error`. `services/profiles.get_profile_orm_by_name_or_id()` lets agents pass a voice by name ("Morgan") instead of UUID.
- **`MCPClientBinding` table** (new) via `Base.metadata.create_all` — no migration needed.
- **Bindings REST:** `GET|PUT /mcp/bindings`, `DELETE /mcp/bindings/{client_id}`.
- **`POST /speak`** REST wrapper for non-MCP callers (shell / ACP / A2A). Same `resolve_profile` precedence, same code path as the MCP tool.
- **Stdio shim** at `backend/mcp_shim/__main__.py` — ~200 lines of `httpx` proxy; reads env (`VOICEBOX_PORT`, `VOICEBOX_HOST`, `VOICEBOX_CLIENT_ID`), waits for `/health`, then streams JSON-RPC ↔ SSE. Rolled our own after the `mcp` SDK's session-management helpers mis-shook-hands. Smoke-tested: `initialize`, `tools/list`, and `tools/call` all round-trip cleanly.
- **Pill SSE:** `GET /events/speak` (`sse-starlette`) emits `speak-start` from the MCP tool and `POST /speak`, `speak-end` from `services/generation.run_generation`'s finally block.
- **PyInstaller:**
  - `backend/build_binary.py` `--shim` flag builds a minimal `voicebox-mcp` binary (torch/transformers/mlx/etc. explicitly excluded, target <20 MB).
  - The main server spec picks up `fastmcp`, `mcp`, `sse_starlette`, and `backend.mcp_server.*` via `--collect-all` / `--hidden-import`.
- **`backend/mcp_server/README.md`** quickstart (Inspector, `.mcp.json` snippets, tool reference).

### Shipped (frontend)
- **`Settings → MCP`** page (`app/src/components/ServerTab/MCPPage.tsx`):
  - Three copy-paste snippets auto-filled with the detected `serverUrl`: HTTP (recommended), Claude Code CLI one-liner, stdio fallback.
  - Default voice picker (bound to `capture_settings.default_playback_voice_id`, shared with Captures-tab "Play as voice").
  - Per-client bindings table with inline profile picker, remove button, and a connection-status indicator that refreshes every 10 s.
  - Add-binding form with client_id / label / profile dropdown.
- **`useMCPBindings`** TanStack hook (optimistic delete, invalidate on upsert).
- **`useSpeakEvents`** hook — auto-reconnecting `EventSource('/events/speak')`, tracks the active generation_id, exposes an elapsed-ms timer that ticks so the pill's clock advances.
- **`CapturePill`** has a new `'speaking'` state + "Speaking" label + playing-bars mode.
- **`DictateWindow`** subscribes to speak events and overrides `pillState` when an agent is speaking. Emits `dictate:show` on speak-start so the Rust side can surface the pill window.
- Router + `ServerTab` tab bar wired to `/settings/mcp`.

### Shipped (native shell)
- **`tauri.conf.json`** — `voicebox-mcp` added to `externalBin` (alongside `voicebox-server`).
- **`dictate:show` listener** in `tauri/src-tauri/src/main.rs` — invokes a new `show_dictate_window(app_handle)` helper that mirrors the hotkey-monitor's position+show logic (undo click-through, reposition to top-center of the current monitor, show). Agent-initiated speech now pops the pill visible on screen.

### Validated end-to-end (this session, via curl)
- `/mcp/` init → `tools/list` → `tools/call voicebox.speak` → actual audio plays (Jarvis, 1.68 s).
- `POST /speak` with `X-Voicebox-Client-Id: claude-code` resolves to the bound Jarvis profile without passing `profile`.
- `/events/speak` emits `ready`, `speak-start`, `speak-end` in order, generation_id threads through both.
- Stdio shim: `echo {…} | python -m backend.mcp_shim` returns valid JSON-RPC for all 4 methods.
- `last_seen_at` auto-stamps on first call; binding row auto-creates.
- Frontend `tsc --noEmit`: clean.
- `cargo check` on the Tauri crate: clean.

### Outstanding (must-do before release)
- **CI build for shim on Windows/Linux** — `python backend/build_binary.py --shim` is wired up and built cleanly for `aarch64-apple-darwin` (18 MB, installed at `tauri/src-tauri/binaries/voicebox-mcp-aarch64-apple-darwin`, Tauri `cargo check` green). The Windows and Linux triples (`x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`) need the same build in their respective CI runners and artifacts dropped alongside the macOS binary.
- **Windows/Linux paths in the stdio snippet** — the Settings page hardcodes the macOS path (`/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp`). Needs a per-OS switch (`%LOCALAPPDATA%\Programs\Voicebox\voicebox-mcp.exe`, Linux bundled-path), ideally with the Tauri shell resolving its own app path at runtime and injecting it into the snippet.

### Nice-to-have (follow-up passes)
- **One-click install buttons** — write/merge into `~/.claude/settings.json`, `~/.cursor/mcp.json`, etc. via a Tauri command. Copy-paste works today; this is pure ergonomics.
- **`.mcpb` desktop extension** for Claude Desktop (single file, double-click to install). Claude Desktop-only, so lower priority than the agent-harness crowd.
- **Refactor the hotkey_monitor.rs show-logic** to call `show_dictate_window()` instead of duplicating the position+show block. Skipped at ship to avoid regressing the well-tested chord path.
- **Source attribution on `Generation.source`** — currently `"manual" | "personality_speak"`; adding `"mcp"` / `"rest"` would let the Captures tab filter by MCP-originated rows.

## Context

Voicebox already ships the I/O surface (Captures, Generate, personality-driven `/profiles/{id}/speak`), but local AI agents can't reach any of it. This plan adds a Model Context Protocol server so Claude Code / Cursor / Cline can call `voicebox.speak`, `voicebox.transcribe`, `voicebox.list_captures`, and `voicebox.list_profiles` — turning Voicebox into the local voice layer for every agent on the user's machine (Phase 5 of `docs/plans/VOICE_IO.md`).

The shortest path to "Claude Code speaks in a cloned voice": mount **FastMCP** inside the existing FastAPI/uvicorn process at `/mcp` (Streamable HTTP), and users install it as a URL (`{"url": "http://127.0.0.1:17493/mcp"}`) — the ecosystem-idiomatic shape for a long-running local service. Per-client voice binding via a new `mcp_client_bindings` table + Settings UI, resolved from an `X-Voicebox-Client-Id` header. A **stdio shim binary** `voicebox-mcp` is bundled as a fallback sidecar for clients that can't speak HTTP MCP. A public `POST /speak` REST wrapper covers non-MCP callers (shell scripts, ACP, A2A). A `speaking` pill state gives agent-initiated audio visibility — trust-critical, non-negotiable.

## Architecture

```
Claude Code / Cursor / Windsurf / VS Code MCP
        │
        ├─ HTTP (primary) ────────────────────┐
        │   {"url": ".../mcp"}                │
        │                                      │
        └─ stdio (fallback) ───────────────▶ [voicebox-mcp shim binary]
           {"command": "/abs/path/voicebox-mcp"}      (absolute path;
                                               │      Settings page
                                               │      copies it for you)
                                               ▼
                                         uvicorn + FastAPI (port 17493)
                                           ├─ /mcp    (FastMCP, Streamable HTTP)
                                           └─ /speak  (REST wrapper for non-MCP callers)
                                                 └─ tools call existing services
```

- **Transport:** Streamable HTTP as primary (Nov-2025 spec, post-SSE). Claude Code, Cursor, Windsurf, and the VS Code MCP extensions all support HTTP — it's the idiomatic shape for a long-running local service, which Voicebox already is.
- **Stdio fallback:** `voicebox-mcp` binary bundled inside the app for clients that can't speak HTTP MCP. The Settings page renders the exact snippet with the detected absolute path — user copies, pastes, done. No PATH manipulation, no custom CLI wrapper.
- **Identity:** HTTP clients set `X-Voicebox-Client-Id` header in their MCP config's `headers` block. Stdio clients set `VOICEBOX_CLIENT_ID` env var, which the shim forwards as the same HTTP header. Server reads it into a `ContextVar`.
- **Profile resolution precedence:** explicit tool arg → per-client `MCPClientBinding.profile_id` → `capture_settings.default_playback_voice_id` → error.
- **Port:** `17493`, matching `tauri/src-tauri/src/main.rs:63` (`SERVER_PORT` constant). Shim default with `VOICEBOX_PORT` env override.
- **Non-MCP access:** `POST /speak` is a thin REST wrapper around the same tool path — one endpoint for shell scripts, ACP, A2A, and anything that isn't MCP-native.

## Library choice

- **`fastmcp`** (PyPI — verify on install whether the canonical import is `fastmcp` standalone or `mcp.server.fastmcp` from the consolidated `mcp` package; the API is identical).
- **`sse-starlette`** for the `/events/speak` pill-state broadcast.
- **`httpx` + `anyio`** already present — used by the shim.

## Data model

New table, **one row per client_id** (not a singleton — scales to unknown clients, maps 1:1 to the Settings UI list):

```python
# backend/database/models.py
class MCPClientBinding(Base):
    __tablename__ = "mcp_client_bindings"
    client_id       = Column(String, primary_key=True)        # "claude-code", "cursor", ...
    label           = Column(String, nullable=True)
    profile_id      = Column(String, ForeignKey("profiles.id"), nullable=True)
    default_engine  = Column(String, nullable=True)
    default_personality = Column(Boolean, nullable=False, default=False)  # rewrite-before-speak default
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

Global default stays in `capture_settings.default_playback_voice_id` — no duplication. Migration: new `_migrate_mcp_client_bindings()` in `backend/database/migrations.py` using `CREATE TABLE IF NOT EXISTS`, mirroring the existing idempotent-add-column pattern.

## File plan

### Backend — new

| File | Purpose |
|---|---|
| `backend/mcp/__init__.py` | Package marker |
| `backend/mcp/server.py` | `build_mcp_server()` + `mount_into(app)`; registers tools, middleware, mount at `/mcp` |
| `backend/mcp/tools.py` | The 4 `@mcp.tool()` functions — thin wrappers over existing services |
| `backend/mcp/context.py` | `current_client_id: ContextVar[str \| None]` + Starlette middleware |
| `backend/mcp/resolve.py` | `resolve_profile(explicit, client_id, db) -> VoiceProfile \| None` |
| `backend/mcp/events.py` | In-memory `asyncio.Queue` pub/sub for speak-start / speak-end |
| `backend/mcp/README.md` | MCP Inspector quickstart + `.mcp.json` snippets |
| `backend/mcp_shim/__init__.py`, `__main__.py` | Stdio ↔ Streamable HTTP proxy (~150 lines) |
| `backend/voicebox-mcp.spec` | PyInstaller spec for the shim (strips torch/transformers from `hiddenimports`) |
| `backend/routes/speak.py` | `POST /speak {text, profile?, engine?, personality?, language?}` — REST wrapper around `resolve_profile()` + `generate_speech()` for non-MCP agents |

### Backend — modified

| File | Change |
|---|---|
| `backend/app.py` | Migrate `@app.on_event("startup"/"shutdown")` (lines 185, 268) to `lifespan=` kwarg on `FastAPI()` using `AsyncExitStack`; call `mount_into(application)` after `register_routers`. Register `ClientIdMiddleware`. |
| `backend/routes/profiles.py` | In `speak_in_character` (line 453): `events.publish("speak-start", {...})` on entry; completion hook publishes `speak-end`. Accept optional `source="mcp"` marker. |
| `backend/services/generation.py` | `run_generation` completion path publishes `speak-end`. |
| `backend/services/profiles.py` | New `async def get_profile_by_name_or_id(name_or_id, db)` — id lookup first, case-insensitive name fallback. |
| `backend/database/models.py` | Add `MCPClientBinding`. |
| `backend/database/migrations.py` | Add `_migrate_mcp_client_bindings`. |
| `backend/models.py` | Add `MCPClientBindingResponse`, `MCPClientBindingUpdate`. |
| `backend/routes/__init__.py` | Register `mcp_bindings_router`, `speak_router`, `events_router`. |
| `backend/routes/mcp_bindings.py` (new) | REST CRUD for bindings (list, upsert, delete). |
| `backend/routes/events.py` (new) | `GET /events/speak` — `EventSourceResponse` subscribed to the events queue. |
| `backend/requirements.txt` | `+ fastmcp` (or `mcp>=1.0`), `+ sse-starlette` |
| `backend/voicebox-server.spec` | `hiddenimports += ['mcp', 'mcp.server', 'fastmcp']` |
| `backend/build_binary.py` | Second PyInstaller invocation for `voicebox-mcp.spec`; copy to `tauri/src-tauri/binaries/` with target-triple suffix |

### Frontend — new

| File | Purpose |
|---|---|
| `app/src/components/ServerSettings/MCPBindings.tsx` | Settings section — default voice + per-client binding rows + `.mcp.json` copy-paste cheatsheet |
| `app/src/lib/hooks/useMCPBindings.ts` | TanStack Query mirror of `useCaptureSettings` |
| `app/src/lib/api/mcp.ts` | `listMCPBindings` / `upsertMCPBinding` / `deleteMCPBinding` |

### Frontend — modified

| File | Change |
|---|---|
| `app/src/components/DictateWindow/DictateWindow.tsx` | Open `EventSource('/events/speak')`; on `speak-start` set pill to `speaking` with profile name; dismiss on `speak-end`. |
| `app/src/components/CapturePill/CapturePill.tsx` | Add `speaking` branch — reuse the active waveform, swap status label to profile name. |
| `app/src/lib/hooks/useCaptureRecordingSession.ts` | Union a `speaking` injection into the derived pill state. |
| `app/src/lib/api/types.ts` | `MCPClientBinding`, `MCPClientBindingUpdate` types. |
| `app/src/components/ServerSettings/index.tsx` | Register the new MCP section in the tab aggregator. |

### Tauri

| File | Change |
|---|---|
| `tauri/src-tauri/tauri.conf.json` | `"externalBin": ["binaries/voicebox-server", "binaries/voicebox-mcp"]` |
| `tauri/src-tauri/binaries/voicebox-mcp-<triple>` | Build artifact from PyInstaller |

## Tool signatures

All tools read `current_client_id.get()` (from middleware). Return JSON-serializable dicts.

Tools are registered with **dotted names** (`voicebox.speak`, etc.) to match the landing page and the industry convention (`filesystem.read_file`, `github.create_issue`). Python function names stay snake_case; the dot goes in the `name=` kwarg.

```python
# backend/mcp/tools.py

@mcp.tool(name="voicebox.speak")
async def speak(text: str,
                profile: str | None = None,     # name OR id
                engine: str | None = None,
                personality: bool | None = None,  # true → rewrite via profile's personality LLM before TTS
                language: str | None = None) -> dict:
    """Speak text in a voice profile. Returns {generation_id, status, profile, poll}."""
    # resolve profile via precedence, delegate to generate_speech — the
    # route honors `personality=True` by running rewrite_as_profile on
    # the input before running the normal TTS pipeline.

@mcp.tool(name="voicebox.transcribe")
async def transcribe(audio_base64: str | None = None,
                     audio_path: str | None = None,    # absolute local path
                     language: str | None = None,
                     model: str | None = None) -> dict:
    """Transcribe audio. Exactly one of audio_base64/audio_path. Returns {text, duration, language}."""
    # validate path readable, size < 200 MB, then call services.transcribe.transcribe_bytes

@mcp.tool(name="voicebox.list_captures")
async def list_captures(limit: int = 20, offset: int = 0) -> dict:
    """Recent captures with transcripts. Returns {captures: [...]}"""

@mcp.tool(name="voicebox.list_profiles")
async def list_profiles() -> dict:
    """Available voice profiles. Returns {profiles: [{id, name, voice_type, has_personality}]}"""
```

### `POST /speak` (non-MCP REST wrapper)

```python
# backend/routes/speak.py
@router.post("/speak", response_model=GenerationResponse)
async def speak(data: SpeakRequest, request: Request, db: Session = Depends(get_db)):
    """Same behavior as the MCP tool — for shell scripts, ACP, A2A, or anything non-MCP."""
    client_id = request.headers.get("X-Voicebox-Client-Id")
    profile = resolve_profile(data.profile, client_id, db)
    if profile is None: raise HTTPException(400, "No voice profile resolved.")
    req = GenerationRequest(profile_id=profile.id, text=data.text,
                            language=data.language or "en",
                            engine=data.engine or "qwen",
                            personality=bool(data.personality))
    return await generate_speech(req, db)
```

`SpeakRequest`: `{ text: str, profile: str | None, engine: str | None, personality: bool | None, language: str | None }`. Accepts name OR id for `profile` (via `resolve_profile`). `personality=None` means "use the per-client binding's `default_personality`"; explicit `true`/`false` always wins. Same precedence as the MCP tool so the two surfaces behave identically.

## Mount point (`backend/app.py`)

```python
# After register_routers(application):
from .mcp.server import mount_into
mount_into(application)
```

`mount_into` installs `ClientIdMiddleware` and calls `app.mount("/mcp", mcp.streamable_http_app())`.

**Lifespan migration is load-bearing** — FastMCP's session manager requires the `lifespan=` kwarg, not `@app.on_event`. Wrap the existing startup/shutdown bodies in an `@asynccontextmanager` using `contextlib.AsyncExitStack` so both Voicebox's init and FastMCP's session manager run. Verify dev + packaged build after the migration.

## Stdio shim (`backend/mcp_shim/__main__.py`)

1. Port: `int(os.environ.get("VOICEBOX_PORT", "17493"))`.
2. Client id: `os.environ.get("VOICEBOX_CLIENT_ID", "unknown")`.
3. Health probe `GET /health` with 30 s tolerance (torch imports slowly). On failure, emit JSON-RPC error on stdout, exit 1.
4. Connect Streamable HTTP MCP client to `http://127.0.0.1:{port}/mcp` with `X-Voicebox-Client-Id: {client_id}` header.
5. Proxy JSON-RPC bidirectionally — stdin → HTTP, SSE → stdout. Use `mcp` SDK's built-in stdio↔HTTP bridge if available; otherwise ~40 lines of asyncio.
6. Stdout = JSON-RPC only. All logs to stderr.

PyInstaller spec keeps only `mcp`, `httpx`, `anyio`, `click` — target binary <20 MB.

## Pill `speaking` state

- `backend/mcp/events.py`: module-level `_subscribers: list[asyncio.Queue]` + `publish(kind, payload)` + `subscribe() -> Queue`.
- `speak_in_character` publishes `speak-start` with `{generation_id, profile_id, profile_name, source}` immediately after `task_manager.start_generation`; `run_generation`'s completion path publishes `speak-end`.
- `/events/speak` → `EventSourceResponse`.
- `DictateWindow` opens `EventSource` next to existing `dictate:*` listeners, maps `speak-start/end` → pill `speaking` mode with profile name.
- Optional filter: only show pill when `source === "mcp"` (avoids pill churn during manual speak flows). Settings toggle later.

## Settings UI (`MCPBindings.tsx`)

- **Global default voice** picker bound to `capture_settings.default_playback_voice_id` (reuses `useCaptureSettings`).
- **Per-client table** — add/edit/remove rows of `{client_id, label, profile_id, default_engine, default_personality}`. Uses `useMCPBindings`.
- **Connection cheatsheet** — two tabs, HTTP (default) and Stdio (fallback), with copy-to-clipboard snippets per known client:

  HTTP form (primary):
  ```json
  {"mcpServers": {"voicebox": {
    "url": "http://127.0.0.1:17493/mcp",
    "headers": {"X-Voicebox-Client-Id": "claude-code"}
  }}}
  ```

  Stdio form (fallback, absolute path auto-filled from detected app location):
  ```json
  {"mcpServers": {"voicebox": {
    "command": "/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp",
    "env": {"VOICEBOX_CLIENT_ID": "claude-code"}
  }}}
  ```

  Plus the Claude-Code-specific one-liner:
  ```
  claude mcp add voicebox --transport http --url http://127.0.0.1:17493/mcp --header "X-Voicebox-Client-Id: claude-code"
  ```
- **One-click install buttons** for known clients (v1: Claude Code via `claude mcp add` invocation, and a config-file writer for Cursor/Windsurf whose config locations are known). Each has a matching "Remove" button. Hide buttons for clients not detected on disk.
- **Connection status** — small indicator next to each binding showing the last time that `client_id` actually called the server (rolling timestamp recorded by middleware), so users can tell their install worked.

## Ordered task list (shortest path first)

1. `fastmcp` + `sse-starlette` → `backend/requirements.txt`; install.
2. Add `backend/mcp/{server,tools,context,resolve}.py` with the 4 tools registered as `voicebox.speak` etc. (no middleware yet — global default profile only).
3. Migrate `app.py` to `lifespan=`; mount FastMCP at `/mcp`.
4. **Milestone:** `npx @modelcontextprotocol/inspector http://127.0.0.1:17493/mcp` — call `voicebox.speak`, hear audio.
5. Add `get_profile_by_name_or_id`; wire the tool's `profile` arg.
6. `MCPClientBinding` model + migration; middleware; full `resolve_profile` precedence.
7. `backend/routes/speak.py` — `POST /speak` REST wrapper, reusing `resolve_profile` + `speak_in_character`.
8. `/mcp/bindings` REST + `MCPBindings.tsx` UI with HTTP and stdio copy-snippets, one-click install for detected clients, and connection-status indicators. **Users can install Voicebox as an MCP server after this step.**
9. `backend/mcp_shim/__main__.py` + PyInstaller spec + `build_binary.py` second pass; register `voicebox-mcp` as a Tauri sidecar. (Fallback path goes live.)
10. Events queue + `/events/speak` SSE + `DictateWindow` `speaking` pill state.
11. `backend/mcp/README.md` quickstart.

Claude Code can call `voicebox.speak` after step 4 (direct HTTP, manual config). Step 8 makes that a one-click experience. Step 9 adds the stdio fallback for clients that don't speak HTTP MCP.

## Verification

- **Step 4 smoke:** `npx @modelcontextprotocol/inspector http://127.0.0.1:17493/mcp`. Call `voicebox.list_profiles`, then `voicebox.speak(text="hello from mcp")`. Audio plays; generation appears in History with `source="personality_speak"` (or new `source="mcp"` if we add one).
- **REST wrapper:** `curl -X POST http://127.0.0.1:17493/speak -d '{"text":"hi","profile":"Morgan"}'` — same behavior, same pill surface.
- **Per-client:** open two Inspector sessions with different `X-Voicebox-Client-Id` headers, bind each to a different profile in Settings, verify distinct voices without `profile` arg.
- **Claude Code end-to-end (HTTP):** `claude mcp add voicebox --transport http --url http://127.0.0.1:17493/mcp --header "X-Voicebox-Client-Id: claude-code"`, then ask Claude Code to speak. Pill shows `speaking: <profile>`, audio plays, capture appears in history.
- **Stdio fallback:** manually paste the stdio snippet from Settings into a client's config, verify same behavior. `VOICEBOX_CLIENT_ID=claude-code python -m backend.mcp_shim` while backend is up; pipe a tools/list JSON-RPC in, verify response over stdout.
- **Transcribe:** point at `/tmp/test.wav`; diff against `POST /transcribe` response.
- **Failure modes:** kill backend mid-speak — shim must surface a JSON-RPC error, not deadlock. When backend isn't running, HTTP clients should get a clear connection-refused surfaced by the client.

## Risks / open decisions

- **`fastmcp` vs `mcp` package name** — confirm on `pip install`; APIs are near-identical, adjust imports.
- **Lifespan migration** touches critical path (DB init, task queue, watchdog). Dev + packaged build both need a smoke after.
- **Shim binary size** — if `mcp` pulls in enough dep weight that PyInstaller output is awkward, fall back to a Rust shim (Tauri shell is already Rust; JSON-RPC framing is trivial).
- **Source attribution** — consider `source="mcp"` on the `Generation` model, or a dedicated `originator_client` column, if the Captures tab should filter MCP-originated generations.
- **`audio_path` in `voicebox_transcribe`** — local-only today, but if the server ever binds beyond 127.0.0.1 we need to restrict reads to `data_dir` + user-whitelist.
- **Auth** — none for now (127.0.0.1 only). If we bind outside, bearer token via `~/.voicebox/secret` + plumb through shim.
- **HTTP MCP client support** — the plan leads with direct HTTP. Claude Code, Cursor, Windsurf, and VS Code MCP extensions all support it as of 2026, but if we discover an important client is stdio-only we still have the shim fallback ready.
- **`.mcpb` desktop extension for Claude Desktop** (v2 polish) — Claude Desktop supports a double-clickable extension bundle format. Worth revisiting after v1 ships for an even cleaner install; skipped for now since Claude Desktop isn't the primary user (Claude Code + IDE users are).

## Critical files

- `backend/app.py`
- `backend/routes/profiles.py`
- `backend/routes/speak.py` (new)
- `backend/database/models.py`
- `backend/database/migrations.py`
- `backend/services/generation.py`
- `backend/build_binary.py`
- `tauri/src-tauri/tauri.conf.json`
- `tauri/src-tauri/src/main.rs` (port constant — no change, just reference)
- `app/src/components/DictateWindow/DictateWindow.tsx`
- `app/src/components/CapturePill/CapturePill.tsx`
- `app/src/components/ServerSettings/`
