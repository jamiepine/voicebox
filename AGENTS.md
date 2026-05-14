# AGENTS.md — Voicebox Agent Reference

Practical reference for AI agents working in this repo. Read this before touching any code.

---

## Project overview

Voicebox is an open-source AI voice studio — a desktop app (Tauri/React) backed by a local FastAPI server that runs TTS, STT, and voice-cloning workloads entirely on-device. It supports multiple TTS engines (MLX on Apple Silicon, PyTorch, Kokoro, Chatterbox, Qwen, HumeAI TADA, LuxTTS), voice profile management with multi-sample cloning, a real-time dictation system, and an MCP server so external agents can drive speech generation.

---

## Repository structure

```
voicebox/
├── backend/           FastAPI Python server (port 17493)
│   ├── backends/      One class per TTS/STT engine, all extend BaseTTSBackend
│   ├── routes/        FastAPI routers (one file per resource)
│   ├── services/      Business logic (profiles, generation, history, …)
│   ├── database/      SQLAlchemy models + migrations
│   ├── mcp_server/    FastMCP server exposing TTS to external agents
│   ├── utils/         Shared helpers (audio, HF cache, platform detect, …)
│   └── tests/         pytest suite — unit/ and integration/ subdirs
├── app/               React/TypeScript frontend (Tauri webview)
│   └── src/lib/api/   AUTO-GENERATED TypeScript client — never edit manually
├── tauri/             Tauri desktop wrapper (Rust)
│   └── src-tauri/     Rust source, Cargo.toml, tauri.conf.json
├── web/               Standalone web app (same React stack, no Tauri)
├── landing/           Marketing site
├── docs/              Documentation site
├── scripts/           Shell scripts (generate-api.sh, build-server.sh, …)
├── .agents/skills/    Reusable agent skill definitions (see below)
├── justfile           Full task runner — cross-platform, power-user tool
├── Makefile           Newcomer-friendly wrapper over just
├── biome.json         Biome config (TS/JS lint + format)
└── backend/pyproject.toml  Python tooling config (ruff, pytest)
```

---

## Auto-generated files — never edit manually

### `app/src/lib/api/`

TypeScript API client generated from the FastAPI OpenAPI spec via `openapi-typescript-codegen`. Every file in this directory is overwritten on regeneration.

**Biome explicitly excludes this directory** (`"!app/src/lib/api"` in `biome.json`).

To regenerate after changing any backend route or model:

```bash
just generate-api        # starts backend if needed, downloads spec, runs codegen
# or manually:
./scripts/generate-api.sh
```

The script fetches `http://localhost:17493/openapi.json` and writes to `app/src/lib/api/`. The backend must be running, or the script will start one temporarily.

### `app/openapi.json`

Snapshot of the OpenAPI schema downloaded during `generate-api`. Committed for reference; regenerated automatically.

### PyInstaller artifacts

`backend/voicebox-server.spec`, `pyi_rth_*.py`, `pyi_hooks/` — PyInstaller packaging files. Don't edit; ruff excludes them.

---

## Development commands

### Setup

```bash
make install             # full install: system deps + Python venv + JS packages
make pre-commit-install  # wire up git hooks (run once after install)
# or with just:
just setup               # Python venv + JS deps
```

Python 3.12 is required. 3.13+ may have ML package incompatibilities.

### Run

```bash
# Two terminals:
make dev-backend         # uvicorn on :17493 with --reload
make dev-frontend        # Tauri desktop app

# One command (just):
just dev                 # starts backend + Tauri together

# Web-only (no Tauri):
just dev-web             # backend + web/ vite dev server
```

### Test

```bash
make test                # all tests: backend pytest + frontend Vitest
make test-unit           # backend unit tests only (fast, no I/O)
make test-integration    # backend integration tests (real SQLite, no network)
make test-frontend       # Vitest only
# or:
just test                # pytest backend/tests
just test-frontend       # bun run test
just test-all            # both
```

Skip the integration gate on a quick WIP commit:
```bash
SKIP=pytest-integration git commit -m "..."
```

### Lint and format

```bash
make lint                # ruff check + biome lint
make format              # ruff format + biome format (auto-fixes)
# or:
just check               # lint + format check, no auto-fix
just fix                 # auto-fix everything (ruff + biome)
just lint                # lint only
just format              # format only
```

### Build

```bash
just build               # server binary + Tauri app
just build-server        # PyInstaller binary only
just build-tauri         # Tauri .app / .dmg / .exe only
```

---

## Before submitting changes

Run both and make sure they pass:

```bash
make lint
make test
```

Pre-commit hooks enforce this automatically after `make pre-commit-install`. Hooks run: ruff lint+format, Biome lint+format, tsc typecheck, pytest unit, pytest integration, Vitest.

**Backend:** ruff (lint + format, 120-char lines, Python 3.12 target)
**Frontend:** Biome (lint + format, 2-space indent, 100-char lines), tsc (strict)
**Rust:** `cargo clippy -- -D warnings` — zero warnings allowed

---

## Architecture notes

### Stack

| Layer | Tech | Port / path |
|-------|------|-------------|
| Backend API | FastAPI + uvicorn | `:17493` |
| Frontend (desktop) | React + TypeScript, bundled by Bun | Tauri webview |
| Frontend (web) | Same React app, served by Vite | varies |
| Desktop shell | Tauri v2 (Rust) | — |
| Database | SQLite via SQLAlchemy | `backend/data/voicebox.db` |
| API client | Auto-generated from OpenAPI | `app/src/lib/api/` |

### TTS engine system

All engines live in `backend/backends/`. Each is a class that implements the `BaseTTSBackend` protocol (defined in `backend/backends/base.py`). Registration is in `backend/backends/__init__.py`.

To add a new engine:
1. Create `backend/backends/<name>_backend.py`, implement `BaseTTSBackend`.
2. Register it in `backend/backends/__init__.py`.
3. Add it to the model config registry — search for existing entries to follow the pattern.
4. Use the `.agents/skills/add-tts-engine` skill for the full step-by-step.

### Voice profiles and samples

One `VoiceProfile` → many `ProfileSample` records. The TTS service combines samples via `combine_voice_prompts()` on the backend (each engine implements its own combiner). Profiles are managed in `backend/services/profiles.py`.

### MLX backend

`backend/backends/mlx_backend.py` is Apple Silicon only. Guard any Apple-specific imports:

```python
if sys.platform == "darwin":
    import mlx
```

Never import `mlx` or `mlx_audio` unconditionally — CI runs on Linux.

### MCP server

`backend/mcp_server/` exposes TTS generation to external MCP clients. `backend/mcp_shim/` handles Tauri-side MCP transport.

---

## Things to avoid

- **Never edit `app/src/lib/api/`** — regenerate with `just generate-api`.
- **Don't mix torch and MLX** — they're separate backends; keep imports in their respective backend files.
- **Don't add `mlx-audio` or `mlx-lm` to `requirements.txt` without `--no-deps`** — they declare `transformers>=5.x` which conflicts with the `transformers<=4.57.6` cap. Install with `--no-deps` as done in `justfile`/`Makefile`.
- **Don't add `chatterbox-tts` or `hume-tada` to `requirements.txt`** — same reason; both are installed with `--no-deps`.
- **Don't use `datetime.utcnow()`** — use `datetime.now(UTC)` (UTC is imported from `datetime`).
- **Don't use `Optional[X]`, `List[X]`, `Tuple[X]`** — use `X | None`, `list[X]`, `tuple[X]` (Python 3.12 style).
- **Don't import `from typing import List, Dict, Optional`** — use built-in generics.
- **Rust: no unawaited futures** — `let _ = some_future` without `.await` will fail `cargo clippy -D warnings`.
- **Don't add heavy imports (torch, transformers) to files that CI tests without ML deps** — the backend CI job installs only lightweight packages. Keep ML imports inside backend class methods (lazy import pattern).

---

## CI

Three jobs in `.github/workflows/ci.yml`, all on `ubuntu-latest`, triggered on every PR and push to `main`/`improvements`.

| Job | What it runs |
|-----|-------------|
| `frontend-ci` | `bun install`, Biome lint + format check, `tsc` typecheck, Vitest, `bun run build:web` |
| `backend-ci` | ruff lint + format check, pytest (lightweight subset — no torch/mlx) |
| `rust-check` | `cargo check` + `cargo clippy -- -D warnings` |

**Critical:** backend CI installs only `fastapi httpx pydantic pytest pytest-asyncio python-multipart ruff sqlalchemy starlette` — no torch, no transformers, no soundfile. If you add an import of a heavy package at module level in a file that is tested, CI will fail. Keep heavy deps as lazy imports inside functions.

Rust CI stubs the external sidecar binaries (`tauri/src-tauri/binaries/`) so `tauri_build` doesn't error on Linux.

---

## `.agents/skills/` directory

Reusable workflow definitions for common tasks:

| Skill | Purpose |
|-------|---------|
| `add-tts-engine` | Full workflow for adding a new TTS engine |
| `draft-release-notes` | Generate release notes from commit history |
| `release-bump` | Bump version across all config files (Cargo.toml, package.json, pyproject.toml, tauri.conf.json) |
| `triage-prs` | Triage open pull requests |

Invoke via your agent harness: `/<skill-name>` or reference the file directly for step-by-step instructions.
