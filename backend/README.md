# Voicebox Backend

FastAPI server powering voice cloning, speech generation, and audio processing. Runs locally as a Tauri sidecar or standalone via `python -m backend.main`.

## Running

```bash
# Via justfile (recommended)
just dev:server

# Standalone
python -m backend.main --host 127.0.0.1 --port 17493

# With custom data directory
python -m backend.main --data-dir /path/to/data
```

The server auto-initializes the SQLite database on first startup. Models are downloaded from HuggingFace on first use.

## Architecture

```
backend/
  app.py                  # FastAPI app factory, CORS, lifecycle events
  main.py                 # Entry point (imports app, runs uvicorn)
  config.py               # Data directory paths and configuration
  models.py               # Pydantic request/response schemas
  server.py               # Tauri sidecar launcher, parent-pid watchdog

  routes/                 # Thin HTTP handlers — validation, delegation, response formatting
  services/               # Business logic, CRUD, orchestration
  backends/               # TTS/STT engine implementations (MLX, PyTorch, etc.)
  database/               # ORM models, session management, migrations, seed data
  utils/                  # Shared utilities (audio, effects, caching, progress tracking)
```

### Request flow

```
HTTP request
  -> routes/        (validate input, parse params)
  -> services/      (business logic, database queries, orchestration)
  -> backends/      (TTS/STT inference)
  -> utils/         (audio processing, effects, caching)
```

Route handlers are intentionally thin. They validate input, delegate to a service function, and format the response. All business logic lives in `services/`.

### Key modules

**services/generation.py** -- Single `run_generation()` function that handles all three generation modes (generate, retry, regenerate). Manages model loading, voice prompt creation, chunked inference, normalization, effects, and version persistence.

**services/task_queue.py** -- Serial generation queue. Ensures only one GPU inference runs at a time. Background tasks are tracked to prevent garbage collection.

**backends/__init__.py** -- Protocol definitions (`TTSBackend`, `STTBackend`), model config registry, and factory functions. Adding a new engine means implementing the protocol and registering a config entry.

**backends/base.py** -- Shared utilities used across all engine implementations: HuggingFace cache checks, device detection, voice prompt combination, progress tracking.

**database/** -- SQLAlchemy ORM models with a re-exporting `__init__.py` for backward compatibility. Migrations run automatically on startup.

### Backend selection

The server detects the best inference backend at startup:

| Platform | Backend | Acceleration |
|----------|---------|-------------|
| macOS (Apple Silicon) | MLX | Metal / Neural Engine |
| Windows / Linux (NVIDIA) | PyTorch | CUDA |
| Linux (AMD) | PyTorch | ROCm |
| Intel Arc | PyTorch | IPEX / XPU |
| Windows (any GPU) | PyTorch | DirectML |
| Any | PyTorch | CPU fallback |

Detection is handled by `utils/platform_detect.py`. Both backends implement the same `TTSBackend` protocol, so the API layer is engine-agnostic.

## API

90 endpoints organized by domain. Full interactive documentation available at `http://localhost:17493/docs` when the server is running.

| Domain | Prefix | Description |
|--------|--------|-------------|
| Health | `/`, `/health` | Server status, GPU info, filesystem checks |
| Profiles | `/profiles` | Voice profile CRUD, samples, avatars, import/export |
| Channels | `/channels` | Audio channel management and voice assignment |
| Generation | `/generate` | TTS generation, retry, regenerate, status SSE |
| History | `/history` | Generation history, search, favorites, export |
| Transcription | `/transcribe` | Whisper-based audio-to-text |
| Stories | `/stories` | Multi-track timeline editor, audio export |
| Effects | `/effects` | Effect presets, preview, version management |
| Audio | `/audio`, `/samples` | Audio file serving |
| Models | `/models` | Load, unload, download, migrate, status |
| Tasks | `/tasks`, `/cache` | Active task tracking, cache management |
| CUDA | `/backend/cuda-*` | CUDA binary download and management |

### Quick examples

```bash
# Generate speech
curl -X POST http://localhost:17493/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "profile_id": "...", "language": "en"}'

# List profiles
curl http://localhost:17493/profiles

# Stream generation status (SSE)
curl http://localhost:17493/generate/{id}/status
```

## Data directory

```
{data_dir}/
  voicebox.db             # SQLite database
  profiles/{id}/          # Voice samples per profile
  generations/            # Generated audio files
  cache/                  # Voice prompt cache (memory + disk)
  backends/               # Downloaded CUDA binary (if applicable)
```

Default location is the OS-specific app data directory. Override with `--data-dir` or the `VOICEBOX_DATA_DIR` environment variable.

## Code quality

Linting and formatting are enforced by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. See `STYLE_GUIDE.md` for conventions.

```bash
just check-python       # lint + format check
just fix-python         # auto-fix lint issues + reformat
just test               # run pytest
```

## Dependencies

Runtime dependencies are in `requirements.txt`. macOS-only MLX dependencies are in `requirements-mlx.txt`. Dev tools (ruff, pytest) are installed automatically by `just setup-python`.
