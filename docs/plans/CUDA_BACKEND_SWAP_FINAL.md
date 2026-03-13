# CUDA Backend Swap — Implementation Summary

> Status: **Complete** | Branch: `feat/cuda-backend-swap` | Created: 2026-03-12

## What This Is

A standalone feature that lets users download a CUDA-enabled backend binary (~2.4 GB) and swap it in via a backend-only restart. The frontend stays running, all UI state is preserved. This solves the #1 user pain point: 19 open issues about "GPU not detected" caused by GitHub's 2 GB release asset limit preventing CUDA binaries from shipping in official releases.

## How It Works

```
User clicks "Download CUDA Backend" in Settings
  → Backend fetches manifest from GitHub Releases
  → Downloads split parts (<2 GB each), concatenates them
  → SHA-256 integrity check on reassembled binary
  → Binary placed in {app_data_dir}/backends/voicebox-server-cuda
  → User clicks "Switch to CUDA Backend"
  → Tauri kills CPU process, launches CUDA binary, frontend reconnects
  → On all future app launches, CUDA binary is auto-detected and used
```

The CUDA binary is functionally identical to the CPU binary — same FastAPI app, same endpoints, same code. The only difference is PyTorch compiled with CUDA 12.1 and bundled CUDA runtime libraries.

## Architecture Decisions

**Backend-only restart, not full app restart.** The Tauri shell kills the current `voicebox-server` process, waits 1 second for port release, and spawns the new binary. The React frontend stays running. Health polling detects the new backend within seconds.

**No provider/subprocess architecture.** This is explicitly not the PR #33 approach (10K+ lines, 136 files, 22 bugs). One process at a time. The CUDA binary replaces the CPU binary — it doesn't run alongside it.

**Data directory, not app bundle.** The CUDA binary lives in `{app_data_dir}/backends/`, which persists across app updates and avoids code-signing issues. The bundled CPU binary in the app bundle is untouched.

**Version mismatch protection.** On startup, Rust runs `voicebox-server-cuda --version` and compares to the app version from `tauri.conf.json`. If they don't match (e.g., after an app update), it falls back to the bundled CPU binary silently.

**GitHub Releases distribution.** The CUDA binary is split into <2 GB chunks (GitHub's asset limit) via `scripts/split_binary.py`. The app downloads a manifest, fetches each part, concatenates them, and runs a SHA-256 integrity check to verify reassembly. No external hosting needed.

## Files Changed

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `backend/cuda_download.py` | ~190 | Download split parts from GitHub Releases, reassemble, verify integrity |
| `scripts/split_binary.py` | ~80 | Split large binary into <2 GB chunks with SHA-256 manifest |
| `.github/workflows/build-cuda.yml` | ~70 | CI workflow: build CUDA binary, split, upload to GitHub Releases |
| `app/src/components/ServerSettings/GpuAcceleration.tsx` | 371 | GPU Acceleration UI card (status, download, restart, delete) |
| `docs/plans/CUDA_BACKEND_SWAP.md` | 581 | Original implementation plan (5 phases with code sketches) |
| `docs/plans/CUDA_BACKEND_SWAP_FINAL.md` | this file | Final implementation summary |
| `docs/plans/PROJECT_STATUS.md` | 462 | Full project triage (all PRs, issues, architecture) |
| `docs/plans/PR33_CUDA_PROVIDER_REVIEW.md` | ~350 | Detailed code review of PR #33 (22 bugs documented) |

### Modified Files

| File | What Changed |
|------|-------------|
| `backend/build_binary.py` | Added `--cuda` flag, parameterized output binary name |
| `backend/server.py` | Added `--version` flag, auto-detect backend variant from binary name (`VOICEBOX_BACKEND_VARIANT` env var) |
| `backend/main.py` | 4 new endpoints (`/backend/cuda-status`, `/backend/download-cuda`, `/backend/cuda`, `/backend/cuda-progress`), health endpoint returns `backend_variant` |
| `backend/models.py` | `HealthResponse` model: added `backend_variant` field |
| `backend/requirements.txt` | Added `httpx>=0.27.0` for async HTTP downloads |
| `tauri/src-tauri/src/main.rs` | `restart_server` command (stop → wait → start), `start_server` checks for CUDA binary in data dir and launches via `shell().command()`, version mismatch check |
| `app/src/platform/types.ts` | `PlatformLifecycle.restartServer()` added |
| `tauri/src/platform/lifecycle.ts` | `restartServer()` implementation via `invoke('restart_server')` |
| `web/src/platform/lifecycle.ts` | `restartServer()` noop for web platform |
| `app/src/lib/api/types.ts` | `CudaStatus`, `CudaDownloadProgress` interfaces; `HealthResponse` updated with `gpu_type`, `backend_type`, `backend_variant` |
| `app/src/lib/api/client.ts` | `getCudaStatus()`, `downloadCudaBackend()`, `deleteCudaBackend()` methods |
| `app/src/components/ServerTab/ServerTab.tsx` | Wired in `<GpuAcceleration />` component (Tauri-only) |

## Backend API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/backend/cuda-status` | Returns `{ available, active, binary_path, downloading, download_progress }` |
| `POST` | `/backend/download-cuda` | Starts background download; returns immediately. Track via SSE. |
| `DELETE` | `/backend/cuda` | Deletes CUDA binary (blocked if CUDA is currently active) |
| `GET` | `/backend/cuda-progress` | SSE stream of download progress (reuses existing `ProgressManager`) |

The existing `GET /health` endpoint now returns two new fields:
- `backend_type`: `"pytorch"` or `"mlx"` (existing detection)
- `backend_variant`: `"cpu"` or `"cuda"` (set from `VOICEBOX_BACKEND_VARIANT` env var)

## Frontend UI States

The `GpuAcceleration` card in Server Settings handles these states:

1. **Native GPU detected** (MPS, MLX, XPU, DirectML) — Shows info message, no download needed
2. **No CUDA binary** — Download button with size estimate, description of requirements
3. **Downloading** — SSE-driven progress bar with bytes/total and percentage
4. **Downloaded, not active** — "Switch to CUDA Backend" button + "Remove" option
5. **CUDA active** — Shows CUDA badge, "Switch to CPU Backend" button
6. **Restarting** — Spinner with phase text, 1s health polling as safety net
7. **Error** — Red error message with details

### Key UX detail: switching to CPU

Since `start_server` always prefers the CUDA binary if it exists on disk, "Switch to CPU" must delete the CUDA binary first, then restart. The user can re-download later. This avoids a persistent configuration mechanism (no new state to manage, no new config file, no DB column).

## Rust: Server Lifecycle

```
start_server
  ├── Check for CUDA binary at {data_dir}/backends/voicebox-server-cuda
  ├── If found: run --version, compare to app version
  │   ├── Match: launch via shell().command() with --data-dir, --port
  │   └── Mismatch: log warning, fall through to CPU
  └── Else: launch bundled sidecar via shell().sidecar()

restart_server
  ├── stop_server (kill process tree)
  ├── wait 1 second for port release
  └── start_server (auto-detects CUDA)
```

## What This Doesn't Cover

- **AMD GPU / ROCm / DirectML binary** — Same pattern, different PyTorch build. Future PR.
- **Linux CUDA** — Same approach, just another CI matrix entry. Can ship same release.
- **Multi-model support** — LuxTTS, Chatterbox, etc. are a separate architectural concern (in-process model registry). Independent of binary variant.
- **Download resume** — If download is interrupted, it restarts from scratch. Acceptable for v1.
- **Remote server CUDA** — Users running voicebox-server on a remote machine manage their own binaries. This feature is for the desktop app.

## Testing Checklist

- [ ] Build CUDA binary locally with `python backend/build_binary.py --cuda`
- [ ] `voicebox-server-cuda --version` prints correct version
- [ ] Place CUDA binary in `{data_dir}/backends/`, launch app → auto-detects and uses it
- [ ] Version mismatch: rename binary to have wrong version → falls back to CPU
- [ ] Frontend: GpuAcceleration card shows correct state for CPU, CUDA available, CUDA active
- [ ] Download flow: POST triggers download, SSE progress works, completion updates status
- [ ] Switch to CUDA: restart works, health endpoint shows `backend_variant: "cuda"`
- [ ] Switch to CPU: deletes binary, restarts, health shows `backend_variant: "cpu"`
- [ ] Delete CUDA while active: returns 409 error
- [ ] Split binary script: `python scripts/split_binary.py` creates manifest + parts + sha256
- [ ] Native GPU (macOS MPS): shows info message, no download section
