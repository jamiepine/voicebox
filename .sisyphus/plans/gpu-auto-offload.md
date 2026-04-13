# GPU Auto-Offload Plan

## Problem

Voicebox holds 2-3GB VRAM indefinitely after any model generation. Models are lazy-loaded on first use but never automatically released. The CUDA context itself retains fragmented memory even after `del model`. No idle detection exists anywhere in the codebase.

## Root Causes

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | No auto-unload timeout | **High** | All backends |
| 2 | No idle timestamp tracking | **High** | All backends |
| 3 | Per-request VRAM leak in `/tts/generate` | **High** | `routes/tts.py:40-42` |
| 4 | CUDA context never released after unload | **Medium** | All PyTorch backends |
| 5 | `gc.collect()` never called after unload | **Medium** | All backends |

## Design

### New Module: `backend/services/gpu_monitor.py`

A lightweight background monitor that:

1. **Tracks last-used timestamps** — Wraps each backend's `generate()`/`transcribe()` to record `time.time()` on every call
2. **Periodic sweep** — `asyncio` loop running every 30s, checking all loaded models against a configurable idle timeout (default: 10 minutes)
3. **Auto-unload** — Calls existing `unload_model()` + `empty_device_cache()` + `gc.collect()` when timeout exceeded
4. **Configurable** — Timeout duration set via `VOICEBOX_GPU_IDLE_TIMEOUT_SECONDS` env var (default: 600 = 10 min). Set to `0` to disable.
5. **Health reporting** — Exposes idle time per model via the existing `/health` endpoint

### Modified Files

| File | Change |
|------|--------|
| `backend/services/gpu_monitor.py` | **NEW** — Idle tracker decorator + background sweep loop |
| `backend/backends/__init__.py` | Inject idle tracking into backend instances at factory time |
| `backend/app.py` | Start/stop GPU monitor in lifecycle events |
| `backend/routes/health.py` | Add `gpu_models_loaded` and `gpu_idle_seconds` to response |
| `backend/routes/tts.py` | Fix per-request VRAM leak — use shared backend + explicit unload |

### Phase Breakdown

#### Phase 1: Idle Tracking + Auto-Unload Monitor

**Files**: `backend/services/gpu_monitor.py` (new), `backend/backends/__init__.py` (modify)

**What it does**:
- `IdleTracker` class: wraps backend `generate()`/`transcribe()` methods, updates `self.last_used_at = time.time()` on each call
- `GPUMonitor` class: asyncio background task, sweeps every 30s
  - Iterates `_tts_backends` dict and `_stt_backend`
  - For each loaded backend, checks `(now - last_used_at) > timeout`
  - If exceeded: calls `unload_model()` + `empty_device_cache()` + `gc.collect()`
  - Logs: `"Auto-unloaded {engine} after {seconds}s idle"`
- Factory hook: `_wrap_backend_with_idle_tracking(backend, engine)` called in `get_tts_backend_for_engine()` and `get_stt_backend()`

**Config**:
- `VOICEBOX_GPU_IDLE_TIMEOUT_SECONDS` (int, default: 600)
- `VOICEBOX_GPU_SWEEP_INTERVAL_SECONDS` (int, default: 30)

**Success criteria**:
- After 10 min idle, `torch.cuda.memory_allocated()` drops to near-zero
- Models reload correctly on next generation request
- No errors in server logs

#### Phase 2: Health Endpoint Enrichment

**Files**: `backend/routes/health.py` (modify), `backend/services/gpu_monitor.py` (add method)

**What it does**:
- Add `gpu_models_loaded: list[dict]` to `/health` response:
  ```json
  {
    "gpu_models_loaded": [
      {"engine": "qwen", "model": "0.6B", "last_used_seconds_ago": 45},
      {"engine": "whisper", "model": "base", "last_used_seconds_ago": 320}
    ]
  }
  ```
- `GPUMonitor.get_loaded_models_status()` — iterates backends, returns engine + model size + idle time

**Success criteria**:
- `/health` endpoint returns model idle info
- No models section when none loaded

#### Phase 3: Fix Per-Request VRAM Leak

**Files**: `backend/routes/tts.py` (modify)

**What it does**:
- The `/tts/generate` endpoint creates `PyTorchTTSBackend()` per request and never unloads it
- Change to use shared backend via `get_tts_backend_for_engine("qwen")`
- Or: explicitly `unload_model()` + `empty_device_cache()` in a `finally` block after generation completes

**Success criteria**:
- Repeated `/tts/generate` calls don't increase VRAM beyond single model size

## Constraints

- Must NOT break existing manual unload API endpoints
- Must NOT unload a model that is currently mid-generation (check if backend has active generation flag)
- Must support all 5 TTS engines + Whisper STT
- Must be disabled by default? No — enabled by default with 10-min timeout, opt-out via env var = 0
- Must not introduce circular imports

## Risks

| Risk | Mitigation |
|------|-----------|
| Unloading mid-generation | Track `is_generating` flag on backends; skip unload if True |
| First-generation latency after auto-unload | Acceptable tradeoff (document in release notes); user can increase timeout |
| torch CUDA context still holds ~100MB after unload | Expected behavior — CUDA runtime always reserves some memory |
| Circular import between `backends/__init__.py` and `gpu_monitor.py` | Use lazy import inside `GPUMonitor` methods |
