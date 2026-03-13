# PR #33 — CUDA Provider System Review

> Branch: `external-provider-binaries` | Created: 2026-02-01 | 34 commits, 136 files, +10,266 lines
> Reviewed: 2026-03-12

---

## The Problem

The CUDA PyTorch binary is ~2.4 GB. GitHub Releases has a 2 GB artifact limit. This means:

- Windows/Linux users with NVIDIA GPUs cannot get GPU acceleration from official releases
- 19 open issues about "GPU not detected" — the single most reported problem category
- Users who want GPU must clone the repo and run from source
- Every app update forces re-download of the entire binary

This is the #1 user pain point by volume.

---

## What PR #33 Does

Splits the monolithic Voicebox binary into two layers:

```
┌──────────────────────────────────────┐
│  Main App (~150MB Win/Lin, ~300 Mac) │
│  Tauri + React + FastAPI + Whisper   │
│  No PyTorch. MLX bundled on macOS.   │
├──────────────────────────────────────┤
│           HTTP (localhost)           │
├──────────────────────────────────────┤
│  Provider Binary (downloaded later)  │
│  PyTorch CPU (~300MB)                │
│  PyTorch CUDA (~2.4GB)              │
│  Hosted on Cloudflare R2             │
└──────────────────────────────────────┘
```

### New Backend Code

| File | Purpose |
|------|---------|
| `backend/providers/__init__.py` (327 lines) | `ProviderManager` — lifecycle management, subprocess spawning, port allocation |
| `backend/providers/base.py` (97 lines) | `TTSProvider` Protocol definition |
| `backend/providers/bundled.py` (144 lines) | `BundledProvider` — wraps existing MLX/PyTorch backends for the new interface |
| `backend/providers/local.py` (191 lines) | `LocalProvider` — HTTP client that talks to external provider processes |
| `backend/providers/installer.py` (262 lines) | Download, extract, delete provider binaries |
| `backend/providers/types.py` (34 lines) | `ProviderType` enum, `ProviderInfo` dataclass |
| `backend/providers/checksums.py` (11 lines) | Checksum dict (currently empty) |

### Provider Servers (Standalone Executables)

| File | Purpose |
|------|---------|
| `providers/pytorch-cpu/main.py` (238 lines) | FastAPI server wrapping PyTorch CPU inference |
| `providers/pytorch-cuda/main.py` (238 lines) | FastAPI server wrapping PyTorch CUDA inference |
| `providers/pytorch-*/build.py` | PyInstaller build scripts |
| `providers/pytorch-*/requirements.txt` | Isolated dependencies |

### Frontend

| File | Purpose |
|------|---------|
| `app/src/components/ServerSettings/ProviderSettings.tsx` (400 lines) | Provider download/start/stop/delete UI |

### Also Included (Scope Creep)

The PR bundles several unrelated changes that inflate the diff:

- `docs2/` — Entire documentation site rewrite (Fumadocs migration, ~3000 lines)
- `Dockerfile`, `Dockerfile.cuda`, `docker-compose.yml` — Docker support
- `landing/` — Banner removal
- UI refactors in Stories, History, Voice Profiles, Audio tab
- Linux audio capture module
- Various dependency bumps

---

## Bug Report

### Critical — Will Crash at Runtime

#### C1. Provider `generate` endpoint can't parse requests

**`providers/pytorch-cpu/main.py:91-97`** (same in pytorch-cuda)

```python
@app.post("/tts/generate")
async def generate(
    text: str,
    voice_prompt: dict,
    language: str = "auto",
    seed: int = None,
    model_size: str = "1.7B"
):
```

Parameters declared as function arguments. FastAPI interprets these as **query parameters**, not JSON body. But `LocalProvider.generate()` sends a JSON body via `httpx`:

```python
# backend/providers/local.py:33-40
response = await self.client.post("/tts/generate", json={
    "text": text,
    "voice_prompt": voice_prompt,
    ...
})
```

**Result:** Every generation call to an external provider returns HTTP 422 (Validation Error). The generation path is completely broken for external providers.

**Fix:** Use a Pydantic request body model:
```python
class GenerateRequest(BaseModel):
    text: str
    voice_prompt: dict
    language: str = "auto"
    seed: Optional[int] = None
    model_size: str = "1.7B"

@app.post("/tts/generate")
async def generate(data: GenerateRequest):
```

#### C2. Timeout error handler references undefined variables

**`backend/providers/__init__.py:82-90`**

```python
stdout_content = ""
stderr_content = ""
# ... threads write to stdout_queue / stderr_queue ...
except TimeoutError:
    while not stdout_queue.empty():
        stdout_lines.append(stdout_queue.get_nowait())  # NameError
    while not stderr_queue.empty():
        stderr_lines.append(stderr_queue.get_nowait())  # NameError
```

`stdout_lines` and `stderr_lines` are never defined. Every provider startup timeout will throw `NameError`, masking the real failure cause. Then `stdout_content` and `stderr_content` are logged but they're still empty strings — the queue data is never assigned back.

#### C3. Sync `get_tts_model()` ignores external provider in async context

**`backend/tts.py:15-29`**

```python
def get_tts_model():
    manager = get_provider_manager()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # We're in an async context, but can't await here
        return manager._get_default_provider()
```

FastAPI routes are async. This function is called from several code paths during generation. In async context it **always returns the bundled provider**, ignoring whatever external provider the user selected. The user downloads and starts a CUDA provider, but generation still runs on CPU.

### Critical — Security

#### C4. Path traversal via `tarfile.extractall()` (CVE-2007-4559)

**`backend/providers/installer.py:115-118`**

```python
with tarfile.open(archive_path, 'r:gz') as tar_ref:
    tar_ref.extractall(providers_dir)
```

No member path filtering. A crafted `.tar.gz` from a compromised CDN can write files anywhere on disk via `../` entries. Python 3.12+ emits a deprecation warning for exactly this pattern.

**Fix:**
```python
tar_ref.extractall(providers_dir, filter='data')  # Python 3.12+
```

Or manually validate each member:
```python
for member in tar_ref.getmembers():
    member_path = os.path.join(providers_dir, member.name)
    if not os.path.commonpath([providers_dir, member_path]).startswith(str(providers_dir)):
        raise ValueError(f"Path traversal attempt: {member.name}")
tar_ref.extractall(providers_dir)
```

#### C5. No checksum verification on downloaded binaries

**`backend/providers/checksums.py`**

```python
PROVIDER_CHECKSUMS = {}
```

Empty dict. `download_provider()` in `installer.py` never calls any verification function. Downloaded binaries are `chmod 0o755`'d and executed without integrity checks. A MitM or CDN compromise delivers arbitrary code.

**Fix:** Populate checksums per release. Verify SHA-256 after download before extraction:
```python
import hashlib
sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if sha256 != expected:
    archive_path.unlink()
    raise ValueError(f"Checksum mismatch for {provider_type}")
```

#### C6. Provider servers have no authentication

**`providers/pytorch-cpu/main.py:18-23`**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ...
)
```

Zero auth. Any local process — including browser JavaScript via localhost — can send requests to the provider on its ephemeral port. Port is discoverable by scanning.

**Fix:** Generate a random token in the parent process, pass via environment variable to the child, validate in middleware:
```python
# Parent (ProviderManager)
token = secrets.token_urlsafe(32)
env = {**os.environ, "VOICEBOX_PROVIDER_TOKEN": token}
process = subprocess.Popen([...], env=env, ...)

# Child (provider server)
EXPECTED_TOKEN = os.environ.get("VOICEBOX_PROVIDER_TOKEN")

@app.middleware("http")
async def verify_token(request, call_next):
    if request.headers.get("X-Provider-Token") != EXPECTED_TOKEN:
        return JSONResponse(status_code=403, content={"error": "unauthorized"})
    return await call_next(request)
```

### Major — Will Cause Problems in Production

#### M1. Leaked file handles on subprocess stdout/stderr

**`backend/providers/__init__.py:68-73`**

```python
process = subprocess.Popen(
    [...],
    stdout=open(stdout_log, 'w'),   # leaked handle
    stderr=open(stderr_log, 'w'),   # leaked handle
)
```

File handles passed directly from `open()` without storing references. They close on GC, not deterministically. On Windows the log files stay locked and unreadable until the process exits.

**Fix:**
```python
stdout_fh = open(stdout_log, 'w')
stderr_fh = open(stderr_log, 'w')
try:
    process = subprocess.Popen([...], stdout=stdout_fh, stderr=stderr_fh)
finally:
    stdout_fh.close()
    stderr_fh.close()
```

#### M2. No subprocess crash detection or recovery

**`backend/providers/__init__.py:56-110`**

Once `start_provider()` succeeds, the `Popen` object is stored but never polled. If the provider process crashes mid-session:
- `LocalProvider` HTTP calls fail with `httpx.ConnectError`
- No auto-restart
- No health-check loop
- User sees cryptic "connection refused" errors
- Must manually restart provider from UI

**Fix:** Background asyncio task that polls `process.poll()` every few seconds. On crash, update provider status and optionally auto-restart:
```python
async def _watch_provider_process(self):
    while self._provider_process and self._provider_process.poll() is None:
        await asyncio.sleep(5)
    if self._provider_process and self._provider_process.returncode != 0:
        logger.error(f"Provider crashed with code {self._provider_process.returncode}")
        self.active_provider = self._default_provider
        # Notify frontend via next health check
```

#### M3. Port allocation race condition (TOCTOU)

**`backend/providers/__init__.py:145-149`**

```python
def _get_free_port(self) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
    # Socket closed here — port is free but unprotected
```

Between this function returning and the provider process binding, another process can claim the port. On busy systems this causes "address already in use" failures.

**Fix options:**
- Pass the socket fd to the child process (complex, platform-specific)
- Retry with a new port on bind failure (simplest)
- Use a fixed port range and try sequentially

#### M4. `delete_provider()` leaves hundreds of MB behind

**`backend/providers/installer.py:155-168`**

```python
provider_path.unlink()  # Deletes just the executable
```

PyInstaller `--onedir` produces a directory with the executable plus all shared libraries. `unlink()` only removes the binary file, leaving behind hundreds of MB of `.so`/`.dll`/`.dylib` files.

**Fix:**
```python
provider_dir = provider_path.parent
shutil.rmtree(provider_dir)
```

#### M5. `LocalProvider.combine_voice_prompts()` bypasses the provider

**`backend/providers/local.py:68-88`**

This method imports from `..utils.audio` and processes locally instead of sending to the provider server. If the user chose an external provider because they lack local dependencies (e.g., no PyTorch on the machine), this will crash with `ImportError`.

#### M6. Download errors silently swallowed

**`backend/main.py:1640`**

```python
asyncio.create_task(download_provider(provider_type))
```

Fire-and-forget. If the download fails, the exception is logged as "Task exception was never retrieved." The frontend SSE progress stream may hang forever showing "downloading" without the error.

**Fix:** Store the task, add an error callback:
```python
task = asyncio.create_task(download_provider(provider_type))
task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
```
And propagate errors through the progress manager so the SSE stream surfaces them.

#### M7. `LocalProvider.is_loaded()` always returns `True`

**`backend/providers/local.py:105-108`**

```python
def is_loaded(self) -> bool:
    return True  # Return True optimistically
```

Health/status checks always report the model as loaded for external providers, even when the provider hasn't loaded anything yet. This breaks the "download model if not cached" logic in the generation flow.

#### M8. `instruct` parameter silently dropped

**`backend/providers/local.py:33-40`**

The `generate()` method accepts `instruct` but never includes it in the JSON payload. The provider server also hardcodes `instruct=None`. Delivery instructions silently do nothing for external providers.

### Minor

| # | Issue | Location |
|---|-------|----------|
| m1 | `pytorch-cpu/main.py` and `pytorch-cuda/main.py` are 95% identical | Both files | 
| m2 | `build.py` scripts also nearly identical | Both build files |
| m3 | `navigator.platform` is deprecated | `ProviderSettings.tsx:20-23` |
| m4 | `console.log('currentProvider', ...)` left in | `ProviderSettings.tsx:151` |
| m5 | `ProviderType` enum defined but never used for validation | `types.py:10-15` |
| m6 | `list_installed()` reimplements platform detection | `__init__.py:129-143` |
| m7 | New `httpx.AsyncClient` created per health poll iteration | `__init__.py:151-165` |
| m8 | `load_model_async()` only stores size, doesn't actually preload | `local.py:95-99` |

---

## Scope Creep

The PR should be split. These are independent changes bundled in:

| Change | Lines | Should Be Separate PR |
|--------|-------|-----------------------|
| `docs2/` site rewrite | ~3000 | Yes |
| Docker support (Dockerfile, compose, docs) | ~600 | Yes — overlaps with PR #161 |
| Landing page banner removal | ~30 | Yes |
| UI refactors (Stories, History, Voices, Audio) | ~400 | Yes |
| Linux audio capture module | ~10 | Yes |
| Dependency bumps | ~100 | Yes |

**Core provider system** (the actual feature) is ~2500 lines across backend + frontend + provider servers. That's the reviewable scope.

---

## What's Well-Designed

These parts should survive any rewrite:

1. **`TTSProvider` Protocol** (`base.py`) — Structural typing via `@runtime_checkable Protocol`. Right pattern. Comprehensive interface.

2. **`BundledProvider` / `LocalProvider` split** — Clean separation between in-process and HTTP-based inference. The wrapper pattern in `BundledProvider` correctly delegates to existing `TTSBackend`.

3. **R2 distribution strategy** — Provider binaries on Cloudflare R2, main app on GitHub Releases. Correct solution to the 2 GB limit.

4. **Progress tracking** — SSE-based download progress integrated with the existing `ProgressManager`. Good UX.

5. **Subprocess log files** — Writing provider stdout/stderr to log files in the data directory is pragmatic and debuggable.

6. **Frontend `ProviderSettings.tsx`** — Clean component structure. Proper loading/disabled states, confirmation dialogs, platform-aware visibility.

7. **CI split** — Separate `build-providers` and `release` jobs. Providers built and uploaded to R2 independently.

---

## Options for Moving Forward

### Option A — Fix and Slim PR #33

Strip the PR down to just the provider system (~2500 lines). Fix the 5 critical and 8 major bugs. Rebase onto current `main`.

**Effort:** ~2-3 days focused work
**Pros:** Full auto-managed provider lifecycle. Foundation for multi-model.
**Cons:** Still complex. Process management is inherently fragile cross-platform.

### Option B — Manual External Server Mode

Skip subprocess management entirely. Ship a "Connect to External Server" feature:

1. User downloads CUDA provider zip from `downloads.voicebox.sh`
2. User runs it manually (`./tts-provider-pytorch-cuda --port 8100`)
3. In Voicebox UI: paste `http://localhost:8100` as the TTS server URL
4. Voicebox routes generation to that URL via `LocalProvider`

This reuses `LocalProvider` from PR #33 but removes:
- `ProviderManager` subprocess spawning (the buggiest part)
- `installer.py` download/extract logic (the security risks)
- Port allocation (user picks the port)
- Process lifecycle management (user's responsibility)

**Effort:** ~1 day. `LocalProvider` + a URL input field + health check.
**Pros:** Simple, reliable, no process management bugs, no security surface.
**Cons:** Manual setup. Not seamless. But CUDA users are already technical (they run from source today).

### Option C — Hybrid (Recommended)

Ship Option B first as v0.2.0. Then iterate toward auto-management:

**Phase 1 (v0.2.0):** Manual external server mode
- `LocalProvider` HTTP client (from PR #33, with the 422 bug fixed)
- Server URL input in Settings
- Health indicator
- CUDA provider published as standalone zip on R2
- One page of docs: "download, unzip, run, paste URL"

**Phase 2 (v0.2.x):** Auto-download + auto-start
- `installer.py` with checksum verification and safe extraction
- `ProviderManager` subprocess spawning with crash detection
- Provider settings UI with download/start/stop buttons

**Phase 3 (v0.3.0):** Multi-model providers
- Provider per model family (not just per hardware)
- LuxTTS provider, Chatterbox provider, etc.
- Provider marketplace / registry

This gets CUDA into users' hands immediately (Phase 1 is ~1 day) while building toward the full vision incrementally. Each phase is independently shippable and testable.

### Option D — GitHub Workaround

Avoid the provider architecture entirely. Host CUDA binaries on R2 and add a download link in the app that opens the user's browser. User downloads the full monolithic CUDA build, replaces their existing install.

**Effort:** Minimal — just hosting + a link.
**Pros:** Zero architecture changes.
**Cons:** Doesn't solve: multi-model, independent app updates, or the re-download-everything-on-update problem. Kicks the can.

---

## Recommendation

**Option C (Hybrid)** is the strongest path. Specifically:

1. **Now:** Close PR #33 as-is. It's too large, too buggy, and too stale to salvage as a single merge.

2. **Extract:** Cherry-pick the good parts into small focused PRs:
   - PR: `TTSProvider` Protocol + `BundledProvider` + `LocalProvider` (the abstractions)
   - PR: Provider settings UI (the frontend)
   - PR: `installer.py` + checksums (the download system)
   - PR: CI changes for R2 upload (the distribution)

3. **Ship Phase 1:** Manual external server mode. One small PR. Unblocks every CUDA user immediately.

4. **Iterate:** Layer in auto-management once the manual mode is proven stable.

The critical bugs in PR #33 (C1-C6) are all fixable, but the PR's size makes review unreliable. Splitting it ensures each piece gets proper attention and nothing ships broken.

---

## Bug Summary

| Severity | Count | Blocks Ship? |
|----------|-------|-------------|
| Critical (runtime crash) | 3 | Yes — C1, C2, C3 |
| Critical (security) | 3 | Yes — C4, C5, C6 |
| Major | 8 | Some — M1, M2, M3 are high risk |
| Minor | 8 | No |
| **Total** | **22** | |
