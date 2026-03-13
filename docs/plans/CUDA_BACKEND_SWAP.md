# CUDA Backend Swap via Binary Replacement

> Status: Plan | Target: v0.2.0 | Created: 2026-03-12

## Problem

The CUDA PyTorch backend binary is ~2.4 GB. GitHub Releases has a 2 GB asset limit. The current release ships CPU-only PyTorch on Windows and Intel Mac — NVIDIA GPU users get no acceleration from official releases. This is the #1 reported issue category (19 open issues).

Users who want GPU today must clone the repo and run from source. That's not acceptable for a desktop app targeting non-technical users.

## Solution

Ship two backend binaries: a default CPU build (~150 MB) bundled with the app, and a downloadable CUDA build (~2.4 GB) hosted externally. When the user downloads the CUDA build, the app kills the current backend process, swaps in the CUDA binary, and relaunches — a backend-only restart. The frontend stays running, all UI state is preserved.

No subprocesses. No HTTP protocol between processes. No port allocation. No provider manager. The backend is still one monolithic process — just a different binary.

## Architecture

### What Exists Today

```
Tauri App
  ├── React Frontend (in-process webview)
  └── voicebox-server (sidecar subprocess on :17493)
        └── One PyInstaller binary: CPU PyTorch or MLX
```

**Sidecar lifecycle** (`tauri/src-tauri/src/main.rs`):
- `start_server` command spawns `voicebox-server` sidecar (line 181)
- Binary located at `tauri/src-tauri/binaries/voicebox-server-{platform-triple}`
- Tauri resolves the sidecar name via `externalBin` in `tauri.conf.json` (line 16)
- Waits up to 120s for "Uvicorn running" in stdout/stderr (line 286)
- `stop_server` kills the process tree (line 466)

**Frontend reconnection** (`app/src/lib/hooks/useServer.ts`):
- Health check polls `GET /health` every 30 seconds
- React Query cache retains data for 10 minutes after disconnect
- All UI state (Zustand stores, form data, open tabs) survives disconnection
- No active reconnect logic — just keeps polling until server responds

This means a backend restart is mostly invisible to the frontend: it sees a few seconds of failed health checks, then the server comes back. The only risk is in-flight operations (generation, transcription) failing mid-request.

### What Changes

```
Tauri App
  ├── React Frontend (in-process webview)
  └── voicebox-server (sidecar subprocess on :17493)
        └── One of:
            ├── voicebox-server-cpu     (bundled, ~150 MB)
            └── voicebox-server-cuda    (downloaded, ~2.4 GB)
```

The CUDA binary is functionally identical to the CPU binary. Same FastAPI app, same endpoints, same code. The only difference is PyTorch is compiled with CUDA 12.1 support and the binary includes CUDA runtime libraries.

The user downloads it once. On every subsequent app launch, Tauri checks which binary variant exists and spawns the appropriate one.

## Implementation Plan

### Phase 1: Build Infrastructure

Build the CUDA binary in CI separately from the main release.

#### 1a. CUDA PyInstaller Build

Add a `build_binary_cuda.py` or parameterize the existing `build_binary.py`:

```python
# backend/build_binary.py — add flag
def build_server(cuda=False):
    args = [
        'server.py',
        '--onefile',
        '--name', f'voicebox-server-{"cuda" if cuda else "cpu"}',
    ]

    if cuda:
        args.extend([
            '--hidden-import', 'torch.cuda',
            '--hidden-import', 'torch.backends.cudnn',
        ])
    # ... rest of existing build
```

The `--onefile` flag is already used, which produces a single executable. This is important — `--onedir` would complicate the swap (replacing a directory vs a file).

#### 1b. CI Workflow for CUDA Binary

New workflow: `.github/workflows/build-cuda.yml`

```yaml
name: Build CUDA Provider
on:
  workflow_dispatch:
  push:
    tags: ["v*"]

jobs:
  build-cuda:
    runs-on: windows-latest  # CUDA is Windows/Linux only
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install dependencies
        run: |
          pip install pyinstaller
          pip install -r backend/requirements.txt
          pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
      - name: Build CUDA binary
        run: python backend/build_binary.py --cuda
      - name: Split binary for GitHub Releases
        run: |
          python scripts/split_binary.py backend/dist/voicebox-server-cuda.exe \
            --chunk-size 1900MB \
            --output release-assets/
      - name: Upload to R2
        # Full binary to R2 (no size limit)
        run: |
          aws s3 cp backend/dist/voicebox-server-cuda.exe \
            s3://voicebox-downloads/cuda/v${{ github.ref_name }}/voicebox-server-cuda.exe \
            --endpoint-url ${{ secrets.R2_ENDPOINT }}
      - name: Upload split parts to GitHub Release
        # Split parts as GitHub Release assets (each <2 GB)
        uses: softprops/action-gh-release@v1
        with:
          files: release-assets/*
```

Two distribution paths for redundancy:
- **Cloudflare R2**: Full binary, direct download, no size limit.
- **GitHub Releases**: Split into <2 GB chunks as fallback.

#### 1c. Binary Splitting Script

```python
# scripts/split_binary.py
"""Split a large binary into chunks for GitHub Releases."""
import hashlib
import argparse
from pathlib import Path

def split(input_path: Path, chunk_size: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = input_path.read_bytes()

    # Write SHA-256 of the complete file
    sha256 = hashlib.sha256(data).hexdigest()
    (output_dir / f"{input_path.stem}.sha256").write_text(
        f"{sha256}  {input_path.name}\n"
    )

    # Split into chunks
    parts = []
    for i in range(0, len(data), chunk_size):
        part_name = f"{input_path.stem}.part{len(parts):02d}{input_path.suffix}"
        part_path = output_dir / part_name
        part_path.write_bytes(data[i:i + chunk_size])
        parts.append(part_name)

    # Write manifest
    (output_dir / f"{input_path.stem}.manifest").write_text(
        "\n".join(parts) + "\n"
    )

    print(f"Split into {len(parts)} parts, SHA-256: {sha256}")
```

### Phase 2: Download & Assemble in App

#### 2a. Backend Download Endpoint

Add to `backend/main.py`:

```python
@app.post("/backend/download-cuda")
async def download_cuda_backend():
    """Download the CUDA backend binary."""
    # Returns immediately, runs download in background
    task = asyncio.create_task(_download_cuda_binary())
    task.add_done_callback(lambda t: logger.error(f"CUDA download failed: {t.exception()}") if t.exception() else None)
    return {"status": "downloading"}

@app.get("/backend/cuda-status")
async def cuda_status():
    """Check if CUDA binary is available."""
    cuda_path = _get_cuda_binary_path()
    return {
        "available": cuda_path is not None and cuda_path.exists(),
        "active": _is_cuda_active(),
        "download_progress": progress_manager.get_progress("cuda-backend"),
    }
```

#### 2b. Download + Assemble + Verify Logic

New file: `backend/cuda_download.py`

Core logic:

```python
import hashlib
from pathlib import Path
from backend.config import get_data_dir
from backend.utils.progress import get_progress_manager

CUDA_DOWNLOAD_URL = "https://downloads.voicebox.sh/cuda/{version}/voicebox-server-cuda{ext}"
CUDA_CHECKSUMS = {
    # Populated per release
    "0.2.0-windows": "sha256:abc123...",
    "0.2.0-linux": "sha256:def456...",
}

def get_cuda_binary_dir() -> Path:
    """Where CUDA binaries live. Inside the app's data directory."""
    return get_data_dir() / "backends"

def get_cuda_binary_path() -> Path | None:
    """Return path to CUDA binary if it exists and is verified."""
    d = get_cuda_binary_dir()
    for name in ["voicebox-server-cuda.exe", "voicebox-server-cuda"]:
        p = d / name
        if p.exists():
            return p
    return None

async def download_cuda_binary(version: str):
    """Download, assemble (if split), and verify the CUDA binary."""
    progress = get_progress_manager()
    dest_dir = get_cuda_binary_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = ".exe" if sys.platform == "win32" else ""
    url = CUDA_DOWNLOAD_URL.format(version=version, ext=ext)

    # Download with progress tracking
    temp_path = dest_dir / f"voicebox-server-cuda{ext}.download"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(temp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update("cuda-backend", downloaded, total)

    # Verify checksum
    sha256 = hashlib.sha256(temp_path.read_bytes()).hexdigest()
    expected = CUDA_CHECKSUMS.get(f"{version}-{sys.platform}")
    if expected and not expected.endswith(sha256):
        temp_path.unlink()
        raise ValueError(f"Checksum mismatch: expected {expected}, got sha256:{sha256}")

    # Atomic move into place
    final_path = dest_dir / f"voicebox-server-cuda{ext}"
    temp_path.rename(final_path)

    # Make executable on Unix
    if sys.platform != "win32":
        final_path.chmod(0o755)

    progress.complete("cuda-backend")
```

Key points:
- Downloads to a `.download` temp file, verifies checksum, then atomically renames. No partial binaries left on crash.
- Progress tracked via the existing `ProgressManager` so the frontend SSE system works unchanged.
- CUDA binary lives in the **app data directory** (`data/backends/`), not alongside the app bundle. This avoids code-signing issues on macOS (though CUDA isn't relevant on macOS) and survives app updates.

#### 2c. Reassembly from Split Parts (GitHub Releases Fallback)

If the R2 download fails, fall back to downloading split parts from GitHub Releases:

```python
async def download_cuda_from_github(version: str):
    """Fallback: download split parts from GitHub Releases, reassemble."""
    base_url = f"https://github.com/jamiepine/voicebox/releases/download/v{version}"

    # Get manifest
    manifest_url = f"{base_url}/voicebox-server-cuda.manifest"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        manifest = (await client.get(manifest_url)).text
        parts = [p.strip() for p in manifest.strip().splitlines()]

        # Download checksum
        sha256_url = f"{base_url}/voicebox-server-cuda.sha256"
        expected_sha = (await client.get(sha256_url)).text.split()[0]

        # Download parts
        dest_dir = get_cuda_binary_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        temp_path = dest_dir / "voicebox-server-cuda.exe.download"

        total_downloaded = 0
        with open(temp_path, "wb") as f:
            for i, part_name in enumerate(parts):
                part_url = f"{base_url}/{part_name}"
                async with client.stream("GET", part_url) as response:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        total_downloaded += len(chunk)
                        get_progress_manager().update(
                            "cuda-backend", total_downloaded, None,
                            message=f"Downloading part {i+1}/{len(parts)}"
                        )

    # Verify reassembled file
    sha256 = hashlib.sha256(temp_path.read_bytes()).hexdigest()
    if sha256 != expected_sha:
        temp_path.unlink()
        raise ValueError(f"Checksum mismatch after reassembly")

    final_path = dest_dir / "voicebox-server-cuda.exe"
    temp_path.rename(final_path)
    get_progress_manager().complete("cuda-backend")
```

### Phase 3: Backend Restart (The Swap)

This is the core of the feature: kill the CPU backend, launch the CUDA backend, frontend reconnects automatically.

#### 3a. New Tauri Command: `restart_server`

Add to `tauri/src-tauri/src/main.rs`:

```rust
#[command]
async fn restart_server(
    app: tauri::AppHandle,
    state: State<'_, ServerState>,
    use_cuda: Option<bool>,
) -> Result<String, String> {
    println!("restart_server: use_cuda={:?}", use_cuda);

    // 1. Stop the current server
    stop_server(state.clone()).await?;

    // 2. Brief wait for port release
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // 3. Start with the appropriate binary
    // The start_server logic needs to check for CUDA binary
    start_server(app, state, None).await
}
```

#### 3b. Modify `start_server` to Prefer CUDA Binary

The existing `start_server` uses `app.shell().sidecar("voicebox-server")` which resolves via Tauri's `externalBin` config. For the CUDA binary (which lives in the data directory, not the app bundle), we need an alternative launch path.

Modify `start_server` in `main.rs`:

```rust
// After the existing sidecar logic, before spawning:

// Check for CUDA binary in data directory
let cuda_binary = data_dir.join("backends")
    .join(if cfg!(windows) { "voicebox-server-cuda.exe" } else { "voicebox-server-cuda" });

let (mut rx, child) = if cuda_binary.exists() {
    println!("Found CUDA backend binary at {:?}", cuda_binary);

    // Launch CUDA binary directly (not as Tauri sidecar)
    let mut cmd = app.shell().command(cuda_binary.to_str().unwrap());
    cmd = cmd.args([
        "--data-dir",
        data_dir.to_str().ok_or("Invalid data dir path")?,
        "--port",
        &SERVER_PORT.to_string(),
    ]);
    if remote.unwrap_or(false) {
        cmd = cmd.args(["--host", "0.0.0.0"]);
    }
    cmd.spawn().map_err(|e| format!("Failed to spawn CUDA backend: {}", e))?
} else {
    // Existing sidecar launch (CPU binary bundled with app)
    sidecar.spawn().map_err(|e| format!("Failed to spawn: {}", e))?
};
```

Key decisions:
- CUDA binary is launched via `app.shell().command()` (arbitrary path), not `app.shell().sidecar()` (bundled path). Tauri's sidecar system only resolves binaries within the app bundle.
- The CUDA binary gets the same args (`--data-dir`, `--port`) as the CPU binary. It's the same `server.py` entry point.
- Preference: if CUDA binary exists, use it. Otherwise fall back to bundled CPU. No user configuration needed.

#### 3c. Frontend: Trigger Restart After Download

Add to the platform lifecycle interface (`app/src/platform/types.ts`):

```typescript
interface PlatformLifecycle {
    startServer(remote?: boolean): Promise<string>;
    stopServer(): Promise<void>;
    restartServer(useCuda?: boolean): Promise<string>;  // new
    // ...
}
```

Implement in `tauri/src/platform/lifecycle.ts`:

```typescript
async restartServer(useCuda?: boolean): Promise<string> {
    const result = await invoke<string>('restart_server', { useCuda });
    this.onServerReady?.();
    return result;
}
```

#### 3d. Frontend: GPU Settings UI

Add a section to the Server Settings page (or Model Management). Minimal UI:

```
┌─────────────────────────────────────────────┐
│  GPU Acceleration                           │
│                                             │
│  Status: CPU only (no CUDA backend)         │
│                                             │
│  [Download CUDA Backend (2.4 GB)]           │
│                                             │
│  Requires an NVIDIA GPU with 4+ GB VRAM.    │
│  The app will restart its backend process   │
│  after download. Your work is preserved.    │
└─────────────────────────────────────────────┘
```

After download:

```
┌─────────────────────────────────────────────┐
│  GPU Acceleration                           │
│                                             │
│  Status: ✓ CUDA backend active (RTX 4090)   │
│                                             │
│  [Switch to CPU]     [Delete CUDA Backend]  │
└─────────────────────────────────────────────┘
```

#### 3e. Frontend: Reconnection During Restart

The current health poll interval is 30 seconds — too slow for a restart UX. During a restart, temporarily increase polling:

```typescript
// In the component that triggers restart:
const restart = async () => {
    setRestarting(true);
    try {
        await platform.lifecycle.restartServer(true);
    } catch (e) {
        // Frontend will show "reconnecting" state
    }
    // Aggressively poll until health check succeeds
    const interval = setInterval(async () => {
        try {
            await apiClient.getHealth();
            clearInterval(interval);
            setRestarting(false);
            queryClient.invalidateQueries(); // Refresh all data
        } catch {}
    }, 1000); // Poll every 1s during restart
    // Safety timeout
    setTimeout(() => clearInterval(interval), 30000);
};
```

### Phase 4: Auto-Detection on Startup

No user action needed on subsequent launches. The preference logic in `start_server` (Phase 3b) handles this:

1. App launches → `start_server` called
2. Check `data/backends/voicebox-server-cuda{.exe}`
3. If exists → launch CUDA binary
4. If not → launch bundled CPU binary

The user downloads CUDA once, and every future app launch (including after updates) uses it automatically. The CUDA binary lives in the data directory, not the app bundle, so app updates don't overwrite it.

### Phase 5: Handling Version Mismatches

When the app updates but the CUDA binary is from an older version, the API might be incompatible. Handle this by:

1. Add `--version` flag to `server.py`:

```python
parser.add_argument("--version", action="store_true")
# If invoked with --version, print version and exit
if args.version:
    from backend import __version__
    print(f"voicebox-server {__version__}")
    sys.exit(0)
```

2. In `start_server` (Rust), before launching the CUDA binary:

```rust
// Quick version check
let version_output = std::process::Command::new(cuda_binary.to_str().unwrap())
    .arg("--version")
    .output();

match version_output {
    Ok(output) => {
        let version = String::from_utf8_lossy(&output.stdout);
        let app_version = env!("CARGO_PKG_VERSION");
        if !version.contains(app_version) {
            println!("CUDA binary version mismatch (app: {}, cuda: {}), falling back to CPU",
                app_version, version.trim());
            // Fall through to CPU sidecar launch
        }
    }
    Err(_) => {
        println!("Failed to check CUDA binary version, falling back to CPU");
    }
}
```

3. Frontend shows a notification: "Your GPU backend needs an update. [Download latest] or [Use CPU for now]"

## Files Changed

### New Files

| File | Purpose |
|------|---------|
| `backend/cuda_download.py` | Download, reassemble, verify CUDA binary |
| `scripts/split_binary.py` | Split binary into <2 GB chunks for GitHub Releases |
| `.github/workflows/build-cuda.yml` | CI: build + upload CUDA binary |

### Modified Files

| File | Change |
|------|--------|
| `tauri/src-tauri/src/main.rs` | Add `restart_server` command, modify `start_server` to check for CUDA binary in data dir |
| `backend/server.py` | Add `--version` flag |
| `backend/main.py` | Add `/backend/download-cuda`, `/backend/cuda-status`, `/backend/progress/cuda-backend` endpoints |
| `backend/build_binary.py` | Accept `--cuda` flag to build CUDA variant |
| `app/src/platform/types.ts` | Add `restartServer` to lifecycle interface |
| `tauri/src/platform/lifecycle.ts` | Implement `restartServer` |
| `app/src/components/ServerSettings/` | New GPU acceleration section |
| `.github/workflows/release.yml` | Trigger CUDA build workflow on tag |

### NOT Changed

| File | Why |
|------|-----|
| `backend/backends/__init__.py` | No changes to the TTSBackend singleton or factory. CUDA binary runs the same code. |
| `backend/backends/pytorch_backend.py` | Already detects CUDA at runtime (line 28-49). No changes needed. |
| `app/src/lib/api/client.ts` | API is identical between CPU and CUDA backends. |
| `app/src/lib/hooks/useGenerationForm.ts` | Generation flow is unchanged. |

## What This Doesn't Solve

- **Multi-model support** — This is purely about GPU acceleration. LuxTTS, Chatterbox, etc. need the in-process model registry, which is an independent workstream.
- **AMD GPU support** — DirectML/ROCm needs a different PyTorch build. Same pattern applies (another binary variant) but deferred.
- **Linux CUDA** — Same approach works, just another CI matrix entry. Can be added in the same release or shortly after.
- **Remote server mode** — Users who want to run TTS on a different machine still need the external provider architecture. Separate concern.

## What This DOES Solve

- **19 "GPU not detected" issues** — Users download the CUDA backend, restart, GPU works.
- **2 GB GitHub Release limit** — Binary splitting + R2 hosting.
- **Update burden** — App updates don't re-download the 2.4 GB CUDA binary. It persists in the data directory.
- **First-run experience** — App works immediately on CPU. GPU is an optional enhancement, not a setup blocker.

## Rollout Plan

1. Build and test CUDA binary locally on Windows with an NVIDIA GPU.
2. Set up R2 bucket at `downloads.voicebox.sh/cuda/`.
3. Ship the backend restart + download UI in v0.2.0.
4. Announce: "GPU acceleration is here — one click in Settings."

## Risks

| Risk | Mitigation |
|------|-----------|
| CUDA binary doesn't work on some GPU/driver combos | `/health` endpoint reports GPU info. Fallback to CPU if CUDA init fails. Clear error message. |
| Antivirus flags downloaded binary (Windows) | Code-sign the CUDA binary in CI. Document AV exceptions. |
| Data dir CUDA binary survives app uninstall | Document in uninstall notes. Not a real problem — it's just a file. |
| Version mismatch after app update | Version check on startup (Phase 5). Auto-fallback to CPU. Prompt to re-download. |
| R2 downtime | GitHub Releases split-binary fallback. |
| Download interrupted | Temp file with `.download` extension. Atomic rename on completion. Resume not implemented in v1 — restart download from scratch. |
