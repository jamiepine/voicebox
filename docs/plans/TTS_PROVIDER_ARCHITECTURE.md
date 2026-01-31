# TTS Provider Architecture

**Status:** Planned for v0.1.13
**Created:** 2025-01-31
**Problem:** GitHub 2GB release limit + poor UX for frequent updates requiring 2.4GB re-downloads

---

## Overview

Split the monolithic backend into modular components:

1. **Main App**:
   - Windows/Linux (~150MB): Tauri + FastAPI backend + Whisper + UI/profiles/history
   - macOS (~300MB): Same + MLX bundled for simplicity
2. **TTS Providers** (Windows/Linux only): Downloadable executables for PyTorch CPU/CUDA inference

This architecture solves:

- ✅ GitHub 2GB release artifact limit
- ✅ Frequent app updates without re-downloading large python binaries (Windows/Linux)
- ✅ User choice of compute backend (CPU/GPU/Cloud) on Windows/Linux
- ✅ Simplified out-of-the-box experience on macOS
- ✅ External provider support (OpenAI, custom servers)
- ✅ Future extensibility

---

## Architecture Diagram

### Windows / Linux
```
┌─────────────────────────────────────────────────────────┐
│  Voicebox App (Tauri + Backend)           ~150MB        │
│  ├─ UI Layer (React)                                    │
│  ├─ Backend (FastAPI)                                   │
│  │  ├─ Voice Profiles                                   │
│  │  ├─ Generation History                               │
│  │  ├─ Audio Editing / Stories                          │
│  │  └─ Provider Manager ◄──────────────┐                │
│  └─ Whisper (bundled, tiny ~50MB)      │                │
└─────────────────────────────────────────┼────────────────┘
                                          │
                            HTTP/IPC      │
                                          │
                    ┌─────────────────────┴─────────────────┐
                    │                                       │
                    ▼                                       ▼
       ┌─────────────────────┐              ┌─────────────────────┐
       │ TTS Provider:       │              │ TTS Provider:       │
       │ PyTorch CPU         │              │ PyTorch CUDA        │
       │                     │              │                     │
       │ ~300MB              │              │ ~2.4GB              │
       │                     │              │                     │
       │ Local inference     │              │ GPU inference       │
       └─────────────────────┘              └─────────────────────┘
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │  Future Providers:         │
                      │  • Remote Server           │
                      │  • OpenAI API              │
                      │  • ElevenLabs              │
                      │  • Custom Docker Container │
                      └────────────────────────────┘
```

### macOS
```
┌─────────────────────────────────────────────────────────┐
│  Voicebox App (Tauri + Backend)           ~300MB        │
│  ├─ UI Layer (React)                                    │
│  ├─ Backend (FastAPI)                                   │
│  │  ├─ Voice Profiles                                   │
│  │  ├─ Generation History                               │
│  │  ├─ Audio Editing / Stories                          │
│  │  └─ MLX Backend (bundled)                            │
│  └─ Whisper (bundled, tiny ~50MB)                       │
│                                                          │
│  No provider downloads needed - works out of the box    │
└─────────────────────────────────────────────────────────┘
```

---

## Problem Statement

### Current Architecture Issues

**Monolithic Binary:**

- CPU version: ~295MB
- CUDA version: ~2.37GB
- GitHub releases: 2GB file size limit (BLOCKED)
- Updates require re-downloading entire binary
- Poor UX: update app → restart → download CUDA update → restart again

**User Pain Points:**

1. Cannot release CUDA version on GitHub (over 2GB)
2. Every app update forces 2.4GB re-download for GPU users
3. No flexibility (can't use OpenAI, remote servers, etc.)
4. Wastes bandwidth for small bug fixes

---

## Solution: Pluggable TTS Providers

### Component Breakdown

#### 1. Main App (voicebox.exe / .app / .AppImage)

**Windows/Linux Size:** ~100-150MB
**macOS Size:** ~300-350MB (includes MLX)

**Includes:**

- Tauri runtime + React UI
- FastAPI backend (pure Python, no PyTorch on Windows/Linux)
- Whisper model (tiny, ~50MB)
- SQLite database
- Profile/history/audio editing logic
- Provider management system (Windows/Linux only)
- **MLX backend (macOS only, bundled)**

**Does NOT include (Windows/Linux only):**

- PyTorch (CPU or CUDA)
- TTS models (Qwen3-TTS)
- Heavy ML dependencies

**Updates frequently:** UI fixes, feature additions, non-ML changes

---

#### 2. TTS Provider: PyTorch CPU

**Binary:** `tts-provider-pytorch-cpu.exe`
**Size:** ~200MB

**Includes:**

- PyTorch CPU build
- Qwen3-TTS package
- Transformers
- No CUDA libraries

**Download source:** Cloudflare R2
**Updates rarely:** Only when model code changes

---

#### 3. TTS Provider: PyTorch CUDA

**Binary:** `tts-provider-pytorch-cuda.exe`
**Size:** ~2.4GB

**Includes:**

- PyTorch CUDA build (cu121)
- Qwen3-TTS package
- CUDA runtime, cuDNN, cuBLAS
- Transformers

**Download source:** Cloudflare R2
**Platform:** Windows + Linux (NVIDIA GPU)
**Updates rarely:** Only when model code or CUDA version changes

---

#### 4. TTS Provider: Remote

**Binary:** None (built-in config)
**Size:** 0MB

**How it works:**

- User provides URL to their own TTS server
- Backend proxies requests to that server
- Implements API spec from `EXTERNAL_PROVIDERS.md`

**Use cases:**

- AMD GPU users running their own server
- Team deployments with shared GPU server
- Cloud hosting (Modal, RunPod, Replicate)

---

#### 5. TTS Provider: OpenAI

**Binary:** None (API wrapper)
**Size:** 0MB

**How it works:**

- User provides OpenAI API key
- Backend wraps OpenAI Audio API
- Voice profiles map to OpenAI voices

**Benefits:**

- Zero local compute
- Pay-per-use
- Instant setup

---

## Communication Protocol

### Provider API Specification

All TTS providers must implement these endpoints:

#### POST /tts/generate

Generate speech from text.

**Request:**

```json
{
	"text": "Hello world!",
	"voice_prompt": {
		/* voice prompt object */
	},
	"language": "en",
	"seed": 12345,
	"model_size": "1.7B"
}
```

**Response:**

```json
{
	"audio": "base64-encoded-audio",
	"sample_rate": 24000,
	"duration": 2.5
}
```

#### POST /tts/create_voice_prompt

Create voice prompt from reference audio.

**Request:** (multipart/form-data)

- `audio`: Audio file
- `reference_text`: Transcript

**Response:**

```json
{
	"voice_prompt": {
		/* serialized prompt */
	}
}
```

#### GET /tts/health

Health check.

**Response:**

```json
{
	"status": "healthy",
	"provider": "pytorch-cuda",
	"version": "1.0.0",
	"model": "Qwen3-TTS-12Hz-1.7B-Base",
	"device": "cuda:0"
}
```

#### GET /tts/status

Model status.

**Response:**

```json
{
	"model_loaded": true,
	"model_size": "1.7B",
	"available_sizes": ["0.6B", "1.7B"],
	"gpu_available": true,
	"vram_used_mb": 1234
}
```

---

## Backend Implementation

### Provider Manager

**File:** `backend/providers/__init__.py`

```python
class ProviderManager:
    """Manages TTS provider lifecycle (Windows/Linux only).

    Note: macOS uses bundled MLX backend directly, no provider management needed.
    """

    def __init__(self):
        self.active_provider: Optional[Provider] = None
        self.config = load_provider_config()

    async def start_provider(self, provider_type: str) -> str:
        """Start a TTS provider process."""
        if provider_type == "pytorch-cpu":
            return await self._start_local_provider("tts-provider-pytorch-cpu.exe")
        elif provider_type == "pytorch-cuda":
            return await self._start_local_provider("tts-provider-pytorch-cuda.exe")
        elif provider_type == "remote":
            return self.config["remote_url"]
        elif provider_type == "openai":
            return None  # No subprocess, API wrapper

    async def _start_local_provider(self, binary_name: str) -> str:
        """Start local provider subprocess."""
        provider_path = get_provider_binary_path(binary_name)

        if not provider_path.exists():
            raise ProviderNotInstalledException(binary_name)

        # Start subprocess on random port
        port = get_free_port()
        process = subprocess.Popen([
            str(provider_path),
            "--port", str(port),
            "--data-dir", str(config.get_data_dir())
        ])

        # Wait for provider to be ready
        await wait_for_provider_health(f"http://localhost:{port}")

        self.active_provider = Provider(process, port)
        return f"http://localhost:{port}"

    async def stop_provider(self):
        """Stop active provider."""
        if self.active_provider:
            self.active_provider.process.terminate()
            self.active_provider = None
```

---

### Provider Abstraction

**File:** `backend/providers/base.py`

```python
class TTSProvider(ABC):
    """Abstract base for TTS providers."""

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str,
        seed: Optional[int]
    ) -> tuple[np.ndarray, int]:
        """Generate speech audio."""
        pass

    @abstractmethod
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str
    ) -> dict:
        """Create voice prompt from reference audio."""
        pass
```

**File:** `backend/providers/local.py`

```python
class LocalProvider(TTSProvider):
    """Provider that communicates with local subprocess via HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def generate(self, text, voice_prompt, language, seed):
        response = await self.client.post(
            f"{self.base_url}/tts/generate",
            json={
                "text": text,
                "voice_prompt": voice_prompt,
                "language": language,
                "seed": seed
            }
        )
        data = response.json()
        audio = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
        return audio, data["sample_rate"]
```

**File:** `backend/providers/openai.py`

```python
class OpenAIProvider(TTSProvider):
    """Provider that wraps OpenAI Audio API."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def generate(self, text, voice_prompt, language, seed):
        # Map voice_prompt to OpenAI voice name
        voice = map_profile_to_openai_voice(voice_prompt)

        response = await self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Convert to numpy array
        audio_data = response.content
        audio, sr = load_audio_from_bytes(audio_data)
        return audio, sr
```

---

## Provider Installation

### Download Manager

**File:** `backend/providers/installer.py`

```python
class ProviderInstaller:
    """Handles provider download and installation (Windows/Linux only)."""

    async def download_provider(self, provider_type: str):
        """Download provider binary from R2."""

        binary_name = {
            "pytorch-cpu": "tts-provider-pytorch-cpu.exe",
            "pytorch-cuda": "tts-provider-pytorch-cuda.exe"
        }[provider_type]

        download_url = f"https://downloads.voicebox.sh/providers/v{PROVIDER_VERSION}/{binary_name}"

        # Download with progress tracking (reuse existing SSE system)
        await download_with_progress(
            url=download_url,
            destination=get_provider_install_path(binary_name),
            progress_key=f"provider-{provider_type}"
        )
```

**Provider Storage Location:**

- Windows: `%APPDATA%/voicebox/providers/`
- macOS: `~/Library/Application Support/voicebox/providers/`
- Linux: `~/.local/share/voicebox/providers/`

---

## Frontend Implementation

### Provider Settings UI

**Component:** `app/src/components/ServerSettings/ProviderSettings.tsx`

```tsx
export function ProviderSettings() {
	const [selectedProvider, setSelectedProvider] =
		useState<ProviderType>("auto");
	const {data: installedProviders} = useQuery({
		queryKey: ["providers", "installed"],
		queryFn: () => apiClient.getInstalledProviders(),
	});

	return (
		<Card>
			<CardHeader>
				<CardTitle>TTS Provider</CardTitle>
				<CardDescription>Choose how Voicebox generates speech</CardDescription>
			</CardHeader>
			<CardContent>
				<RadioGroup
					value={selectedProvider}
					onValueChange={setSelectedProvider}
				>
					{/* Auto-detect */}
					<div className="flex items-center space-x-2">
						<RadioGroupItem value="auto" id="auto" />
						<Label htmlFor="auto">
							<div className="font-medium">Auto-detect (Recommended)</div>
							<div className="text-sm text-muted-foreground">
								Automatically choose the best available provider
							</div>
						</Label>
					</div>

					{/* PyTorch CUDA */}
					<div className="flex items-center justify-between">
						<div className="flex items-center space-x-2">
							<RadioGroupItem
								value="pytorch-cuda"
								id="cuda"
								disabled={!gpuAvailable}
							/>
							<Label htmlFor="cuda">
								<div className="font-medium">PyTorch CUDA (NVIDIA GPU)</div>
								<div className="text-sm text-muted-foreground">
									4-5x faster inference on NVIDIA GPUs
								</div>
							</Label>
						</div>
						{!installedProviders?.includes("pytorch-cuda") && gpuAvailable && (
							<Button
								onClick={() => downloadProvider("pytorch-cuda")}
								size="sm"
							>
								Download (2.4GB)
							</Button>
						)}
					</div>

					{/* PyTorch CPU (Windows/Linux only) */}
					{!isMacOS && (
						<div className="flex items-center justify-between">
							<div className="flex items-center space-x-2">
								<RadioGroupItem value="pytorch-cpu" id="cpu" />
								<Label htmlFor="cpu">
									<div className="font-medium">PyTorch CPU</div>
									<div className="text-sm text-muted-foreground">
										Works on any system, slower inference
									</div>
								</Label>
							</div>
							{!installedProviders?.includes("pytorch-cpu") && (
								<Button onClick={() => downloadProvider("pytorch-cpu")} size="sm">
									Download (300MB)
								</Button>
							)}
						</div>
					)}

					{/* MLX bundled (macOS only) */}
					{isMacOS && (
						<div className="p-3 bg-muted rounded-md">
							<div className="text-sm">
								<div className="font-medium">MLX (Apple Silicon)</div>
								<div className="text-muted-foreground mt-1">
									Bundled with the app - optimized for M1/M2/M3 chips
								</div>
							</div>
						</div>
					)}

					{/* Remote */}
					<div className="space-y-2">
						<div className="flex items-center space-x-2">
							<RadioGroupItem value="remote" id="remote" />
							<Label htmlFor="remote">
								<div className="font-medium">Remote Server</div>
								<div className="text-sm text-muted-foreground">
									Connect to your own TTS server
								</div>
							</Label>
						</div>
						{selectedProvider === "remote" && (
							<Input placeholder="http://your-server:8000" className="ml-6" />
						)}
					</div>

					{/* OpenAI */}
					<div className="space-y-2">
						<div className="flex items-center space-x-2">
							<RadioGroupItem value="openai" id="openai" />
							<Label htmlFor="openai">
								<div className="font-medium">OpenAI API</div>
								<div className="text-sm text-muted-foreground">
									Use OpenAI's TTS API (requires API key)
								</div>
							</Label>
						</div>
						{selectedProvider === "openai" && (
							<Input type="password" placeholder="sk-..." className="ml-6" />
						)}
					</div>
				</RadioGroup>
			</CardContent>
		</Card>
	);
}
```

---

## File Structure

```
voicebox/
├── backend/
│   ├── main.py                    # Main FastAPI app (no TTS on Win/Linux)
│   ├── backends/
│   │   ├── __init__.py            # Backend abstraction (existing)
│   │   ├── pytorch_backend.py     # PyTorch backend (existing, for reference)
│   │   └── mlx_backend.py         # MLX backend (bundled in macOS build only)
│   ├── providers/
│   │   ├── __init__.py            # ProviderManager (Windows/Linux)
│   │   ├── base.py                # TTSProvider Protocol
│   │   ├── local.py               # LocalProvider (subprocess)
│   │   ├── remote.py              # RemoteProvider (HTTP)
│   │   ├── openai.py              # OpenAIProvider (API wrapper)
│   │   └── installer.py           # Provider download logic (Windows/Linux)
│   ├── profiles.py                # Voice profile management
│   ├── history.py                 # Generation history
│   ├── transcribe.py              # Whisper (still bundled)
│   └── ... (other backend modules)
│
├── providers/
│   ├── pytorch-cpu/
│   │   ├── main.py                # FastAPI server for TTS
│   │   ├── tts_backend.py         # PyTorch TTS logic
│   │   ├── requirements.txt       # torch (CPU), qwen-tts, transformers
│   │   └── build.spec             # PyInstaller spec
│   │
│   └── pytorch-cuda/
│       ├── main.py                # FastAPI server for TTS
│       ├── tts_backend.py         # PyTorch TTS logic
│       ├── requirements.txt       # torch+cu121, qwen-tts, transformers
│       └── build.spec             # PyInstaller spec
│
├── app/                           # Frontend (Tauri + React)
│   └── src/
│       └── components/
│           └── ServerSettings/
│               └── ProviderSettings.tsx  # Only shown on Windows/Linux
│
└── tauri/
    └── src-tauri/
        └── tauri.conf.json        # No externalBin for providers (Windows/Linux)
                                   # MLX bundled in macOS build
```

---

## Migration Path

### Phase 1: Refactor Backend (No User Changes)

**Goal:** Abstract TTS behind provider interface

1. Create `backend/providers/` module structure
2. Implement `TTSProvider` abstract base class
3. Create `LocalProvider` wrapper for current PyTorch code
4. Modify `backend/tts.py` to use provider abstraction
5. Keep PyTorch bundled in main app

**Result:** Code is prepared, but user experience unchanged

---

### Phase 2: Build Provider Binaries

**Goal:** Create standalone TTS provider executables (Windows/Linux only)

1. Create separate PyInstaller specs for each provider
2. Build provider executables:
   - `tts-provider-pytorch-cpu.exe` (~300MB)
   - `tts-provider-pytorch-cuda.exe` (~2.4GB)
3. Test subprocess communication
4. Upload providers to Cloudflare R2

**Result:** Provider binaries exist but aren't used yet

**Note:** macOS keeps MLX bundled in main app - no separate provider needed

---

### Phase 3: Remove PyTorch from Main App

**Goal:** Split main app from providers (Windows/Linux only)

1. Exclude PyTorch/Qwen3-TTS from Windows/Linux main app PyInstaller spec
2. Windows/Linux app now requires provider download
3. Update GitHub CI to build multiple artifacts:
   - `voicebox-{version}-windows.exe` (~150MB, no TTS)
   - `voicebox-{version}-linux.AppImage` (~150MB, no TTS)
   - `voicebox-{version}-macos.app` (~300MB, MLX bundled)
   - `tts-provider-pytorch-cpu-{version}.exe`
   - `tts-provider-pytorch-cuda-{version}.exe`

**Result:** Windows/Linux apps are small with downloadable providers, macOS app is self-contained

---

### Phase 4: Add Provider UI

**Goal:** User-facing provider management

1. Create Provider Settings page
2. Implement provider download UI
3. Add provider status indicators
4. Show active provider in UI

**Result:** Users can choose and download providers

---

### Phase 5: External Providers

**Goal:** Enable remote and cloud providers

1. Implement `RemoteProvider` (HTTP client)
2. Implement `OpenAIProvider` (API wrapper)
3. Add provider configuration UI (URLs, API keys)
4. Document external provider API spec

**Result:** Full provider ecosystem

---

## Provider Versioning

### Independent Versioning

Providers have their own version numbers, independent of the main app:

- **App version:** `v0.2.0` (frequent updates)
- **Provider version:** `v1.0.0` (rare updates)

### Compatibility Matrix

**Example:**

| App Version | Min Provider Version | Max Provider Version |
| ----------- | -------------------- | -------------------- |
| v0.2.0      | v1.0.0               | v1.x.x               |
| v0.3.0      | v1.0.0               | v1.x.x               |
| v0.4.0      | v1.2.0               | v1.x.x               |
| v1.0.0      | v2.0.0               | v2.x.x               |

**Backend checks compatibility:**

```python
async def check_provider_compatibility(provider_version: str) -> bool:
    """Check if provider version is compatible with current app."""
    min_version = "1.0.0"
    max_version = "1.999.999"
    return min_version <= provider_version < max_version
```

**UI shows warning if incompatible:**

```
⚠️ Provider version 0.9.0 is outdated. Update to v1.0.0+
```

---

## User Flows

### First-Time Setup (Windows/Linux)

1. User downloads and installs Voicebox (~150MB)
2. App launches → detects no TTS provider installed
3. Shows setup wizard:

   ```
   Choose your TTS provider:

   [ ] PyTorch CUDA (2.4GB)    [Download]
       ✓ Fastest on NVIDIA GPUs
       ✗ Requires NVIDIA GPU

   [●] PyTorch CPU (300MB)     [Download]
       ✓ Works on any system
       ✗ Slower inference

   [ ] Remote Server
       URL: ___________________

   [ ] OpenAI API
       API Key: ________________
   ```

4. User selects provider → downloads with progress bar
5. Provider installs to AppData/Application Support
6. App starts provider → ready to use

### First-Time Setup (macOS)

1. User downloads and installs Voicebox (~300MB with MLX bundled)
2. App launches → MLX backend is ready immediately
3. No provider setup needed - works out of the box

---

### App Update Flow (No Provider Change)

**Scenario:** Bug fix in UI, no backend changes

**Windows/Linux:**
1. User gets update notification: "Voicebox v0.2.1 available"
2. Downloads update (~150MB, not 2.4GB!)
3. Installs and restarts
4. **Provider stays the same** (no re-download needed)
5. App starts using existing provider

**macOS:**
1. User gets update notification: "Voicebox v0.2.1 available"
2. Downloads update (~300MB with MLX bundled)
3. Installs and restarts - ready to use

**User experience:** Fast updates, no multi-GB downloads (especially for CUDA users)

---

### Provider Update Flow

**Scenario:** New Qwen3-TTS model version released

1. User opens Settings → Provider tab
2. Sees notification: "Provider update available (v1.1.0)"
3. Clicks "Update Provider"
4. Downloads new provider binary
5. Old provider binary is replaced
6. Restart app to use new provider

**Frequency:** Rare (only when TTS model/backend changes)

---

### Switching Providers

**Scenario:** User upgrades to NVIDIA GPU

1. User goes to Settings → Provider
2. Selects "PyTorch CUDA"
3. Clicks "Download" → downloads 2.4GB
4. Download completes → restarts app
5. App now uses CUDA provider

---

## Benefits

| Benefit                       | Details                                                   |
| ----------------------------- | --------------------------------------------------------- |
| **GitHub Releases Work**      | Main app ~150MB (Win/Linux), ~300MB (macOS) << 2GB limit  |
| **Fast Updates**              | UI/feature updates don't require re-downloading providers |
| **User Choice**               | CPU, CUDA, OpenAI, remote server (Win/Linux)              |
| **macOS Simplicity**          | MLX bundled - works immediately, no provider setup needed |
| **External Provider Support** | Users can run their own TTS servers                       |
| **Bandwidth Savings**         | Only download provider once, app updates are small        |
| **Future-Proof**              | Easy to add new providers (ElevenLabs, custom models)     |
| **Team Deployments**          | Multiple users share one remote provider                  |
| **Cloud-Ready**               | Works with Modal, Replicate, RunPod, etc.                 |

---

## Open Questions

### 1. Provider Versioning

**Question:** Should providers have independent versions or match app version?

**Options:**

- A. Independent (providers: v1.x, app: v0.2.x)
- B. Matched (both use v0.2.x)

**Recommendation:** Independent versioning with compatibility matrix

---

### 2. Auto-Update Providers

**Question:** Should providers auto-update separately from app?

**Options:**

- A. Manual updates only (user clicks "Update Provider")
- B. Optional auto-update (user can enable)
- C. Always auto-update

**Recommendation:** Optional auto-update (default off)

---

### 3. Provider Discovery

**Question:** How does app find installed providers?

**Options:**

- A. Check standard paths in AppData/Application Support
- B. Registry (Windows) / plist (macOS)
- C. Config file with provider locations

**Recommendation:** Standard paths + config fallback

---

### 4. Fallback Behavior

**Question:** What if no provider is installed?

**Options:**

- A. Show setup wizard on first launch
- B. Block app until provider installed
- C. Allow app to run in "demo mode" (transcription only)

**Recommendation:** Setup wizard on first launch

---

### 5. Provider Auto-Start

**Question:** Should provider start automatically with app?

**Options:**

- A. Always start selected provider on app launch
- B. Start on-demand (when user generates speech)
- C. User preference

**Recommendation:** Auto-start (configurable in settings)

---

## Future Enhancements

- [ ] **Provider Marketplace:** Built-in directory of community providers
- [ ] **Multi-Provider Support:** Use different providers per voice/language
- [ ] **Provider Health Monitoring:** Automatic failover if provider crashes
- [ ] **Cost Tracking:** Monitor API usage for OpenAI/cloud providers
- [ ] **Performance Metrics:** Latency, throughput, VRAM usage dashboards
- [ ] **Docker Providers:** Run providers in Docker containers
- [ ] **Provider Plugins:** Load custom providers from user scripts

---

## Related Documents

- [EXTERNAL_PROVIDERS.md](./EXTERNAL_PROVIDERS.md) - External provider support plan
- [OPENAI_SUPPORT.md](./OPENAI_SUPPORT.md) - OpenAI API compatibility
- [github-2gb-limit-issue.md](../github-2gb-limit-issue.md) - Original problem
- [r2-setup.md](../r2-setup.md) - Cloudflare R2 configuration

---

## Contributing

If you want to build a custom TTS provider:

1. Implement the provider API spec (see above)
2. Test with Voicebox locally
3. Package as executable (PyInstaller, Docker, etc.)
4. Share in GitHub Discussions

**Questions?**

- GitHub Issues: [voicebox/issues](https://github.com/jamiepine/voicebox/issues)
- Discord: Coming soon
