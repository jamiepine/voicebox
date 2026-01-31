# TTS Provider Architecture

This document explains how Voicebox's modular TTS provider system works.

## Overview

Voicebox uses a **pluggable provider architecture** that separates the main application from TTS inference. This solves several problems:

- **GitHub's 2GB release limit** - CUDA builds are ~2.4GB, too large for GitHub releases
- **Faster app updates** - UI/feature updates don't require re-downloading heavy ML binaries
- **User choice** - Users can pick CPU, CUDA, or external providers based on their hardware

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Voicebox App                                               │
│  ├─ UI (React)                                              │
│  ├─ Backend (FastAPI)                                       │
│  │   ├─ Voice Profiles                                      │
│  │   ├─ Generation History                                  │
│  │   ├─ Whisper STT (bundled)                               │
│  │   └─ Provider Manager ◄────────────────┐                 │
│  │                                         │                │
│  └─ providers/                             │                │
│      ├─ bundled.py (wraps backends/)       │                │
│      └─ local.py (HTTP client)─────────────┼───┐            │
│                                            │   │            │
└────────────────────────────────────────────┼───┼────────────┘
                                             │   │
                        ┌────────────────────┘   │
                        │                        │ HTTP
                        ▼                        ▼
              ┌──────────────────┐    ┌──────────────────────┐
              │ backends/        │    │ Standalone Provider  │
              │ (bundled on Mac) │    │ (subprocess)         │
              │                  │    │                      │
              │ - mlx_backend    │    │ - FastAPI server     │
              │ - pytorch_backend│    │ - PyTorch + Qwen-TTS │
              └──────────────────┘    │ - Runs on localhost  │
                                      └──────────────────────┘
```

## Platform Behavior

| Platform | App Size | TTS Backend | Provider Download |
|----------|----------|-------------|-------------------|
| macOS (Apple Silicon) | ~300MB | MLX bundled | Not needed |
| macOS (Intel) | ~300MB | PyTorch bundled | Not needed |
| Windows | ~150MB | None bundled | Required |
| Linux | ~150MB | None bundled | Required |

### macOS (Apple Silicon)
- MLX backend is **bundled** in the app
- Works immediately after install
- Uses Metal for GPU acceleration

### macOS (Intel)
- PyTorch backend is **bundled** in the app
- Works immediately after install
- Uses CPU inference

### Windows / Linux
- **No TTS bundled** - keeps app small (~150MB)
- On first use, prompts to download a provider
- Provider options:
  - **PyTorch CPU** (~300MB) - Works on any system
  - **PyTorch CUDA** (~2.4GB) - Fast inference on NVIDIA GPUs

## Directory Structure

```
voicebox/
├── backend/
│   ├── backends/                 # Actual TTS implementations
│   │   ├── __init__.py          # TTSBackend Protocol
│   │   ├── mlx_backend.py       # MLX implementation (macOS)
│   │   └── pytorch_backend.py   # PyTorch implementation
│   │
│   └── providers/               # Provider abstraction layer
│       ├── __init__.py          # ProviderManager
│       ├── base.py              # TTSProvider Protocol
│       ├── bundled.py           # Wraps backends/ for bundled use
│       ├── local.py             # HTTP client for subprocess providers
│       ├── installer.py         # Downloads providers from R2
│       └── types.py             # Shared types
│
└── providers/                   # Standalone provider builds
    ├── pytorch-cpu/
    │   ├── main.py              # FastAPI server
    │   ├── build.py             # PyInstaller build script
    │   └── requirements.txt
    │
    └── pytorch-cuda/
        ├── main.py              # FastAPI server
        │   build.py              # PyInstaller build script
        └── requirements.txt
```

## How Providers Work

### 1. BundledProvider (macOS)

On macOS, the `BundledProvider` directly calls the bundled `backends/` code:

```python
# backend/providers/bundled.py
class BundledProvider:
    def __init__(self):
        self._backend = get_tts_backend()  # MLX or PyTorch
    
    async def generate(self, text, voice_prompt, ...):
        return await self._backend.generate(text, voice_prompt, ...)
```

### 2. LocalProvider (Windows/Linux)

On Windows/Linux, the `LocalProvider` communicates with a standalone provider via HTTP:

```python
# backend/providers/local.py
class LocalProvider:
    def __init__(self, base_url: str):
        self.base_url = base_url  # e.g., "http://127.0.0.1:8765"
    
    async def generate(self, text, voice_prompt, ...):
        response = await self.client.post(
            f"{self.base_url}/tts/generate",
            json={"text": text, "voice_prompt": voice_prompt, ...}
        )
        # Decode audio from response
        return audio, sample_rate
```

### 3. Standalone Provider Server

The standalone providers are self-contained FastAPI servers:

```python
# providers/pytorch-cpu/main.py
@app.post("/tts/generate")
async def generate(text: str, voice_prompt: dict, ...):
    audio, sr = await backend.generate(text, voice_prompt, ...)
    return {"audio": base64_encode(audio), "sample_rate": sr}
```

## Provider API Specification

All providers (local or remote) must implement these HTTP endpoints:

### POST /tts/generate
Generate speech from text.

**Request:**
```json
{
    "text": "Hello world!",
    "voice_prompt": { /* voice embedding */ },
    "language": "en",
    "seed": 12345,
    "model_size": "1.7B"
}
```

**Response:**
```json
{
    "audio": "base64-encoded-wav",
    "sample_rate": 24000,
    "duration": 2.5
}
```

### POST /tts/create_voice_prompt
Create voice embedding from reference audio.

**Request:** `multipart/form-data`
- `audio`: Audio file
- `reference_text`: Transcript

**Response:**
```json
{
    "voice_prompt": { /* voice embedding */ },
    "was_cached": false
}
```

### GET /tts/health
Health check.

**Response:**
```json
{
    "status": "healthy",
    "provider": "pytorch-cuda",
    "version": "1.0.0",
    "model": "1.7B",
    "device": "cuda:0"
}
```

### GET /tts/status
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

## Provider Lifecycle

### Startup Flow (Windows/Linux)

```
1. App launches
2. ProviderManager checks for installed providers
3. If none installed:
   └─ Show setup wizard, prompt download
4. If installed:
   ├─ Start provider subprocess on random port
   ├─ Wait for /tts/health to return 200
   └─ Create LocalProvider with that URL
5. Generation requests go through LocalProvider → subprocess
```

### Download Flow

```
1. User clicks "Download PyTorch CUDA"
2. Installer downloads from Cloudflare R2:
   https://downloads.voicebox.sh/providers/v1.0.0/tts-provider-pytorch-cuda-windows.exe
3. Saved to:
   - Windows: %APPDATA%/voicebox/providers/
   - Linux: ~/.local/share/voicebox/providers/
4. Provider is now available to start
```

## Building Providers

### Prerequisites
- Python 3.12
- PyInstaller

### Build PyTorch CPU Provider
```bash
cd providers/pytorch-cpu
pip install -r requirements.txt
python build.py
# Output: dist/tts-provider-pytorch-cpu.exe
```

### Build PyTorch CUDA Provider
```bash
cd providers/pytorch-cuda
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python build.py
# Output: dist/tts-provider-pytorch-cuda.exe (~2.4GB)
```

## Provider Versioning

Providers have **independent versions** from the app:

- **App version:** `v0.2.0` (frequent updates)
- **Provider version:** `v1.0.0` (rare updates)

Providers only need updates when:
- TTS model changes (new Qwen3-TTS version)
- API spec changes
- Bug fixes in inference code

The app checks provider compatibility on startup.

## Future Providers

The architecture supports additional providers:

- **Remote Server** - Connect to your own TTS server
- **OpenAI API** - Use OpenAI's TTS (requires API key)
- **ElevenLabs** - Cloud TTS service
- **Docker** - Run providers in containers

These would implement the same HTTP API spec.
