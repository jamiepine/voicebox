# Voicebox Developer Guide

Local development environment for [Voicebox](https://github.com/jamiepine/voicebox) on this machine.

## Environment Summary

| Item | Value |
|------|--------|
| **Machine** | ASUS TUF A15 (Windows) |
| **RAM** | 16 GB |
| **GPU** | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM) |
| **Repo path** | `F:\VoiceBox\voicebox-src` |
| **Python venv** | `backend\venv` |
| **Backend URL** | `http://127.0.0.1:17493` |
| **PyTorch** | CUDA 12.8 (`cu128`) — Ampere-compatible |
| **Package manager** | Bun (JS), pip (Python), Cargo (Rust/Tauri) |

### What `just setup` installs

- Python virtual environment and FastAPI backend dependencies
- CUDA-enabled PyTorch when an NVIDIA GPU is detected
- TTS engines (Qwen3-TTS, Kokoro, Chatterbox, LuxTTS, TADA, etc.)
- Bun workspaces for `app/`, `tauri/`, and `web/`
- Tauri dev sidecar placeholders (`bun run setup:dev`)

### Useful commands

```powershell
just --list          # all recipes
just dev-backend     # API only
just dev-frontend    # Tauri only (backend must be running)
just kill            # stop dev processes
just test-models --only kokoro   # smoke-test Kokoro engine
```

---

## How to Start

Open PowerShell, then from the repo root:

```powershell
cd F:\VoiceBox\voicebox-src
just dev
```

This starts the Python backend on port **17493** and launches the Tauri desktop app.

**Two-terminal alternative:**

```powershell
# Terminal 1 — backend
cd F:\VoiceBox\voicebox-src
.\backend\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --port 17493

# Terminal 2 — desktop UI
cd F:\VoiceBox\voicebox-src\tauri
bun run tauri dev
```

**Verify the backend:**

```powershell
curl http://127.0.0.1:17493/health
```

API docs: http://127.0.0.1:17493/docs

### Optional VRAM helper (recommended before `just dev`)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

---

## VRAM Survival Guide (RTX 3060 — 6 GB)

Six gigabytes is enough for daily dev work if you pick the right engines. The app **defaults to Qwen3-TTS 1.7B**, which needs ~6 GB VRAM and will OOM on this GPU.

### Engine cheat sheet

| Engine | VRAM | Safe on 6 GB? |
|--------|------|----------------|
| **Kokoro 82M** | ~150 MB | Yes — **use this as your default** |
| LuxTTS | ~1 GB | Yes |
| Chatterbox Turbo | ~1.5 GB | Yes |
| Qwen3-TTS **0.6B** | ~2 GB | Yes (for cloning) |
| Chatterbox Multilingual | ~3 GB | Usually OK (unload others first) |
| Qwen3-TTS **1.7B** | ~6 GB | **No — will OOM** |
| TADA 1B / 3B | ~4–8 GB | Avoid |

### Do this immediately after launch

1. **Switch the TTS engine to Kokoro**
   - In the generate box, open the **engine dropdown**
   - Select **Kokoro 82M** (not Qwen3-TTS 1.7B)
   - For preset voices: **Profiles → + New Profile → Kokoro** → pick a voice

2. **Never load Qwen 1.7B for TTS on this GPU**
   - If you need Qwen cloning, choose **Qwen3-TTS 0.6B** in the engine/model selector
   - Do not select 1.7B or Qwen CustomVoice 1.7B

3. **Unload unused models when switching engines**
   - **Settings → Models**
   - Click **Unload** on any engine you are not actively using
   - Only one generation runs at a time, but **loaded models stay in VRAM** until unloaded

4. **Keep Whisper small for dictation**
   - Use **Whisper Base** or **Small** in transcription settings
   - Unload TTS models before heavy dictation sessions

5. **Close other GPU apps** (games, browsers with hardware acceleration) before generating

### If you still hit OOM

- Unload all models in Settings → Models, then restart the app
- Use Kokoro or LuxTTS only
- Lower **Settings → Generation → chunk limit** for long scripts
- Check VRAM: `nvidia-smi` in a separate terminal

---

## Notes from initial setup

- **`just setup` on this machine:** The first run failed building `pyopenjtalk` (needs MSVC `nmake`). Setup was completed with CUDA PyTorch (`torch 2.11.0+cu128`) and `misaki[en]` (English Kokoro — no Japanese `pyopenjtalk` build required).
- **`just dev` (Tauri desktop):** Requires **Visual Studio C++ Build Tools** with `link.exe` on PATH. If `just dev` fails with `linker link.exe not found`, open **"Developer PowerShell for VS 2022"** or install [Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**, then retry.
- **Until Tauri builds:** Use `just dev-web` for the browser UI (backend + Vite on port 5173).
- **GPU detection in automated shells:** If `just setup` skips CUDA PyTorch, install manually:
  ```powershell
  .\backend\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
- **`just` binary:** Also available at `F:\VoiceBox\tools\just.exe` if winget PATH is stale.
