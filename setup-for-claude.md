# Voicebox Fixed Backend — Setup Instructions for Claude Code

You are helping a user set up a fixed version of the Voicebox AI voice studio on their Mac (Apple Silicon). The fix resolves a transcription bug that causes voice cloning to fail with "Error -3 while decompressing data: incorrect header check". Work through the steps below in order. Check each one before proceeding to the next.

---

## What You're Setting Up

- A local Python backend (FastAPI) cloned from a patched fork of Voicebox
- The user's installed Voicebox.app will connect to this backend instead of its own bundled server
- The fix is already in the code — you just need to get the environment running

---

## Step 0 — Install Official Voicebox App

The fixed backend plugs into the official Voicebox desktop app — the app itself is not built from source here. The user must have it installed first.

Ask the user: **"Do you have Voicebox.app installed in your Applications folder?"**

If not, have them:
1. Download the DMG from **https://voicebox.sh/download/mac-arm** (Apple Silicon) or **https://voicebox.sh/download/mac-intel** (Intel Mac)
2. Open the DMG and drag Voicebox to Applications
3. Launch it once and let it finish its first-run setup
4. Close it again before continuing — we need to start our backend before the app opens

---

## Step 1 — Check Prerequisites

Run the following checks and install anything missing:

**Homebrew:**
```bash
which brew
```
If not found, install it:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Python 3.12:**
```bash
python3.12 --version
```
If not found:
```bash
brew install python@3.12
```

**just (task runner):**
```bash
just --version
```
If not found:
```bash
brew install just
```

**Git:**
```bash
git --version
```
Git is pre-installed on macOS. If missing, it will prompt to install Xcode Command Line Tools — follow that prompt.

---

## Step 2 — Clone the Repo

Clone the patched fork directly onto the fix branch into the home directory:
```bash
git clone --branch fix/transcription-audio-format https://github.com/kashmo/voicebox.git ~/voicebox
```

Then move into the project folder — all remaining commands run from here:
```bash
cd ~/voicebox
```

---

## Step 3 — Install Python Dependencies

This downloads and installs all ML libraries (torch, MLX, transformers, librosa, etc.). It takes 20-40 minutes depending on internet speed. Run it and wait for it to finish:
```bash
just setup
```

When it finishes it will print: `Python environment ready.`

If it errors, check the error message carefully:
- If it says Python version not found, make sure Step 1 completed successfully
- If a package fails to install, report the exact error message to the user

---

## Step 4 — Install Two Additional Packages

These are intentionally excluded from the main setup to avoid a dependency conflict, so they must be installed separately:
```bash
backend/venv/bin/pip install --no-deps mlx-lm==0.31.1 mlx-audio==0.4.1
```

Confirm both installed successfully before continuing.

---

## Step 5 — Verify the Setup

Do a quick sanity check that the key packages are importable:
```bash
backend/venv/bin/python -c "import mlx_audio; import librosa; import soundfile; print('OK')"
```

It should print `OK`. If it errors, report what's missing.

---

## Step 6 — Start the Backend

Start the backend server. It must be running BEFORE the user opens Voicebox.app:
```bash
backend/venv/bin/uvicorn backend.main:app --port 17493 &
```

Wait 4 seconds, then confirm it's responding:
```bash
sleep 4 && curl -sf http://127.0.0.1:17493/health && echo "Backend is up"
```

If `Backend is up` prints, the server is ready.

If it fails, check the startup logs:
```bash
# The backend was started in the background — check for errors with:
ps aux | grep uvicorn
```

---

## Step 7 — Tell the User to Open Voicebox

Tell the user:

> The backend is running. Now open Voicebox from your Applications folder. Once it loads, try recording audio and clicking the transcribe button — it should work without the "Error -3" message.

After they open the app, confirm it connected by checking for HTTP requests in the process:
```bash
curl -sf http://127.0.0.1:17493/health && echo "Still up — app should be connected"
```

---

## How to Start It Next Time

Setup is one-time only. Every subsequent time the user wants to use Voicebox with this fix:

1. Start the backend first:
```bash
cd ~/voicebox && backend/venv/bin/uvicorn backend.main:app --port 17493 &
```

2. Then open Voicebox.app normally.

To stop the backend: `pkill -f uvicorn`

---

## Background Context

- **Bug:** macOS records audio as mp4/AAC (not webm). The original backend passed the raw file directly to Whisper, which couldn't decode it — causing the "Error -3" zlib error.
- **Fix:** The patched backend decodes the upload with librosa (handles any format) and re-encodes as proper PCM WAV before handing to Whisper.
- **PR:** https://github.com/jamiepine/voicebox/pull/602 — once merged, the fix will be in the official DMG and this local setup won't be needed.
- **Port:** The backend runs on 17493. Voicebox.app checks this port at startup and reuses an existing server if found, so no rebuild of the app is needed.
- **Python:** The venv uses Python 3.12 — the system Python (3.14) is too new for the ML packages.
