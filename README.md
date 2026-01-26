<p align="center">
  <img src=".github/assets/icon-dark.webp" alt="Voicebox" width="120" height="120" />
</p>

<h1 align="center">Voicebox</h1>

<p align="center">
  <strong>The open-source voice synthesis studio.</strong><br/>
  Clone voices. Generate speech. Build voice-powered apps.<br/>
  All running locally on your machine.
</p>

<p align="center">
  <a href="https://voicebox.sh">voicebox.sh</a> •
  <a href="#download">Download</a> •
  <a href="#features">Features</a> •
  <a href="#api">API</a> •
  <a href="#roadmap">Roadmap</a>
</p>

<br/>

<p align="center">
  <img src=".github/assets/screenshot.webp" alt="Voicebox App Screenshot" width="800" />
</p>

<br/>

## Why Voicebox?

Voice AI is exploding, but most tools are either cloud-locked, expensive, or a nightmare to set up. Voicebox is different:

- **100% Local** — Your voice data never leaves your machine
- **Lightweight** — No bloated Electron, native Tauri performance
- **Fast** — Near-instant on CUDA, optimized for Apple Silicon
- **Flexible** — Use the app, integrate the API, or both
- **Open Source** — No subscriptions, no limits, no lock-in

Built with **Tauri** (Rust), **TypeScript**, **React**, and **Python**. Native performance meets modern DX.

---

## Download

Voicebox is available now for macOS, Windows, and Linux.

| Platform | Download |
|----------|----------|
| macOS (Apple Silicon) | [voicebox-macos-arm64.dmg](https://github.com/voicebox-sh/voicebox/releases/latest) |
| macOS (Intel) | [voicebox-macos-x64.dmg](https://github.com/voicebox-sh/voicebox/releases/latest) |
| Windows | [voicebox-windows-x64.exe](https://github.com/voicebox-sh/voicebox/releases/latest) |
| Linux | [voicebox-linux-x64.AppImage](https://github.com/voicebox-sh/voicebox/releases/latest) |

---

## See it in action...

<video width="800" controls>
  <source src="https://voicebox.sh/voicebox-demo.webm" type="video/webm">
  Your browser does not support the video tag. <a href="https://voicebox.sh">Watch on voicebox.sh</a>
</video>

---

## Features

### Voice Cloning with Qwen3-TTS

Powered by Alibaba's **Qwen3-TTS** — a breakthrough model that achieves near-perfect voice cloning from just a few seconds of audio.

- **Instant cloning** — Upload a sample, get a voice profile
- **High fidelity** — Natural prosody, emotion, and cadence
- **Multi-language** — English, Chinese, and more coming

### Voice Profile Management

- **Create profiles** from audio files or record directly in-app
- **Import/Export** profiles to share or backup
- **Organize** with descriptions and language tags

### Speech Generation

- **Text-to-speech** with any cloned voice
- **Batch generation** for long-form content
- **Smart caching** — regenerate instantly with voice prompt caching

### Recording & Transcription

- **In-app recording** with waveform visualization
- **Automatic transcription** powered by Whisper
- **Export recordings** in multiple formats

### Generation History

- **Full history** of all generated audio
- **Search & filter** by voice, text, or date
- **Re-generate** any past generation with one click

### Flexible Deployment

- **Local mode** — Everything runs on your machine
- **Remote mode** — Connect to a GPU server on your network
- **One-click server** — Turn any machine into a Voicebox server

---

## API

Voicebox exposes a full REST API, so you can integrate voice synthesis into your own apps.

```bash
# Generate speech
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "profile_id": "abc123"}'

# List voice profiles
curl http://localhost:8000/api/profiles

# Create a profile from audio
curl -X POST http://localhost:8000/api/profiles \
  -F "audio=@voice-sample.wav" \
  -F "name=My Voice"
```

**Use cases:**

- Game dialogue systems
- Podcast/video production pipelines
- Accessibility tools
- Voice assistants
- Content creation automation

Full API documentation available at `http://localhost:8000/docs` when running.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Desktop App | Tauri (Rust) |
| Frontend | React, TypeScript, Tailwind CSS |
| State | Zustand, React Query |
| Backend | FastAPI (Python) |
| Voice Model | Qwen3-TTS |
| Transcription | Whisper |
| Database | SQLite |
| Audio | WaveSurfer.js, librosa |

**Why this stack?**

- **Tauri over Electron** — 10x smaller bundle, native performance, lower memory
- **FastAPI** — Async Python with automatic OpenAPI schema generation
- **Type-safe end-to-end** — Generated TypeScript client from OpenAPI spec

---

## Roadmap

Voicebox is the beginning of something bigger. Here's what's coming:

### Coming Soon

| Feature | Description |
|---------|-------------|
| **Real-time Synthesis** | Stream audio as it generates, word by word |
| **Conversation Mode** | Multi-speaker dialogues with automatic turn-taking |
| **Voice Effects** | Pitch shift, reverb, M3GAN-style effects |
| **Timeline Editor** | Audio studio with word-level precision editing |
| **More Models** | XTTS, Bark, and other open-source voice models |

### Future Vision

- **Voice Design** — Create new voices from text descriptions
- **Project System** — Save and load complex multi-voice sessions
- **Plugin Architecture** — Extend with custom models and effects
- **Mobile Companion** — Control Voicebox from your phone

Voicebox aims to be the **one-stop shop for everything voice** — cloning, synthesis, editing, effects, and beyond.

---

## Development

### Prerequisites

- [Bun](https://bun.sh) (package manager)
- [Rust](https://rustup.rs) (for Tauri)
- [Python 3.11+](https://python.org) (for backend)
- CUDA-capable GPU recommended (CPU inference supported but slower)

### Setup

```bash
# Clone the repo
git clone https://github.com/voicebox-sh/voicebox.git
cd voicebox

# Install dependencies
bun install

# Install Python dependencies
cd backend && pip install -r requirements.txt && cd ..

# Start development
bun run dev
```

### Project Structure

```
voicebox/
├── app/              # Shared React frontend
├── tauri/            # Desktop app (Tauri + Rust)
├── web/              # Web deployment
├── backend/          # Python FastAPI server
├── landing/          # Marketing website
└── scripts/          # Build & release scripts
```

---

## Contributing

Contributions welcome! Whether it's bug fixes, new features, or documentation improvements.

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a PR

---

## License

MIT License — use it however you want.

---

<p align="center">
  <a href="https://voicebox.sh">voicebox.sh</a>
</p>
