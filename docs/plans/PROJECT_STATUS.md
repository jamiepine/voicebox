# Voicebox Project Status & Roadmap

> Last updated: 2026-03-13 | Current version: **v0.1.13** | 13.1k stars | ~176 open issues | 25 open PRs

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current State](#current-state)
3. [Open PRs — Triage & Analysis](#open-prs--triage--analysis)
4. [Open Issues — Categorized](#open-issues--categorized)
5. [Existing Plan Documents — Status](#existing-plan-documents--status)
6. [New Model Integration — Landscape](#new-model-integration--landscape)
7. [Architectural Bottlenecks](#architectural-bottlenecks)
8. [Recommended Priorities](#recommended-priorities)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Tauri Shell (Rust)                                 │
│  ┌───────────────────────────────────────────────┐  │
│  │  React Frontend (app/)                        │  │
│  │  Zustand stores · API client · Generation UI  │  │
│  │  Stories Editor · Voice Profiles · Model Mgmt │  │
│  └──────────────────────┬────────────────────────┘  │
│                         │ HTTP :17493                │
│  ┌──────────────────────▼────────────────────────┐  │
│  │  FastAPI Backend (backend/)                   │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │ TTSBackend Protocol                     │  │  │
│  │  │  ┌──────────┐ ┌───────┐ ┌───────────┐  │  │  │
│  │  │  │ Qwen3-TTS│ │LuxTTS │ │Chatterbox │  │  │  │
│  │  │  │(Py/MLX)  │ │       │ │(MTL+Turbo)│  │  │  │
│  │  │  └──────────┘ └───────┘ └───────────┘  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │  ┌───────────┐  ┌─────────┐                   │  │
│  │  │ STTBackend│  │ Profiles│                   │  │
│  │  │ (Whisper) │  │ History │                   │  │
│  │  └───────────┘  │ Stories │                   │  │
│  │                  └─────────┘                   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Key Files

| Layer | File | Purpose |
|-------|------|---------|
| Backend entry | `backend/main.py` | FastAPI app, all API routes (~2100 lines) |
| TTS protocol | `backend/backends/__init__.py:14-81` | `TTSBackend` Protocol definition |
| TTS factory | `backend/backends/__init__.py:138-178` | Thread-safe engine registry (double-checked locking) |
| PyTorch TTS | `backend/backends/pytorch_backend.py` | Qwen3-TTS via `qwen_tts` package |
| MLX TTS | `backend/backends/mlx_backend.py` | Qwen3-TTS via `mlx_audio.tts` |
| LuxTTS | `backend/backends/luxtts_backend.py` | LuxTTS — fast, CPU-friendly |
| Chatterbox MTL | `backend/backends/chatterbox_backend.py` | Chatterbox Multilingual — 23 languages |
| Chatterbox Turbo | `backend/backends/chatterbox_turbo_backend.py` | Chatterbox Turbo — English, paralinguistic tags |
| Platform detect | `backend/platform_detect.py` | Apple Silicon → MLX, else → PyTorch |
| API types | `backend/models.py` | Pydantic request/response models |
| HF progress | `backend/utils/hf_progress.py` | HFProgressTracker (tqdm patching for download progress) |
| Audio utils | `backend/utils/audio.py` | `trim_tts_output()`, normalize, load/save audio |
| Frontend API | `app/src/lib/api/client.ts` | Hand-written fetch wrapper |
| Frontend types | `app/src/lib/api/types.ts` | TypeScript API types |
| Generation form | `app/src/components/Generation/GenerationForm.tsx` | TTS generation UI |
| Floating gen box | `app/src/components/Generation/FloatingGenerateBox.tsx` | Compact generation UI |
| Model manager | `app/src/components/ServerSettings/ModelManagement.tsx` | Model download/status/progress UI |
| GPU acceleration | `app/src/components/ServerSettings/GpuAcceleration.tsx` | CUDA backend swap UI |
| Gen form hook | `app/src/lib/hooks/useGenerationForm.ts` | Form validation + submission |
| Language constants | `app/src/lib/constants/languages.ts` | Per-engine language maps |

### How TTS Generation Works (Current Flow)

```
POST /generate
  1. Look up voice profile from DB
  2. Resolve engine from request (qwen | luxtts | chatterbox | chatterbox_turbo)
  3. Get backend: get_tts_backend_for_engine(engine)  # thread-safe singleton per engine
  4. Check model cache → if missing, trigger background download, return HTTP 202
  5. Load model (lazy): tts_backend.load_model(model_size)
  6. Create voice prompt: profiles.create_voice_prompt_for_profile(engine=engine)
       → tts_backend.create_voice_prompt(audio_path, reference_text)
  7. Generate: tts_backend.generate(text, voice_prompt, language, seed, instruct)
  8. Post-process: trim_tts_output() for Chatterbox engines
  9. Save WAV → data/generations/{id}.wav
  10. Insert history record in SQLite
  11. Return GenerationResponse
```

---

## Current State

### What's Shipped (v0.1.13 + recent merges)

**Core TTS:**
- Qwen3-TTS voice cloning (1.7B and 0.6B models)
- MLX backend for Apple Silicon, PyTorch for everything else
- Multi-engine TTS architecture with thread-safe backend registry (PR #254)
- LuxTTS integration — fast, CPU-friendly English TTS (PR #254)
- Chatterbox Multilingual TTS — 23 languages including Hebrew (PR #257)
- Delivery instructions (instruct parameter, Qwen only)
- Single flat model dropdown (Qwen 1.7B, Qwen 0.6B, LuxTTS, Chatterbox, Chatterbox Turbo)

**Infrastructure:**
- CUDA backend swap via binary download and restart (PR #252)
- GPU acceleration settings UI
- Voice profiles with multi-sample support
- Stories editor (multi-track DAW timeline)
- Whisper transcription (base, small, medium, large variants)
- Model management UI with inline download progress bars (HFProgressTracker)
- Download cancel/clear UI with error panel (PR #238)
- Generation history with caching
- Streaming generation endpoint (MLX only)
- Duplicate profile name validation (PR #175)
- Linux NVIDIA GBM buffer + WebKitGTK microphone fix (PR #210)

### What's In-Flight

| Feature | Branch/PR | Status |
|---------|-----------|--------|
| Chatterbox Turbo + per-engine language lists | `feat/chatterbox-turbo` / PR #258 | Open, ready for review |

### TTS Engine Comparison

| Engine | Model Name | Languages | Size | Key Features |
|--------|-----------|-----------|------|-------------|
| Qwen3-TTS 1.7B | `qwen-tts-1.7B` | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) | ~3.5 GB | Instruct mode, highest quality |
| Qwen3-TTS 0.6B | `qwen-tts-0.6B` | 10 | ~1.2 GB | Lighter, faster |
| LuxTTS | `luxtts` | English | ~300 MB | CPU-friendly, 48 kHz, fast |
| Chatterbox | `chatterbox-tts` | 23 (incl. Hebrew, Arabic, Hindi, etc.) | ~3.2 GB | Zero-shot cloning, multilingual |
| Chatterbox Turbo | `chatterbox-turbo` | English | ~1.5 GB | Paralinguistic tags ([laugh], [cough]), 350M params, low latency |

### Multi-Engine Architecture (Shipped)

The singleton TTS backend blocker described in the previous version of this doc has been **resolved**. The architecture now supports:

- **Thread-safe backend registry** (`_tts_backends` dict + `_tts_backends_lock`) with double-checked locking
- **Per-engine backend instances** — each engine gets its own singleton, loaded lazily
- **Engine field on GenerationRequest** — frontend sends `engine: 'qwen' | 'luxtts' | 'chatterbox' | 'chatterbox_turbo'`
- **Per-engine language filtering** — `ENGINE_LANGUAGES` map in frontend, backend regex accepts all languages
- **Per-engine voice prompts** — `create_voice_prompt_for_profile()` dispatches to the correct backend
- **Trim post-processing** — `trim_tts_output()` for Chatterbox engines (cuts trailing silence/hallucination)

### Known Limitations

- **HF XET progress**: Large files downloaded via `hf-xet` (HuggingFace's new transfer backend) report `n=0` in tqdm updates. Progress bars may appear stuck for large `.safetensors` files even though the download is proceeding. This is a known upstream limitation.
- **Chatterbox Turbo upstream token bug**: `from_pretrained()` passes `token=os.getenv("HF_TOKEN") or True` which fails without a stored HF token. Our backend works around this by calling `snapshot_download(token=None)` + `from_local()`.
- **chatterbox-tts must install with `--no-deps`**: It pins `numpy<1.26`, `torch==2.6.0`, `transformers==4.46.3` — all incompatible with our stack (Python 3.12, torch 2.10, transformers 4.57.3). Sub-deps listed explicitly in `requirements.txt`.
- **Streaming generation** only works for Qwen on MLX. Other engines use the non-streaming `/generate` endpoint.
- **dicta-onnx** (Hebrew diacritization) not included — upstream Chatterbox bug requires `model_path` arg but calls `Dicta()` with none. Hebrew works fine without it.

---

## Open PRs — Triage & Analysis

### Recently Merged (Since Last Update)

| PR | Title | Merged |
|----|-------|--------|
| **#257** | feat: Chatterbox TTS engine with multilingual voice cloning | 2026-03-13 |
| **#254** | feat: LuxTTS integration — multi-engine TTS support | 2026-03-13 |
| **#252** | feat: CUDA backend swap via binary download and restart | 2026-03-13 |
| **#238** | Download cancel/clear UI, fixed model downloading | 2026-03-13 |
| **#250** | docs: align local API port examples | 2026-03-13 |
| **#210** | fix: Linux NVIDIA GBM buffer crash | 2026-03-13 |
| **#175** | Fix #134: duplicate profile name validation | 2026-03-13 |

### In-Flight (Our Work)

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| **#258** | feat: Chatterbox Turbo engine + per-engine language lists | Open | Ready for review. Adds Turbo engine + dynamic language dropdown. |

### Merge-Ready / Near-Ready (Bug Fixes & Small Features)

| PR | Title | Risk | Notes |
|----|-------|------|-------|
| **#230** | docs: fix README grammar | None | Docs-only |
| **#243** | a11y: screen reader and keyboard improvements | Low | Accessibility, no backend changes |
| **#178** | Fix #168 #140: generation error handling | Low | Error handling improvements |
| **#152** | Fix: prevent crashes when HuggingFace unreachable | Medium | Monkey-patches HF hub; solves real offline bug (#150, #151) |
| **#218** | fix: unify qwen tts cache dir on Windows | Low | Windows-specific path fix |
| **#214** | fix: panic on launch from tokio::spawn | Low | Rust-side Tauri fix |
| **#88** | security: restrict CORS to known local origins | Low | Security hardening |
| **#133** | feat: network access toggle | Low | Wires up existing plumbing |

### Significant Feature PRs

| PR | Title | Complexity | Notes |
|----|-------|-----------|-------|
| **#253** | Enhance speech tokenizer with 48kHz version | Medium | Qwen tokenizer upgrade |
| **#97** | fix: pass language parameter to TTS models | Medium | May be partially obsoleted by multi-engine work — needs review |
| **#99** | feat: chunked TTS with quality selector | Medium | Solves 500-char limit. Addresses #191, #203, #69, #111. |
| **#154** | feat: Audiobook tab | Medium | Full audiobook workflow. Depends on #99 concepts. |
| **#91** | fix: CoreAudio device enumeration | Medium | macOS audio device handling |

### Architectural PRs (Need Careful Review)

| PR | Title | Complexity | Notes |
|----|-------|-----------|-------|
| **#225** | feat: custom HuggingFace model support | High | Arbitrary HF repo loading. May need rework given multi-engine arch is now shipped. |
| **#194** | feat: Hebrew + Chatterbox TTS | High | **Superseded** by PR #257 which shipped Chatterbox multilingual (23 langs incl. Hebrew). May be closeable. |
| **#195** | feat: per-profile LoRA fine-tuning | Very High | Training pipeline, adapter management, 15 new endpoints. Depends on #194 (now superseded). |
| **#161** | feat: Docker + web deployment | High | 3-stage Dockerfile, SPA serving. Independent of TTS engine work. |
| **#124** / **#123** | Docker (simpler attempts) | Low-Medium | Overlap with #161 |
| **#227** | fix: harden input validation & file safety | Medium | Coupled to #225 (custom models) |

### PRs That Need Author Action / Are Stale

| PR | Title | Notes |
|----|-------|-------|
| **#237** | fix: bundle qwen_tts source files in PyInstaller | Build system, needs review |
| **#215** | Update prerequisites with Tauri deps | Branch is `main` — will have conflicts |
| **#89** | Linux Support | Branch is `main` — will have conflicts. Broad scope. |
| **#83** | Update download links for v0.1.12 | Outdated (we're on v0.1.13) |

### PRs Likely Superseded

| PR | Superseded By | Notes |
|----|--------------|-------|
| **#194** (Hebrew + Chatterbox) | PR #257 (merged) | #257 ships Chatterbox multilingual with 23 languages including Hebrew. #194 took a different approach (route by language). Can likely be closed. |
| **#33** (External provider binaries) | PR #252 (merged) | #252 shipped CUDA backend swap. #33's broader provider architecture may still have value but needs reassessment. |

---

## Open Issues — Categorized

### GPU / Hardware Detection (19 issues)

The single most reported category. Users on Windows with NVIDIA GPUs frequently report "GPU not detected."

**Root causes (likely):**
- PyInstaller binary doesn't bundle CUDA correctly → falls back to CPU
- DirectML/Vulkan path not implemented (AMD on Windows)
- Binary size limit means CUDA can't ship in the main release

**Key issues:** #239, #222, #220, #217, #208, #198, #192, #167, #164, #141, #130, #127

**Fix path:** PR #252 (CUDA backend swap) is now merged. Users can download the CUDA binary separately from the GPU acceleration settings. Many of these issues may now be resolvable — needs triage to confirm.

### Model Downloads (20 issues)

Second most reported. Users get stuck downloads, can't resume, no offline fallback.

**Key issues:** #249, #240, #221, #216, #212, #181, #180, #159, #150, #149, #145, #143, #135, #134

**Fix path:** PR #238 (cancel/clear UI) is now merged. PR #152 (offline crash fix) still open. Inline progress bars now show for all engines. Resume support not yet addressed.

### Language Requests (18 issues)

Strong demand for: Hindi (#245), Indonesian (#247), Dutch (#236), Hebrew (#199), Greek (#188), Portuguese (#183), Persian (#162), and many more.

**Key issues:** #247, #245, #236, #211, #205, #199, #189, #188, #187, #183, #179, #162

**Fix path:** Chatterbox Multilingual (merged via #257) now supports 23 languages including many of the requested ones: Arabic, Danish, German, Greek, Finnish, Hebrew, Hindi, Dutch, Norwegian, Polish, Swedish, Swahili, Turkish. Per-engine language filtering (PR #258) ensures the UI shows correct options. Several of these issues may be closeable.

### New Model Requests (5 explicit issues)

| Issue | Model Requested |
|-------|----------------|
| #226 | GGUF support |
| #172 | VibeVoice |
| #138 | Export to ONNX/Piper format |
| #132 | LavaSR (transcription) |
| #76 | (General model expansion) |

Community also requests: XTTS-v2, Fish Speech, CosyVoice, Kokoro. The multi-engine architecture is now in place, making new model integration significantly easier.

### Long-Form / Chunking (5 issues)

Users hitting the ~500 character practical limit.

**Key issues:** #234 (queue system), #203 (500 char limit), #191 (auto-split), #111, #69

**Fix path:** PR #99 (chunked TTS + quality selector) directly addresses this. PR #154 (Audiobook tab) builds on it.

### Feature Requests (23 issues)

Notable requests:
- **#234** — Queue system for batch generation
- **#182** — Concurrent/multi-thread generation
- **#173** — Vocal intonation/inflection control
- **#165** — Audiobook mode
- **#144** — Copy text to clipboard
- **#184** — Cancel button for progress bar
- **#242** — Seed value pinning for consistency
- **#228** — Always use 0.6B option
- **#233** — Transcribe audio API improvements
- **#235** — Finetuned Qwen3-TTS tokenizer

### Bugs (19 issues)

| Category | Issues |
|----------|--------|
| Generation failures | #248 (broken pipe), #219 (unsupported scalarType), #202 (clipping error), #170 (load failed) |
| UI bugs | #231 (history not updating), #190 (mobile landing), #169 (blank interface) |
| File operations | #207 (transcribe file error), #168 (no such file), #142 (download audio fail) |
| Server lifecycle | #166 (server processes remain), #164 (no auto-update) |
| Database | #174 (sqlite3 IntegrityError) |
| Dependency | #131 (numpy ABI mismatch), #209 (import error) |

---

## Existing Plan Documents — Status

| Document | Target Version | Status | Relevance |
|----------|---------------|--------|-----------|
| `TTS_PROVIDER_ARCHITECTURE.md` | v0.1.13 | **Partially superseded** by multi-engine arch + CUDA swap | Core concepts implemented differently than planned |
| `CUDA_BACKEND_SWAP.md` | — | **Shipped** (PR #252) | CUDA binary download + backend restart |
| `CUDA_BACKEND_SWAP_FINAL.md` | — | **Shipped** (PR #252) | Final implementation plan |
| `EXTERNAL_PROVIDERS.md` | v0.2.0 | **Not started** | Remote server support |
| `MLX_AUDIO.md` | — | **Shipped** | MLX backend is live |
| `DOCKER_DEPLOYMENT.md` | v0.2.0 | **PR exists** (#161) | Waiting on review |
| `OPENAI_SUPPORT.md` | v0.2.0 | **Not started** | OpenAI-compatible API layer |
| `PR33_CUDA_PROVIDER_REVIEW.md` | — | **Reference** | Analysis of the original provider approach |

---

## New Model Integration — Landscape

### Models Worth Supporting (2026 SOTA)

| Model | Cloning | Speed | Sample Rate | Languages | VRAM | Integration Ease | Status |
|-------|---------|-------|-------------|-----------|------|-----------------|--------|
| **Qwen3-TTS** | 10s zero-shot | Medium | 24 kHz | 10 | Medium | **Shipped** | v0.1.13 |
| **LuxTTS** | 3s zero-shot | 150x RT, CPU ok | 48 kHz | English | <1 GB | **Shipped** | PR #254 |
| **Chatterbox MTL** | 5s zero-shot | Medium | 24 kHz | 23 | Medium | **Shipped** | PR #257 |
| **Chatterbox Turbo** | 5s zero-shot | Fast | 24 kHz | English | Low | **PR #258** | In review |
| **XTTS-v2** | 6s zero-shot | Mid-GPU | 24 kHz | 17+ | Medium | Ready | Multi-engine arch in place |
| **Fish Speech** | 10-30s few-shot | Real-time | 24-44 kHz | 50+ | Medium | Ready | Multi-engine arch in place |
| **CosyVoice2-0.5B** | 3-10s zero-shot | Very fast | 24 kHz | Multilingual | Low | Ready | Multi-engine arch in place |
| **Kokoro-82M** | 3s instant | CPU realtime | 24 kHz | English | Tiny | Ready | Multi-engine arch in place |

### Adding a New Engine (Now Straightforward)

With the multi-engine architecture shipped, adding a new TTS engine requires:

1. **Create `backend/backends/<engine>_backend.py`** — implement `TTSBackend` protocol (~200-300 lines)
2. **Register in `backend/backends/__init__.py`** — add to `TTS_ENGINES` dict + factory function
3. **Update `backend/models.py`** — add engine name to regex
4. **Update `backend/main.py`** — add engine cases in generate, stream, model-status, download, delete (5 dispatch points)
5. **Update frontend** — add to engine union type, form schema, model dropdown, language map (5-6 files)

Total effort: **~1 day** for a well-documented model with a PyPI package.

---

## Architectural Bottlenecks

### ~~1. Single Backend Singleton~~ — RESOLVED

The singleton TTS backend was replaced with a thread-safe per-engine registry in PR #254. Multiple engines can now be loaded simultaneously.

### 2. `main.py` is 2100+ Lines

All API routes, all model configs, all business logic in one file. Five separate dispatch points for each engine. Any new engine touches this file in 5 places. A model config registry pattern would reduce duplication.

### 3. Model Config is Scattered (Improved)

Model identifiers are still duplicated across `main.py` (3 dicts), backend files, frontend components, and the languages constant. However, the pattern is now consistent and well-understood. A centralized model registry would help but isn't blocking.

### 4. Voice Prompt Cache Assumes PyTorch Tensors

`backend/utils/cache.py` uses `torch.save()` / `torch.load()`. LuxTTS and Chatterbox backends work around this by storing reference audio paths instead of tensors in their voice prompt dicts. Not ideal but functional.

### 5. ~~Frontend Assumes Qwen Model Sizes~~ — RESOLVED

The generation form now uses a flat model dropdown with engine-based routing. Per-engine language filtering is in place. Model size is only sent for Qwen.

---

## Recommended Priorities

### Tier 1 — Ship Now (Low Risk)

| Priority | PR/Item | Impact | Effort |
|----------|---------|--------|--------|
| 1 | **#258** — Chatterbox Turbo + per-engine languages | Paralinguistic tags, proper language filtering | Review only |
| 2 | **#152** — Offline mode crash fix | Fixes #150, #151 | Low |
| 3 | **#99** — Chunked TTS + quality selector | Removes 500-char limit, addresses 5 issues | Medium |
| 4 | **#218** — Windows HF cache dir fix | Windows-specific pain | Low |
| 5 | **#178** — Generation error handling | Error UX | Low |
| 6 | **#230** — Docs fixes | Zero risk | None |
| 7 | **#133** — Network access toggle | Wires up existing code | Low |
| 8 | **#88** — CORS restriction | Security improvement | Low |
| 9 | **#214** — Tauri window close panic fix | Stability | Low |
| 10 | Triage GPU issues | Many may be resolved by CUDA swap (#252) | Low |
| 11 | Close superseded PRs | #194 (superseded by #257), #83 (outdated) | None |

### Tier 2 — Next Release (v0.2.0)

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | **#253** — 48kHz speech tokenizer | Quality improvement | Medium |
| 2 | **#161** — Docker deployment | Server/headless users | Medium |
| 3 | **#154** — Audiobook tab | Long-form users | Medium |
| 4 | **Model config registry** | Reduce 5-dispatch-point duplication in main.py | Medium |
| 5 | **#225** — Custom HuggingFace models | User-supplied models | High (needs rework for multi-engine) |

### Tier 3 — Future (v0.3.0+)

| Item | Notes |
|------|-------|
| XTTS-v2 / Fish Speech / CosyVoice | Multi-engine arch is ready; just needs backend implementation |
| OpenAI-compatible API (plan doc exists) | Low effort once API is stable |
| LoRA fine-tuning (PR #195) | Complex, needs rework for multi-engine |
| External/remote providers | Depends on use case demand |
| GGUF support (#226) | Depends on model ecosystem maturity |
| Queue system (#234) | Batch generation |
| Streaming for non-MLX engines | Currently MLX-only |
| Kokoro-82M | Tiny model, great for CPU-only machines |

---

## Branch Inventory

| Branch | PR | Status | Notes |
|--------|-----|--------|-------|
| `feat/chatterbox-turbo` | #258 | Open | Chatterbox Turbo + per-engine languages |
| `feat/chatterbox` | #257 | **Merged** | Chatterbox Multilingual |
| `feat/luxtts` | #254 | **Merged** | LuxTTS + multi-engine arch |
| `external-provider-binaries` | #33 | Superseded by #252 | Original CUDA provider approach |
| `feat/dual-server-binaries` | — | No PR | Related to provider split |
| `fix-multi-sample` | — | No PR | Voice profile multi-sample fix |
| `fix-dl-notification-...` | — | No PR | Model download UX |

---

## Quick Reference: API Endpoints

<details>
<summary>All current endpoints</summary>

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, model/GPU status |
| `/profiles` | POST, GET | Create/list voice profiles |
| `/profiles/{id}` | GET, PUT, DELETE | Profile CRUD |
| `/profiles/{id}/samples` | POST, GET | Add/list voice samples |
| `/profiles/{id}/avatar` | POST, GET, DELETE | Avatar management |
| `/profiles/{id}/export` | GET | Export profile as ZIP |
| `/profiles/import` | POST | Import profile from ZIP |
| `/generate` | POST | Generate speech (engine param selects TTS backend) |
| `/generate/stream` | POST | Stream speech (MLX only) |
| `/history` | GET | List generation history |
| `/history/{id}` | GET, DELETE | Get/delete generation |
| `/history/{id}/export` | GET | Export generation ZIP |
| `/history/{id}/export-audio` | GET | Export audio only |
| `/transcribe` | POST | Transcribe audio (Whisper) |
| `/models/status` | GET | All model statuses (Qwen, LuxTTS, Chatterbox, Chatterbox Turbo, Whisper) |
| `/models/download` | POST | Trigger model download |
| `/models/download/cancel` | POST | Cancel/dismiss download |
| `/models/{name}` | DELETE | Delete downloaded model |
| `/models/load` | POST | Load model into memory |
| `/models/unload` | POST | Unload model |
| `/models/progress/{name}` | GET | SSE download progress |
| `/tasks/active` | GET | Active downloads/generations (with inline progress) |
| `/stories` | POST, GET | Create/list stories |
| `/stories/{id}` | GET, PUT, DELETE | Story CRUD |
| `/stories/{id}/items` | POST, GET | Story items CRUD |
| `/stories/{id}/export` | GET | Export story audio |
| `/channels` | POST, GET | Audio channel CRUD |
| `/channels/{id}` | PUT, DELETE | Channel update/delete |
| `/cache/clear` | POST | Clear voice prompt cache |
| `/server/cuda/status` | GET | CUDA binary availability |
| `/server/cuda/download` | POST | Download CUDA binary |
| `/server/cuda/switch` | POST | Switch to CUDA backend |

</details>
