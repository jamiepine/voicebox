# Voicebox Project Status & Roadmap

> Last updated: 2026-04-18 | Current version: **v0.4.1** | 232 open issues | 12 open PRs

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

**Tauri shell (Rust)** hosts a **React frontend** (`app/`) that talks over HTTP on `localhost:17493` to a **FastAPI backend** (`backend/`).

The backend exposes:

- **`TTSBackend` Protocol** with seven concrete engine implementations:
  - Qwen3-TTS (PyTorch or MLX depending on platform)
  - Qwen CustomVoice (predefined speakers with instruct)
  - LuxTTS (fast, CPU-friendly)
  - Chatterbox Multilingual (23 languages)
  - Chatterbox Turbo (English, paralinguistic tags)
  - TADA (1B English, 3B multilingual via HumeAI)
  - Kokoro 82M (pre-built voices, CPU realtime)
- **`STTBackend` Protocol** for Whisper (PyTorch or MLX-Whisper)
- **Profiles / History / Stories** services for persistence and timeline editing

### Key Files

| Layer | File | Purpose |
|-------|------|---------|
| Backend entry | `backend/main.py` | FastAPI app, all API routes (~2850 lines) |
| TTS protocol | `backend/backends/__init__.py:32-101` | `TTSBackend` Protocol definition |
| Model registry | `backend/backends/__init__.py:17-29,153-366` | `ModelConfig` dataclass + registry helpers |
| TTS factory | `backend/backends/__init__.py:382-426` | Thread-safe engine registry (double-checked locking) |
| PyTorch TTS | `backend/backends/pytorch_backend.py` | Qwen3-TTS via `qwen_tts` package |
| MLX TTS | `backend/backends/mlx_backend.py` | Qwen3-TTS via `mlx_audio.tts` |
| LuxTTS | `backend/backends/luxtts_backend.py` | LuxTTS — fast, CPU-friendly |
| Chatterbox MTL | `backend/backends/chatterbox_backend.py` | Chatterbox Multilingual — 23 languages |
| Chatterbox Turbo | `backend/backends/chatterbox_turbo_backend.py` | Chatterbox Turbo — English, paralinguistic tags |
| TADA | `backend/backends/hume_backend.py` | HumeAI TADA — 1B English + 3B Multilingual |
| Kokoro | `backend/backends/kokoro_backend.py` | Kokoro 82M — CPU realtime, pre-built voices |
| Qwen CustomVoice | `backend/backends/qwen_custom_voice_backend.py` | Qwen CustomVoice — predefined speakers with instruct |
| Platform detect | `backend/platform_detect.py` | Apple Silicon → MLX, else → PyTorch |
| API types | `backend/models.py` | Pydantic request/response models |
| HF progress | `backend/utils/hf_progress.py` | HFProgressTracker (tqdm patching for download progress) |
| Audio utils | `backend/utils/audio.py` | `trim_tts_output()`, normalize, load/save audio |
| Frontend API | `app/src/lib/api/client.ts` | Hand-written fetch wrapper |
| Frontend types | `app/src/lib/api/types.ts` | TypeScript API types |
| Engine selector | `app/src/components/Generation/EngineModelSelector.tsx` | Shared engine/model dropdown |
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
  2. Resolve engine from request (qwen | qwen_custom_voice | luxtts | chatterbox | chatterbox_turbo | tada | kokoro)
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

### What's Shipped (v0.4.x)

**New since v0.3.0:**
- Kokoro 82M TTS engine + voice profile type system (PR #325)
- Qwen CustomVoice preset engine — predefined speakers with instruct support (PR #328)
- Intel Arc (XPU) GPU support (PR #320)
- Blackwell GPU (sm_120) CUDA support (PR #401)
- Generation cancellation flow (PR #444)
- Frontend quality gates + TypeScript hardening (PR #418)
- macOS Intel (x86_64) PyTorch compatibility (PR #416)
- Frozen-binary import fixes for Kokoro / Chatterbox Multilingual / scipy / transformers (PR #438)
- Linux PipeWire/PulseAudio monitor detection (PR #457)
- Server survives GUI close on Windows (PR #402)
- GPU arch compatibility warning on startup (catches unsupported PyTorch builds)
- cpal Stream playback reliability (PR #405), clip-splitting stability (PR #403)
- torch.from_numpy crash with numpy 2.x in frozen binary (PR #361)
- Async CUDA download lock (PR #428), NUMBA_CACHE_DIR env var (PR #425)
- "Clear failed" history button (PR #412)
- External server GUI startup + data refresh (PR #319)
- Force offline mode for cached Qwen/Whisper models (PR #318)
- macOS 11 ScreenCaptureKit launch crash fix (PR #424)

**Core TTS (cumulative):**
- Qwen3-TTS voice cloning (1.7B and 0.6B models, MLX + PyTorch)
- Qwen CustomVoice (preset speakers, instruct)
- LuxTTS — fast, CPU-friendly English TTS (PR #254)
- Chatterbox Multilingual — 23 languages including Hebrew (PR #257)
- Chatterbox Turbo — paralinguistic tags, low latency English (PR #258)
- HumeAI TADA — 1B English + 3B Multilingual (PR #296)
- Kokoro 82M — CPU-realtime, 8 languages, Apache 2.0 (PR #325)
- Multi-engine architecture with thread-safe backend registry (PR #254)
- Chunked TTS generation — engine-agnostic, removes ~500 char limit (PR #266)
- Async generation queue (PR #269)
- Post-processing audio effects system (PR #271)
- Voice profile type system (preset vs cloned, engine compatibility gating)
- Centralized `ModelConfig` registry — no per-engine dispatch maps
- Shared `EngineModelSelector` component

**Infrastructure (cumulative):**
- CUDA backend swap via binary download (PR #252), cu128 upgrade (PR #316), Blackwell/sm_120 (PR #401)
- CUDA backend split into independently versioned server + libs archives (PR #298)
- Intel Arc XPU support (PR #320)
- Docker + web deployment (PR #161)
- Backend refactor: modular architecture, style guide, tooling (PR #285)
- Settings overhaul: routed sub-tabs, server logs, changelog, about page (PR #294)
- Windows support: CUDA detection, cross-platform justfile, server lifecycle (PR #272, #402)
- Linux audio capture via pactl monitor detection (PR #457)
- macOS Intel x86_64 compatibility (PR #416)
- Voice profiles with multi-sample support
- Stories editor (multi-track DAW timeline)
- Whisper transcription (base, small, medium, large, turbo variants)
- Model management UI with inline download progress + folder migration (PR #268)
- Download cancel/clear UI with error panel (PR #238)
- Generation history with caching and cancellation (PR #444)
- Streaming generation endpoint (MLX only)
- Audio player freeze fix + UX improvements (PR #293)
- CORS restriction to known local origins (PR #88)

### Abandoned / Backlogged Integrations

| Model | PR / Branch | Reason |
|-------|-------------|--------|
| **CosyVoice2/3** | PR #311 | Output quality too poor. Heavy deps, no PyPI, needed 5+ shims. PR should be closed. |
| **VoxCPM 1.5 / VoxCPM2** | `voicebox-new-models` research (2026-04-18) | **Backlogged.** See detailed analysis below. |

#### VoxCPM — Evaluation Notes (2026-04-18)

**Project:** [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — tokenizer-free TTS, 2B params (VoxCPM2), end-to-end diffusion autoregressive architecture, 30 languages, 48 kHz output, Apache 2.0, `pip install voxcpm`.

**Why it looked interesting:**
- Clean PyPI install (`pip install voxcpm`)
- Apache 2.0 — commercially safe
- Voice cloning via `reference_wav_path` with optional `prompt_wav_path` + `prompt_text` for "ultimate" cloning
- Streaming API via `generate_streaming()`
- Zero-shot cloning + style control via parenthetical prefixes in text (`(slightly faster, cheerful tone)...`)
- Relatively high-quality output per demos

**Why we backlogged it:**
- **Effectively CUDA-only.** README states `CUDA ≥ 12.0` as hard requirement. Source code's `from_pretrained(device=None|"auto")` claims "preferring CUDA, then MPS, then CPU," but in practice:
  - **MPS (Apple Silicon) broken upstream** — OpenBMB/VoxCPM issues #232 (`NotImplementedError: Output channels > 65536 not supported at the MPS device`) and #248 (`IndexError` on M3 Mac) are both open with no resolution.
  - **CPU unsupported in the Python package** — issue #256 shows `voxcpm --device cpu` rejected with `unrecognized arguments`. The only CPU path is the third-party **VoxCPM.cpp** GGML engine, which is a separate ecosystem project, not `pip install voxcpm`.
  - **macOS source install fails** — issue #233 open with no resolution.
- Would require CUDA-only gating in UI (new `requires_cuda` flag on `ModelConfig`, lock icon + "Requires NVIDIA GPU" in `ModelManagement.tsx` / `EngineModelSelector.tsx`) plus a hard error at `load_model()` as safety net. Doable but adds first-class platform gating that doesn't exist for any other engine today.
- Voicebox's user base skews Apple Silicon (MLX is a primary backend). Shipping a CUDA-only model sets a precedent worth a separate scoping discussion (see issues #419 engine sprawl, #420 platform tiers, PR #465).

**What would change the decision:**
- Upstream fixes MPS crashes (watch issues #232, #248).
- We define an "experimental / CUDA-only" engine tier as part of issue #419 / PR #465, and decide it's acceptable to ship engines that are hidden on non-NVIDIA platforms.
- VoxCPM.cpp matures into a viable CPU path we can wrap (currently separate project, C++/GGML, unclear ergonomics).

**Integration shape if we revive it:** Zero-shot cloning maps naturally to the Chatterbox-style backend (store `ref_audio` + `ref_text` paths in the voice prompt dict, process at generate time). Est. ~250 lines for `voxcpm_backend.py` + one `ModelConfig` entry + engine registration in `backends/__init__.py`. Frontend UI gating is the bigger lift.

### What's In-Flight

| Feature | Branch/PR | Status |
|---------|-----------|--------|
| Platform support tiers | PR #465, issue #420 | Defining tier-1 (supported) vs tier-2 (community) platforms |
| Engine sprawl cleanup | issue #419 | First-class vs experimental TTS backends distinction |
| Frontend tech-debt burn-down | issue #421 | Biome + a11y debt before gating CI |
| Docker registry auto-publish | PR #463, issue #453 | ghcr.io image on tag push |
| New model research | `voicebox-new-models` branch | Evaluating Fish Speech, XTTS-v2, Pocket TTS, VibeVoice, Fish Audio S2, index-tts2 |

### TTS Engine Comparison

| Engine | Model Name | Profile Type | Languages | Size | Key Features | Instruct Support |
|--------|-----------|--------------|-----------|------|-------------|-----------------|
| Qwen3-TTS 1.7B | `qwen-tts-1.7B` | Cloned | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) | ~3.5 GB | Highest quality, voice cloning | None (Base model has no instruct path) |
| Qwen3-TTS 0.6B | `qwen-tts-0.6B` | Cloned | 10 | ~1.2 GB | Lighter, faster | None |
| Qwen CustomVoice 1.7B | `qwen-custom-voice-1.7B` | Preset | 10 | ~3.5 GB | Predefined speakers, instruct support | **Yes** |
| Qwen CustomVoice 0.6B | `qwen-custom-voice-0.6B` | Preset | 10 | ~1.2 GB | Predefined speakers, instruct support | **Yes** |
| LuxTTS | `luxtts` | Cloned | English | ~300 MB | CPU-friendly, 48 kHz, fast | None |
| Chatterbox | `chatterbox-tts` | Cloned | 23 (incl. Hebrew, Arabic, Hindi, etc.) | ~3.2 GB | Zero-shot cloning, multilingual | Partial — `exaggeration` float (0-1) |
| Chatterbox Turbo | `chatterbox-turbo` | Cloned | English | ~1.5 GB | Paralinguistic tags ([laugh], [cough]), 350M params, low latency | Partial — inline tags only |
| TADA 1B | `tada-1b` | Cloned | English | ~4 GB | HumeAI speech-language model, 700s+ coherent audio | None |
| TADA 3B Multilingual | `tada-3b-ml` | Cloned | 10 (en, ar, zh, de, es, fr, it, ja, pl, pt) | ~8 GB | Multilingual, text-acoustic dual alignment | None |
| Kokoro 82M | `kokoro` | Preset | 8 (en, es, fr, hi, it, pt, ja, zh) | ~350 MB | 82M params, CPU realtime, Apache 2.0, pre-built voices | None |

### Multi-Engine Architecture (Shipped)

- **Thread-safe backend registry** (`_tts_backends` dict + `_tts_backends_lock`) with double-checked locking
- **Per-engine backend instances** — each engine gets its own singleton, loaded lazily
- **Engine field on GenerationRequest** — frontend sends `engine: 'qwen' | 'qwen_custom_voice' | 'luxtts' | 'chatterbox' | 'chatterbox_turbo' | 'tada' | 'kokoro'`
- **Per-engine language filtering** — `ENGINE_LANGUAGES` map in frontend, backend regex accepts all languages
- **Per-engine voice prompts** — `create_voice_prompt_for_profile()` dispatches to the correct backend
- **Profile type system** — preset vs cloned profiles, UI grays out incompatible engines and auto-switches on selection
- **Trim post-processing** — `trim_tts_output()` for Chatterbox engines (cuts trailing silence/hallucination)

### Known Limitations

- **HF XET progress**: Large files downloaded via `hf-xet` (HuggingFace's new transfer backend) report `n=0` in tqdm updates. Progress bars may appear stuck for large `.safetensors` files even though the download is proceeding. This is a known upstream limitation.
- **Chatterbox Turbo upstream token bug**: `from_pretrained()` passes `token=os.getenv("HF_TOKEN") or True` which fails without a stored HF token. Our backend works around this by calling `snapshot_download(token=None)` + `from_local()`.
- **chatterbox-tts must install with `--no-deps`**: It pins `numpy<1.26`, `torch==2.6.0`, `transformers==4.46.3` — all incompatible with our stack (Python 3.12, torch 2.10, transformers 4.57.3). Sub-deps listed explicitly in `requirements.txt`.
- **Instruct parameter partially shipped** (#224, #303): Qwen CustomVoice (PR #328) now provides real instruct support via predefined speakers. Other backends still silently drop the instruct field — the UI exposes the field broadly but most engines ignore it. The floating generate box was patched to restore instruct for CustomVoice (commit `106aec4`).
- **Streaming generation** only works for Qwen on MLX. Other engines use the non-streaming `/generate` endpoint.
- **dicta-onnx** (Hebrew diacritization) not included — upstream Chatterbox bug requires `model_path` arg but calls `Dicta()` with none. Hebrew works fine without it.
- **Blackwell (RTX 50-series) CUDA**: cu128 + sm_120 kernel support shipped (PR #401, #316), but users still report `cudaErrorNoKernelImageForDevice` (#417, #400, #396, #395, #390, #362) — likely a stale CUDA binary on upgraded installs. Needs a follow-up diagnostic / forced re-download path.
- **Long text 50k character limit** (#464, #365, #354): Still hit on GPU despite chunking (PR #266). Chunking reliability needs another pass.
- **ROCm on RDNA 3/4** (#469): `HSA_OVERRIDE_GFX_VERSION` is hardcoded and harms newer cards.
- **`flash-attn is not installed` warning on every platform (cosmetic, common user complaint)**: Our transformer-based engines (Chatterbox / Qwen) emit `Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.` on every startup, on every platform — we don't pin `flash-attn` in requirements because installing it is fragile and version-sensitive. Fallback is PyTorch SDPA, which is near-FA2 throughput on Ampere+ and is what actually runs. **Per-platform reality:** (a) **macOS/Apple Silicon** — FlashAttention is CUDA-only, irrelevant here; MLX has its own attention kernels. (b) **Linux** — `pip install flash-attn --no-build-isolation` works but takes 20+ min to compile. (c) **Windows** — no official support (Dao-AILab README still says only "Might work"; source builds routinely fail on recent CUDA/MSVC, issues #1715, #1828, #2395). Windows users can install community prebuilt wheels from `kingbri1/flash-attention` or `bdashore3/flash-attention` (latest v2.8.3, Aug 2025; `win_amd64` wheels for CUDA 12.4/12.8, Torch 2.6–2.9, Python 3.10–3.13) matching their exact CUDA/Torch/Python, or use WSL2. **Native-Windows alternatives worth considering as a build-time swap:** SageAttention (thu-ml, Apache 2.0, claims 2–5× over FA2) and xformers (official Windows wheels). **Action for us:** troubleshooting doc now covers it (see `docs/content/docs/overview/troubleshooting.mdx`), and we should optionally suppress the warning via `logging.getLogger(...).setLevel(ERROR)` at backend import since the fallback is functionally fine.
- **WebAudio playback dies after audio-session interruption** (#41, plus an internal repro where the app is backgrounded long enough): WaveSurfer's `AudioContext` gets suspended by macOS — either because another app grabs the audio output, or because the WKWebView throttles when backgrounded. `play()` resolves and `timeupdate` can still fire, but no audio reaches the output. Only app restart fixes it. **Things already tried that didn't work:** (a) swapping WaveSurfer backend away from WebAudio — introduced more bugs, not an option; (b) remount hook on the player — doesn't help because a freshly-created `AudioContext` is born suspended and only resumes on a user gesture. PR #293 was a prior partial fix that doesn't cover this path. **Next thing to try** (not yet attempted — confirmed via grep of `AudioPlayer.tsx`): call `wavesurfer.getMediaElement().getGainNode().context.resume()` on the play button click (the click itself is a valid user gesture), plus a `visibilitychange` + `statechange` listener as belt-and-suspenders. The `ctx.resume()` pattern already exists in the codebase at `useStoryPlayback.ts:52` — just not wired into the main player.

---

## Open PRs — Triage & Analysis

### Recently Merged (Since Last Update — 2026-03-18 → 2026-04-18)

| PR | Title | Merged |
|----|-------|--------|
| **#481** | fix(build): pin transformers in MLX requirements to prevent 5.x upgrade | 2026-04-19 |
| **#470** | fix(api-client): declare moved + errors on migrateModels response type | 2026-04-18 |
| **#457** | fix(linux): use pactl to detect PipeWire/PulseAudio monitor | 2026-04-18 |
| **#450** | docs: clarify paralinguistic tag support in quick start | 2026-04-18 |
| **#447** | fix: delete version rows and files in delete_generations_by_profile | 2026-04-18 |
| **#444** | Fix generation cancellation flow | 2026-04-18 |
| **#440** | fix(paths): strip legacy "data/" prefix when resolving stored paths | 2026-04-18 |
| **#439** | Fix migration dialog hanging when no models are present | 2026-04-18 |
| **#438** | fix(build): repair frozen-binary imports for kokoro/chatterbox-multilingual/scipy/transformers | 2026-04-18 |
| **#433** | fix: warn user when no models to migrate during storage change | 2026-04-18 |
| **#425** | Add NUMBA_CACHE_DIR environment variable | 2026-04-16 |
| **#424** | fix: avoid ScreenCaptureKit launch crash on macOS 11 | 2026-04-16 |
| **#418** | Frontend quality gates + TypeScript hardening | 2026-04-18 |
| **#416** | fix(deps): relax PyTorch requirement for macOS Intel (x86_64) | 2026-04-16 |
| **#412** | feat(history): add "Clear failed" button | 2026-04-16 |
| **#405** | fix: keep cpal Stream alive until playback completes | 2026-04-16 |
| **#403** | fix: prevent intermittent clip splitting failures | 2026-04-16 |
| **#402** | fix: reliably keep server alive after GUI close on Windows | 2026-04-16 |
| **#401** | feat: add Blackwell GPU (sm_120) CUDA support | 2026-04-16 |
| **#394** | fix(history): populate status/error/engine fields from DB row | 2026-04-16 |
| **#384** | Fix: Resolve ModuleNotFoundError in effects service | 2026-04-16 |
| **#361** | fix: torch.from_numpy crash with numpy 2.x in frozen binary | 2026-04-16 |
| **#345** | Fix: "Failed to Save" preset error by resolving backend import path | 2026-03-22 |
| **#344** | fix: include changelog in docker web build | 2026-03-27 |
| **#332** | Fix links in Get Started section of index.mdx | 2026-03-21 |
| **#328** | feat: add Qwen CustomVoice preset engine | 2026-03-27 |
| **#325** | feat: Kokoro 82M TTS engine + voice profile type system | 2026-03-20 |
| **#321** | fix: allows deletion of failed generations | 2026-03-19 |
| **#320** | feat: Intel Arc (XPU) GPU support | 2026-03-21 |
| **#319** | fix: GUI startup with external server + data refresh on server switch | 2026-03-27 |
| **#318** | fix: force offline mode when loading cached models (Qwen TTS & Whisper) | 2026-03-21 |
| **#316** | Upgrade CUDA backend from cu126 to cu128, fix GPU settings UI | 2026-03-18 |

### Currently Open (12 PRs)

| PR | Title | Status | Notes |
|----|-------|--------|-------|
| **#465** | docs: define tier-1 and tier-2 platform support targets | Community PR | Pairs with issue #420. Important for scoping. |
| **#463** | feat(actions): add docker-registry.yml for automatic ghcr.io publishing | Community PR | Pairs with issue #453. Low risk. |
| **#443** | fix: prevent infinite retry loop in offline mode (#434) | Community PR | Fixes reported bug. |
| **#430** | feat: add MiniMax TTS provider support | Community PR | Cloud TTS provider — new direction (external API). Superset of #331? |
| **#331** | feat: add MiniMax Cloud TTS as a built-in engine | Community PR | Likely superseded by #430. Dedupe. |
| **#311** | feat: add CosyVoice2/3 TTS engine | **Close** | Abandoned — output quality too poor. |
| **#253** | Enhance speech tokenizer with 48kHz version | Community PR | Qwen tokenizer upgrade. Still worth reviewing. |
| **#227** | fix: harden input validation & file safety | Community PR | Coupled to #225 (custom models). |
| **#225** | feat: custom HuggingFace voice model support | Community PR | Needs rework for multi-engine arch. |
| **#195** | feat: per-profile LoRA fine-tuning | Draft | Complex. 15 new endpoints. |
| **#154** | feat: Audiobook tab | Community PR | Chunked generation now shipped (#266). |
| **#91** | fix: CoreAudio device enumeration | Draft | macOS audio device handling. |

---

## Open Issues — Categorized

### GPU / Hardware Detection — still the top category

**RTX 50-series (Blackwell / sm_120) cluster — NEW:** #417, #400, #396, #395, #390, #362 all report `cudaErrorNoKernelImageForDevice` / "no kernel image available." sm_120 support shipped in PR #401 + cu128 in PR #316, but users on upgraded installs still hit it — likely stale CUDA binary. Needs a diagnostic that detects binary/GPU-arch mismatch and prompts re-download.

**AMD / ROCm — NEW:** #469 `HSA_OVERRIDE_GFX_VERSION` is hardcoded and breaks RDNA 3/4 cards. #313 DirectML on AMD Ryzen AI Max+ 395 not working.

**Intel Arc:** PR #320 shipped XPU support — may resolve #119.

**General GPU-not-detected (older):** #368, #310, #330, #324, #326, #355 (multi-GPU / eGPU).

**Fix path:** CUDA backend swap (PR #252) + cu128 (PR #316) + sm_120 (PR #401) + GPU-arch warning (`73170d0`) are all in. Remaining work is diagnostics + re-download prompts for users whose binary predates the kernel updates.

### Model Downloads

Still reported. Users get stuck downloads, can't resume, offline mode edge cases.

**Key issues:** #475 (MAC CustomVoice install error), #449 (infinite loading macOS), #445 (can't download CustomVoice), #462 (Qwen requires internet even when loaded — regression from #150), #434 (infinite retry loop offline — PR #443 open), #432 (storage location change hangs when empty — partly fixed by PR #439/#433), #348 (TADA 3B Multilingual download fails), #336 (TADA model not listed in app), #275 (`No module named 'chatterbox'` on download), #304 (whisper-base feature extractor load error), #287 (macOS ARM `check_model_inputs` ImportError on new version), #181, #180.

**Fix path:** PR #443 addresses infinite offline retry. CustomVoice-specific download failures (#475, #445) need triage — likely related to frozen-binary import fixes in PR #438. TADA cluster (#336, #348) and macOS ARM import regressions (#287, #275, #304) need a dedicated triage pass.

**Qwen 0.6B-downloads-1.7B reports:** **#485** (2026-04-19), **#423** (macOS M1), **#329**. Platform-dependent:

- **On MLX (Apple Silicon) — not a bug.** `mlx-community` only publishes 1.7B-Base-bf16 weights, so the 0.6B Base option intentionally resolves to the same repo (`backend/backends/__init__.py:180` — `# 0.6B not available in MLX, falls back`). UX gap: the selector offers a size that doesn't exist on the active backend. Fix: (a) hide the 0.6B option on MLX, or (b) label it "0.6B (uses 1.7B on Apple Silicon)".
- **On PyTorch (Windows/Linux/CUDA/ROCm/XPU/CPU) — real bug if reported.** Both 0.6B and 1.7B have distinct repos (`Qwen/Qwen3-TTS-12Hz-0.6B-Base` vs `-1.7B-Base`). Triage each report by platform before merging into the MLX cluster.
- **Qwen CustomVoice (either platform)** — no fallback, both sizes always have dedicated repos.

### Language Requests (ongoing)

Strong demand: Hungarian (#479), Indonesian (#458, #247), Thai (#455), Bangla (#454), Arabic (#379), Persian (#162), IndicF5 (#339 — Indian languages), Ukrainian (#109), Chinese UI (#392, #261).

**Fix path:** Chatterbox Multilingual (PR #257) covers Arabic, Danish, German, Greek, Finnish, Hebrew, Hindi, Dutch, Norwegian, Polish, Swedish, Swahili, Turkish. Still missing: Hungarian, Indonesian, Thai, Bangla, Ukrainian. Issue #411 offers a PR for UI i18n foundation.

### New Model Requests (growing)

| Issue | Model Requested |
|-------|----------------|
| #478 | CosyVoice3 (we tried & abandoned CosyVoice2/3 — see #311) |
| #407, #347 | RVC-style voice-to-voice / seed voice conversion (STS) |
| #385 | Fish Audio S2 |
| #380 | OmniVoice |
| #370 | index-tts2 |
| #364 | Voxtral-TTS |
| #335 | Faster-Qwen-TTS |
| #346 | Multi-model batch request |
| #381 | Microsoft MAI models |
| #339 | IndicF5 |
| #226 | GGUF support |
| #172 | VibeVoice |
| #138 | Export to ONNX/Piper format |
| #132 | LavaSR (transcription) |
| #147 | Facebook Omnilingual ASR |
| #338 | Default voices |

The multi-engine architecture makes integration straightforward — see [`content/docs/developer/tts-engines.mdx`](content/docs/developer/tts-engines.mdx). Platform-specific gating (e.g. VoxCPM CUDA-only) doesn't exist yet and would need design.

### Platform Scope & Quality Debt — NEW category

Awareness issues filed this cycle — ties into engine sprawl and platform tier work.

- **#419** — Engine sprawl: define first-class vs experimental TTS backends
- **#420** — Formalize tier-1 vs tier-2 platform support targets (PR #465 open)
- **#421** — Track & burn down frontend Biome + a11y debt before gating CI
- **#422** — Code-split web build (main bundle > 1 MB)

### Long-Form / Chunking

Still reported despite chunking + queue being merged.

**Key issues:** #464 (50k char limit on GPU despite 16 GB VRAM — v0.4.0), #365 (FR: >50k chars), #363 (smart chunking to prevent robotic artifacts), #354 (50k limit v0.3.0).

**Fix path:** Chunking (#266) and queue (#269) shipped. Remaining work is raising/removing the 50k guard and tuning chunk boundaries for prosody.

### Feature Requests (ongoing)

Notable:
- **#480** — Noise removal on uploaded recordings
- **#448** — API for non-Qwen models (external integrations)
- **#427** — Task status control
- **#407, #347** — Voice-to-voice / audio-to-audio conversion
- **#387** — Location of downloaded generated voices
- **#383** — Concatenate partial reference audio into generated audio
- **#382** — Lightning.ai support
- **#376** — Remote mode
- **#353** — Audio transcoding
- **#317** — Voice pitch control
- **#189** — "Auto" language option
- **#173** — Vocal intonation/inflection control
- **#165, #270** — Audiobook mode (PR #154 open)
- **#242** — Seed value pinning
- **#228** — Always use 0.6B option
- **#235** — Finetuned Qwen3-TTS tokenizer (PR #253 open)
- **#144** — Copy text to clipboard

### Housekeeping / Triage Needed

| Issue | Reason |
|-------|--------|
| **#431**, **#408** | Spam — Chinese "free Claude API" promos. Close. |
| **#398** ("Excelente") | Non-issue. Close. |
| **#357** | Informational — project featured in Awesome MLX. Close after acknowledgement. |
| **#374**, **#377** | Version-release questions, no bug. Close. |
| **#306** ("voice model"), **#389** ("New model"), **#473** ("New functionality") | Title-only issues, no content. Request details or close. |
| **#309** | Uninstall/cleanup question. Answer and close. |
| **#241** | "How to use in Colab" — support question, not a bug. |
| **#423** / **#485** / **#329** | Platform-dependent. On MLX: not a bug (0.6B weights don't exist upstream, fallback is intentional — fix UX). On PyTorch: real bug if reproducible. Classify each by reporter's platform before deduping. |
| **#336** / **#348** | TADA download/registration cluster — triage together. |
| **#287** / **#275** / **#304** | macOS ARM import regressions on new version — likely one root cause. |
| **#292**, **#349** | Possibly already fixed by merged PRs (#321/#412 and #345). Verify + close. |

**~70 older issues (pre-#170) not individually categorized above.** Most are long-tail support questions or duplicates of problems now addressed by the multi-engine / model-registry work. A dedicated backlog-sweep pass is overdue.

### Bugs (ongoing)

| Category | Issues |
|----------|--------|
| Generation failures | #476, #467, #452, #459 (voice clone fetch error), #468 (tada-1b marked error), #437, #300, #301, #282 |
| Audio quality | #456 (clipping errors v0.4.0), #436 (emotion labels), #333 (pitch/echo), #307 (by-model breakdown), #340 (all generations say "www...") |
| Transcription | #371 (fails every time), #291 (extract transcription from generated audio) |
| Effects / presets | #349 ("Failed to save" when creating effects presets — possibly fixed by merged #345) |
| File ops | #477 (spacy_pkuseg dict missing on frozen Windows build), #472 (storage location change), #283 (allow longer files for voice creation + in-app trim), #350 (failed to add sample) |
| History | #292 (can't delete failed generations — possibly fixed by merged #321/#412) |
| Windows | #466 (install problem), #375 (WinError 5 access denied), #273 (port 8000 conflict), #201 (model doesn't stay loaded) |
| Linux | #471 (thread-safe PULSE_SOURCE), #413 (Arch build), #409 (Kubuntu build), #351, #341 |
| macOS | #441 (older macOS), #369 (malware flag), #334 (microphone permission), #287 (`check_model_inputs` ImportError — regression), #171 (ARM64 binary won't open) |
| Profile/UI | #360 (Kokoro profile hides others — partly addressed by auto-switch), #299 (drag-drop on Win11), #329 (size selector state bug), #393 (stuck loading screen after reinstall to new dir) |
| Integrations | #397 (SAMMI-bot 422 Unprocessable Entity) |
| Audio playback / session | **#41** (macOS: Voicebox goes silent after another app takes audio output; restart restores it) — see deep-dive below |
| Database | #174 (sqlite3 IntegrityError) |

---

## Existing Plan Documents — Status

| Document | Target Version | Status | Relevance |
|----------|---------------|--------|-----------|
| `TTS_PROVIDER_ARCHITECTURE.md` | v0.1.13 | **Partially superseded** by multi-engine arch + CUDA swap | Core concepts implemented differently than planned |
| `CUDA_BACKEND_SWAP.md` | — | **Shipped** (PR #252) | CUDA binary download + backend restart |
| `CUDA_BACKEND_SWAP_FINAL.md` | — | **Shipped** (PR #252) | Final implementation plan |
| `EXTERNAL_PROVIDERS.md` | v0.2.0 | **Not started** | Remote server support |
| `MLX_AUDIO.md` | — | **Shipped** | MLX backend is live |
| `DOCKER_DEPLOYMENT.md` | v0.2.0 | **Shipped** (PR #161) | Docker + web deployment |
| `OPENAI_SUPPORT.md` | v0.2.0 | **Not started** | OpenAI-compatible API layer |
| `PR33_CUDA_PROVIDER_REVIEW.md` | — | **Reference** | Analysis of the original provider approach |

---

## New Model Integration — Landscape

### Status Snapshot (2026-04-18)

| Model | Cloning | Speed | Sample Rate | Languages | VRAM | Instruct | Cross-platform? | Status |
|-------|---------|-------|-------------|-----------|------|----------|-----------------|--------|
| **Qwen3-TTS** | 10s zero-shot | Medium | 24 kHz | 10 | Medium | None | MLX + PyTorch | **Shipped** |
| **Qwen CustomVoice** | Preset speakers | Medium | 24 kHz | 10 | Medium | **Yes** | PyTorch | **Shipped** (PR #328) |
| **LuxTTS** | 3s zero-shot | 150x RT, CPU ok | 48 kHz | English | <1 GB | None | All | **Shipped** (PR #254) |
| **Chatterbox MTL** | 5s zero-shot | Medium | 24 kHz | 23 | Medium | Partial — `exaggeration` | CPU/CUDA | **Shipped** (PR #257) |
| **Chatterbox Turbo** | 5s zero-shot | Fast | 24 kHz | English | Low | Partial — inline tags | CPU/CUDA | **Shipped** (PR #258) |
| **HumeAI TADA 1B/3B** | Zero-shot | 5x faster than LLM-TTS | 24 kHz | EN (1B), 10 (3B) | Medium | Partial — prosody | PyTorch | **Shipped** (PR #296) |
| **Kokoro-82M** | Preset voices | CPU realtime | 24 kHz | 8 | Tiny (82M) | None | All | **Shipped** (PR #325) |
| ~~**CosyVoice2-0.5B**~~ | 3-10s zero-shot | Very fast | 24 kHz | Multilingual | Low | **Yes** | — | **Abandoned** (PR #311) — poor output quality |
| ~~**VoxCPM2**~~ | Zero-shot | ~0.15 RTF streaming | 48 kHz | 30 | Medium | Partial — parenthetical style | **CUDA-only in practice** | **Backlogged** (2026-04-18) — see notes above |
| **Fish Speech** | 10-30s few-shot | Real-time | 24-44 kHz | 50+ | Medium | **Yes** — word-level inline | All | Candidate — license TBD |
| **Fish Audio S2** | — | — | — | — | — | — | — | Candidate (#385) |
| **XTTS-v2** | 6s zero-shot | Mid-GPU | 24 kHz | 17+ | Medium | Partial — style transfer from ref | All | Candidate — CPML license likely blocker |
| **Pocket TTS** (Kyutai) | Zero-shot + streaming | >1x RT on CPU | — | English + several European (FR/DE/PT/IT/ES added by Feb 2026) | ~100M | None | CPU-first | Candidate — MIT |
| **MOSS-TTS-Nano** | Zero-shot | **Realtime on 4 CPU cores** | 48 kHz stereo | 20 | 0.1B | Partial — MOSS-VoiceGenerator companion does text-to-voice design | All (ONNX CPU path dropped 2026-04-17) | **Top candidate** — Apache 2.0, released 2026-04-13, streaming |
| **VibeVoice** (Microsoft) | — | — | — | Multi-speaker long-form (up to 90 min, 4 speakers) | 1.5B | — | — | Candidate (#172) — Stories-editor fit |
| **index-tts2** | — | — | — | — | — | — | — | Candidate (#370) |
| **Voxtral TTS** (Mistral) | Zero-shot (short clips) + 20 preset voices | Single-GPU | — | — | 4B (`Voxtral-4B-TTS-2603`) | Presets + cloning | CUDA (16 GB+ VRAM) | Candidate (#364) — frontier quality claim, open-weight |
| **Dia / Dia2** | — | — | — | — | — | — | — | Watch — emotion-forward, but "rough edges" / artifacts per April reviews |
| **IndicF5** | — | — | — | Indian languages | — | — | — | Candidate (#339) — fills Indic gap |
| **MiniMax Cloud TTS** | — | Cloud | — | — | N/A (API) | — | N/A | Community PR #430, #331 — new direction (external API) |
| **OmniVoice** | — | — | — | — | — | — | — | Candidate (#380) |
| **RVC voice conversion** | N/A (STS) | — | — | — | — | N/A | All | New modality, not TTS (#407, #347) |

**Watch list:** MioTTS-2.6B (fast LLM-based EN/JP, vLLM compatible), Oolel-Voices (Soynade Research, expressive modular control), Faster-Qwen-TTS (#335), Orpheus / Sesame CSM (on-device fine-tuning discussions), Fish Audio S2 Pro / Fish Speech V1.5 (benchmark leader but research/non-commercial license — same blocker as Fish Speech).

**Deep-research pass (2026-04-18):** MOSS-TTS-Nano identified as the freshest high-alignment candidate — verified via [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) README (0.1B params, Apache 2.0, 48 kHz stereo, 4-core CPU realtime, streaming, released 2026-04-13). Dedicated repo: [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano). Voxtral TTS verified on HF as `mistralai/Voxtral-4B-TTS-2603`.

#### Active Evaluation Criteria (learned from cycle)

1. **Cross-platform first.** MLX is a primary backend for our Apple Silicon user base. CUDA-only models require platform gating that doesn't exist yet — shipping one sets a precedent (see VoxCPM notes, issues #419/#420).
2. **PyPI + Apache/MIT licensing preferred.** Heavy deps, git-only installs, and `--no-deps` workarounds are expensive to maintain (Chatterbox taught us this).
3. **Output quality is non-negotiable.** CosyVoice was abandoned despite the best instruct API.
4. **Instruct support fills a real gap** (#173, #224, #303). Qwen CustomVoice partially addresses it with preset speakers; zero-shot clone-with-instruct is still unmet.
5. **Long-form + streaming are user-requested** (#363, #365, #464). Candidates with native streaming (Pocket TTS, Fish Speech) get extra weight.

### Adding a New Engine (Now Straightforward)

With the model config registry and shared `EngineModelSelector` component, adding a new TTS engine requires:

1. **Create `backend/backends/<engine>_backend.py`** — implement `TTSBackend` protocol (~200-300 lines)
2. **Register in `backend/backends/__init__.py`** — add `ModelConfig` entry + `TTS_ENGINES` entry + factory elif
3. **Update `backend/models.py`** — add engine name to regex
4. **Update frontend** — add to engine union type, `EngineModelSelector` options, form schema, language map, profile type gating (icons/labels ~9 files per grep of `kokoro`)

`main.py` requires **zero changes** — the registry handles all dispatch automatically.

**Platform gating doesn't exist yet.** If we add a CUDA-only model (e.g. VoxCPM), we need a new `requires_cuda` (or more generally `requires: list[device]`) flag on `ModelConfig`, plumbed through `/models` API and surfaced in `ModelManagement.tsx` and `EngineModelSelector.tsx` as a lock icon + "Requires NVIDIA GPU" state. Backend should hard-error at `load_model()` as a safety net.

Total effort: **~1 day** for a well-documented model with a PyPI package, cross-platform. **~2 days** if platform gating is required. See [`content/docs/developer/tts-engines.mdx`](content/docs/developer/tts-engines.mdx) for the full guide.

---

## Architectural Bottlenecks

### ~~1. Single Backend Singleton~~ — RESOLVED

The singleton TTS backend was replaced with a thread-safe per-engine registry in PR #254. Multiple engines can now be loaded simultaneously.

### ~~2. `main.py` Dispatch Point Duplication~~ — RESOLVED

Previously, each engine required updates to 6+ hardcoded dispatch maps across `main.py` (~320 lines of if/elif chains). A model config registry in `backend/backends/__init__.py` now centralizes all model metadata (`ModelConfig` dataclass) with helper functions (`load_engine_model()`, `check_model_loaded()`, `engine_needs_trim()`, etc.). Adding a new engine requires zero changes to `main.py`.

### ~~3. Model Config is Scattered~~ — RESOLVED

Model identifiers, HF repo IDs, display names, and engine metadata are now consolidated in the `ModelConfig` registry. Backend-aware branching (e.g. MLX vs PyTorch Qwen repo IDs) happens inside the registry. Frontend model options are centralized in `EngineModelSelector.tsx`.

### 4. Voice Prompt Cache Assumes PyTorch Tensors

`backend/utils/cache.py` uses `torch.save()` / `torch.load()`. LuxTTS, Chatterbox, and Kokoro backends work around this by storing reference audio paths (or preset voice IDs) instead of tensors in their voice prompt dicts. Not ideal but functional.

### 5. ~~Frontend Assumes Qwen Model Sizes~~ — RESOLVED

The generation form now uses a flat model dropdown with engine-based routing. Per-engine language filtering is in place. Model size is only sent for Qwen / Qwen CustomVoice.

### 6. No Platform Gating on Models — NEW

`ModelConfig` has no way to express hardware requirements. Every engine is shown to every user, regardless of whether it'll actually load. Users on non-CUDA platforms discover failure at load time (or not at all — some fall back silently to CPU and never complete). Blocks shipping CUDA-only engines (VoxCPM) and would improve the Intel Arc / ROCm / CPU-only UX today. See `ModelConfig` TODO: add `requires: list[Literal["cuda", "mps", "xpu", "cpu", "rocm"]]` or equivalent, plumb through `/models` API, render in `ModelManagement.tsx` + `EngineModelSelector.tsx`.

### 7. Engine Sprawl — NEW

Seven TTS engines shipped, more candidates queued. Issue #419 asks for a first-class vs experimental distinction. Related: issue #420 asks for formalized platform support tiers. Combined, these would let us ship more engines more confidently with clearer expectations for users.

---

## Recommended Priorities

### Tier 1 — Ship Now

| Priority | PR/Item | Impact | Effort |
|----------|---------|--------|--------|
| 1 | **RTX 50-series / Blackwell diagnostic** — detect stale CUDA binary vs GPU arch, prompt re-download (#417, #400, #396, #395, #390, #362) | Large cluster of user-blocking errors | Medium |
| 2 | **CustomVoice download failures** (#475, #445) | New engine blocked on MAC/Win — regression triage | Medium |
| 3 | **50k char limit on GPU** (#464) | Regression — chunking should handle this | Medium |
| 4 | Close PR #311 (CosyVoice) and dedupe #331/#430 (MiniMax) | Housekeeping | None |
| 5 | **PR #443** — infinite offline retry loop | Bug fix, reviewable | Low |
| 6 | **PR #465** — define tier-1 / tier-2 platforms | Unblocks engine-sprawl decision (#419) | Low |
| 7 | **PR #463** — docker registry auto-publish | Community PR, low risk | Low |
| 8 | **#253** — 48kHz speech tokenizer | Quality improvement for Qwen | Medium |
| 9 | **Kokoro profile UX** (#360) — partially addressed by auto-switch | Polish | Low |

### Tier 2 — Feature Work

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | **Engine tier system** (#419) — first-class vs experimental, platform gating in `ModelConfig` | Unblocks CUDA-only engines (VoxCPM, etc.) and frontend polish | Medium |
| 2 | **Frontend tech-debt burn-down** (#421) + code-split (#422) | Before gating CI on Biome | Medium |
| 3 | **#154** — Audiobook tab | Long-form users. Chunking + queue shipped. | Medium |
| 4 | **UI i18n** (#411 PR offer, #392, #261) | Chinese UI + general localization | Medium |
| 5 | **#225** — Custom HuggingFace models | User-supplied models. Needs rework. | High |
| 6 | OpenAI-compatible API (plan doc exists) — see also #448 (API for non-Qwen) | Low effort once API is stable | Low |
| 7 | LoRA fine-tuning (PR #195) | Complex, needs rework for multi-engine | Very High |
| 8 | Streaming for non-MLX engines | Currently MLX-only | Medium |
| 9 | Voice-to-voice / RVC (#407, #347) | New modality — different arch shape | High |

### Tier 3 — Future Engines (cross-platform preferred)

| Priority | Item | Notes |
|----------|------|-------|
| 1 | **MOSS-TTS-Nano** | 0.1B, Apache 2.0, 4-core CPU realtime, 48 kHz stereo, streaming, 20 langs, released 2026-04-13. Best alignment with our criteria. Verify install ergonomics before committing. |
| 2 | **Pocket TTS** (Kyutai) | CPU-first 100M model. MIT. Fills streaming gap without CUDA dependency. Several European langs added by Feb 2026. |
| 3 | **IndicF5** | Fills Indian-language gap (#339). Closes many language-request issues. |
| 4 | **VibeVoice** (Microsoft, #172) | 1.5B, long-form multi-speaker (up to 90 min, 4 speakers). Strong Stories-editor fit. |
| 5 | **Voxtral TTS** (Mistral, #364) | 4B presets+cloning. Frontier quality claim, but 16 GB+ VRAM — would need the platform-tier work first. |
| 6 | **Fish Speech / Fish Audio S2** | 50+ langs, word-level instruct. **License clarification first.** (#385) |
| 7 | **XTTS-v2** | 17+ langs, mature pip. CPML likely kills commercial use — verify. |
| 8 | **index-tts2** (#370) | Unvetted. |
| — | ~~**VoxCPM2**~~ | **Backlogged** — CUDA-only upstream. Revisit when tier system ships or MPS bugs are fixed upstream. |

### ~~Previously Prioritized — Now Done~~

- ~~Kokoro 82M — finish integration~~ **Shipped** (PR #325)
- ~~Qwen CustomVoice~~ **Shipped** (PR #328)
- ~~Intel Arc (XPU) support~~ **Shipped** (PR #320)
- ~~Blackwell CUDA~~ **Shipped** (PR #401, follow-up work open)
- ~~Generation cancellation~~ **Shipped** (PR #444)
- ~~macOS Intel x86_64~~ **Shipped** (PR #416)

---

## Branch Inventory

| Branch | PR | Status | Notes |
|--------|-----|--------|-------|
| `voicebox-new-models` | — | **Active** | New model research (Fish Speech, Pocket TTS, VibeVoice, etc.); VoxCPM evaluated & backlogged |
| `fix/kokoro-pyinstaller-source-files` | — | Active | Kokoro frozen-build source bundling (parent of `voicebox-new-models`) |
| `feat/cosyvoice-engine` | #311 | Open — closing | CosyVoice2/3 — abandoned, poor quality |
| `feat/kokoro` | #325 | **Merged** | Kokoro 82M + voice profile type system |
| `feat/qwen-custom-voice` | #328 | **Merged** | Qwen CustomVoice preset engine |
| `feat/chatterbox-turbo` | #258 | **Merged** | Chatterbox Turbo + per-engine languages |
| `feat/chatterbox` | #257 | **Merged** | Chatterbox Multilingual |
| `feat/luxtts` | #254 | **Merged** | LuxTTS + multi-engine arch |

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
| `/models/status` | GET | All model statuses (Qwen, LuxTTS, Chatterbox, Chatterbox Turbo, TADA, Whisper) |
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
