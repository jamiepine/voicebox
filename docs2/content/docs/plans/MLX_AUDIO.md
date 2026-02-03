---
title: "MLX Audio Integration"
description: "MLX Audio integration for Voicebox (Validated)"
---

**Status:** Validated ✅
**Context:** [mlx-audio v0.3.1 release](https://github.com/Blaizzy/mlx-audio)

## Validation Results

We validated mlx-audio in an isolated environment (`mlx-test/`). Key findings:

| Metric          | Result                                      |
| --------------- | ------------------------------------------- |
| MLX Version     | 0.30.4                                      |
| Model Load Time | ~1s (after initial download)                |
| Generation RTF  | **0.5-0.6x** (1.7-2x faster than real-time) |
| Test Hardware   | Apple Silicon Mac                           |

### Model Mapping

| voicebox (PyTorch)              | mlx-audio (MLX)                               |
| ------------------------------- | --------------------------------------------- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | (not yet converted)                           |

### mlx-audio API

The API uses a **generator-based streaming pattern**:

```python
from mlx_audio.tts import load

model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")

# generate() yields GenerationResult objects
for result in model.generate("Hello world"):
    audio = result.audio           # numpy array of samples
    sample_rate = result.sample_rate  # 24000
    rtf = result.real_time_factor  # e.g., 0.55
```

### Known Warnings (harmless)

```
You are using a model of type qwen3_tts to instantiate a model of type .
The tokenizer you are loading... with an incorrect regex pattern...
```

These warnings appear but don't affect functionality or output quality.

### Demo Script

Run `mlx-test/demo.py` to test:

```bash
cd mlx-test && source venv/bin/activate && python demo.py "Your text here"
```

## Problem

Apple Silicon users are stuck on CPU inference while Windows and Linux users get CUDA acceleration. The current PyTorch MPS backend has stability issues (lines 34-36 in `backend/tts.py` and `backend/transcribe.py`), forcing a CPU fallback that makes voicebox significantly slower on M1/M2/M3 Macs.

This creates a poor experience for a large portion of users who bought Apple Silicon specifically for ML workloads.

## Solution

Integrate [mlx-audio](https://github.com/Blaizzy/mlx-audio) as the inference engine for macOS Apple Silicon builds. MLX is Apple's native ML framework, optimized for Metal and the unified memory architecture. It's fast, stable, and already supports the same Qwen3-TTS models we use.

**Key wins:**

- Native GPU acceleration on Apple Silicon (no more CPU fallback)
- Streaming TTS support (faster perceived latency)
- Memory optimizations (run larger models on less RAM)
- Fixed 0.6B silence bug that we currently ship
- Same Qwen3-TTS models (zero migration cost for users)

## Architecture

### Current Stack

```
┌─────────────────────────┐
│   PyTorch + Qwen3-TTS   │
│   (CPU only on macOS)   │
└─────────────────────────┘
```

### Proposed Stack

```
┌─────────────────────────────────────────┐
│  Platform Detection at Runtime          │
└─────────────────────────────────────────┘
           │
           ├─── Apple Silicon (aarch64-darwin)
           │    ┌─────────────────────────┐
           │    │  MLX Audio Backend      │
           │    │  - Qwen3-TTS (mlx)      │
           │    │  - Whisper (mlx)        │
           │    │  - Streaming support    │
           │    └─────────────────────────┘
           │
           └─── Other (x86_64, Windows, Linux)
                ┌─────────────────────────┐
                │  PyTorch Backend        │
                │  - Qwen3-TTS (pytorch)  │
                │  - Whisper (pytorch)    │
                │  - CUDA if available    │
                └─────────────────────────┘
```

## Implementation Phases

### Phase 1: Platform Detection & Dependency Management

Create a backend that switches between PyTorch and MLX based on runtime platform detection.

**New files:**

- `backend/platform.py` - Detect Apple Silicon, return backend type
- `backend/backends/__init__.py` - Backend factory pattern
- `backend/requirements-mlx.txt` - MLX-specific deps (macOS only)

**Modified files:**

- `backend/requirements.txt` - Keep PyTorch as default
- `backend/main.py` - Import from backend factory instead of direct imports

**Platform detection logic:**

```python
def get_backend_type() -> str:
    """Detect best backend for current platform."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon detected
        try:
            import mlx
            return "mlx"
        except ImportError:
            return "pytorch"  # Fallback if mlx not installed
    return "pytorch"
```

### Phase 2: MLX Backend Implementation

Create parallel implementations of TTS and STT using mlx-audio.

**New files:**

- `backend/backends/mlx_backend.py` - MLX inference engine
- `backend/backends/pytorch_backend.py` - Refactor current code into backend

**Interface both backends must implement:**

```python
class TTSBackend(Protocol):
    async def load_model(self, model_size: str) -> None: ...
    async def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict: ...
    async def generate(self, text: str, voice_prompt: dict, **kwargs) -> Tuple[np.ndarray, int]: ...
    async def generate_streaming(self, text: str, voice_prompt: dict, **kwargs) -> AsyncIterator[bytes]: ...
    def unload_model(self) -> None: ...

class STTBackend(Protocol):
    async def load_model(self, model_size: str) -> None: ...
    async def transcribe(self, audio_path: str, language: Optional[str]) -> str: ...
    def unload_model(self) -> None: ...
```

**MLX backend implementation notes:**

mlx-audio's `generate()` returns a generator by default (streaming is built-in):

```python
# MLX backend wrapper
from mlx_audio.tts import load

class MLXTTSBackend:
    def __init__(self):
        self.model = None

    async def load_model(self, model_size: str) -> None:
        model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            # "0.6B": needs conversion to mlx format
        }
        self.model = load(model_map[model_size])

    async def generate(self, text: str, voice_prompt: dict, **kwargs) -> Tuple[np.ndarray, int]:
        # Collect all chunks from generator
        chunks = []
        for result in self.model.generate(text):  # TODO: add voice_prompt support
            chunks.append(np.array(result.audio))
        return np.concatenate(chunks), 24000
```

**MLX-specific features to expose:**

- Streaming TTS (new endpoint: `/api/generate/stream`)
- Memory-optimized model loading
- Qwen3-ASR for transcription (in addition to Whisper)

### Phase 3: API Layer Updates

Update FastAPI endpoints to support new streaming capabilities and maintain backward compatibility.

**Modified files:**

- `backend/main.py` - Add streaming endpoints
- `backend/tts.py` - Refactor to use backend abstraction
- `backend/transcribe.py` - Refactor to use backend abstraction

**New endpoints:**

```python
@app.post("/api/generate/stream")
async def generate_stream(...) -> StreamingResponse:
    """Stream TTS chunks as they're generated (MLX only)."""
    backend = get_backend()
    if not hasattr(backend, 'generate_streaming'):
        raise HTTPException(501, "Streaming not supported on this backend")
    return StreamingResponse(backend.generate_streaming(...), media_type="audio/wav")
```

**Backward compatibility:**

- Keep all existing `/api/generate` endpoints unchanged
- PyTorch backend users see no behavior change
- MLX users automatically get faster inference, streaming is opt-in

### Phase 4: Frontend Integration

Add UI indicators for backend type and streaming progress.

**Modified files:**

- `app/src/hooks/useGenerationForm.tsx` - Add streaming support
- `app/src/components/GenerationForm.tsx` - Show backend badge, streaming toggle
- `app/src/lib/api.ts` - Add streaming API client

**UI additions:**

- Badge showing current backend ("MLX" or "PyTorch")
- Toggle for streaming mode (disabled if PyTorch)
- Real-time streaming playback (WaveSurfer progressive loading)

### Phase 5: Build & Distribution

Create separate installers for MLX (Apple Silicon) and PyTorch (Universal).

**Modified files:**

- `tauri/src-tauri/tauri.conf.json` - Add target-specific builds
- `.github/workflows/release.yml` - Build both variants

**Build matrix:**

```yaml
- target: aarch64-apple-darwin
  backend: mlx
  installer: voicebox-macos-silicon-{version}.dmg

- target: x86_64-apple-darwin
  backend: pytorch
  installer: voicebox-macos-intel-{version}.dmg

- target: x86_64-pc-windows-msvc
  backend: pytorch
  installer: voicebox-windows-{version}.exe
```

**Installation flow:**

- Auto-detect architecture, recommend correct installer
- MLX installer includes `mlx-audio` in embedded Python
- PyTorch installer includes `torch` in embedded Python
- Both can coexist (different backend, same profile format)

### Phase 6: Testing & Validation

Ensure both backends produce compatible outputs.

**New files:**

- `backend/tests/test_backend_parity.py` - Verify both backends produce similar audio
- `backend/tests/test_streaming.py` - Streaming-specific tests

**Test scenarios:**

- Same voice prompt on both backends → similar (not identical) audio output
- Profile created on MLX → loads on PyTorch (and vice versa)
- Streaming chunks assemble into valid WAV file
- Model downloads work on both backends
- Memory usage stays within bounds

### Phase 7: Documentation

Update user-facing docs and developer guides.

**New files:**

- `docs/developer/BACKENDS.md` - Guide for adding new backends
- `docs/overview/performance.md` - Backend comparison benchmarks

**Modified files:**

- `README.md` - Note Apple Silicon acceleration
- `docs/TROUBLESHOOTING.md` - Add MLX-specific issues

**Key docs to write:**

- Which installer to download (architecture detection)
- Performance comparison (MLX vs PyTorch on same M2 hardware)
- How streaming mode works
- How to force PyTorch on Apple Silicon (for debugging)

## Technical Decisions

### Why Dual Backend Instead of MLX-Only?

**Pros of dual backend:**

- Windows and Intel Mac users unaffected
- Easier testing (can compare outputs)
- Fallback if MLX has issues

**Cons of dual backend:**

- More code to maintain
- Two dependency trees
- Build complexity (separate installers)

**Decision:** Dual backend. The maintenance cost is worth it to avoid breaking existing users and to have a fallback.

### Why Separate Installers Instead of Runtime Detection?

**Pros of separate installers:**

- Smaller bundle size (don't ship both PyTorch and MLX)
- Clearer to users which version they have
- Easier to debug (no "which backend am I running?" confusion)
- Can optimize each build for its target

**Cons:**

- More installers to build and test
- Users might download the wrong one

**Decision:** Separate installers. Bundle size matters (PyTorch + MLX would be huge), and we can auto-detect architecture on the download page.

### Streaming vs Batch Generation

MLX supports streaming, PyTorch doesn't (without significant work). Should streaming be:

1. MLX-only feature (✅ chosen)
2. Implemented for both (lots of work)
3. Not exposed at all (wasted opportunity)

**Decision:** MLX-only. Expose as opt-in feature with graceful degradation (button disabled on PyTorch backend).

## Migration Path

Nothing needs migrating, macos users will just notice a speed-boost in inference

**Data format compatibility:**

- Profiles (SQLite) → no schema changes needed
- Voice prompts (cached) → backend-agnostic (just numpy arrays)
- Audio files → unchanged

## Performance Expectations

### Measured Results (from validation)

| Metric                  | MLX (measured) | PyTorch CPU (estimated) |
| ----------------------- | -------------- | ----------------------- |
| **6s audio generation** | ~3-4s          | ~10-15s                 |
| **Real-time factor**    | 0.5-0.6x       | 2-3x                    |
| **Model load (cached)** | ~1s            | ~3-5s                   |

### TTS Generation (1.7B model, ~20s output)

- **PyTorch CPU (M2 Max):** ~45-60s (slower than real-time)
- **MLX (M2 Max):** ~8-12s (faster than real-time)
- **Improvement:** ~4-5x faster

### Whisper Transcription (10s audio clip)

- **PyTorch CPU:** ~5-8s
- **MLX:** ~1-2s
- **Improvement:** ~3-4x faster

### Memory Usage (1.7B model)

- **PyTorch:** ~8-10GB (no GPU offload, so CPU RAM)
- **MLX:** ~4-6GB (unified memory, better optimization)
- **Improvement:** ~40% less RAM

Full benchmarks will be in `docs/overview/performance.md` after Phase 6.

## Open Questions

- **Should we support Qwen3-ASR (MLX-only) in addition to Whisper?** Adds another model option but increases complexity. Probably phase 8+. - Sure
- **Should we backport streaming to PyTorch?** Would require chunking and callback-based generation. Probably not worth it given mlx-audio already has it. - No
- **What's the auto-update UX for migrating PyTorch→MLX users?** Needs design. Don't want to force reinstall, but also want to make upgrade obvious. - it just updates, users see nothing
- **Do we expose backend selection in settings or hide it?** Leaning toward auto-detect only, with env var override for power users.

## Success Metrics

How we'll know this worked:

1. **Performance:** Apple Silicon users report generation faster than real-time
2. **Adoption:** >80% of macOS downloads are MLX build within 1 month
3. **Stability:** <5% increase in bug reports (backend abstraction doesn't introduce regressions)
4. **Feedback:** Positive sentiment in Discord/GitHub about macOS performance

## Related Work

- [PyTorch MPS tracking issue](https://github.com/pytorch/pytorch/issues/77764) - Why we can't use MPS directly
- [mlx-audio server implementation](https://github.com/Blaizzy/mlx-audio/blob/main/examples/server.py) - Reference for streaming API
- [MLX Whisper benchmarks](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Performance data

## Next Steps

1. ~~Validate mlx-audio can load Qwen3-TTS models (quick test)~~ ✅ Done - see `mlx-test/`
2. Get approval on dual-backend architecture
3. Start Phase 1 (platform detection)

## Questions?

Feedback welcome in GitHub discussions or Discord.
