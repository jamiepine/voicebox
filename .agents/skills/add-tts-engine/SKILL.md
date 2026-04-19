---
name: add-tts-engine
description: Use this skill to add a new TTS engine to Voicebox. It walks through dependency research, backend implementation, frontend wiring, PyInstaller bundling, and frozen-build testing. Always start with Phase 0 (dependency audit) before writing any code.
---

# Add TTS Engine

## Goal

Integrate a new text-to-speech engine into Voicebox end-to-end: dependency research, backend protocol implementation, frontend UI wiring, PyInstaller bundling, and frozen-build verification. The user should only need to test the final build locally.

## Reference Doc

The full phased guide lives at `docs/content/docs/developer/tts-engines.mdx`. **Read this file in its entirety before starting.** It contains:

- Phase 0: Dependency research (mandatory before writing code)
- Phase 1: Backend implementation (`TTSBackend` protocol)
- Phase 2: Route and service integration (usually zero changes)
- Phase 3: Frontend integration (5 files)
- Phase 4: Dependencies (`requirements.txt`, justfile, CI, Docker)
- Phase 5: PyInstaller bundling (`build_binary.py` + `server.py`)
- Phase 6: Common upstream workarounds
- Implementation checklist (gate between phases)

## Workflow

### 1. Read the guide

```bash
# Read the full TTS engines doc
cat docs/content/docs/developer/tts-engines.mdx
```

Internalize all phases, especially Phase 0 and Phase 5. The v0.2.3 release was three patch releases because Phase 0 was skipped.

### 2. Dependency research (Phase 0)

Clone the model library into a temporary directory and audit it. Do NOT skip this.

```bash
mkdir /tmp/engine-research && cd /tmp/engine-research
git clone <model-library-url>
```

Run the grep searches from Phase 0.2 in the guide against the cloned source and its transitive dependencies. Produce a written dependency audit covering:

1. PyPI vs non-PyPI packages
2. PyInstaller directives needed (`--collect-all`, `--copy-metadata`, `--hidden-import`)
3. Runtime data files that must be bundled
4. Native library paths that need env var overrides in frozen builds
5. Monkey-patches needed (`torch.load`, float64, MPS, HF token)
6. Sample rate
7. Model download method (`from_pretrained` vs `snapshot_download` + `from_local`)

Test model loading and generation on CPU in the throwaway venv before proceeding.

### 3. Implement (Phases 1–4)

Follow the guide's phases in order. Key files to modify:

**Backend (Phase 1):**
- Create `backend/backends/<engine>_backend.py`
- Register in `backend/backends/__init__.py` (ModelConfig + TTS_ENGINES + factory)
- Update regex in `backend/models.py`

**Frontend (Phase 3):**
- `app/src/lib/api/types.ts` — engine union type
- `app/src/lib/constants/languages.ts` — ENGINE_LANGUAGES
- `app/src/components/Generation/EngineModelSelector.tsx` — ENGINE_OPTIONS, ENGINE_DESCRIPTIONS
- `app/src/lib/hooks/useGenerationForm.ts` — Zod schema, model-name mapping
- `app/src/components/ServerSettings/ModelManagement.tsx` — MODEL_DESCRIPTIONS

**Dependencies (Phase 4):**
- `backend/requirements.txt`
- `justfile` (setup-python, setup-python-release targets)
- `.github/workflows/release.yml`
- `Dockerfile` (if applicable)

### 4. PyInstaller bundling (Phase 5)

Register the engine in `backend/build_binary.py`:
- `--hidden-import` for the backend module and model package
- `--collect-all` for packages using `inspect.getsource`, shipping data files, or native libraries
- `--copy-metadata` for packages using `importlib.metadata`

If the engine has native data paths, add `os.environ.setdefault()` in `backend/server.py` inside the `if getattr(sys, 'frozen', False):` block.

### 5. Verify in dev mode

```bash
just dev
```

Test the full chain: model download → load → generate → voice cloning.

### 6. Use the checklist

Walk through the Implementation Checklist at the bottom of `tts-engines.mdx`. Every item must be checked before handing the build to the user.

## Key Lessons (from v0.2.3)

These are the most common failure modes. Phase 0 research catches all of them:

| Pattern | Symptom in Frozen Build | Fix |
|---------|------------------------|-----|
| `@typechecked` / `inspect.getsource()` | "could not get source code" | `--collect-all <package>` |
| Package ships pretrained model files | `FileNotFoundError` for `.pth.tar`, `.yaml` | `--collect-all <package>` |
| C library with hardcoded system paths | `FileNotFoundError` for `/usr/share/...` | `--collect-all` + env var in `server.py` |
| `importlib.metadata.version()` | "No package metadata found" | `--copy-metadata <package>` |
| `torch.load` without `map_location` | CUDA device not available on CPU build | Monkey-patch `torch.load` |
| `torch.from_numpy` on float64 data | dtype mismatch RuntimeError | Cast to `.float()` |
| `token=True` in HF download calls | Auth failure without stored HF token | Use `snapshot_download(token=None)` + `from_local()` |

## Notes

- The route and service layers have zero per-engine dispatch points. `main.py` requires zero changes.
- The model config registry in `backends/__init__.py` handles all dispatch automatically.
- Use `get_torch_device()` and `model_load_progress()` from `backends/base.py` — don't reimplement device detection or progress tracking.
- Always test with a **clean HuggingFace cache** (no pre-downloaded models from dev).
- Do NOT push or create a release. Hand the build to the user for local testing.
