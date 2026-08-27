# Implementation Plan: MiniCPM5-1B LLM Engine Support

**Branch**: `001-minicpm5-llm-engine` | **Date**: 2026-08-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-minicpm5-llm-engine/spec.md`

**Baseline note**: Re-assessed against `origin/main` at commit `34b8433` (fast-forwarded 2026-08-26, no rebase needed — this branch had zero unique commits). One required addition surfaced: `MLXMiniCPMLLMBackend.unload_model()` must call the newly-added `empty_mlx_cache()` helper, matching a fix just landed for `MLXQwenLLMBackend`. No other decision in this plan changed. Full detail in research.md's "Baseline update" section.

## Summary

Add MiniCPM5-1B as a second local LLM engine (`minicpm_llm`) alongside the existing single-engine Qwen3 LLM (`qwen_llm`), with full UI parity (settings picker, model management page, Compose/Rewrite/Refine). The core technical move is generalizing the LLM subsystem's identifier from a bare Qwen3 size string (`"0.6B"`) to the codebase's existing cross-engine-unique `ModelConfig.model_name` (`"qwen3-0.6b"`, `"minicpm5-1b"`), which is already how the TTS side of this app handles multiple engines. MiniCPM5-1B itself loads via the exact same code shape as Qwen3 (plain `LlamaForCausalLM`, `transformers.AutoModelForCausalLM` / `mlx_lm.load`, same `enable_thinking` chat-template kwarg) — confirmed via research, not assumed. A new cross-engine unload guard is required (see data-model.md) because today's "switching model = unload the old one" behavior is an accident of both Qwen3 sizes sharing one Python object; it does not extend to a second engine's separate singleton without an explicit fix.

## Technical Context

**Language/Version**: Python 3.x (FastAPI backend, PyInstaller-bundled), TypeScript/React (Tauri desktop frontend)

**Primary Dependencies**: `transformers` (`AutoModelForCausalLM`/`AutoTokenizer`), `mlx` + `mlx-lm` (Apple Silicon path), FastAPI, SQLAlchemy, Pydantic v2, React + the app's existing settings-hook layer (`useSettings.ts`)

**Storage**: SQLite (single-user desktop app, one file per install), schema/migrations managed by hand-rolled idempotent `_migrate_*` functions in `backend/database/migrations.py` (no Alembic — see file header rationale)

**Testing**: pytest for backend (`backend/tests/`); no dedicated unit tests exist today for `qwen_llm_backend.py` or the LLM entries in the config registry (see research.md Decision 6) — new pytest coverage follows the plain-unit-test style of `test_model_status_pending_downloads.py`. No frontend test infra exists for the equivalent Qwen3 picker UI; none is added for parity per spec.md's non-goals.

**Target Platform**: macOS (Apple Silicon → MLX path), macOS/Windows/Linux general case (PyTorch CPU/GPU path) — same platform matrix Qwen3 already runs on, no new platform requirement

**Project Type**: Desktop app — Tauri frontend (`app/`) + FastAPI backend (`backend/`), packaged as a single binary

**Performance Goals**: N/A beyond "loads and generates at a speed comparable to Qwen3 0.6B/1.7B on the same hardware" — no new performance target introduced by this feature; MiniCPM5-1B's own inference speed is a property of the model, not something this plan can tune

**Constraints**: `mlx-lm>=0.31` required for MiniCPM5-1B's dual-`eos_token_id` handling (not currently pinned anywhere in the repo — see research.md); official MLX weights only, no community quant, per explicit user decision; single global "active LLM" (no per-request engine matrix beyond what already exists for model_size)

**Scale/Scope**: One new backend module, ~8 modified backend files, ~4 modified frontend files, one new migration function. No new services, no new external integrations beyond one new Hugging Face repo pair.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` is still the unfilled template for this project — no project-specific principles have been ratified yet. No constitution-specific gates apply beyond the global development rules already governing this session (YAGNI/KISS, explicit error handling at boundaries, function-header comments, no speculative abstraction). This plan complies with those: it introduces exactly one new abstraction point (the cross-engine unload guard in `get_llm_model()`), justified by an explicit functional requirement (FR-005) that did not exist before a second engine did, not a hypothetical future need.

**Re-check after Phase 1 design**: No new violations surfaced during data-model/contracts design. The one addition beyond a pure mirror-of-Qwen3 (the cross-engine unload guard) is justified in [data-model.md](data-model.md)'s "State / relationships" section with a concrete failure scenario it prevents (double-loaded models holding memory simultaneously), not spec-work for its own sake.

## Project Structure

### Documentation (this feature)

```text
specs/001-minicpm5-llm-engine/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── llm-model-selection.md
└── tasks.md             # Phase 2 output (/speckit-tasks — not created by this command)
```

### Source Code (repository root)

```text
backend/
├── backends/
│   ├── __init__.py              # MODIFIED: LLM_ENGINES, _get_minicpm_llm_configs, get_llm_model_configs,
│   │                             #   get_llm_backend_for_engine, unload_model_by_config/check_model_loaded/
│   │                             #   get_model_load_func generalized off qwen_llm-only branches
│   ├── qwen_llm_backend.py       # MODIFIED: _progress_name generalized to be engine-aware (or superseded
│   │                             #   by keying progress off ModelConfig.model_name directly — decide in tasks.md)
│   └── minicpm_llm_backend.py    # NEW: PyTorchMiniCPMLLMBackend, MLXMiniCPMLLMBackend
├── services/
│   ├── llm.py                    # MODIFIED: get_llm_model(engine=...)/unload_llm_model(engine=...) +
│   │                             #   cross-engine unload guard
│   ├── personality.py            # MODIFIED: resolve model_name -> (engine, model_size) before get_llm_model()
│   └── refinement.py             # MODIFIED: same resolution
├── routes/
│   ├── llm.py                    # MODIFIED: model_name-based validation + engine resolution, progress naming
│   └── captures.py                # MODIFIED: model_size == saved.llm_model -> model_name == saved.llm_model
├── models.py                      # MODIFIED: static regex fields -> dynamic membership validator (3 fields)
├── database/
│   ├── models.py                  # MODIFIED: llm_model column default "0.6B" -> "qwen3-0.6b"
│   └── migrations.py              # MODIFIED: new _migrate_llm_model_names, wired into run_migrations()
├── requirements-mlx.txt           # MODIFIED: explicit mlx-lm>=0.31 pin
└── tests/
    ├── test_minicpm_llm_backend.py        # NEW
    ├── test_llm_backend_registry.py       # NEW (or extend an existing backends/__init__ test if one exists)
    └── test_migrate_llm_model_names.py    # NEW

app/src/
├── lib/api/types.ts               # MODIFIED: Qwen3ModelSize -> broadened model_name union covering both engines
└── components/
    ├── ServerTab/CapturesPage.tsx         # MODIFIED: picker options + type
    ├── CapturesTab/CapturesTab.tsx        # MODIFIED: picker options + type
    └── ServerSettings/ModelManagement.tsx  # MODIFIED: language-models filter includes minicpm5
```

**Structure Decision**: No new top-level directories or projects. This is a same-shape addition within the existing `backend/backends/` (one-file-per-engine convention) and `app/src/components/` structure — Option 2 (web/desktop app with `backend/` + `app/` frontend) already matches this repo's existing layout, so no structural decision beyond "keep using the layout that's already there" is needed.

## Complexity Tracking

*No constitution violations requiring justification. The one new piece of logic beyond mirroring Qwen3 (the cross-engine unload guard in `backend/services/llm.py: get_llm_model()`) is not a "violation" of any stated principle — it's the minimal fix required to keep FR-005 true once a second engine exists, and it lives at the single existing chokepoint every caller already passes through rather than introducing a new abstraction layer.*
