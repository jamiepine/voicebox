# Implementation Plan: ModelScope Download Source

**Branch**: `001-modelscope-download-source` | **Date**: 2026-08-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-modelscope-download-source/spec.md`

## Summary

Let users pick a model download source (HuggingFace / ModelScope) in
Settings so mainland-China users can download models directly without a
VPN. ModelScope uses the official `modelscope` SDK to download the 13/17
model entries that have a verified ModelScope mirror into a Voicebox-managed
local directory, from which backends load without any network access
(reusing the existing `force_offline_if_cached` mechanism); the 4 entries
without a mirror (Chatterbox ×2, TADA ×2) download directly from
huggingface.co, identical to today, even when ModelScope is selected. The
setting is stored in a small JSON file (not the DB, still simpler than a
schema for one string) and is read fresh on every download, so it applies
live — no restart needed.

**Amendment (2026-08-27)**: The original plan also had a third source, "HF
Mirror" (`HF_ENDPOINT=https://hf-mirror.com`), both as its own option and as
the fallback for the 4 unmirrored models — with a restart-required UX
because applying it meant mutating a process env var at startup. Real-world
verification (running the actual backend, not mocks) found `hf-mirror.com`
now just 308-redirects to the real huggingface.co without proxying content,
which `huggingface_hub` refuses to trust and which would fail for genuinely
blocked users anyway. Removed rather than shipped unverified — see
research.md §2 and §4 for the full account. This plan reflects the
resulting two-source, live-apply design.

## Technical Context

**Language/Version**: Python 3.12 (backend), existing FastAPI app — no new language/runtime

**Primary Dependencies**: `huggingface_hub` (existing, unchanged), `modelscope` (new — official ModelScope Python SDK)

**Storage**: SQLite (existing, unused by this feature) + a new flat JSON file (`data/model_source.json`), read live on every download (no caching/apply-at-startup); models downloaded via ModelScope land under `data/models/modelscope/<safe-repo-id>/`

**Testing**: pytest / pytest-asyncio (existing `backend/tests/`)

**Target Platform**: Same as the rest of the backend — macOS (MLX/Metal), Windows (CUDA), Linux, AMD ROCm, Intel Arc, Docker

**Project Type**: Existing web application (FastAPI backend + Tauri/React frontend) — this feature is backend-plus-Settings-UI, no new project

**Performance Goals**: N/A — download speed is bounded by network/CDN, not something this feature controls; no new hot-path code (model loading after download is unchanged)

**Constraints**: Must not add network calls to huggingface.co when ModelScope is selected and a model has a mirror (SC-001/FR-009); must not change behavior at all for users who never touch the setting (SC-004); must not require symlinks (Windows compatibility, constitution Principle V)

**Scale/Scope**: 17 `ModelConfig` entries across 8 backend files, 2 new/changed settings endpoints, `/models/status` and `DELETE /models/{name}` route changes, 1 new frontend Settings control

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Model registry is the single source of truth | `ms_repo_id` added to `ModelConfig`, not a second parallel list | PASS |
| II. Backend abstraction via Protocol, not inheritance | New resolver lives in `backend/backends/base.py` (existing shared-logic home), called from each backend's existing `_get_model_path()` — no duplication per engine | PASS |
| III. Explicit configuration over hidden runtime magic | No dependency on a third-party proxy whose behavior we can't verify (dropped after real-world testing showed it doesn't work — see research.md §2); ModelScope downloads land in an explicit, Voicebox-owned directory rather than disguising as HF cache internals; the one deviation from the DB-settings pattern (JSON file) is documented with rationale in research.md §4 | PASS |
| IV. Local-first privacy is non-negotiable | Only outbound calls are the user-initiated model download itself, to whichever source the user explicitly chose — no telemetry added | PASS |
| V. Cross-platform parity without platform-specific hacks | Explicitly rejected the "materialize as HF cache" approach because it needs symlinks (fragile on Windows); ModelScope path uses plain file copies via SDK's `local_dir` | PASS |

No violations — Complexity Tracking table is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-modelscope-download-source/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── settings-model-source.md
└── tasks.md              # Phase 2 output (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
backend/
├── backends/
│   ├── __init__.py              # ModelConfig gains ms_repo_id; 13/17 configs get a value
│   ├── base.py                  # new: resolve_model_source(), source-aware is_model_cached() helper
│   ├── chatterbox_backend.py    # _get_model_path() routed through resolver
│   ├── chatterbox_turbo_backend.py
│   ├── hume_backend.py
│   ├── kokoro_backend.py
│   ├── luxtts_backend.py
│   ├── mlx_backend.py
│   ├── pytorch_backend.py
│   └── qwen_custom_voice_backend.py
│   └── qwen_llm_backend.py
├── utils/
│   └── model_source.py          # new: read/write data/model_source.json (get/set only — no env-var application)
├── models.py                    # new Pydantic models for the settings request/response
├── routes/
│   ├── settings.py              # new GET/PUT /settings/model-source
│   └── models.py                # /models/status, DELETE /models/{name} become source-aware
├── services/
│   └── settings.py              # unchanged (model-source setting bypasses this DB-row pattern — see research.md §4)
└── requirements.txt              # add `modelscope`

app/src/components/ServerSettings/
└── ModelManagement.tsx           # add the download-source selector (existing home for HF cache dir / migrate UI)
```

**Structure Decision**: Extends the existing single FastAPI backend +
frontend structure — no new services, no new top-level directories beyond
the standard `backend/utils/` module and the data-dir subfolder created at
runtime (`data/models/modelscope/`).

## Complexity Tracking

*No constitution violations — table not needed.*
