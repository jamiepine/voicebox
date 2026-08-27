---
description: "Task list for MiniCPM5-1B LLM Engine Support"
---

# Tasks: MiniCPM5-1B LLM Engine Support

**Input**: Design documents from `specs/001-minicpm5-llm-engine/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/llm-model-selection.md, quickstart.md

**Tests**: TDD is mandatory for this feature (explicit user requirement). Every implementation task has a preceding test task that must be written and observed failing first.

**Organization**: Phase 2 (Foundational) carries everything every user story depends on — the new engine's backend classes, the registry/factory generalization, the identifier-scheme validation change, and the cross-engine unload guard — because US1 and US2 cannot be exercised at all until MiniCPM5-1B exists as a loadable, selectable engine. US3 (migration / backward compatibility) gets its own phase because it is specifically about historical-data correctness, independently testable from a legacy-shaped DB fixture.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Maps to spec.md's US1/US2/US3

## Phase 1: Setup

- [x] T001 Pin `mlx-lm>=0.31` explicitly in `backend/requirements-mlx.txt` (currently arrives only transitively via `mlx-audio`, no version floor — see research.md's "Constraint check" section; MiniCPM5-1B's dual `eos_token_id` list needs >=0.31). Verified installable in the project venv (`mlx-lm-0.31.1`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: MiniCPM5-1B must exist as a loadable engine with a resolvable identifier before any user story can be exercised.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Backend classes for the new engine

- [x] T002 [P] Write failing tests in `backend/tests/test_minicpm_llm_backend.py` for `PyTorchMiniCPMLLMBackend`: `load_model` calls `AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", ...)` and `AutoTokenizer.from_pretrained(...)` (both mocked), `generate` builds messages and calls `tokenizer.apply_chat_template(..., enable_thinking=False)` then decodes output, `unload_model` clears state and calls `empty_device_cache`. Mirror the shape of what `qwen_llm_backend.py`'s equivalent classes do — mock `transformers.AutoModelForCausalLM`/`AutoTokenizer` the same way any other backend test in `backend/tests/` mocks external SDKs (e.g. `test_rocm_backends.py`'s mocking style).
- [x] T003 [P] Write failing tests in `backend/tests/test_minicpm_llm_backend.py` for `MLXMiniCPMLLMBackend`: `load_model` calls `mlx_lm.load("openbmb/MiniCPM5-1B-MLX")` (mocked), `generate` follows the same `_build_messages`/`enable_thinking=False` pattern, `unload_model` clears state **and calls `empty_mlx_cache()`** — mirror `backend/tests/test_mlx_unload_clears_cache.py`'s `test_mlx_llm_backend_unload_clears_mlx_cache` pattern (monkeypatch `empty_mlx_cache` on the `minicpm_llm_backend` module, assert it's called once, and assert the already-unloaded no-op case doesn't call it — this pattern landed on `main` after this plan was first drafted, see research.md's "Baseline update" note). No real MLX runtime invoked in CI.
- [x] T004 Implement `backend/backends/minicpm_llm_backend.py` with `PyTorchMiniCPMLLMBackend` and `MLXMiniCPMLLMBackend`, copying the structure of `backend/backends/qwen_llm_backend.py` (`_build_messages` helper — reused via import rather than copy-pasted, since it's a pure engine-agnostic function; `LLMBackend` protocol methods; `PYTORCH_HF_REPOS = {"1B": "openbmb/MiniCPM5-1B"}`; `MLX_HF_REPOS = {"1B": "openbmb/MiniCPM5-1B-MLX"}`; and `MLXMiniCPMLLMBackend.unload_model()` calling `empty_mlx_cache()`, matching `MLXQwenLLMBackend`'s current implementation) until T002 and T003 pass. 12/12 tests green. Did not copy `qwen_llm_backend.py`'s unused `manual_seed` import — pre-existing dead code there, not worth propagating into new code.
- [x] T005 Investigated and decided **no code change needed here**: each backend class already constructs its own progress-name string (`_progress_name()` in `qwen_llm_backend.py`, inline `f"minicpm5-{model_size.lower()}"` in the new `minicpm_llm_backend.py`), and both already produce values matching their `ModelConfig.model_name` (`"qwen3-0.6b"`, `"minicpm5-1b"`) — two small per-family functions aren't yet duplication worth abstracting (rule of three). The actual bug is `backend/routes/llm.py`'s hardcoded `f"qwen3-{model_size.lower()}"`, which would silently mis-tag a MiniCPM5-1B download as `"qwen3-1b"` — that fix requires knowing the resolved engine, which only exists once T016 resolves `model_name -> (engine, model_size)`. Folded into T016; not a separate change.

### Registry & factory generalization

- [x] T006 [P] Write failing tests (new file `backend/tests/test_llm_backend_registry.py`) asserting: `get_llm_model_configs()` returns entries for both `qwen_llm` and `minicpm_llm` engines including one with `model_name == "minicpm5-1b"`; `get_llm_backend_for_engine("minicpm_llm")` returns `MLXMiniCPMLLMBackend` when `get_backend_type()` is mocked to `"mlx"` and `PyTorchMiniCPMLLMBackend` otherwise; `unload_model_by_config`, `check_model_loaded`, and `get_model_load_func` all behave correctly (matching their existing `qwen_llm` behavior) when given a `minicpm_llm` `ModelConfig`.
- [x] T007 In `backend/backends/__init__.py`: added `"minicpm_llm": "MiniCPM5 LLM"` to `LLM_ENGINES`; added `_get_minicpm_llm_configs() -> list[ModelConfig]` mirroring `_get_qwen_llm_configs()`'s backend-type-aware repo selection; folded into `get_llm_model_configs()`/`get_all_model_configs()`; added the `elif engine == "minicpm_llm":` branch to `get_llm_backend_for_engine()`; generalized all three `if config.engine == "qwen_llm":` blocks to `if config.engine in LLM_ENGINES:` (also now passing `config.engine` to `llm_service.get_llm_model(...)` — this makes T007 depend on T011's signature change to actually work end-to-end at runtime, though T006's tests mock `get_llm_model` directly so they pass independently). 7/7 tests green.

### Identifier-scheme validation (dynamic membership replaces static regex)

- [x] T008 [P] Write failing tests for `backend/models.py`'s affected fields: a value like `"minicpm5-1b"` is accepted, a bogus value is rejected, and (with `get_llm_model_configs()` mocked to exclude MiniCPM5) `"minicpm5-1b"` is correctly rejected too. **Correction from the original plan**: re-scanning `models.py` for the `pattern="^(0\\.6B|1\\.7B|4B)$"` regex during implementation found **four** affected fields, not three — `LLMGenerateRequest.model_size` (the actual `/llm/generate` request field) was missed in the original research/plan and is now covered alongside `CaptureRefineRequest.model_size`, `CaptureSettingsResponse.llm_model`, `CaptureSettingsUpdate.llm_model`. (`GenerationRequest.model_size`, pattern `^(1\.7B|0\.6B|1B|3B)$`, is a separate TTS field and is correctly untouched.)
- [x] T009 In `backend/models.py`, replaced all four regex constraints with a shared `_validate_llm_model_name` function applied via `field_validator` on each field, checking membership in `{cfg.model_name for cfg in get_llm_model_configs()}`. Added the naming-mismatch comment at each field per contracts/llm-model-selection.md. 9/9 tests green.

### Cross-engine unload guard (new logic, not a mirror — see data-model.md)

- [x] T010 Write a failing test in `backend/tests/test_llm_service_engine_switch.py` proving: given a `qwen_llm` backend instance that reports `is_loaded() == True`, calling `llm_service.get_llm_model(engine="minicpm_llm")` calls `unload_model()` on the `qwen_llm` instance before returning the `minicpm_llm` instance. Also assert the reverse direction and that requesting the *same* already-active engine does not unload anything.
- [x] T011 In `backend/services/llm.py`, added `engine: str = "qwen_llm"` to `get_llm_model()`/`unload_llm_model()`; implemented the cross-engine unload guard per data-model.md. Also removed `backends/__init__.py`'s now-fully-unused `get_llm_backend()` zero-arg wrapper (its only caller was replaced) rather than leave new dead code. 5/5 tests green.

**Checkpoint**: MiniCPM5-1B is now a fully registered, loadable, validated engine with correct single-active-model semantics. No user-facing surface uses it yet.

---

## Phase 3: User Story 1 - Select MiniCPM5-1B for Compose/Rewrite/Refine (Priority: P1) 🎯 MVP

**Goal**: A user can select MiniCPM5-1B in the picker and have Compose, Rewrite, and Refine actually run on it.

**Independent Test**: With Phase 2 complete, call `POST /llm/generate` with a MiniCPM5-1B identifier directly (no UI needed yet) and confirm generation succeeds and the response identifies MiniCPM5-1B; separately confirm the UI picker offers it once T017 lands.

### Tests for User Story 1

- [x] T012 [P] [US1] Write a failing test for `backend/routes/llm.py`'s generate endpoint: a request identifying `"minicpm5-1b"` resolves to the `minicpm_llm` engine and the response reports the model used (`test_llm_routes_engine_resolution.py`, 3 tests, calling the async route function directly rather than through TestClient — matches this repo's lightweight-unit-test style since there's no existing precedent for full-app route tests on this endpoint).
- [x] T013 [P] [US1] Write a failing test for `compose_as_profile`/`rewrite_as_profile` (`test_personality_engine_resolution.py`, 3 tests).
- [x] T014 [P] [US1] Write a failing test for the refinement path (`test_refinement_engine_resolution.py`, 2 tests).
- [x] T015 [P] [US1] Write a failing test for `capture_readiness_endpoint`'s model lookup (`test_capture_readiness_engine_lookup.py`, 1 test).
- [x] (unplanned, discovered mid-implementation) Wrote `test_llm_service_resolve.py` (2 tests) for a shared `resolve_backend_and_size()` helper extracted into `services/llm.py` — see T017 note below for why this became necessary.

### Implementation for User Story 1

- [x] T016 [US1] In `backend/routes/llm.py`, resolve `request.model_size` (a model_name despite the field name) to a `ModelConfig` via `get_model_config()`, call `llm_service.get_llm_model(config.engine)`, pass `config.model_size` (bare) to the backend, and key download-progress tracking off `model_name` directly — this is the fix that was deferred from T005. 3/3 tests green.
- [x] T017 [US1] **Went beyond the original plan**: implementing T013/T014 surfaced a real bug — `personality.py`/`refinement.py` originally each inlined `model_size or backend.model_size`, and `refine_transcript`'s returned "resolved size" was the *bare* size (e.g. `"1B"`), which `services/captures.py` persists as `row.llm_model`. A bare size is ambiguous once two engines exist (both "1B"-shaped and otherwise), so capture history would have silently stored the wrong kind of value. Fixed by extracting a shared `resolve_backend_and_size(model_name) -> (backend, bare_size, model_name)` into `services/llm.py` (reused by both files, replacing near-duplicate inline logic), which also reverse-resolves the `None` (default) case to its own model_name via `get_llm_model_configs()`. `personality.py`'s `PersonalityResult.model_size` and `refinement.py`'s `refine_transcript` return value now report `model_name`, not bare size. Removed the two files' import of `get_model_config` (no longer needed directly — routed through `resolve_backend_and_size` instead). 5+2 tests green.
- [x] T018 [US1] In `backend/routes/captures.py`, changed `c.model_size == saved.llm_model` to `c.model_name == saved.llm_model` in `capture_readiness_endpoint`; updated the adjacent stale comment ("pattern-validated" → "validated", since the mechanism changed in T009). 1/1 test green.
- [x] T019 [US1] In `app/src/lib/api/types.ts`, replaced `Qwen3ModelSize` with `LlmModelName = 'qwen3-0.6b' | 'qwen3-1.7b' | 'qwen3-4b' | 'minicpm5-1b'`, updated `CaptureRefineRequest.model_size` and `CaptureSettings.llm_model` (which `CaptureSettingsUpdate` inherits via `Partial<CaptureSettings>`). Verified via `tsc --noEmit` — clean except one pre-existing unrelated error (`StoryContent.tsx`'s missing `react-loaders` module, untouched by this feature).
- [x] T020 [US1] In `app/src/components/ServerTab/CapturesPage.tsx`, added "MiniCPM5 1B" as a picker option, switched all three existing option values from bare sizes (`"0.6B"`) to model_names (`"qwen3-0.6b"`), fixed the stale `'0.6B'` default fallback to `'qwen3-0.6b'`, added the `sizeMiniCPM5` translation key to all 9 locale files (`900 Mo` for French per its existing MB→Mo/GB→Go localization convention, `900 MB` elsewhere — verified all 9 JSON files still parse).
- [x] T021 [US1] `CapturesTab.tsx` turned out to have **no dedicated model picker of its own** (only `CapturesPage.tsx`/Settings does) — it just reads the shared `llm_model` setting and displays whatever string comes back. Fixed its own stale `'0.6B'` default fallback to `'qwen3-0.6b'`; left the raw-identifier display in `transcript.refinedHint` as-is (shows the model_name directly, e.g. `"minicpm5-1b"`, with no separate display-name lookup layer to hook into — introducing one would be new scope beyond what was asked).

**Checkpoint**: User Story 1 is fully functional — a user can select, download (via existing generic download flow, already engine-agnostic per Phase 2), and use MiniCPM5-1B for Compose/Rewrite/Refine.

---

## Phase 4: User Story 2 - Manage MiniCPM5-1B on the Model Page (Priority: P2)

**Goal**: MiniCPM5-1B is visible and manageable (download/delete/status) on the model management page.

**Independent Test**: Open the model management page and confirm MiniCPM5-1B appears under language models with working controls, independent of whether Phase 3's picker flow has been exercised.

### Implementation for User Story 2

*No test task: `ModelManagement.tsx` has no existing frontend test infrastructure for the equivalent Qwen3 rows (per plan.md's non-goals — matching existing coverage level, not adding new frontend test infra for parity).*

- [x] T022 [US2] **Correction from the original plan**: `ModelStatus` (both the TS interface and the backend's `/models/status` response) carries no `engine` field at all today — matching by engine as originally proposed would require adding a new field to the backend response model and threading it through, which is bigger than this task's scope. Went with the minimal, consistent fix instead: extended the existing string-prefix filter to `m.model_name.startsWith('qwen3-') || m.model_name.startsWith('minicpm5-')`, matching the file's existing style (the voice-generation section above it already does the same multi-`startsWith` pattern for several engines). Also added a `minicpm5-1b` entry to the per-model description map used by the detail modal, for full parity with the Qwen3 entries. Verified via `tsc --noEmit` — clean.

**Checkpoint**: MiniCPM5-1B has full management-page parity with Qwen3 sizes.

---

## Phase 5: User Story 3 - Existing Qwen3 Selections Keep Working After the Upgrade (Priority: P1)

**Goal**: Pre-existing installs retain their Qwen3 selection and capture history across the identifier-scheme change, automatically.

**Independent Test**: Start the app against a DB seeded with legacy-format values (`"0.6B"` etc.) and confirm they read back as the correct new-format identifiers with no user action.

### Tests for User Story 3

- [x] T023 [P] [US3] Wrote 4 failing tests in `backend/tests/test_migrate_llm_model_names.py` using an in-memory SQLite engine + the real `Capture`/`CaptureSettings` ORM models (`Base.metadata.create_all`) rather than hand-rolled schema: legacy rewrite, already-migrated values left alone, idempotency on a second run, and a no-op when tables don't exist yet (fresh-install safety net).
- [x] T024 [US3] Added `_migrate_llm_model_names(engine, tables)` to `backend/database/migrations.py` per data-model.md's migration table, following the file's documented `_migrate_*` convention (plain `UPDATE`, logs row counts when it does real work), wired into `run_migrations()`. 4/4 tests green.
- [x] T025 [US3] `backend/database/models.py`: `CaptureSettings.llm_model` column default `"0.6B"` → `"qwen3-0.6b"` (plus the naming-mismatch comment). `backend/models.py`'s `CaptureSettingsResponse.llm_model` default was already fixed as part of T009 (found and corrected while touching that field for the validator change, rather than left for a separate pass).

**Checkpoint**: All three user stories are independently functional; existing installs are unaffected in any user-visible way.

---

## Phase 6: Polish & Cross-Cutting

- [x] T026 [P] Backend-only real-hardware verification (Apple Silicon, this machine), fully completed: `mlx-lm>=0.31` installs cleanly (`0.31.1`); real download of `openbmb/MiniCPM5-1B-MLX` completed (608MB weight file, ~79 min on this sandbox's throttled bandwidth); **the real download surfaced a genuine bug** — `AutoTokenizer.from_pretrained` refused to load the repo's declared `tokenizer_class` ("TokenizersBackend", a transformers-5.x name incompatible with this project's pinned transformers<=4.57.6). Root-caused, fixed (`_ensure_compatible_mlx_snapshot` — see research.md's "Post-implementation finding" section), covered by 3 new unit tests, and **re-verified end-to-end through the real `MLXMiniCPMLLMBackend` class**: load → `enable_thinking=False` chat template → `generate()` → real text output ("Hello!") → clean `unload_model()`. This is real evidence, not a mock — the single strongest verification in this feature that the approved design is actually viable on Apple Silicon. **Still not verified in this session**: the PyTorch (non-Apple-Silicon) path end-to-end, and the full UI click-through — no non-Mac machine was available; both remain covered only by the mocked unit tests (12/12 for the backend classes, all passing).
- [x] T027 Ran the full backend pytest suite repeatedly after each implementation step (not just once at the end) — final count: 210 passed, 4 skipped, 1 failed. The 1 failure (`test_progress.py::test_hf_progress_tracker`) and the 1 collection error excluded from the run (`test_profile_duplicate_names.py`, an import-path issue unrelated to this feature) were both confirmed pre-existing against the baseline taken before any change in this feature (162 passed/1 failed/1 error-excluded at that point) — not introduced by this work.

---

## Dependencies & Execution Order

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Phase 1. BLOCKS Phases 3, 4, 5. Internally: T002/T003 (tests) → T004 (impl) → T005; T006 (tests) → T007 (impl); T008 (tests) → T009 (impl); T010 (tests) → T011 (impl). T005's `routes/llm.py` half is completed together with T016 in Phase 3 — noted as a cross-reference, not a duplicate task.
- **User Story 1 (Phase 3)**: Depends on Phase 2 only. T012–T015 (tests, parallel) → T016–T018 (backend impl, can run in parallel across the three files since they touch different modules) → T019 (frontend type) → T020/T021 (frontend UI, parallel, both depend on T019).
- **User Story 2 (Phase 4)**: Depends on Phase 2 only (does not depend on Phase 3 — the management page's filter fix is independent of the picker/generation flow). Can be done in parallel with Phase 3.
- **User Story 3 (Phase 5)**: Depends on Phase 2 only (specifically the identifier scheme existing). Can be done in parallel with Phases 3 and 4 — flagged per the user's instruction that migration work can proceed independently/early; it is ordered last here only for narrative clarity, not because of a real code dependency.
- **Polish (Phase 6)**: Depends on Phases 3, 4, and 5 all being complete.

## Parallel Execution Examples

```text
# Phase 2, backend-class tests (different concerns, same new test file — coordinate to avoid merge conflicts):
Task: T002 PyTorchMiniCPMLLMBackend tests
Task: T003 MLXMiniCPMLLMBackend tests

# Phase 2, once T004/T007/T009/T011 land, these are independent verification tasks:
Task: T006 registry tests
Task: T008 validation tests
Task: T010 cross-engine guard test

# Phase 3, backend resolution tests (three different files, fully parallel):
Task: T012 routes/llm.py test
Task: T013 personality.py test
Task: T014 refinement.py test
Task: T015 routes/captures.py test

# Phases 3, 4, 5 can be staffed in parallel once Phase 2's checkpoint is reached.
```

## Implementation Strategy

### MVP First

1. Phase 1 (Setup) → Phase 2 (Foundational — the largest phase; this is where all the real design risk lives).
2. Phase 3 (User Story 1) → **STOP and validate**: this alone delivers the feature's core value (spec.md SC-001).
3. Ship/demo here if time-constrained; Phases 4 and 5 are additive and independently deliverable afterward.

### Incremental Delivery

Phase 2 → Phase 3 (MVP, P1) → Phase 5 (P1, migration safety net — should not ship without this, despite being organizationally separate) → Phase 4 (P2) → Phase 6.

Note: both remaining P1 stories (US1 and US3) should land before calling this feature complete — US3 isn't optional polish, it's the guarantee that shipping this doesn't regress every existing user's setup.
