# Tasks: ModelScope Download Source

> **Amendment (2026-08-27)**: Real-environment verification (T043) found the
> "HF Mirror" mechanism (`hf_mirror` source, and the fallback path for
> unmirrored models) doesn't actually work — `hf-mirror.com` redirects to
> the real huggingface.co instead of proxying. It was removed rather than
> shipped unverified; see spec.md's Amendment note and research.md §2/§4
> for the full account. Below, tasks whose description still says
> "HF Mirror" describe what was originally built, not current behavior —
> Phase 5 (User Story 3, "HF Mirror as its own source") was removed
> entirely along with the mechanism it tested. `T004`/`T005`/`T029` were
> re-implemented against the corrected 2-source design (see current source
> for ground truth); their `[X]` here reflects that the *task* (add a test
> for X, wire up Y) is done, not that the original mirror-specific
> description is still accurate.

**Input**: Design documents from `/specs/001-modelscope-download-source/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/settings-model-source.md](contracts/settings-model-source.md), [quickstart.md](quickstart.md)

**Tests**: Required — every implementation task is preceded by a test task per project convention (TDD). Tests live flat under `backend/tests/` (matching this repo's existing convention — no `tests/unit`/`tests/integration` subfolders).

**Organization**: Tasks are grouped by user story (P1–P4 from spec.md) so each can be implemented and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1–US4)

## Path Conventions

Existing FastAPI backend at `backend/`, frontend at `app/src/`. No new top-level directories.

---

## Phase 1: Setup

**Purpose**: Add the new dependency; nothing else to scaffold (extends an existing app).

- [X] T001 Add `modelscope` to `backend/requirements.txt` (see [research.md](research.md) §5)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The registry field, the source-resolution primitive, and the persisted setting that every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Tests for Foundational (write first, confirm they fail)

- [X] T002 [P] Test `ModelConfig` accepts and defaults `ms_repo_id` to `None` in `backend/tests/test_model_registry.py`
- [X] T003 [P] Test `backend/utils/model_source.py`'s `get_model_source()`/`set_model_source()` round-trip through a JSON file, and default to `"huggingface"` when the file is absent, in `backend/tests/test_model_source.py`
- [X] T004 [P] Test `apply_model_source_to_env()` sets `HF_ENDPOINT=https://hf-mirror.com` for both `"hf_mirror"` and `"modelscope"` sources, and leaves it unset for `"huggingface"` in `backend/tests/test_model_source.py`
- [X] T005 [P] Test `resolve_model_source(hf_repo_id, ms_repo_id, model_name)` in `backend/tests/test_model_source_resolution.py`: returns `hf_repo_id` unchanged when source is `"huggingface"` or `"hf_mirror"` regardless of `ms_repo_id`; returns `hf_repo_id` unchanged when source is `"modelscope"` and `ms_repo_id` is `None`; returns a local directory path under `get_models_dir()/modelscope/` when source is `"modelscope"` and `ms_repo_id` is set (mock the actual download call)

### Implementation for Foundational

- [X] T006 Add `ms_repo_id: Optional[str] = None` field to the `ModelConfig` dataclass in `backend/backends/__init__.py` (makes T002 pass)
- [X] T007 [P] Populate `ms_repo_id` on the 4 Qwen TTS configs in `_get_qwen_model_configs()` (`backend/backends/__init__.py`) — same repo id as `hf_repo_id` per [research.md](research.md) §1, for both the `Qwen/*` and `mlx-community/*` branches
- [X] T008 [P] Populate `ms_repo_id` on the 2 Qwen CustomVoice configs in `_get_qwen_custom_voice_configs()` (`backend/backends/__init__.py`) — same repo id as `hf_repo_id`
- [X] T009 [P] Populate `ms_repo_id` on `luxtts` (`hf/YatharthS-LuxTTS`) and `kokoro` (`AI-ModelScope/Kokoro-82M`) in `_get_non_qwen_tts_configs()` (`backend/backends/__init__.py`); leave the 4 Chatterbox/TADA entries unset
- [X] T010 [P] Populate `ms_repo_id` on the 5 Whisper configs in `_get_whisper_configs()` (`backend/backends/__init__.py`) using the `openai-mirror/whisper-*` ids from [research.md](research.md) §1
- [X] T011 [P] Populate `ms_repo_id` on the 3 Qwen3 LLM configs in `_get_qwen_llm_configs()` (`backend/backends/__init__.py`) — same repo id as `hf_repo_id`, for both the `Qwen/*` and `mlx-community/*` branches
- [X] T012 Create `backend/utils/model_source.py`: `get_model_source()` / `set_model_source(value)` reading/writing `get_data_dir()/model_source.json`, and `apply_model_source_to_env()` that sets/unsets `HF_ENDPOINT` (makes T003, T004 pass)
- [X] T013 Call `apply_model_source_to_env()` from `backend/config.py`, in the same startup block that already handles `HF_HUB_CACHE` (before any other backend import) — see [research.md](research.md) §4
- [X] T014 Implement `resolve_model_source(hf_repo_id, ms_repo_id, model_name)` in `backend/backends/base.py`: for the ModelScope+mirror-available case, download via `modelscope.snapshot_download(ms_repo_id, local_dir=get_models_dir()/"modelscope"/safe_name, progress_callback=...)` (lazy-imported) and return the local dir path; otherwise return `hf_repo_id` unchanged (makes T005 pass)
- [X] T015 Add a source-aware `is_model_cached_at(path_or_repo)` helper in `backend/backends/base.py` that checks a local directory's contents when given a filesystem path, and falls back to the existing `is_model_cached()` HF-cache check when given a repo id string

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Choose ModelScope and download a mirrored model (Priority: P1) 🎯 MVP

**Goal**: A user can select ModelScope in Settings, restart, and successfully download+use a model that has a ModelScope mirror.

**Independent Test**: Per [quickstart.md](quickstart.md) Scenario 2 — switch source, restart, download `kokoro`, confirm no huggingface.co traffic and the model loads.

### Tests for User Story 1 (write first, confirm they fail)

- [X] T016 [P] [US1] Contract test `GET /settings/model-source` returns `{"source": "huggingface", "active_source": "huggingface"}` by default, in `backend/tests/test_model_source_settings.py`
- [X] T017 [P] [US1] Contract test `PUT /settings/model-source` persists a valid value and rejects an invalid one with 422, in `backend/tests/test_model_source_settings.py`
- [X] T018 [P] [US1] Integration test: with a mocked `modelscope.snapshot_download`, downloading `kokoro` while source is `"modelscope"` writes files under `get_models_dir()/modelscope/AI-ModelScope--Kokoro-82M/` and `_get_model_path()` on `KokoroBackend` returns that local path, in `backend/tests/test_modelscope_download.py`

### Implementation for User Story 1

- [X] T019 [US1] Add `ModelSourceResponse` / `ModelSourceUpdate` Pydantic models to `backend/models.py`
- [X] T020 [US1] Add `GET /settings/model-source` and `PUT /settings/model-source` handlers to `backend/routes/settings.py`, backed by `backend/utils/model_source.py` (makes T016, T017 pass)
- [X] T021 [US1] Wire download progress from `modelscope.snapshot_download`'s `progress_callback` into the existing `ProgressManager` (`backend/utils/progress.py`) inside `resolve_model_source()`/its download helper in `backend/backends/base.py`, reusing the same `model_name` keys the SSE endpoint already serves
- [X] T022 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/kokoro_backend.py`
- [X] T023 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/luxtts_backend.py`
- [X] T024 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/pytorch_backend.py` (Qwen TTS + Whisper)
- [X] T025 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/mlx_backend.py`
- [X] T026 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/qwen_custom_voice_backend.py`
- [X] T027 [P] [US1] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/qwen_llm_backend.py`
- [X] T028 [US1] Verify/adjust `KokoroBackend._is_model_cached()` and `load_model()` to use `is_model_cached_at()` (T015) so a ModelScope-resolved local path is recognized as cached and loads offline (makes T018 pass)
- [X] T029 [US1] Add the download-source selector (HuggingFace / HF Mirror / ModelScope) to `app/src/components/ServerSettings/ModelManagement.tsx`, calling the new `/settings/model-source` endpoints and showing "restart required" after a change

**Checkpoint**: User Story 1 fully functional — ModelScope-mirrored models download and run end to end.

---

## Phase 4: User Story 2 - Transparent fallback for unmirrored models (Priority: P1)

**Goal**: Selecting ModelScope never breaks Chatterbox/TADA — they silently download via the HF mirror instead.

**Independent Test**: Per [quickstart.md](quickstart.md) Scenario 3 — with source `"modelscope"`, download `chatterbox-tts` and confirm it succeeds with no error.

### Tests for User Story 2 (write first, confirm they fail)

- [X] T030 [P] [US2] Test `resolve_model_source(hf_repo_id="ResembleAI/chatterbox", ms_repo_id=None, ...)` with source `"modelscope"` returns `hf_repo_id` unchanged (not an error, not a local path) in `backend/tests/test_model_source_resolution.py`
- [X] T031 [P] [US2] Test that `HF_ENDPOINT` is already pointed at the mirror in this scenario (i.e. the fallback is actually fast, not just non-erroring) — extends T004's coverage in `backend/tests/test_model_source.py`

### Implementation for User Story 2

- [X] T032 [US2] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/chatterbox_backend.py` (confirms the `ms_repo_id=None` fallback path end-to-end)
- [X] T033 [US2] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/chatterbox_turbo_backend.py`
- [X] T034 [US2] Route `_get_model_path()` through `resolve_model_source()` in `backend/backends/hume_backend.py` (both TADA variants)

**Checkpoint**: All 17 model entries are downloadable regardless of source selection (SC-002).

---

## Phase 5: User Story 3 - HF Mirror as its own source (Priority: P2) — REMOVED

> This entire phase was removed 2026-08-27. `hf-mirror.com` turned out to
> just redirect to the real huggingface.co rather than proxying, which
> `huggingface_hub` rejects and which wouldn't have helped a genuinely
> blocked user anyway — see the amendment note at the top of this file.
> Kept here (rather than deleted) as a record of what was tried and why it
> didn't ship.

**Goal**: A user who wants speed without ModelScope specifically can select "HF Mirror" and get uniform behavior across all 17 models.

**Independent Test**: Per [quickstart.md](quickstart.md) Scenario 4 — select `"hf_mirror"`, download `whisper-base`, confirm traffic goes to the mirror endpoint using the existing byte-level progress UI.

### Tests for User Story 3 (write first, confirm they fail)

- [X] T035 [P] [US3] Test `resolve_model_source(...)` with source `"hf_mirror"` always returns `hf_repo_id` unchanged regardless of whether `ms_repo_id` is set — extends `backend/tests/test_model_source_resolution.py`

### Implementation for User Story 3

- [X] T036 [US3] Confirm (and adjust if needed) that no backend-specific code path is required beyond T012/T013 — this story is fully satisfied by the env-var-only mechanism already built in Phase 2; if the test from T035 already passes with no changes, mark this task as verification-only

**Checkpoint**: HF Mirror source works uniformly across all model entries.

---

## Phase 6: User Story 4 - Status and deletion recognize ModelScope-downloaded models (Priority: P2)

**Goal**: `/models/status` and `DELETE /models/{name}` correctly report and remove models downloaded via ModelScope.

**Independent Test**: Per [quickstart.md](quickstart.md) Scenario 5 — after downloading via ModelScope, status shows `downloaded: true` with a size; delete removes it and status flips back.

### Tests for User Story 4 (write first, confirm they fail)

- [X] T037 [P] [US4] Test `GET /models/status` reports `downloaded: true` and a populated `size_mb` for a model with files under `get_models_dir()/modelscope/<safe-id>/` in `backend/tests/test_model_status_modelscope.py`
- [X] T038 [P] [US4] Test `DELETE /models/{model_name}` removes `get_models_dir()/modelscope/<safe-id>/` and a subsequent status check reports `downloaded: false` in `backend/tests/test_model_status_modelscope.py`

### Implementation for User Story 4

- [X] T039 [US4] Extend `get_model_status()` in `backend/routes/models.py` to also check the ModelScope-managed directory (via `is_model_cached_at()` / a size-on-disk helper) alongside the existing HF-cache check (makes T037 pass)
- [X] T040 [US4] Extend `delete_model()` in `backend/routes/models.py` to delete from the ModelScope-managed directory when that's where the model actually lives, alongside the existing HF-cache deletion (makes T038 pass)

**Checkpoint**: All 4 user stories independently functional (SC-005).

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T041 [P] Update `docs/content/docs/developer/model-management.mdx` with the new download-source setting, the `ms_repo_id` column, and the fallback behavior
- [X] T042 [P] Run `ruff check` / `ruff format` over all touched `backend/` files
- [X] T043 Run the [quickstart.md](quickstart.md) scenarios end-to-end manually against a real backend process — Scenario 1 and Scenario 2 (ModelScope download + real load + real audio generation for Kokoro) fully verified and pass. Scenario 3/4 (HF Mirror path, both the standalone `hf_mirror` source and the `modelscope`-fallback for unmirrored models) uncovered a real infrastructure problem: `hf-mirror.com` now returns a bare redirect to the real huggingface.co (no `X-Repo-Commit`/`ETag` on its own response) instead of proxying, which `huggingface_hub`'s metadata validation rejects — confirmed against both the pinned and an older `huggingface_hub` version, so this is not a version regression on our side. Reported to the user rather than silently patched; needs a decision (different mirror host, or a custom redirect-following resolution path) before Scenario 3/4 can be called verified.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational only. This is the MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational; reuses US1's settings endpoints conceptually but touches disjoint backend files (`chatterbox_backend.py`, `chatterbox_turbo_backend.py`, `hume_backend.py`) — can run in parallel with US1 once Foundational is done.
- **User Story 3 (Phase 5)**: Depends on Foundational only; trivially small since the mechanism is already built there.
- **User Story 4 (Phase 6)**: Depends on Foundational; benefits from US1 existing (something to have actually downloaded via ModelScope to test against) but the route changes themselves don't depend on US1's code.
- **Polish (Phase 7)**: Depends on all four user stories being complete.

### Parallel Opportunities

- T007–T011 (populating `ms_repo_id` across the 5 config-builder functions) are all `[P]` — different functions in the same file but non-overlapping edits; if working solo, do them sequentially in one pass through the file instead to avoid merge noise.
- T022–T027 (routing 6 backends through the resolver) are `[P]` — six independent files.
- T032–T034 (the 3 fallback-path backends) can run in parallel with T022–T027.
- All test tasks marked `[P]` within a phase can be written in parallel before that phase's implementation tasks.

---

## Implementation Strategy

### MVP First

1. Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (User Story 1).
2. **STOP and VALIDATE**: run quickstart.md Scenario 1 and Scenario 2. This alone delivers the headline capability (ModelScope downloads working for the 13 mirrored models).

### Incremental Delivery

1. Foundational → US1 (MVP: ModelScope works for mirrored models).
2. Add US2 (nothing breaks when ModelScope is selected — the other 4 models still work).
3. Add US3 (HF Mirror as a standalone, simpler option).
4. Add US4 (status/delete correctness) — needed before calling the feature done, but doesn't block US1–US3 from being demoed first.
5. Polish.
