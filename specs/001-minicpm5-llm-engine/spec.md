# Feature Specification: MiniCPM5-1B LLM Engine Support

**Feature Branch**: `001-minicpm5-llm-engine`

**Created**: 2026-08-26

**Status**: Draft

**Input**: User description: "Add MiniCPM5-1B as a second local LLM engine alongside the existing Qwen3 LLM engine, with full feature parity in both backend and UI."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select MiniCPM5-1B as the personality/refinement model (Priority: P1)

A user who wants faster or higher-quality local text generation for voice-profile personality (Compose / Rewrite) and transcript refinement opens the settings picker where they currently choose a Qwen3 size (0.6B / 1.7B / 4B) and sees "MiniCPM5 1B" listed as an additional option. Selecting it and downloading it (if not already cached) makes it the active model for Compose, Rewrite, and Refine, exactly as any Qwen3 size is today.

**Why this priority**: This is the core value of the feature — without it, MiniCPM5-1B is invisible to users no matter how well it's wired up in the backend.

**Independent Test**: Can be fully tested by opening the LLM model picker, selecting "MiniCPM5 1B", triggering a download, waiting for it to finish, and confirming subsequent Compose/Rewrite/Refine calls run against it (e.g. via the returned `model_size`/model identifier in the response, or backend logs).

**Acceptance Scenarios**:

1. **Given** MiniCPM5-1B is not yet downloaded, **When** the user selects it in the LLM picker, **Then** the system starts a download and reports progress the same way it does for a Qwen3 size.
2. **Given** MiniCPM5-1B is downloaded and selected, **When** the user clicks Compose or submits text for Rewrite/Refine, **Then** the generated text comes from MiniCPM5-1B and the response identifies which model produced it.
3. **Given** the user previously selected a Qwen3 size, **When** they switch the picker to MiniCPM5-1B, **Then** the previously loaded Qwen3 model is unloaded and MiniCPM5-1B is loaded, following the same one-model-active-at-a-time behavior Qwen3 sizes already have with each other.

---

### User Story 2 - Manage MiniCPM5-1B like any other downloadable model (Priority: P2)

A user browsing the model download/management page sees MiniCPM5-1B listed under the language-models section alongside the Qwen3 sizes, with its own download/delete/loaded-status controls.

**Why this priority**: Users need visibility into disk usage and the ability to remove a model they no longer want, matching the existing management experience for every other model in the app.

**Independent Test**: Open the model management page and confirm MiniCPM5-1B appears in the "language models" grouping with working download, loaded-state, and delete controls, independent of whether User Story 1's picker flow has been exercised.

**Acceptance Scenarios**:

1. **Given** the model management page is open, **When** the user views the language-models section, **Then** MiniCPM5-1B is listed with its display name, size, and current cached/loaded state.
2. **Given** MiniCPM5-1B is cached on disk, **When** the user chooses to delete it, **Then** it is removed and the picker in User Story 1 falls back to requiring a fresh download before it can be selected again.

---

### User Story 3 - Existing Qwen3 selections keep working after the upgrade (Priority: P1)

A user who already had a Qwen3 size selected as their LLM (from before this feature shipped) opens the app after the update and finds their previous selection, downloaded models, and past capture history untouched and still functional.

**Why this priority**: The identifier used to record "which model is active" changes shape as part of this feature (see Assumptions); if existing installs aren't carried forward correctly, users lose their configured model and past captures show broken/blank model attribution — this must not regress for any existing user on the very first run after the update.

**Independent Test**: Start the app against a database created by a pre-feature build with a Qwen3 size already selected and some existing captures, confirm after startup that the selection and each capture's recorded model still resolve to the correct Qwen3 model with no manual re-selection required.

**Acceptance Scenarios**:

1. **Given** a database written by a previous version with the LLM setting recorded in its old form, **When** the app starts after the update, **Then** the setting is automatically carried forward and the picker shows the same Qwen3 size the user had selected before, with no user action required.
2. **Given** past captures recorded a model in the old form, **When** the user views that capture's history/detail after the update, **Then** it still displays which Qwen3 model produced it.

### Edge Cases

- What happens if the user starts a Compose/Rewrite/Refine call while MiniCPM5-1B is still downloading? System must respond the same way it does today for an in-progress Qwen3 download (reject with a "still downloading" message, not a crash or silent fallback).
- What happens if MiniCPM5-1B download fails partway (network loss, disk full)? System must surface the failure the same way a failed Qwen3 download is surfaced today, and must not leave the picker's selection in a state that silently and permanently points at an unusable model.
- What happens if a user deletes MiniCPM5-1B while it is the currently active/loaded model? It must unload cleanly, matching existing behavior for deleting the active Qwen3 model.
- What happens on a platform without Apple Silicon (Windows/Linux, or Intel Mac)? MiniCPM5-1B must be available via the CPU/GPU (PyTorch) path exactly as Qwen3 already is on those platforms — it is not an Apple-Silicon-only feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST offer MiniCPM5-1B as a selectable local LLM option everywhere a Qwen3 size is currently selectable (settings picker, per-capture override, model management page).
- **FR-002**: System MUST support running MiniCPM5-1B on every platform Qwen3 currently runs on — the Apple Silicon accelerated path and the general CPU/GPU path — with no feature gap between the two paths beyond what already exists for Qwen3.
- **FR-003**: System MUST let a user download, view cached/loaded status, and delete MiniCPM5-1B independently of any Qwen3 size, using the same management affordances as every other downloadable model.
- **FR-004**: System MUST use MiniCPM5-1B for Compose, Rewrite, and Refine when it is the selected model, producing output through the same request/response shape those features already use for Qwen3 (including reporting which model produced the output).
- **FR-005**: System MUST treat MiniCPM5-1B and each Qwen3 size as mutually exclusive with respect to "currently loaded" — selecting one unloads the other, matching today's behavior when switching between Qwen3 sizes.
- **FR-006**: System MUST reject requests to use MiniCPM5-1B before it has finished downloading, with the same class of error already used when a Qwen3 size isn't downloaded yet.
- **FR-007**: System MUST preserve every existing user's previously selected Qwen3 model and previously recorded per-capture model attribution across the upgrade with no manual re-selection or visible data loss.
- **FR-008**: System MUST NOT offer or expose the community-maintained quantization of MiniCPM5-1B as a selectable option in this feature — only the official pre-quantized weights are in scope.
- **FR-009**: System MUST continue to expose exactly the Qwen3 sizes and behavior that exist today — this feature is additive and must not remove or rename any existing user-facing Qwen3 option.

### Key Entities

- **LLM model option**: A selectable local text-generation model. Has a unique identifier, a human-readable display name, an underlying model family/engine (e.g. Qwen3, MiniCPM5), a size, a download source, and a cached/loaded status. MiniCPM5-1B is a new instance of this entity; Qwen3 0.6B/1.7B/4B are existing instances.
- **Active LLM selection**: The single record of which LLM model option is currently chosen for Compose/Rewrite/Refine, persisted across app restarts. Exactly one is active at a time, app-wide, unless a specific request explicitly overrides it for that one call.
- **Capture model attribution**: A historical record, attached to each saved capture, of which LLM model option produced its refined transcript. Must remain resolvable to a real model option after the upgrade.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can go from "MiniCPM5-1B not installed" to "generating personality text with MiniCPM5-1B" using only the existing model-picker and download flow, with zero steps that differ in kind from selecting a Qwen3 size.
- **SC-002**: 100% of pre-existing installs retain their previously selected Qwen3 model and previously recorded capture model attribution after the first run of the updated app — zero manual re-selection required, zero captures left with unresolvable model attribution.
- **SC-003**: MiniCPM5-1B is usable end-to-end (download, select, generate) on both an Apple Silicon Mac and a non-Apple-Silicon machine, with no platform where it is listed but non-functional.
- **SC-004**: Every control that exists for a Qwen3 size on the model management page (download, delete, loaded-status indicator) is present and functional for MiniCPM5-1B, with no missing control.

## Assumptions

- "Full feature parity" means MiniCPM5-1B is reachable through every existing Qwen3-LLM surface (settings picker, capture-level override, model management page, Compose/Rewrite/Refine) — it does not mean adding new surfaces that don't exist for Qwen3 today (e.g. no new benchmarking/comparison UI).
- The internal identifier used to record "which LLM is active" changes shape as part of this feature (from a bare size label to a model-family-qualified identifier, since a bare size label like "1B" is no longer unique once a second model family exists). This is an internal representation change; User Story 3 and FR-007/SC-002 hold regardless of the specific representation chosen, and existing user-visible values (the Qwen3 size names shown in the UI, past capture history) must not change or break.
- Only the official pre-quantized MiniCPM5-1B weights are in scope for the Apple Silicon accelerated path; community-maintained alternative quantizations are explicitly out of scope for this feature (FR-008) and may be revisited later as a separate decision.
- MiniCPM5-1B ships as a single size/variant for this feature — no equivalent to Qwen3's multiple size tiers (0.6B/1.7B/4B) is in scope.
- No new quality-comparison or benchmarking tooling between MiniCPM5-1B and Qwen3 is in scope — this feature is about making MiniCPM5-1B usable, not about helping users choose between the two.
- Non-Apple-Silicon platforms already run Qwen3 via a general-purpose (CPU/GPU) path; MiniCPM5-1B is expected to reuse that same general-purpose path rather than requiring a new execution mode.
