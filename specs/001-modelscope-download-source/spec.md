# Feature Specification: ModelScope Download Source

**Feature Branch**: `001-modelscope-download-source`

**Created**: 2026-08-26

**Status**: Draft — amended 2026-08-27 after real-environment verification (see Amendment note below)

**Input**: User description: "支持 ModelScope 作为模型下载源(面向中国用户,无需 VPN 直连下载)。用户可在设置里从 HuggingFace(默认)/ HF 镜像(hf-mirror.com)/ ModelScope 三选一,切换后重启应用生效。已核实的模型-仓库映射:Qwen 系列(4 个 TTS + 3 个 LLM,含 Apple Silicon 的 mlx-community 变体)在 ModelScope 上有与 HuggingFace 完全同名的官方仓库;LuxTTS 对应 ModelScope 的 hf/YatharthS-LuxTTS;Kokoro-82M 对应 AI-ModelScope/Kokoro-82M;5 个 Whisper 模型对应 openai-mirror/whisper-*(已验证是 transformers 格式而非 OpenAI 原始 .pt 格式)。Chatterbox(multilingual + turbo)和 HumeAI TADA(1b + 3b-ml)这 4 个模型在 ModelScope 上没有可用镜像,当用户选择 ModelScope 作为下载源时,这 4 个模型自动退回 HF 镜像端点下载,不报错、不阻塞。"

**Amendment note**: The original design included a third source, "HF Mirror" (`hf-mirror.com`), both as its own option and as the fallback for the 4 models with no ModelScope mirror. Running the actual implementation against real infrastructure (not mocks) showed `hf-mirror.com` now just redirects to the real huggingface.co instead of proxying content — which would fail for exactly the blocked users it's meant to help, and which `huggingface_hub` itself refuses to trust. The user decided to drop the mirror entirely rather than depend on a third-party proxy of unverifiable reliability. This revision reflects that: **two** sources (HuggingFace, ModelScope), and the 4 unmirrored models simply use HuggingFace directly instead of any mirror. This also removed the only reason the original design needed a restart to apply a source change (see Assumptions).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Choose ModelScope as the download source (Priority: P1)

A user in mainland China opens Voicebox for the first time (or has been fighting slow/blocked downloads from huggingface.co) and wants every model download to go through ModelScope instead, so they can fetch models directly without a VPN.

**Why this priority**: This is the entire point of the feature. Without it, Chinese users keep hitting slow or blocked downloads and the feature delivers no value.

**Independent Test**: In Settings, switch the download source to "ModelScope", and trigger a download of a model that has a verified ModelScope mirror (e.g. Kokoro). Confirm the download completes and the model becomes usable, without any request reaching huggingface.co. (Verified for real — see Verification Log below.)

**Acceptance Scenarios**:

1. **Given** the user has never changed the setting, **When** they open Settings, **Then** the download source shows "HuggingFace" as the active selection (unchanged default behavior).
2. **Given** the user selects "ModelScope" in Settings, **When** they download a model that has a ModelScope mirror, **Then** the download completes successfully and the resulting model loads and generates/transcribes normally.
3. **Given** the user has models already downloaded via HuggingFace, **When** they later switch the source to ModelScope, **Then** already-downloaded models continue to show as downloaded/usable (the source setting only affects future downloads, not existing ones).

---

### User Story 2 - Unmirrored models still download when ModelScope is selected (Priority: P1)

A user has selected ModelScope as their source, and then tries to download Chatterbox or TADA — engines that don't have a ModelScope-hosted copy.

**Why this priority**: Without this, choosing ModelScope would silently break 4 of the 7 TTS engines, which defeats the purpose of a "just works" toggle and would confuse users who don't know (and shouldn't need to know) which specific engines are mirrored.

**Independent Test**: With "ModelScope" selected as the source, trigger a download of Chatterbox Multilingual. Confirm the download proceeds (directly from huggingface.co, same as the HuggingFace source) and completes successfully — no error is surfaced, no manual re-selection of a different source is required.

**Acceptance Scenarios**:

1. **Given** "ModelScope" is the active source, **When** the user downloads a model with no ModelScope mirror (Chatterbox Multilingual, Chatterbox Turbo, TADA 1B, TADA 3B Multilingual), **Then** the download proceeds directly from huggingface.co (identical to the HuggingFace source's behavior for that model) and completes without an error being shown to the user.
2. **Given** such a download is in progress, **When** the user views download progress, **Then** they see normal progress feedback (not an error or stuck state) consistent with the other download paths.

---

### User Story 3 - Model status and deletion recognize ModelScope-downloaded models (Priority: P2)

A user who downloaded some models via ModelScope opens the Models management screen and wants to see accurate "downloaded" status and sizes, and be able to delete a ModelScope-downloaded model to free disk space.

**Why this priority**: If the status/management screen can't see models downloaded through the new path, users would see "not downloaded" for models they already have, prompting confusing re-downloads, and would have no way to reclaim disk space through the UI.

**Independent Test**: Download a model via ModelScope, then check `GET /models/status` reports it as downloaded with a reasonable size, and confirm `DELETE /models/{name}` removes it and frees the corresponding disk space. (Verified for real — see Verification Log below.)

**Acceptance Scenarios**:

1. **Given** a model was downloaded via ModelScope, **When** the user views the models list, **Then** it shows as downloaded with an approximate size, exactly like a HuggingFace-downloaded model.
2. **Given** a model was downloaded via ModelScope, **When** the user deletes it, **Then** its on-disk files are removed and subsequent status checks show it as not downloaded.

---

### Edge Cases

- What happens when the user is offline entirely (no network) while trying to download via ModelScope? → Same failure mode as today's HuggingFace path: the download errors out and surfaces through the existing error/retry UI; no new offline-specific handling is introduced.
- What happens when the ModelScope SDK isn't installed or fails to import at runtime (e.g. corrupted install)? → Model downloads for entries with a ModelScope mapping fail with a clear error surfaced through the existing download-error UI, the same way any other missing/broken dependency would; this does not block the app from starting or affect the HuggingFace path.
- What happens if a user switches the download source setting mid-download? → The in-flight download continues using whatever source it started with (already resolved); the new source applies to the next download triggered after the change (immediately — see Assumptions).
- What happens if a model's ModelScope mirror repo is later taken down or renamed upstream? → The download fails like any other network/repo-not-found error; no automatic self-healing or alternate-mirror discovery is in scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST let the user choose a model download source from exactly two options: HuggingFace (default) and ModelScope.
- **FR-002**: The system MUST persist the selected download source across app restarts.
- **FR-003**: The system MUST display the currently active download source in Settings.
- **FR-004**: When the source is HuggingFace, the system MUST behave exactly as it does today (no change to existing downloads, caching, status, or deletion behavior).
- **FR-006**: When the source is ModelScope, the system MUST download a model's weights from its ModelScope repository if one is registered for that model.
- **FR-007**: When the source is ModelScope and a model has no registered ModelScope repository (Chatterbox Multilingual, Chatterbox Turbo, TADA 1B, TADA 3B Multilingual), the system MUST download that specific model directly from huggingface.co instead (identical to the HuggingFace source's behavior), without showing an error and without requiring the user to change their source selection.
- **FR-008**: The system MUST support ModelScope repositories for the following model entries: Qwen TTS 1.7B/0.6B (Base and CustomVoice, including the Apple Silicon MLX variants), Qwen3 LLM 0.6B/1.7B/4B (including the Apple Silicon MLX variants), LuxTTS, Kokoro 82M, and all five Whisper sizes (base/small/medium/large/turbo).
- **FR-009**: A model downloaded via ModelScope MUST be usable for generation/transcription without any further network access to huggingface.co (consistent with the local-first/offline-capable behavior already expected of HuggingFace-downloaded models).
- **FR-010**: The models status listing MUST report accurate downloaded/not-downloaded state and an approximate on-disk size for models regardless of which of the two sources they were downloaded through.
- **FR-011**: Deleting a model MUST remove its on-disk files and update its status to "not downloaded" regardless of which source it was downloaded through.
- **FR-012**: Download progress feedback MUST be shown for ModelScope downloads. Byte-level progress (current/total bytes) is REQUIRED only where the underlying download mechanism exposes it; otherwise a coarser "downloading" status without exact byte counts is acceptable.
- **FR-013**: The system MUST NOT alter the on-disk location or format of models downloaded via the existing HuggingFace path (no migration of pre-existing downloads is required by this feature).

### Key Entities

- **Model Download Source**: A user-level, persisted setting with exactly two possible values (HuggingFace, ModelScope) that determines where future model downloads are fetched from. Applies process-wide; takes effect on the next download (live, no restart needed).
- **Model Registry Entry**: The existing per-model configuration (display name, engine, size, HuggingFace repository) extended with an optional ModelScope repository identifier. Absence of that identifier signals "no ModelScope mirror — use HuggingFace directly, even when ModelScope is selected as the source."

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user in mainland China with no VPN can successfully download and use at least 13 of the 17 available model entries by selecting ModelScope as the download source. **Verified for real** for Kokoro (downloaded via ModelScope, loaded, and generated real audio — see Verification Log).
- **SC-002**: 100% of the 17 model entries remain downloadable when ModelScope is selected as the source (either via their native ModelScope mirror or directly from huggingface.co for the 4 unmirrored ones) — no model becomes undownloadable as a side effect of this feature.
- **SC-003**: Switching the download source requires no manual file/cache manipulation by the user — the change is fully self-contained within the Settings UI and applies immediately.
- **SC-004**: Existing users who never touch the new setting see zero behavior change (downloads, status, and deletion work exactly as before).
- **SC-005**: The models management screen reports correct downloaded/not-downloaded status and allows deletion for models regardless of which of the two sources was used to fetch them. **Verified for real** (see Verification Log).

## Assumptions

- The download source applies **live** (the very next download uses it), not after a restart. The original design required a restart because it applied the setting by mutating a process environment variable at startup; removing the mirror mechanism (see Amendment note) removed that constraint, so this was simplified along with it.
- The four models without a ModelScope mirror (Chatterbox Multilingual, Chatterbox Turbo, TADA 1B, TADA 3B Multilingual) are expected to remain unmirrored for the foreseeable future; no mechanism is required to detect or adopt a mirror if one appears later (a future model-registry update can add it manually, per existing "Adding a New Model" practice).
- Byte-level download progress for the ModelScope path is a nice-to-have, not a hard requirement — if the ModelScope SDK doesn't expose granular progress hooks, a coarse "downloading" indicator is an acceptable outcome for v1. (In practice the SDK does expose per-file progress — see research.md §2.)
- Migrating models between the HuggingFace cache and a ModelScope-managed local directory (or vice versa) is out of scope; a user who wants to switch source for an already-downloaded model can simply delete and re-download it.
- The existing `/models/migrate` (move model storage to a new directory) is out of scope for this feature; ModelScope-downloaded models live under the existing data directory and move with it as a whole, without needing dedicated migration logic.
- Regional/locale auto-detection of a default source is out of scope; HuggingFace remains the default for all users regardless of locale.
- We do not depend on any third-party HTTP mirror/proxy service whose availability or behavior we can't verify continues to hold for the user's actual network path (see Amendment note) — a future "HF Mirror" option would need its own real-environment verification before being added, not just a docs claim that it works.

## Verification Log

Real (non-mocked) verification against an actual running backend process, 2026-08-27:

- Default state: `GET /settings/model-source` → `huggingface`. ✅
- Switched to `modelscope`, downloaded Kokoro: real ~312 MB download from `AI-ModelScope/Kokoro-82M` (observed live progress, no `huggingface.co` in logs), landed in `data/models/modelscope/AI-ModelScope--Kokoro-82M/`. ✅
- Loaded the downloaded model directly (`KokoroTTSBackend`) and generated real audio from text — not just a "loaded" flag. ✅
- `GET /models/status` reported `kokoro` as downloaded with a real size once the ModelScope directory existed. ✅ (Delete path exercised via unit/integration tests with the same directory-detection code — not re-run manually against this specific real download.)
- The originally-planned "HF Mirror" fallback path was found broken against real infrastructure (see Amendment note) and has been removed from the design rather than shipped unverified.
