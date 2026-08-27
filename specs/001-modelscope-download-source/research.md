# Phase 0 Research: ModelScope Download Source

## 1. Model → repository mapping (verified live against modelscope.cn)

**Decision**: Extend each `ModelConfig` with an optional `ms_repo_id`. 13 of 17
entries get a value; 4 stay `None` (Chatterbox Multilingual, Chatterbox
Turbo, TADA 1B, TADA 3B Multilingual) and download straight from
huggingface.co — unchanged from today — when ModelScope is selected (see §2
for why there's no mirror fallback for these).

| model_name | HF repo_id | ms_repo_id |
|---|---|---|
| qwen-tts-1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (or `mlx-community/...-bf16` on Apple Silicon) | same id (verified on ModelScope under both orgs) |
| qwen-tts-0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (or mlx-community variant) | same id |
| qwen-custom-voice-1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | same id |
| qwen-custom-voice-0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | same id |
| qwen3-0.6b / 1.7b / 4b (LLM) | `Qwen/Qwen3-*` (or `mlx-community/Qwen3-*-4bit`) | same id |
| luxtts | `YatharthS/LuxTTS` | `hf/YatharthS-LuxTTS` |
| kokoro | `hexgrad/Kokoro-82M` | `AI-ModelScope/Kokoro-82M` |
| whisper-base/small/medium/large/turbo | `openai/whisper-*` | `openai-mirror/whisper-*` |
| chatterbox-tts | `ResembleAI/chatterbox` | *(none)* |
| chatterbox-turbo | `ResembleAI/chatterbox-turbo` | *(none)* |
| tada-1b | `HumeAI/tada-1b` | *(none)* |
| tada-3b-ml | `HumeAI/tada-3b-ml` | *(none)* |

**Rationale**: Confirmed via `GET https://www.modelscope.cn/api/v1/models/{ns}/{name}`
(200 = exists) plus a files listing check to rule out non-transformers-format
mirrors (the `iic/Whisper-*` org publishes OpenAI's original `.pt`
checkpoints, which `transformers.WhisperForConditionalGeneration` can't load
directly — `openai-mirror/whisper-*` publishes proper
`config.json` + `model.safetensors` + tokenizer files, matching what
`pytorch_backend.py` already expects).

**Alternatives considered**: Relying on ModelScope's on-demand `hf/<org>-<repo>`
auto-mirror for every model (observed working for LuxTTS) was rejected —
requesting it for Chatterbox/TADA returned 404, meaning the mirror is not
reliably auto-created on first request; building the architecture around an
unconfirmed, undocumented lazy-mirror behavior is too fragile.

## 2. Download mechanism per source

**Decision**:
- **HuggingFace**: no change.
- **ModelScope**: use the official `modelscope` PyPI package's
  `modelscope.snapshot_download(model_id, local_dir=..., progress_callbacks=[...])`.
  `local_dir` downloads straight to a directory we choose (no dependency on
  ModelScope's own `~/.cache/modelscope/hub` layout). The SDK's
  `progress_callbacks` takes a list of `ProgressCallback` *classes*
  (instantiated per-file as `callback_cls(filename, file_size)`, with
  `update(size)`/`end()` methods) — confirmed against the installed 1.39.1
  SDK, not the function-callback shape some docs describe. Progress is
  reported per-file, not aggregated across the whole snapshot (the SDK
  doesn't expose a repo-wide total up front).
- **No mirror fallback for the 4 unmirrored models** (Chatterbox ×2, TADA
  ×2): they download straight from huggingface.co in every source mode,
  same as today.

**Rationale**: `local_dir` avoids needing to understand or replicate
ModelScope's internal cache directory format. Storing under Voicebox's own
`get_models_dir()/modelscope/<safe-name>/` (rather than disguising it as a
HuggingFace cache entry) avoids relying on `huggingface_hub`'s internal
blob/symlink cache format — which isn't officially a stable public
contract, and symlinks are unreliable on Windows without dev mode.

**Alternatives considered**: Materializing ModelScope downloads into the
exact `models--org--repo/snapshots/<rev>/blobs` layout so every existing
consumer (`is_model_cached`, `scan_cache_dir`, delete, migrate) works with
zero changes was considered and rejected — it depends on `huggingface_hub`
internals that aren't a public contract, requires real symlinks (fragile on
Windows), and is exactly the kind of hidden magic the project constitution
(Principle III) says to avoid. Instead, `is_model_cached`-equivalent checks
and the status/delete routes are made explicitly source-aware.

**Superseded — "HF Mirror" source and mirror-based fallback**: The initial
design (see git history) had a third source, `hf_mirror` (setting
`HF_ENDPOINT=https://hf-mirror.com`), and used it both as its own
selectable option and as the fallback path for the 4 models with no
ModelScope mirror. **Removed after real-world verification** (running the
actual backend against real network, not mocks — spec.md's Assumptions
originally waived this to "acceptable for v1", but it turned out to matter):
`hf-mirror.com`'s resolve endpoint now returns a bare HTTP 308 redirect
straight back to the real `huggingface.co`, with none of the
`X-Repo-Commit`/`ETag` headers `huggingface_hub` requires on the *first*
hop to trust a mirror response. Confirmed against both the pinned
`huggingface_hub` (0.36.2) and an older one (0.25.2) — not a library
version regression, a property of the mirror's current server behavior. A
redirect to the real huggingface.co would also just fail for a user who is
actually blocked from reaching it, defeating the purpose. Decision: don't
depend on a third-party proxy whose availability/behavior we don't
control and can't verify holds for the user's real network. The 4
unmirrored models fall back to plain HuggingFace instead — identical to
today's behavior, not a regression for them.

## 3. Loading a ModelScope-downloaded model without hitting the network

**Decision**: No new mechanism needed. `backend/utils/hf_offline_patch.py`
already forces `HF_HUB_OFFLINE=1` for the duration of any `from_pretrained`
call when the model is already cached (`force_offline_if_cached`). Backends
already call this today. Once a backend's `_get_model_path()` returns a
local directory path (instead of a HF repo id) for a ModelScope-sourced
model, `from_pretrained(local_dir)` loads straight from disk — this is a
standard, well-supported `transformers`/`huggingface_hub` code path (loading
from a local directory instead of a repo id) and never touches the network
regardless of offline-mode state.

**Rationale**: Reuses an existing, tested mechanism instead of adding a new
one.

## 4. Where the download-source setting is persisted, and when it applies

**Decision**: A plain JSON file at `get_data_dir()/model_source.json`
(`{"source": "huggingface" | "modelscope"}`), read directly with
`pathlib`/`json` — not a SQLAlchemy-backed settings row like
`CaptureSettings`/`GenerationSettings`. `resolve_model_source()` reads it
fresh on every call (no caching), so a change **applies immediately, live**
— no restart required.

**Rationale (superseded from the original design)**: The original decision
here was "JSON file because `HF_ENDPOINT` must be set before any ML
library import, earlier than the DB layer is ready" — and, following from
that, a restart-required UX (config.py applied the setting once, at
process start). Both of those were specifically about the now-removed
`hf_mirror` mechanism's `HF_ENDPOINT` env var (see §2). With that gone,
there is no env var to set and no import-ordering constraint to work
around — `resolve_model_source()` already reads
`backend.utils.model_source.get_model_source()` lazily, at the moment a
model is actually loaded, not at process startup. Keeping the
restart-required framing after removing its only justification would be
dead ceremony (misleading UX, an `active_source` field that's always
trivially equal to `source`) — so this iteration drops it: the setting is
still a flat JSON file (still simpler than a DB row for a single string,
and this project has no other reason to reach for SQLAlchemy here), but it
takes effect on the very next model load, not the next restart.

**Alternatives considered**: A DB-backed row read via a raw `sqlite3`
connection was rejected as needless complexity for a single string value —
a JSON file achieves the same result with far less code and no
schema/migration to maintain. (This reasoning holds regardless of the
`hf_mirror` removal — it's about avoiding a DB dependency for one flag, not
about import-ordering timing.)

## 5. New dependency

**Decision**: Add `modelscope` to `backend/requirements.txt`. Import it
lazily (inside the function that performs a ModelScope download), not at
module load time — mirrors how other optional/heavy engine dependencies
are already handled in this codebase (e.g. `huggingface_hub.snapshot_download`
is imported inline in `chatterbox_turbo_backend.py` / `hume_backend.py`,
not at module top).

**Rationale**: Keeps startup fast and avoids a hard import-time dependency
on a package that's irrelevant unless the user has actually selected
ModelScope as their source.
