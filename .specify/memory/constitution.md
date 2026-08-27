<!--
Sync Impact Report
- Version change: (none) → 1.0.0
- Modified principles: n/a (initial ratification)
- Added sections: Core Principles (5), Technical Constraints, Governance
- Removed sections: none
- Templates requiring updates: none (first ratification, no dependent
  templates existed before this constitution)
- Deferred TODOs: RATIFICATION_DATE set to the date this constitution was
  first authored, since no earlier project constitution existed to recover
  a true adoption date from.
-->

# Voicebox Constitution

## Core Principles

### I. Model registry is the single source of truth
Every downloadable model (TTS, STT, LLM) is described by exactly one
`ModelConfig` entry in `backend/backends/__init__.py`. Routes, status
checks, and the frontend model list all derive from
`get_all_model_configs()` — they MUST NOT hardcode a second list of model
names or repo IDs. Adding a model variant means adding a `ModelConfig`
entry, not touching route handlers.

### II. Backend abstraction via Protocol, not inheritance
TTS/STT/LLM engines implement the `TTSBackend` / `STTBackend` / `LLMBackend`
Protocols defined in `backend/backends/__init__.py`. Shared logic (device
detection, cache checks, progress tracking, voice-prompt combination) lives
in `backend/backends/base.py` and is called by each engine, not
copy-pasted per engine. Engine-specific quirks stay inside that engine's own
file.

### III. Explicit configuration over hidden runtime magic
Startup-time behavior (data directory, model cache location, model download
source) is resolved once from environment/settings at process start,
following the existing `VOICEBOX_MODELS_DIR` precedent — not toggled via
runtime monkey-patching unless the underlying library leaves no other
option. The only sanctioned exception is `backend/utils/hf_offline_patch.py`,
which patches `huggingface_hub`/`transformers` cached constants for a
narrow, documented reason (those libraries read `HF_HUB_OFFLINE` once at
import time). Any new patch of this kind MUST document why a
non-invasive alternative isn't possible.

### IV. Local-first privacy is non-negotiable
Models, voice profiles, and captures never leave the user's machine except
for an explicit, user-initiated model download from a configured source
(HuggingFace, an HF mirror, or ModelScope) or the opt-in Voicebox Cloud
sync. No telemetry, analytics, or generation content is sent anywhere by
default. Any feature that introduces a new outbound network call must be
justified against this principle before being added.

### V. Cross-platform parity without platform-specific hacks
Voicebox ships on macOS (MLX/Metal), Windows (CUDA), Linux, AMD ROCm, Intel
Arc, and Docker. Features MUST work — or degrade explicitly and visibly —
on all of them. Do not rely on platform behavior that silently fails
elsewhere (e.g. symlinks, which require elevated privileges on Windows) or
special-case a subset of platforms in the review's blocking issues.

## Technical Constraints

- Backend: Python / FastAPI (`backend/`). Frontend: Tauri (Rust) + web UI
  (`app/`, `web/`). Model downloads currently assume `huggingface_hub`'s
  cache layout (`HF_HUB_CACHE`, `models--org--repo`, `scan_cache_dir()`);
  any new download source must either conform to that layout or be clearly
  isolated from it so existing cache-scanning code isn't silently broken.
- New third-party SDKs (e.g. a model-hub client) are added to
  `requirements.txt` only when the functionality can't be achieved through
  the existing `huggingface_hub`-based path (e.g. pointing `HF_ENDPOINT` at
  a compatible mirror).

## Governance

This constitution governs Voicebox backend/frontend development practices
specific to this project. General coding, review, and process rules come
from the developer's global agent rules and take precedence where this
document is silent; where the two conflict, the stricter constraint wins.

Amendments: propose the change in the PR/commit that motivates it, update
this file in the same change, and bump the version per semantic versioning
(MAJOR: incompatible principle removal/redefinition; MINOR: new principle
or materially expanded guidance; PATCH: wording/clarification only).

**Version**: 1.0.0 | **Ratified**: 2026-08-26 | **Last Amended**: 2026-08-26
