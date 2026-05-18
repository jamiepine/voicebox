# Debug / Code Review Follow-up

This document tracks the CodeRabbit review items handled during the SRT2Voice
stabilization pass, plus the items intentionally deferred to avoid regressions.

Scope:

- development folder: `voicebox_v2`
- no user database, voices, generations, or AppData assets are part of this file
- this is a working debug/recap document, not release notes

## Integrated Fixes

### Dead migration block

CodeRabbit label:

> Dead migration block in `backend/database/migrations.py`.

Action:

- removed the unreachable duplicate migration block in `_migrate_dubbing`
- kept existing migration behavior intact

Validation:

- backend `py_compile`: OK

### Unknown generation mode can leave `final_path` unbound

CodeRabbit label:

> `run_generation` can leave `final_path` unbound if mode is unknown.

Action:

- added an explicit `ValueError` for unknown generation modes
- prevents silent undefined-path failures

Validation:

- backend `py_compile`: OK

### Dubbing project language default

CodeRabbit label:

> `DubbingProject.language` default should not silently default to French.

Action:

- changed database model default from `fr` to `en`
- import behavior remains controlled by request/project data

Validation:

- backend `py_compile`: OK

### Project serialization N+1 queries

CodeRabbit label:

> `_serialize_project` performs repeated per-segment database queries.

Action:

- added contextual segment serialization
- batch-loaded linked generation rows for project serialization
- removed an unnecessary `db.refresh(project)` no-op

Note:

- file-based cut bounds are still read per segment because they are currently
  stored outside the DB; this is acceptable for now and can be optimized later
  if needed.

Validation:

- backend `py_compile`: OK

### Auto-fit status commit safety

CodeRabbit label:

> `auto_fit_segment` commits status before scheduling background work.

Action:

- wrapped the status update / task scheduling section in guarded transaction
  logic
- on scheduling failure, segment/project state is restored instead of leaving
  a stale generating state

Validation:

- backend `py_compile`: OK

### Qwen VoiceDesign prompt handling

CodeRabbit label:

> Qwen VoiceDesign should not pretend to combine audio voice prompts.

Action:

- removed unused prompt-combine import
- `create_voice_prompt` now logs that audio paths are ignored by VoiceDesign
- `combine_voice_prompts` now raises `NotImplementedError`

Rationale:

- VoiceDesign is prompt-driven; it does not support audio voice prompt
  combination as a real backend feature

Validation:

- backend `py_compile`: OK

### Retry versioning

CodeRabbit label:

> `_save_retry` does not create a generation version.

Action:

- `_save_retry` now creates a `GenerationVersion`
- retry output is marked as default for the generation

Validation:

- backend `py_compile`: OK

### Cache key hashing

CodeRabbit label:

> Cache key uses full audio read / MD5-style hashing.

Action:

- cache key now streams audio files in chunks
- switched to SHA-256
- includes reference text in the cache key

Validation:

- backend `py_compile`: OK

### SRT import decoding

CodeRabbit label:

> SRT import falls from UTF-8 directly to CP1252 and can corrupt UTF-16 files.

Action:

- import now tries `utf-8-sig`
- then `utf-16`
- then `cp1252`
- fallback path is logged

Validation:

- backend `py_compile`: OK

### Startup stale status reset

CodeRabbit label:

> Startup reset misses failed dubbing segment generations.

Action:

- startup cleanup now includes `failed` alongside `generating` and
  `loading_model`
- table/column existence checks were added before running the SQL

Validation:

- backend `py_compile`: OK

### Pending vs generating counts

CodeRabbit label:

> Generating segments are counted as pending.

Action:

- project list pending count now includes only `pending`
- `acceptable_count` is now exposed in the project list response

Validation:

- backend `py_compile`: OK

### Cancel task count

CodeRabbit label:

> Cancel count can increment even when no task was cancelled.

Action:

- cancel count now increments only when a cancellation job actually exists

Validation:

- backend `py_compile`: OK

### Model size fallback on retry/regenerate

CodeRabbit label:

> Retry/regenerate uses hardcoded Qwen `1.7B` model size fallback.

Action:

- added engine-aware fallback:
  - existing model size if present
  - `tada` defaults to `1B`
  - size-aware engines default to `1.7B`
  - other engines use `default`

Validation:

- backend `py_compile`: OK

### DirectML pipeline device handling

CodeRabbit label:

> Passing `device=self.device` can fail on Windows DirectML.

Action:

- pipeline creation now omits the `device` kwarg for DirectML devices
- non-DirectML device handling remains unchanged

Validation:

- backend `py_compile`: OK

### Optional temperature forwarding

CodeRabbit label:

> `chunked_tts` forwards `temperature` to backends that may not support it.

Action:

- added signature-based detection before passing `temperature`
- temperature is forwarded only when the backend `generate()` accepts it

Validation:

- backend `py_compile`: OK

### Auto-fit temperature

CodeRabbit label:

> `DubbingAutoFitRequest` is missing `temperature`.

Action:

- added `temperature` with the same bounds as segment generation:
  `0.1 <= temperature <= 1.2`

Validation:

- backend `py_compile`: OK

### `drop_tts_backend_for_engine` exception masking

CodeRabbit label:

> `finally: return True` suppresses unload exceptions.

Action:

- removed the `finally` return behavior
- function now returns `True` only after successful unload
- unload exceptions are logged and re-raised

Validation:

- backend `py_compile`: OK

### Floating generate box VoiceDesign sync

CodeRabbit label:

> Early `return` after selecting VoiceDesign prevents downstream profile sync.

Action:

- removed the short-circuit return after setting `qwen_voice_design` /
  `1.7B`
- effects/personality synchronization can continue

Validation:

- frontend typecheck: OK
- frontend build: OK

### Audio timeline accessibility

CodeRabbit label:

> Icon-only transport/zoom buttons need accessible names.

Action:

- play/pause, stop, zoom out, and zoom in buttons have `aria-label`
- playhead has keyboard support:
  - ArrowLeft / ArrowRight
  - PageUp / PageDown
  - Home / End
- playhead exposes slider semantics and current time text

Validation:

- frontend typecheck: OK
- frontend build: OK

### FFmpeg lookup portability

CodeRabbit label:

> `find_ffmpeg()` is Windows-biased.

Action:

- added non-Windows candidates:
  - sidecar `ffmpeg`
  - `_internal/ffmpeg`
  - `/usr/local/bin/ffmpeg`
  - `/usr/bin/ffmpeg`
- kept existing Windows paths

Validation:

- backend `py_compile`: OK

### SRT parser strictness and metadata

CodeRabbit label:

> SRT parser accepts invalid seconds/minutes and does not tolerate position metadata.

Action:

- minutes and seconds are now restricted to `00-59`
- optional metadata after timestamp values is tolerated
- parsed project still stores clean timecode values

Validation:

- backend `py_compile`: OK

### CUDA docstring / manual download behavior

CodeRabbit label:

> CUDA docs/comments imply auto-download behavior.

Action:

- updated comments/docstring to state that startup only checks status
- CUDA replacement remains manual through the GPU settings action

Validation:

- backend `py_compile`: OK

### Build script NumPy pin visibility

CodeRabbit label:

> NumPy pin happens silently.

Action:

- added a log before pinning `numpy==2.0.0`

Validation:

- backend `py_compile`: OK

### Sidebar label

CodeRabbit label:

> Sidebar route label is still `Dubbing`.

Action:

- renamed sidebar label to `SRT2Voice`

Validation:

- frontend typecheck: OK
- frontend build: OK

### NumPy documentation consistency

CodeRabbit label:

> Documentation mentions inconsistent NumPy versions.

Action:

- aligned VoiceDesign and SRT2Voice docs to the current packaging pin:
  `numpy==2.0.0`

## Validation Summary

Backend:

```text
python -m py_compile
```

Status:

```text
OK
```

Frontend:

```text
bun run typecheck
bun run build
```

Status:

```text
OK
```

Notes:

- root `npm run build` is not a reliable Windows validation command in this
  workspace because it calls `./scripts/build-server.sh`
- direct `app` build is the correct validation for the touched React/TS files

## Deferred / Technical Debt

These items are intentionally not fixed in this pass. They are not ignored;
they are deferred because they touch global behavior or create disproportionate
regression risk compared with the immediate CodeRabbit fixes.

### Scrollbar keyboard support

Status:

```text
Deferred
```

Reason:

- this appears to be absent from base Voicebox as well
- it is therefore not a SRT2Voice regression
- keep as future accessibility improvement

### Lifecycle / concurrent GPU refcount

CodeRabbit label:

> `unload_after` can unload a shared backend while another generation is using it.

Status:

```text
Deferred
```

Benefit:

- avoids unloading a model while another generation still uses it
- would make model lifetime safer if real concurrent generation is introduced

Risk:

- this is a sensitive global Voicebox area, not just SRT2Voice
- a bad implementation can recreate VRAM leaks, infinite generations, or unloads
  that stop working

Project rule:

- no concurrent generation for now
- engines are loaded per task and unloaded when the task completes
- generation must be serialized / queue-managed
- parallel generation is not part of the supported behavior, even if technically
  possible
- the current `load -> generate -> unload` model only works reliably under this
  serialized-task assumption

Point to re-evaluate:

- the unload/load behavior that was added when entering SRT2Voice may interfere
  with global Voicebox state
- this should be rechecked later against the strict rule:
  no concurrent generation, queued tasks only, unload after task completion

### Tauri startup readiness loop

CodeRabbit label:

> Startup should wait for `/health`, not only for child-process presence.

Status:

```text
Deferred
```

Benefit:

- more reliable startup readiness
- backend would be considered ready only when `/health` responds
- avoids false positives where a process exists but the server is not actually
  listening

Risk:

- this is the global Voicebox startup mechanism
- a bad change can prevent the app from starting
- it can recreate backend ghosting / sidecar launch issues

Recommendation:

- treat as a dedicated task
- test startup, shutdown, relaunch, CPU backend, CUDA backend, and absence of
  residual processes

### ProfileForm i18n refactor

CodeRabbit label:

> ProfileForm contains hardcoded strings.

Status:

```text
Deferred
```

Benefit:

- replaces hardcoded UI strings with proper translation keys
- improves consistency with Voicebox i18n

Risk:

- low functional risk, but broad UI surface
- can create missing labels, break existing translation strings, or pollute the
  diff with non-essential changes

Recommendation:

- handle separately as an i18n cleanup PR/task

## Watch List

### VoiceDesign / cloned voice stuck generation

Observation:

- one cloned profile (`jeanne moreau new`) produced repeated failed or
  interrupted generations
- recreated voice profile worked better

Current interpretation:

- no proof of delivery-instruction failure
- possible profile/cache corruption or problematic cloned profile state

Rule:

- do not assume delivery instructions are the cause unless the same failure
  reproduces across multiple healthy cloned profiles

### CUDA / VRAM policy

Current principle:

- SRT2Voice relies on explicit unload after full narration and after auto-cut
- Whisper Turbo has a much lower VRAM footprint than Large
- no hidden server restart should be used as the normal unload strategy

Future work:

- if VRAM unload becomes unreliable again, first inspect model lifecycle and
  queue behavior before reintroducing server restart
