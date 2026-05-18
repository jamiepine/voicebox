# SRT2Voice Module Notes

This document is the minimal handover for the `SRT2Voice` module, with a
strong focus on the Windows sidecar/server setup.

The main rule is simple:

- `SRT2Voice` must behave like original Voicebox on startup.
- `voicebox.exe` must start its own backend sidecar.
- No external `.bat`, no manual uvicorn, no fallback launcher in normal use.


## Version Log

### 2026-05-17 - Technical state for rebuild/extraction

- Added a technical reconstruction checklist so SRT2Voice can be extracted,
  rebased, or reconnected without rediscovering the same Windows/CUDA/backend
  pitfalls.
- The checklist distinguishes SRT2Voice-specific files from global Voicebox
  corrections that must also be carried forward.

### 2026-05-17 - Profile cache resilience note

- Keep full SRT2Voice narration generations visible when useful, because they
  provide a reusable full WAV trace similar to Stories.
- Internal SRT2Voice artifacts should remain filtered or scoped to SRT2Voice:
  auto cuts, retries, debug files, temporary alignment assets.
- Added a recovery note for old cloned profiles that appear to hang during Qwen
  generation while a freshly recreated clone from the same source audio works.
  The likely suspects are stale profile metadata, reference text mismatch, or a
  bad cached voice prompt, not necessarily a corrupted source WAV.
- Future tooling should allow profile-scoped `Rebuild voice prompt cache` and
  `Clear voice prompt cache for this voice` actions.


## Scope

This fork adds a `SRT2Voice` module on top of Voicebox `v0.5.0`.

It must not:

- change the original Tauri startup model
- require a separate backend launch command
- leak SRT2Voice-specific behavior into other modules

It may:

- add backend routes and services under `backend/routes/dubbing.py` and
  `backend/services/dubbing.py`
- add frontend UI under `app/src/components/DubbingTab`
- add database models for SRT2Voice projects/segments


## Current Contract / Do Not Break

This section overrides older exploratory notes when there is any ambiguity.

SRT2Voice current stable workflow:

- import SRT into a dedicated SRT2Voice project
- keep editable SRT segments as the timing/text source of truth
- generate one full narration WAV from cleaned SRT text
- keep the full narration WAV as the stable voice-continuity source
- use Auto Cut/manual cut to mount the full WAV back onto the SRT timeline
- export the mounted timeline WAV, not blindly the raw full WAV
- export package includes full WAV, mounted WAV, SRT, debug/alignment files

Never break:

- SRT segments must not disappear when regenerating a full WAV
- regenerating a full WAV invalidates old full WAV timeline clips, cuts, and
  debug files, but not the SRT segment rows
- helper/reference clips are immutable visual SRT references
- deleted clips must not remain in playback/export as ghosts
- SRT2Voice must remain isolated from normal Voicebox generation logic
- full narration generations may remain visible as useful audio history, but
  internal artifacts/retries/cuts/debug files should remain scoped to SRT2Voice
- `unload_after=True` and explicit CUDA/cache cleanup must remain active after
  full narration and Auto Cut work
- project selection must use project IDs, not names, because duplicate names are
  allowed


## Technical Reconstruction / Rebranch Checklist

Use this section when rebuilding from a clean Voicebox v0.5 source tree or when
extracting SRT2Voice into another fork.

### Backend SRT2Voice files

Carry forward these module-specific backend files and changes:

- `backend/routes/dubbing.py`
- `backend/services/dubbing.py`
- `backend/services/srt_parser.py`
- SRT2Voice database models/fields in `backend/database/models.py`
- SRT2Voice request/response schemas in `backend/models.py`
- SRT2Voice router registration in the FastAPI app
- export/package endpoints under `/dubbing/projects/...`
- memory release endpoint `/dubbing/release-memory`

Critical backend behavior:

- full narration endpoint creates a `generations` row with
  `source = dubbing_full_narration`
- deterministic full narration ids start with `dubbing-full-narration-`
- Auto Cut derived ids start with `dubbing-cut-`
- full narration clean text is persisted before generation
- timing/debug JSON files live under
  `generations/dubbing_full_narration_timing`
- cut/debug JSON files live under `generations/dubbing_cuts/<project_id>`
- `Export Timeline WAV` prefers mounted timeline/cuts, then full narration,
  then legacy segment audio only as fallback

### Frontend SRT2Voice files

Carry forward these module-specific frontend files and changes:

- `app/src/components/DubbingTab/DubbingTab.tsx`
- shared timeline components under `app/src/components/AudioTimeline`
- SRT2Voice navigation/menu entry
- SRT2Voice API client additions
- project-ID persistence for selected SRT2Voice project
- UI logic that distinguishes "server unavailable" from "no projects"

Critical frontend behavior:

- duplicate project names are allowed; selection must persist by ID
- changing project unloads/refreshes the current timeline view
- Generate narration, Auto Cut, export actions must reflect real task state
- Suggested Tempo appears only when Auto Cut data exists
- Qwen-only controls such as delivery instructions, temperature, and pace must
  not be shown as if they apply to every engine

### Global Voicebox corrections that must be kept

These are not SRT2Voice-only, but this fork depends on them:

- voice prompt cache must not keep CUDA tensors alive between generations
- cached cloned voice prompts should be stored/reloaded on CPU
- cached prompts are moved to the active device only immediately before Qwen
  generation
- global model unload should clear backend references and CUDA cache when
  requested
- Voicebox v0.5 engines must remain registered: Qwen, Qwen CustomVoice, Qwen
  VoiceDesign, Chatterbox, Chatterbox Turbo, LuxTTS, Kokoro, TADA 1B, TADA 3B
  Multilingual
- TADA uses the local DAC shim rather than requiring the full
  `descript-audio-codec` dependency chain
- Kokoro/Misaki Windows packaging needs the working phonemizer/Misaki path
- LuxTTS needs defensive text normalization/padding so short inputs do not trip
  Conv1d kernel-size errors

### Windows runtime / CUDA contract

Do not change these runtime locations casually. The examples below are
intentionally generic and must resolve through the normal Voicebox app data
directory at runtime:

- app data root:
  `%APPDATA%\sh.voicebox.app`
- runtime CUDA backend:
  `%APPDATA%\sh.voicebox.app\backends\cuda`
- expected CUDA exe:
  `%APPDATA%\sh.voicebox.app\backends\cuda\voicebox-server-cuda.exe`
- source build output:
  `backend/dist/voicebox-server-cuda`
- local DB:
  `%APPDATA%\sh.voicebox.app\voicebox.db`

Before replacing the runtime CUDA backend:

- stop running `voicebox.exe`, `voicebox-server.exe`, and
  `voicebox-server-cuda.exe`
- backup the existing AppData CUDA backend directory
- copy the rebuilt `backend/dist/voicebox-server-cuda` contents into the AppData
  CUDA backend directory
- smoke test `/health` on a temporary port before normal use

### Build / deploy sequence

Preferred safe sequence:

1. Build frontend: `npm.cmd run build`
2. Compile backend files with Python if only Python files changed.
3. Rebuild CUDA backend when backend/runtime dependencies changed.
4. Backup and deploy CUDA backend to AppData.
5. Build Tauri from the repository `tauri` directory, not from `app`.
6. Treat NSIS bundler failure separately if `voicebox.exe` and MSI were already
   produced.
7. Launch app and verify:
   - server starts by itself
   - `/health` returns `200`
   - `/dubbing/projects` returns `200`
   - CUDA is detected in Settings > GPU
   - existing SRT2Voice projects still load by ID


## Timing / Pace Rules

For natural dubbing, pace correction must not be treated as a per-segment
micro-adjustment.

Rules:

- preferred pace correction range: `0.8x` to `1.2x`
- pace should be computed on a **project-level context** or a **phrase/group of
  segments**
- do **not** treat each segment as an isolated acceleration / slowdown target
- avoid abrupt per-segment pace jumps, because they create audible
  acceleration/deceleration artifacts between adjacent subtitles

Manual control policy:

- expose pace control **only inside the SRT2Voice module**
- allow manual pace override at project level
- allow manual pace override at phrase / segment-group level
- do **not** expose pace override at single-segment level
- if a manual override exists, it must take priority over automatic timing
  logic
- if no manual override exists, automatic timing logic may suggest or apply a
  pace factor inside the allowed range

Implementation notes:

- project override field: `dubbing_projects.pace_override`
- group override field: `dubbing_projects.group_pace_overrides`
- segment group field: `dubbing_segments.pace_group_id`
- group assignment is based on phrase punctuation, not on isolated SRT block
  timing
- manual pace is applied during SRT2Voice generation/regeneration, immediately
  after the TTS WAV is produced
- pace processing must preserve pitch: do not use sample-rate tricks that alter
  voice height or character
- current implementation uses FFmpeg `atempo` when a local `ffmpeg.exe` is
  available; this is scoped to SRT2Voice only
- do not use `librosa.effects.time_stretch` as the automatic production
  fallback: it preserves pitch but can add phase / reverb / wet artifacts on
  generated speech
- if FFmpeg is not available, skip destructive pace processing rather than
  degrading the voice
- FFmpeg `rubberband` should not be assumed available because it requires an
  FFmpeg build compiled with `--enable-librubberband`

API:

- `PUT /dubbing/projects/{project_id}/settings`
- `PUT /dubbing/projects/{project_id}/groups/{group_id}/pace`

Priority order:

1. group manual override
2. project manual override
3. automatic group pace
4. neutral `1.0x`

Do not apply these rules to normal Voicebox generation.


## SRT Readability Metrics

The Dubbing / SRT2Voice module should help the user identify SRT segments that
are too dense before generation.

These metrics are editing aids, not hard generation constraints.

Reference targets:

- global subtitle readability standard: about `15 CPS` (characters per second)
- French narration target: about `2.2 words per second` on average
- these values should be treated as guidance for professional training videos,
  not as automatic failure thresholds

Recommended UI behavior:

- compute CPS for every SRT segment:
  `visible_character_count / segment_duration_seconds`
- compute words per second for every SRT segment:
  `word_count / segment_duration_seconds`
- show the metrics in the Segments panel near each segment timing/status; this
  is currently calculated client-side immediately after SRT import from the
  returned segment text and timecodes
- use a gentle warning when a segment is above the target range
- suggest that the user edits the SRT text or timecodes when density is too
  high
- do not mark a segment as failed only because CPS or words/second is high

Counting policy:

- count visible text only, not SRT index or timecode
- ignore leading/trailing whitespace
- collapse repeated whitespace before counting words
- preserve French accents
- apostrophes may split words for matching/alignment (`j'ai` -> `j ai`), but
  for user-facing readability metrics either policy is acceptable if consistent

Why this matters:

- high CPS usually predicts a delivery that feels rushed
- high words/second often explains why the generated narration overflows the
  SRT time window
- exposing the metric lets the user manually redistribute words between
  adjacent segments before regenerating the full narration
- this is especially useful for training videos where interface demonstrations
  must stay synchronized with narration


## AI Dubbing V2 Goal: Whole-SRT Narration Homogeneity

### Current limitation

The current Dubbing implementation is functional but still too mechanical:

- one SRT block becomes one TTS generation
- each segment is treated independently
- Qwen receives no stable linguistic/prosodic context between adjacent
  subtitles
- tone, energy, phrasing, breath placement, and sentence contour can drift from
  one segment to the next

This is especially audible when a complete spoken sentence is split across
several SRT blocks. In that case, segment-by-segment generation creates cuts
inside what should be one continuous phrase.

Further observation:

```text
Phrase groups are still not enough.
```

Even if segments are grouped by sentence, generation remains split into several
independent model calls. With Qwen VoiceDesign/CustomVoice this can still reset
or weaken the delivery instruction every one or two generations, producing
audible drift in tone, phrasing, intensity, and narration posture.

Therefore the only reliable logical generation unit for high-quality Dubbing is
the complete SRT project.

Compatibility note:

- this whole-SRT generation mode is also useful with cloned voices
- for cloned voices, do **not** rely on delivery instructions for style control
- the benefit comes from one continuous TTS call, persistent reference prosody,
  punctuation, and cleaned text continuity
- for VoiceDesign and CustomVoice, delivery instructions remain useful and are
  sent as a single prompt for the full narration

Instruction limits:

- Qwen's official VoiceDesign `voice_prompt` limit is 2048 characters
- Alibaba Qwen instruction control documents `instructions` as 1600 tokens
- the app accepts up to 2000 characters for dubbing `instruct` / `style_prompt`
- recommended practical prompt length remains short: roughly 10 to 40 words

### Target behavior

The Dubbing module must process the SRT as one continuous narration, not as
isolated subtitle rows or independent phrase groups.

Beyond timing, the voice must remain constant across the whole dubbing project:

- same perceived speaker identity
- same tone
- same phrasing style
- same intensity/energy
- same articulation level
- same narration posture

### V1 cleaned SRT input

Before the full narration is sent to TTS, the SRT is cleaned internally. This is
transparent to the user.

Input kept by the app:

- segment id
- SRT index
- start timecode
- end timecode
- editable segment text

Input sent to Qwen:

- natural text only
- no SRT index
- no timecode
- no `-->`
- blank line between SRT blocks for now

Example:

```text
Bonjour, j'ai le plaisir de vous proposer ce bref tutoriel ayant pour titre : Introduction au fond de dossier. C'est parti ! Dans portefeuille,
```

The cleaned text is generated from persisted segment rows, not by sending raw
SRT or JSON to Qwen. JSON remains an internal application structure only.

Current persistence/debug rule:

- Before full WAV generation, SRT2Voice persists the cleaned narration text as
  a debug/audit artifact.
- Primary path:
  `%APPDATA%\\sh.voicebox.app\generations\srt2voice_clean_text\<project_id>.txt`
- Human-readable debug copy:
  `%APPDATA%\\sh.voicebox.app\generations\dubbing_full_narration_timing\<safe_project_name>__<full_narration_generation_id>.txt`
- The human-readable copy includes the stable full narration timing JSON id, so
  it can always be associated with
  `dubbing_full_narration_timing/<full_narration_generation_id>.json`.
- The same text is still stored in the `generations.text` database field for
  the `dubbing_full_narration` row.
- The clean text file is transparent to the user and is included in export
  packages as `debug/clean_srt_narration.txt`.
- The clean text is a single flattened line: SRT timecodes are removed,
  `\n`, `\r`, and `\t` become regular spaces, repeated whitespace collapses,
  and light typography normalization is applied for the selected language
  before sending text to TTS.

Current beta endpoint:

- `POST /dubbing/projects/{project_id}/generate-full-narration`
- creates one `generations` row with source `dubbing_full_narration`
- uses a deterministic generation id prefixed with `dubbing-full-narration-`
- does not delete or overwrite segment-level generations
- while generation is active, the UI must show a visible running state in the
  header controls, the Generation Controls panel, and the timeline lane
- when generation completes or fails, the backend records the real task runtime
  with a dedicated monotonic timer, not by comparing database `created_at` to
  the WAV file timestamp
- the UI can display both:
  `Duration: xx.xxx s` for the generated narration audio length and
  `Generated in xx.x seconds` for the actual generation runtime
- timing metadata is stored as a sidecar JSON under
  `generations/dubbing_full_narration_timing/<generation_id>.json`
- this sidecar is reset before each new full narration run so a reused stable
  generation id cannot report stale multi-hour generation times
- `POST /dubbing/projects/{project_id}/post-process`
- cuts the completed full narration WAV into deterministic SRT-segment WAV
  files in the current pre-Whisper pass
- stores each cut as a derived `generations` row with source
  `dubbing_segment_cut`
- uses deterministic ids prefixed with `dubbing-cut-`
- does not require a database migration; cuts can be rebuilt from the full
  narration and the current SRT timing
- `Export Timeline WAV` prefers post-processed cuts when they exist, then the
  full narration audio, then legacy segment-level audio

### V3.1 isolation rule

SRT2Voice must stay stateless across project switches and generation cycles:

- switching `project_id` unloads the current SRT2Voice timeline view before
  loading the next project
- the frontend defaults are `pace = 1.0` and `temperature = 0.9`
- active values are loaded from the active project database row only
- regenerating a full narration purges the persisted SRT2Voice timeline clips
  for that project before the new audio is queued
- regenerating a full narration invalidates Auto Cut/manual cut artifacts and
  resets full-narration timing metadata before work starts

Primary constraint:

```text
The SRT timecodes remain the timeline contract.
```

This means:

- the complete SRT text is used to generate one coherent narration
- the SRT timecodes are then used as alignment/export constraints
- segment start times remain the reference grid for remounting against the
  source video
- the module must not lose the SRT start timing contract required by UI
  demonstrations and training videos

The external video remounting step is out of scope. Voicebox Dubbing only needs
to export an audio WAV that can be aligned with the source video by another
tool.

### Export package requirements

The final Dubbing export should favor one complete package instead of multiple
individual downloads.

Required package behavior:

- provide a dedicated `Export Package` action
- generate one `.zip` archive
- include the original full narration WAV generated from the cleaned SRT text
- include the post-processed / resequenced timeline WAV
- include every cut segment as an individual WAV under a `segments/` directory
- include an updated SRT file if the user edited segment text after import
- include a machine-readable `manifest.json`
- expose `GET /dubbing/projects/{project_id}/export-package`

Recommended package layout:

- `audio/full_narration.wav`
- `audio/resequenced_timeline.wav`
- `segments/segment_0001.wav`
- `segments/segment_0002.wav`
- `srt/original.srt`
- `srt/edited.srt`
- `manifest.json`

The edited SRT must reflect the current Dubbing project state:

- current segment order
- current editable text
- current editable start/end timecodes
- no stale text from the originally imported file

Current validated implementation note:

- the stable source for SRT2Voice is the complete full narration WAV generated
  from the cleaned SRT text
- this full WAV gives the best voice persistence, because Qwen keeps one
  continuous delivery context across the whole project
- the SRT timecodes remain the visual and export reference grid
- the full narration WAV must remain accessible after cuts are created
- the full narration WAV is generated on timeline lane `0` by default

Validated workflow 1: manual cut

- the user generates the full SRT narration first
- the user manually cuts the full WAV in place on the timeline, like in Stories
- the cut operation must behave like a real scissors operation: no duplicated
  ghost clip, no hidden stale audio, no playback of deleted audio
- manual cuts must stay at their real timeline positions
- moving a cut clip changes its playback/export position
- deleted cuts must be removed from playback and export immediately
- this workflow remains the quality fallback when automatic alignment is not
  good enough

Validated workflow 2: Auto Cut

- `Auto Cut` also starts from the full narration WAV
- Voicebox's existing Whisper backend is used to request word-level timestamps
  from the full narration WAV
- Auto Cut does not hard-code Whisper Large: it selects the best locally cached
  Whisper model in this order: `turbo`, `large`, `medium`, `small`, `base`
- therefore, if the user has installed Whisper Turbo from the Models /
  Transcription screen, Auto Cut should use Turbo automatically and only fall
  back to Large when Turbo is not cached
- this keeps alignment local/offline and avoids downloading a different
  Whisper model during Auto Cut
- the language selected in the SRT2Voice project must match the SRT/narration
  language used for alignment
- if the project language and detected/expected SRT language do not match,
  Auto Cut should show a warning and create no cuts, because forcing Whisper
  with the wrong language produces unreliable word timestamps and bad cuts
- matching is case-insensitive and punctuation-insensitive; apostrophes become
  spaces (`j'ai` -> `j ai`) while French accents are preserved
- automatic boundaries are not cut directly on a word end timestamp
- the system identifies the boundary between the last matched word of segment
  `N` and the first matched word of segment `N + 1`
- punctuation drives the boundary strategy:
  - hard punctuation (`.`, `!`, `?`, `…`) uses RMS/ZCR acoustic detection to
    preserve natural sentence-final breathing
  - soft punctuation (`,`, `;`, `:`) and no-punctuation continuations use a
    hybrid rule: prefer the mathematical midpoint between matched words when
    there is no reliable silence, but trust RMS/ZCR when it finds a clean,
    stable low-energy gap between the true tail of the previous word and the
    true attack of the next word
- this avoids artificial silence on continuous phrases while still protecting
  long French endings, nasals, fricatives, aspirations, and trailing phonemes
- if the acoustic gap is shorter than the safety threshold or drifts too far
  from the semantic midpoint, Auto Cut uses the semantic midpoint and relies on
  the tiny micro-fade used during export/playback to avoid clicks
- after source cuts are computed, each cut is placed on the timeline by matching
  the acoustic attack of its first matched word to the SRT segment start
- this first-word placement step must not create new cuts or alter cut source
  bounds; it only repositions already computed clips on the timeline
- the first-word placement uses RMS energy around the Whisper first-word
  timestamp; the clip may start slightly before the SRT segment so the real
  spoken word begins on the SRT timecode
- timeline placement then applies punctuation-specific adjacency:
  - no punctuation means strict continuity, but the next segment remains the
    anchor: clip `N+1` keeps its SRT/first-word attack placement, and clip `N`
    is shifted so its end reaches that anchor; no artificial delay is inserted
  - soft punctuation (`,`, `;`, `:`) now follows the same adjacency rule as no
    punctuation: clip `N+1` keeps its SRT/first-word attack placement, and clip
    `N` is shifted so its end reaches that anchor; this avoids audible timeline
    gaps that vary by voice
  - hard punctuation keeps the first-word/SRT attack placement because a real
    sentence break can legitimately contain a larger pause
- SRT helper blocks are never modified by Auto Cut placement; they remain fixed
  visual references derived only from the current SRT segment text and timecodes
- if word matching or RMS gap detection fails, the system falls back to the
  proportional SRT-ratio estimate and marks the cut source as fallback
- if the resulting cut is longer than the SRT window, the audio is preserved and
  the segment is marked as `timing overflow`; it must not be truncated
- every Auto Cut run writes an inspection file at
  `generations/dubbing_cuts/<project_id>/word_matching_debug.json`
- the export package also includes this file as
  `debug/word_matching_debug.json`
- the debug file includes `placements` entries with
  `first_word_start_ms`, `refined_first_word_attack_ms`,
  `cut_source_start_ms`, `leading_offset_ms`, `timeline_start_ms`, and
  `placement_source`
- boundary debug entries include `punctuation_kind`, `semantic_mid_ms`,
  `semantic_gap_ms`, `acoustic_cut_ms`, `acoustic_gap_ms`,
  `acoustic_drift_ms`, and `cut_method` so soft/hard decisions can be audited
  without guessing from the UI

Shared workflow rules:

- `Export Timeline WAV` must export the current mounted timeline result, not
  blindly export the raw full narration WAV
- `Export Package` must include the full narration WAV, the mounted timeline
  WAV, segment/cut assets, SRT files, manifest, and debug files
- segment start/end timecodes are editable directly in the Segments panel,
  alongside the editable SRT text, for manual recut/reposition workflows
- users can delete an SRT segment from the Segments panel when they merge its
  text into a neighboring segment and adjust the remaining timecodes
- any editable SRT structural change, including text edit, timecode edit, or
  segment deletion, invalidates and deletes the full narration WAV and all
  derived cuts; the project must regenerate them from the updated SRT
- future UI work must add mute / unmute per timeline line

Future alignment notes:

- WhisperX remains a possible refinement layer, but it is no longer required to
  validate the current Auto Cut concept
- if WhisperX is added, it must be visible in `Models > Transcription` rather
  than acting as a hidden dependency

Future tempo-fit note:

- after TTS or future V2V generation, measure the generated audio duration
  `D_ia` against the target SRT duration `D_srt`
- if the difference is small, for example below roughly `10%`, a light
  post-processing pass may use FFmpeg `atempo` or SoX to fit the audio duration
  more closely
- this must preserve pitch and perceived voice character
- this should remain optional and conservative; do not use it to hide badly
  overcrowded SRT text
- if the required correction is larger than the safe range, prefer surfacing
  CPS / words-per-second warnings and asking the user to edit text or timecodes
- this idea belongs after the full narration / cut workflow, not inside the
  prompt as delivery instructions

The manifest should map each exported segment back to:

- SRT index
- segment id
- start/end timecode
- source text
- edited text
- generated audio filename
- actual duration
- delta / overflow status
- source track, e.g. full narration or post-processed cut

### SRT linguistic analysis

SRT segments should still be analyzed linguistically, but this analysis must not
define the main generation unit.

Purpose of linguistic analysis:

- preserve punctuation and sentence continuity in the full script
- help the UI show phrase/sentence boundaries
- support future word/phrase alignment
- help users understand where text edits affect the narration

Initial grouping rules:

- continue a group until terminal punctuation is reached
- terminal punctuation includes `.`, `!`, `?`, `...`, and closing quotes or
  parentheses after them
- commas, semicolons, colons, parentheses, and quotes are rhythm markers, not
  necessarily group terminators
- manual text edits must invalidate/recompute the affected group

Example:

```text
Segment 1: Bonjour, j'ai le plaisir de vous proposer ce bref tutoriel ayant
Segment 2: pour titre : Introduction au fond de dossier...
```

These two SRT rows should be treated as one sentence/phrase for script
construction and future alignment, but not as an independent generation unit in
the high-quality mode.

### Generation strategy

The current stable mode remains available:

```text
mode = segment
one SRT segment -> one generation
```

The V2/Beta mode should add:

```text
mode = whole_srt
complete SRT script -> one coherent TTS generation
```

The whole-SRT generation text is the concatenation of all editable segment
texts, preserving punctuation and natural sentence boundaries.

Important limitation:

```text
Phrase grouping alone does not guarantee voice persistence across the full project.
```

Generating one phrase group after another can still cause drift between groups:

- slightly different speaker color
- different emotional intensity
- inconsistent rhythm
- changed narration posture
- abrupt energy reset at phrase boundaries

Therefore phrase grouping must be considered an intermediate/diagnostic layer,
not the target generation architecture.

### Project-level voice/session layer

Dubbing needs a stable generation context that is reused across all phrase
groups in the same project.

Conceptual target:

```text
Dubbing project -> one voice session/style contract -> one full narration
```

The session contract should include:

- selected profile id
- resolved engine
- language
- voice/design prompt or reference voice metadata
- short delivery instruction
- punctuation policy
- optional manual pace override
- optional reference generation/audio anchor

The session contract must be built once per project/generation batch and used
for the complete narration. It must not be rebuilt with different wording for
every segment or phrase group, because that reintroduces drift.

Recommended instruction shape:

```text
Professional documentary narration with clear articulation, natural French prosody, punctuation-aware pauses, and steady tone.
```

Keep it short and stable. Do not append retry/timing text dynamically.

The generation instruction should stay short and natural. It should focus on
voice continuity and punctuation-aware delivery, not on hard timing:

```text
Use natural human prosody with realistic pauses, punctuation-aware pacing, and smooth conversational intonation.
```

Do not reintroduce forced timing instructions such as:

```text
Timing fit retry...
Speak noticeably faster...
Minimize pauses...
Keep the sentence very compact...
```

Those instructions caused unnatural pacing and may create hallucinations or
truncated delivery.

### Engine-specific expectation

VoiceDesign and Qwen CustomVoice are the best targets for delivery instruction
control.

VoiceDesign:

- use the same `design_prompt` for the whole project
- use the same delivery instruction for the whole narration
- do not mutate delivery instructions per segment
- this is currently the best candidate for project-level voice consistency

Qwen CustomVoice:

- use the same preset voice for the whole project
- use the same delivery instruction for the whole narration
- expect better instruction control than cloned/Base voices

For Qwen Base/cloned voices:

- delivery instructions may be ignored or have weak effect
- continuity must rely mostly on punctuation, text chunking, and reference
  audio/prosody
- do not assume bracket tags like `[sad]`, `[slow]`, `[laugh]` work with Qwen
  Base cloning

No engine-specific behavior may leak outside Dubbing unless it is part of the
general Voicebox engine contract.

### Mapping full narration audio back to SRT timing

The hard problem is not only generating coherent audio. The result must be
mapped back to a timeline constrained by the SRT.

V2 should use a conservative first implementation:

1. Generate the complete SRT script as one audio file.
2. Store this full narration generation separately from individual segment
   generations.
3. Place the full narration audio at the first SRT start time.
4. Keep each SRT segment's original start time as metadata and UI reference.
5. Do not split audio internally until alignment is implemented.

This gives maximum voice/delivery persistence while preserving the SRT project
start anchor.

Later, if needed, add alignment:

- use WhisperX or another forced aligner to map generated words back to segment
  boundaries
- derive per-segment audio spans from word timings
- keep the generated full narration as the source of truth

### Post-generation Whisper/WhisperX alignment

Whole-SRT generation solves voice persistence, but it does not by itself tell
us where each original SRT segment appears inside the generated narration.

After generating the full narration WAV, Dubbing should run a transcription /
alignment step:

```text
full SRT text
-> full narration WAV
-> Whisper or WhisperX transcription/alignment
-> fuzzy matching against editable SRT segments
-> segment-to-audio span map
-> timeline WAV export
```

Purpose:

- re-identify the spoken text inside the generated full narration
- associate each detected audio span with the corresponding SRT segment
- avoid relying on naive proportional duration splitting
- make the final WAV remountable against the original video timeline

Recommended V1:

1. Generate one full narration WAV.
2. Transcribe/align that WAV locally.
3. Extract word-level or phrase-level timestamps when available.
4. Normalize both SRT text and transcription text for comparison.
5. Use fuzzy matching to map each SRT segment to the closest transcription
   span.
6. Store the resulting segment/audio span map.
7. Use that map to cut/place audio on the export timeline.

WhisperX is the preferred candidate because it can provide finer alignment than
plain Whisper. Plain Whisper can remain a fallback if WhisperX is unavailable.

Matching policy:

- preserve SRT segment order as a strong constraint
- allow small text differences caused by TTS pronunciation or transcription
  errors
- prefer monotonic matching: later SRT segments should not map before earlier
  segments
- log low-confidence matches for user review instead of failing the project
- expose the transcript/SRT word rematch map in debug data so bad matches can
  be inspected and corrected
- add a manual full-narration cut editor, similar to Stories, so the user can
  zoom into the full WAV waveform and create or adjust cuts by hand when ASR
  alignment is not reliable enough

This alignment step is the key bridge between:

```text
natural full narration
```

and:

```text
timecode-constrained SRT export
```

### Future stronger persistence options

If whole-SRT generation is too long for quality or model limits, test stronger
approaches behind the same Beta switch:

1. Generate larger narration chunks, such as paragraph/scene blocks, as a
   fallback only when full-SRT generation is impractical.
2. Generate a short calibration phrase at the start of the project and reuse it
   as a prosody/reference anchor when the engine supports it.
3. For cloned voices, select or create reference audio already recorded in the
   target narration style.
4. Add a project-level voice consistency check based on loudness, duration,
   and optional speaker embedding similarity.

The likely best long-term quality path is:

```text
larger coherent generation -> alignment -> SRT/timeline placement
```

This is closer to how high-end dubbing systems maintain continuity while still
respecting subtitle timing.

### Timeline/export policy

When whole-SRT generation is active:

- timeline placement uses the first SRT segment's `start_ms`
- subsequent SRT segment boxes remain visible as text/time references
- the exported WAV uses the full narration audio, not isolated regenerated
  snippets
- if narration audio exceeds an intermediate segment boundary, do not mark it
  failed
- overflows remain warnings only

Failure must mean:

```text
no audio file was generated
```

Timing overflow is not a generation failure.

### Rollback requirement

Phrase-aware Dubbing must be introduced behind a switch:

```text
Dubbing generation mode:
- Stable: segment-by-segment
- Beta: whole-SRT narration
```

Rollback must be possible by switching back to Stable without database surgery.

Implementation rule:

- do not overwrite existing per-segment generation behavior
- add whole-narration generation fields or tables separately
- keep existing segment generation endpoints working
- avoid schema changes that make older Dubbing projects unreadable

### Suggested data model additions

The existing `pace_group_id` is useful for UI analysis, but it is not sufficient
as the full generation source of truth.

Suggested fields/table:

```text
dubbing_narrations
- id
- project_id
- start_ms
- end_ms
- text
- generation_id
- status
- actual_duration_ms
- delta_ms
- alignment_status
- alignment_json
```

Alternative minimal V1:

```text
dubbing_projects.generation_mode
dubbing_projects.narration_generation_id
full narration stored as a Generation row with source = dubbing_full_narration
```

The cleaner long-term path is a dedicated `dubbing_narrations` table, with
future alignment data stored separately from editable SRT segments.

### Acceptance criteria

V2/Beta is acceptable when:

- importing an SRT still creates editable segments
- Stable mode still generates exactly as before
- Beta mode generates the complete SRT script as one narration
- a phrase split across several SRT rows sounds like one continuous spoken
  sentence
- voice identity, tone, intensity, and narration posture remain stable across
  the full project
- the same project-level voice/session contract is used for the full narration
- generated audio is placed at the first SRT start time
- timeline/export use the same generated narration audio
- timing overflow is warning-only
- failed means the audio file was not generated
- VoiceDesign delivery instructions are passed through in Dubbing
- normal Voicebox generation outside Dubbing is not affected


## Critical Startup Rule

Do **not** hack `tauri/src-tauri/src/main.rs` to compensate for a broken sidecar.

If startup is broken, the first thing to verify is:

1. the packaged `voicebox-server` sidecar itself
2. then the Tauri wiring

In this branch, startup was restored **without** changing the original startup
flow in `main.rs`.


## Files That Matter

Backend Dubbing:

- [backend/routes/dubbing.py](backend/routes/dubbing.py)
- [backend/services/dubbing.py](backend/services/dubbing.py)
- [backend/services/srt_parser.py](backend/services/srt_parser.py)
- [backend/database/models.py](backend/database/models.py)
- [backend/models.py](backend/models.py)

Sidecar build:

- [backend/build_binary.py](backend/build_binary.py)
- [backend/voicebox-server.spec](backend/voicebox-server.spec)

Tauri packaging:

- [tauri/src-tauri/binaries/voicebox-server-x86_64-pc-windows-msvc.exe](tauri/src-tauri/binaries/voicebox-server-x86_64-pc-windows-msvc.exe)
- [tauri/src-tauri/src/main.rs](tauri/src-tauri/src/main.rs)


## Known Good Result

The target healthy state is:

- launching [voicebox.exe](tauri/src-tauri/target/release/voicebox.exe)
- automatically starts backend on `127.0.0.1:17493`
- `GET /health` returns `200`
- `GET /dubbing/projects` returns `200`

### Release build rule

For a local Windows executable, do **not** use plain:

- `cargo build --release`

Plain Cargo can produce a binary that still falls back to the Tauri dev URL
(`http://localhost:5173`) because the `custom-protocol` feature is not enabled.
If another Vite project is running on that port, Voicebox may display the wrong
frontend.

Use the Tauri build path instead:

- `cd the development workspace/tauri`
- `npm.cmd run tauri -- build --no-bundle`

This is the current safe local build command because:

- it runs the frontend build
- it uses `build.frontendDist`
- it enables the correct Tauri release protocol
- it skips only the installer/bundler stage

The `devUrl` value in `tauri.conf.json` is normal and should remain:

- `http://localhost:5173`

That URL is for development only. It is not the release UI source when the app
is built through the Tauri command above.

Do not change the app identifier just to isolate this fork:

- current identifier: `sh.voicebox.app`

Changing it would create a separate AppData namespace and diverge from the
official Voicebox path contract. This may be useful later for a true forked
product, but it is not the current compatibility target.

### Normal Windows process shape

On Windows, the packaged `voicebox-server.exe` uses a PyInstaller `onefile`
layout. In Task Manager this normally appears as:

- one parent `voicebox-server.exe` bootloader/extractor process
- one child `voicebox-server.exe` process that runs the actual backend

This is **normal** and matches the official installed Voicebox build.

Do not treat `2 voicebox-server.exe` alone as a duplicate-startup bug.

The real checks are:

- `GET /health` responds on `127.0.0.1:17493`
- the backend becomes usable
- there is no second independent listener or conflicting backend instance

### CPU / CUDA note

The standard packaged sidecar is named `voicebox-server`. The official
installed Windows build uses this same name and is the reference for normal
startup behavior.

`backend/build_binary.py` also supports a CUDA build via `--cuda`, which
produces `voicebox-server-cuda`. Tauri has a code path for such a binary, but
do not rename or force the standard sidecar to pretend it is CUDA.

For this fork, the CUDA backend must also come from `the development workspace` sources.

Why:

- the official CUDA backend starts correctly but does **not** contain the
  Dubbing routes
- if `the development workspace` launches the official CUDA backend, `/dubbing/projects`
  returns `404`
- the Dubbing UI then appears failed even though CUDA itself is healthy

Expected Windows CUDA path:

- `%APPDATA%/sh.voicebox.app/backends/cuda/voicebox-server-cuda.exe`

This path is intentionally the same path used by official Voicebox `v0.5.0`.
Do not change the app identifier or invent a second CUDA path unless the
product decision is to isolate the fork from the official installation.

Current fork rule:

- build CUDA with `backend/build_binary.py --cuda`
- install the resulting onedir folder at the same official AppData path
- keep `cuda-libs.json` in that folder with `{"version": "cu128-v1"}`
- do **not** let startup auto-download the official CUDA backend
- CPU is only a runtime fallback when CUDA is absent or unavailable; never
  package a CPU-only `torch` inside `voicebox-server-cuda`
- do **not** use `backend/.venv_cuda`; on this machine it is obsolete because
  it points to a removed `Python310` installation
- current validated build environment:
  `backend/.conda_build312`
- current validated build Python:
  `backend/.conda_build312/python.exe`
- validated runtime versions:
  `Python 3.12.13`, `torch 2.11.0+cu128`, `CUDA 12.8`,
  `numpy 1.26.4`, `numba 0.60.0`, `PyInstaller 6.20.0`

`backend/build_binary.py --cuda` now has a hard guard against fake CUDA builds:
it must fail if the active Python environment imports `torch` but
`torch.version.cuda` is empty. This is intentional. A CUDA sidecar that starts
with `torch +cpu` is not acceptable because the app will show `CPU Only` while
the filename still says `voicebox-server-cuda.exe`.

The startup auto-download was disabled in
[backend/services/cuda.py](backend/services/cuda.py).
Manual CUDA download from the GPU settings page may still replace the backend
with the official one; only use it intentionally.

Release packaging requirement:

- this local AppData replacement is acceptable for development only
- for a real fork release, rebuild and package the CUDA backend as a proper
  release artifact using the same naming contract as Voicebox
- expected server artifact name: `voicebox-server-cuda`
- expected Windows executable inside the artifact: `voicebox-server-cuda.exe`
- expected install/extract layout: `backends/cuda/voicebox-server-cuda.exe`
- the release artifact must include the fork's Dubbing routes and migrations
- the release artifact must include a valid `cuda-libs.json`

Do not ship instructions that ask users to copy a manually patched AppData
folder as the release path. That is only a dev-machine workaround.

Before claiming CUDA support in a rebuilt package, verify the build venv:

- `backend/.conda_build312/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

Expected result on this machine:

- `torch 2.11.0+cu128`
- `torch.version.cuda == "12.8"`
- `torch.cuda.is_available() == True`

If this prints `+cpu`, `None`, or `False`, do not build or install the CUDA
sidecar. Fix the build environment first.

After installing CUDA, verify:

- `GET /health` reports `backend_variant: cuda`
- `GET /health` reports `gpu_available: true`
- `GET /health` reports a real NVIDIA GPU in `gpu_type`
- `GET /dubbing/projects` returns `200`
- `GET /openapi.json` contains `/dubbing/projects/{project_id}/generate-full-narration`
- Task Manager shows `voicebox-server-cuda.exe` from the AppData CUDA path

Important: `backend_variant: cuda` alone is not enough. If `gpu_available` is
`false`, the CUDA sidecar is present but CUDA is not actually usable.

Known local CUDA backup:

- `%APPDATA%/sh.voicebox.app/backends/cuda_official_backup_20260505_1617`
- `%APPDATA%/sh.voicebox.app/backends/cuda_backup_20260506_1440`

If rollback to official CUDA is needed, restore that folder to
`backends/cuda`, but remember that Dubbing routes will disappear until the CUDA
backend is rebuilt from this fork again.


## What Broke Before

The real failure was not `Dubbing` routes.

The failure chain was:

1. broken packaged sidecars
2. missing Python metadata for `fastmcp` / `mcp`
3. PyInstaller `onefile` extraction failures on Windows
4. a compiled `charset_normalizer` binary (`__mypyc...pyd`) that made the
   sidecar extraction unstable in this environment

Symptoms seen:

- app starts but nothing listens on `17493`
- `Dubbing` UI shows `Not Found`
- direct sidecar run fails before HTTP server starts


## Mandatory Sidecar Rules

When touching the backend build, keep these rules:

1. Keep original Tauri startup behavior.
2. Fix the sidecar itself first, not `main.rs`.
3. Keep `fastmcp` and `mcp` metadata bundled.
4. On this machine, keep a valid runtime extraction directory for PyInstaller
   onefile builds.
5. Avoid mixing random Python environments when building.


## Important Build Adjustments In This Branch

These adjustments are currently required in
[backend/build_binary.py](backend/build_binary.py):

- `--copy-metadata fastmcp`
- `--copy-metadata mcp`
- support for env var `VOICEBOX_RUNTIME_TMPDIR`
- support for env var `VOICEBOX_SKIP_CPU_TORCH_SWAP`
- support for env var `VOICEBOX_DEBUG_CONSOLE`

Why:

- without `fastmcp/mcp` metadata, the packaged backend crashes at import time
- without a stable runtime temp dir on this machine, `onefile` extraction may
  fail before the backend starts


## Local Windows Build Constraint

On this machine, the `venv` used for packaging contained compiled
`charset_normalizer` artifacts that contributed to sidecar extraction issues.

To stabilize the build, these files were disabled locally in the build env:

- `voicebox/backend/venv/Lib/site-packages/81d243bd2c585b0f4821__mypyc.cp310-win_amd64.pyd`
- `voicebox/backend/venv/Lib/site-packages/charset_normalizer/md.cp310-win_amd64.pyd`
- `voicebox/backend/venv/Lib/site-packages/charset_normalizer/cd.cp310-win_amd64.pyd`

They were renamed with `.disabled`.

This is a **build-environment workaround**, not a product feature.

If a new dev rebuilds on another clean machine, this workaround may not be
necessary. But if the sidecar starts failing with PyInstaller extraction
errors again, check this first.


## Safe Rebuild Procedure

If the Dubbing UI works in source but not in packaged app, follow this order.

1. Verify source backend:
   - `/health`
   - `/dubbing/projects`
2. Rebuild `voicebox-server` sidecar only.
3. Run the sidecar directly before touching Tauri.
4. Replace the packaged sidecar in `tauri/src-tauri/binaries`.
5. Launch `voicebox.exe`.
6. Recheck:
   - port `17493`
   - `/health`
   - `/dubbing/projects`

Do not jump directly to frontend debugging if `17493` is not up.

CUDA rebuild addendum:

1. Build with a CUDA-capable Python environment:
   - `backend/.conda_build312/python.exe build_binary.py --cuda`
2. Test the rebuilt CUDA on a temporary port before installing it:
   - `backend/dist/voicebox-server-cuda/voicebox-server-cuda.exe --port 17495`
3. Verify:
   - `GET /health` reports `backend_variant: cuda`
   - `GET /dubbing/projects` returns `200`
4. Replace the AppData CUDA onedir folder only after that test passes.
5. Keep a backup of the previous AppData CUDA folder.

Validated on 2026-05-06:

- built CUDA from `the development workspace/backend/.conda_build312`
- installed to
  `%APPDATA%/sh.voicebox.app/backends/cuda`
- `voicebox.exe` auto-started
  `%APPDATA%/sh.voicebox.app/backends/cuda/voicebox-server-cuda.exe`
- `GET /health` returned `backend_variant: cuda`
- `GET /health` detected `CUDA (NVIDIA GeForce RTX 5090 Laptop GPU)`
- `GET /dubbing/projects` returned `200`


## Direct Validation Commands

The most useful validations are:

1. Direct sidecar run:
   - `the development workspace/backend/dist/voicebox-server.exe --port 17493`
2. Direct CUDA sidecar run:
   - `the development workspace/backend/dist/voicebox-server-cuda/voicebox-server-cuda.exe --port 17495`
3. Health check:
   - `http://127.0.0.1:17493/health`
4. Dubbing route:
   - `http://127.0.0.1:17493/dubbing/projects`

If direct sidecar run fails, Tauri is not the root cause.


## Dubbing Isolation Rules

Keep `Dubbing` isolated from the rest of Voicebox.

Do not:

- patch global cloned-voice behavior for Dubbing-only needs
- modify other modules to compensate for Dubbing timing logic
- put Dubbing-specific fallbacks into generic generation flows

Do:

- keep Dubbing routes and services local
- keep Dubbing UI/API local
- avoid touching unrelated startup/runtime logic unless the sidecar itself is broken


## Rollback Guidance

If startup breaks again:

1. compare current `voicebox-server` sidecar behavior with a direct run
2. inspect `backend/build_binary.py`
3. inspect `backend/voicebox-server.spec`
4. revalidate `fastmcp/mcp` metadata
5. revalidate runtime temp extraction behavior

If needed, rollback should target:

- the sidecar build chain
- not the Dubbing routes/UI first


## Current Functional Intent

The Dubbing module currently aims to support:

- project list
- project delete
- SRT import
- segment edit
- segment generation
- auto-fit
- timeline WAV export

But none of this matters if the sidecar does not boot.

So the permanent priority order is:

1. sidecar starts
2. backend responds
3. Dubbing routes respond
4. features are debugged


## Stabilization Roadmap

At this stage, stabilization should focus on the `Dubbing` module itself, not
on the global app bootstrap.

The startup/server layer should now be treated as frozen unless it regresses
again.

The remaining work should be limited to functional module behavior.

### 1. Generation and Auto-Fit

Validate and stabilize:

- manual segment generation
- sequential auto-fit batch
- segment status transitions
- retry behavior
- correct routing for both cloned voices and `Qwen CustomVoice`

Expected rule:

- if the server is healthy, a Dubbing failure should now be treated first as a
  module logic problem, not an app startup problem

### 2. Dubbing UX

Polish only inside the Dubbing module:

- project deletion
- segment text editing
- segment player
- contextual `...` menu
- batch progress
- responsive/scroll behavior
- readable errors and warnings

### 3. Timeline WAV Export

Stabilize export logic:

- export reliability
- no silent truncation unless explicitly desired
- segment overflow behavior
- correct sequencing when one segment exceeds and the next must shift
- predictable output timeline
- Implemented: timeline export now preserves generated segment audio as-is.
  The exporter should place/mix the segment files on the SRT timeline; it must
  not apply `time_stretch_audio()` / `librosa.effects.time_stretch` during WAV
  export because that produced caverneous / phasey voices while the individual
  generated segments sounded natural.
- Pace controls remain project/group metadata for Dubbing decisions, but export
  must not destructively transform audio unless a future feature explicitly
  exposes and validates that behavior.

### 4. Audio Quality

Business-quality topics should remain local to Dubbing:

- continuity between segments
- phrase continuity
- delivery instruction behavior
- clone vs custom voice suitability

This is a product-quality layer, not a server-startup layer.

Implemented: Dubbing delivery instructions are sanitized before generation.

Current observation:

- generated speech can hallucinate or cut phrases when delivery instructions are
  polluted with dynamic timing retry text
- avoid instructions like:
  `Timing fit retry 3: target the subtitle window precisely. Speak noticeably faster, minimize pauses, and keep the sentence very compact.`
- do not use delivery instructions to force exact SRT fit
- delivery should focus on natural voice continuity, stable tone, and
  punctuation-aware phrasing
- timing pressure should be handled separately by project/group pace controls
  and warnings, not by repeatedly rewriting the delivery prompt
- old `Timing fit retry ...` fragments are stripped in the Dubbing backend
  before being saved or sent to Qwen

Preferred direction:

- keep user delivery instructions clean
- add only short continuity hints when needed
- respect punctuation as the main rhythm guide
- let manual project/group pace sliders handle timing compromises

Important limitation: Qwen cloned voices / Base model:

- for Qwen3-TTS cloned voices using the Base model, do not rely on
  `instructions` / `instruct` for emotion, pacing, or delivery control
- the Base voice cloning path mainly follows the reference audio timbre, the
  reference audio prosody, the target text, and punctuation/text segmentation
- delivery instructions may be ignored or have only a very weak effect on
  cloned voices
- this is different from `Qwen CustomVoice` and VoiceDesign-style paths, where
  instruction control is explicitly supported and usually more audible

Recommended workaround for cloned voices:

- use reference audio already recorded in the desired style
- provide accurate reference text for the cloned voice prompt
- avoid `x_vector_only_mode=True` when prosody similarity matters
- encode delivery through punctuation, sentence grouping, and chunking
- for strong style/emotion control, use `Qwen CustomVoice` or VoiceDesign
  instead of Base voice cloning

Prompt guidance:

- keep Dubbing instructions short and actor-like, ideally 10-40 words
- prefer natural-language acting directions over keyword lists or SSML-like tags
- good default:
  `Professional documentary narration with clear articulation, natural conversational prosody, realistic pauses, and punctuation-aware pacing.`
- for cloned voices, treat this as a soft hint only; the stronger controls are
  the reference audio, punctuation, and segment/phrase structure
- example text-level control:
  `We should leave now... before it's too late.`
  is more likely to affect cloned voice rhythm than
  `We should leave now before it's too late.`

### 5. Persistence and Cleanup

Stabilize Dubbing project behavior:

- save/reopen projects
- delete project cleanly
- retry failed generations cleanly
- avoid polluting main History with internal Dubbing retries


## What "Stabilize the Module" Means

From this point onward:

- do not reopen the startup/server architecture unless it breaks again
- do not add new global app workarounds for Dubbing problems
- do not patch unrelated Voicebox modules to compensate for Dubbing issues

The correct approach is:

1. keep app/server startup stable
2. isolate bugs to Dubbing behavior
3. fix them locally inside Dubbing


## Dubbing UI Direction

The Dubbing UI should support both text correction and timing correction.

Current rules:

- the imported SRT creates the initial timeline
- the user must be able to edit segment text after import
- the user must be able to manually realign segment timing on a timeline
- `start_ms` / `end_ms` are the editable timing values used by Dubbing
- timeline edits update `start_tc`, `end_tc`, and `target_duration_ms`
- timing edits should not delete edited segment text

Generation controls should stay visible even when the app window is not full
screen:

- voice selection
- language
- Qwen model display
- prosody / punctuation instructions
- selected segment generation
- sequential batch generation
- cancel tasks

Timing policy:

- do not use prompt text to force speed or exact time fitting
- do not inject `Timing fit retry ...` instructions
- delivery instructions should focus on prosody, articulation, punctuation, and
  continuity across adjacent segments
- manual pace sliders remain the only user-facing speed/debit control for now
- sequential batch generation should run one natural pass per segment unless a
  future explicit retry mode is added

Priority TODO: phrase-level generation for natural continuity:

- Dubbing already computes phrase-like `pace_groups` by joining consecutive SRT
  segments until terminal punctuation is found
- currently these groups are used for pacing/UI only; generation still runs one
  isolated Qwen request per SRT segment
- this causes audible prosody cuts when one sentence spans multiple SRT blocks
- implement a dedicated phrase/group generation mode that sends the full
  punctuation-bounded sentence to Qwen in one request
- after generation, map the resulting phrase audio back onto the underlying SRT
  segment windows/timeline
- keep segment text editable; edited text must update the phrase group content
- keep this logic Dubbing-only and do not alter the global Voicebox generation
  behavior

Required safety guard before implementation:

- add a project-level generation mode selector in Dubbing `Generation Controls`
- mode `Segment by segment - stable` keeps the current behavior and must remain
  the rollback/fallback path
- mode `Phrase groups - beta` enables the new phrase-level generation pipeline
- `Generate All Segments` must use the selected mode
- segment-level regeneration must remain available for local corrections
- do not remove or overwrite the stable mode until beta output is validated

### Timeline UI state, 2026-05-06

Architecture rule, updated 2026-05-08:

- Dubbing must not maintain a separate timeline implementation when Stories
  already has the required behavior.
- Shared timeline pieces live under `app/src/components/AudioTimeline`.
- Implemented shared pieces:
  - `ClipWaveform`: visual WaveSurfer waveform used by Stories and Dubbing
  - `TimelineScrollbar`: Stories-style horizontal chariot with pan and edge
    zoom handles, now used by Stories and Dubbing
- `AudioTrackEditor`: generic Stories-derived track editor for tracks,
  playhead, drag, trim handles, split, duplicate, volume, delete, regenerate,
  resize, zoom, and scrollbar.
- `StoryTrackEditor` is now an adapter over `AudioTrackEditor`.
- Dubbing has a first adapter over `AudioTrackEditor` for generated clips,
  post-processed cuts, and full narration. The previous local Dubbing timeline
  remains temporarily behind it as a rollback/safety layer until the backend
  fully persists Dubbing trim/split/volume metadata.
- Dubbing must keep SRT theoretical blocks visible as permanent reference
  clips on the shared timeline, even after full narration, cuts, or segment
  generations exist. These reference clips are non-audio and non-editable.
- Next cleanup step: remove the old Dubbing-specific timeline JSX once
  `AudioTrackEditor` covers all Dubbing-only overlays and persisted actions.
- Stories remains the reference implementation. Any new Dubbing timeline
  behavior should first check whether the Stories implementation can be reused
  or adapted.

Current expected Dubbing timeline behavior:

- the main timeline Play button plays generated segments sequentially
- the Play icon must become Pause during playback
- Stop remains a separate square button
- moving the playhead while stopped must **not** start playback
- moving the playhead while playing should continue playback from the new
  position
- double-clicking a generated clip starts playback from that clip
- the playhead should keep moving through gaps between generated clips
- generated clips are shown on timeline lanes `1`, `0`, and `-1`
- SRT reference clips remain visible on their own upper lane for visual
  alignment against generated/cut audio
- dragging a clip horizontally updates its timing
- dragging a clip vertically may move it between lanes `1`, `0`, and `-1`
- lane `+` is reserved for adding more lanes later

Current floating generation box behavior:

- it must stay visible when the app is not full-screen
- it should align visually with the Segments panel
- it should be compact enough not to hide its own controls
- its primary action is `Generate All Segments`
- voice, language, Qwen model, and prosody/punctuation display remain visible

Current Segments panel behavior:

- keep approximately two segment cards visible
- use vertical scrolling to reach the rest
- selecting a segment in the list also selects it in the timeline
- selecting a generated clip in the timeline exposes clip actions

Current generated-clip toolbar:

- Cut icon: visible for parity with Stories, but **not persisted yet**
- Volume icon: visible for parity with Stories, but **not persisted yet**
- Trash icon: deletes the generated audio for that Dubbing segment
- Regenerate icon: regenerates the selected Dubbing segment

Do not claim Cut or Volume are functional until the backend has explicit
Dubbing support for:

- segment split / cut
- per-segment volume persistence
- timeline WAV export honoring per-segment volume

The current UI intentionally avoids silently pretending that Cut/Volume changed
the exported result.

## V3 Exploration: Voice-To-Voice Prosody Transfer

This is a research track, not part of the current stable Dubbing chain.

Hypothesis:

- ElevenLabs-style SRT dubbing likely uses more than isolated TTS
- a voice-to-voice or prosody-transfer pass may help preserve pauses,
  intonation, rhythm, and phrase continuity after sequencing
- the candidate local architecture is a cascade:
  source/generated narration -> audio understanding/alignment ->
  style/prosody transfer -> final TTS/resynthesis

Possible Qwen-oriented directions:

- Qwen-Audio / Qwen2.5-Omni-style audio input could eventually inspect a
  sequenced narration track and condition a more coherent regenerated output
- Qwen-TTS VoiceDesign / CustomVoice would remain the preferred final voice
  synthesis target when delivery instructions matter
- cloned/Base voices should not be assumed to obey delivery instructions; for
  those, the value would come mostly from reference audio/prosody, punctuation,
  and segmentation

Important guardrails:

- do not mix V3 experiments into the stable segment/full-narration workflow
- keep a project-level selector or beta flag before enabling this path
- keep rollback to the current full narration + phrase-aware post-process path
- do not change global Voicebox generation behavior
- document every extra dependency before adding it to the release flow

Open questions:

- whether a local Qwen voice-to-voice path can preserve the selected target
  voice better than the current TTS-only path
- whether the pass should use the original video audio, the generated full WAV,
  or the resequenced post-processed WAV as prosody reference
- whether the gain in natural continuity justifies the extra processing time


## Qwen Sampling Controls

Current state in `the development workspace`:

- `instruct` is supported
- `seed` is supported
- `max_chunk_chars` is supported
- `crossfade_ms` is supported
- `temperature` is now exposed at project level in SRT2Voice for Qwen engines
  only

Current state not yet wired:

- `top_p`
- `top_k`
- `repetition_penalty`

Top-P / nucleus sampling note:

- Voicebox / SRT2Voice does not currently set or expose `top_p` for Qwen
- the Qwen library therefore keeps its own default behavior
- local package inspection showed Qwen3-TTS defaults to `top_p = 1.0`
- `temperature` defaults to `0.9` inside Qwen3-TTS when no explicit override is
  sent
- do not add a `top_p` UI control yet; if needed later, keep it Qwen-only and
  evaluate a conservative range such as `0.8` to `1.0`

The SRT2Voice temperature slider is hidden for Chatterbox and other non-Qwen
engines. For Qwen, it is persisted on the SRT2Voice project, sent with full
narration / segment generation requests, and forwarded to:

- `generate_voice_clone`
- `generate_custom_voice`
- `generate_voice_design`

Default behavior remains the Qwen library default when the project temperature
is reset. Recommended working range for narration tests is usually `0.3` to
`0.7`; lower values should be steadier, higher values may be more variable.

Files checked for this:

- [backend/services/generation.py](backend/services/generation.py)
- [backend/backends/pytorch_backend.py](backend/backends/pytorch_backend.py)
- [backend/backends/qwen_custom_voice_backend.py](backend/backends/qwen_custom_voice_backend.py)
- [backend/utils/chunked_tts.py](backend/utils/chunked_tts.py)

Recommended rule:

- if sampling controls are added, add them **for Dubbing only**
- do not change global Voicebox generation behavior

Current policy:

- keep `temperature` isolated to SRT2Voice
- do not expose `top_p`, `top_k`, or repetition penalty until there is a
  measured need
- do not change global Voicebox generation behavior

Why:

- Dubbing needs stable, disciplined delivery more than creativity
- a lower `temperature` may help with punctuation discipline and reduce
  over-fluid delivery
- but this should remain isolated to the Dubbing module


## Apply Suggested Tempo Workflow

Status: implemented as a functional SRT2Voice beta workflow.

Goal:

- keep TTS generation natural by generating the full SRT as one continuous
  narration first
- avoid forcing the model itself to speak faster/slower when that damages
  prosody or creates artifacts
- apply a global, pitch-preserved tempo adjustment after generation
- re-run alignment after tempo processing so first-word attacks can be snapped
  precisely to the SRT timeline

Terminology:

- `D_srt`: target SRT project duration, from the first SRT `start_ms` to the
  last SRT `end_ms`
- `D_proj`: projected mounted timeline duration from the same Auto Cut clips
  that will be exported; this reuses word matching, punctuation strategy,
  RMS/ZCR boundaries, and the rule that the next segment stays anchored
- `M`: suggested tempo multiplier, computed as `D_proj / D_srt`

Expected user-facing ranges:

- safe: `0.9x` to `1.1x`, shown green
- warning: `0.8x` to `0.9x` or `1.1x` to `1.2x`, shown amber
- critical: below `0.8x` or above `1.2x`, shown red with a warning that quality
  degradation is likely and the user should consider editing SRT text/timecodes
  using CPS/WPS hints before regenerating

Current flow:

1. Generate the full SRT narration naturally.
2. Run Whisper Turbo word matching and RMS/ZCR acoustic boundary detection.
3. Compute `D_proj`, `D_srt`, and suggested global multiplier `M` from the
   mounted Auto Cut clips, not from a separate theoretical placement model.
4. Display the suggestion only. Do not apply it automatically.
5. If the user clicks `Apply Suggested Tempo`, write the confirmed multiplier to
   project metadata (`pace_override`) and process the current full narration WAV.
6. Process the full narration WAV in-place with FFmpeg `atempo`.
7. Re-run Whisper Turbo word matching on the tempo-processed audio.
8. Re-run RMS/ZCR boundary detection on the tempo-processed audio.
9. Reposition clips so each refined first-word attack snaps to its original SRT
   `start_ms`.

Implementation details:

- API:
  - `POST /dubbing/projects/{project_id}/tempo-suggestion`
  - `POST /dubbing/projects/{project_id}/apply-tempo`
- Backend:
  - suggestion and application logic live in
    [backend/services/dubbing.py](backend/services/dubbing.py)
  - the suggestion reuses `word_matching_debug.json` when it matches the current
    project/audio revision and debug schema
  - if no valid debug file exists, the backend runs Auto Cut alignment first to
    produce fresh word/boundary data
  - applying tempo invalidates old cut artifacts, processes the full WAV with
    FFmpeg `atempo`, then re-runs Auto Cut on the processed WAV
  - Whisper/STT is unloaded after suggestion/application endpoints so VRAM is
    released again after alignment work
- Frontend:
  - the Generation Controls card exposes `Suggest` and `Apply` under
    `Suggested Tempo`
  - suggestion colors follow the safe/warning/critical ranges
  - after applying tempo, the timeline is rebuilt from the refreshed Auto Cut
    clips

Implementation constraints:

- keep this local to SRT2Voice
- backend logic belongs in `backend/services/dubbing.py`
- UI belongs in `app/src/components/DubbingTab/DubbingTab.tsx`
- do not fall back to librosa time-stretching, because prior tests showed
  phase/reverb/wet artifacts
- use FFmpeg `atempo` only; if FFmpeg is missing, do not apply tempo
- project-level tempo must avoid per-segment speed jumps
- applying tempo invalidates previous Auto Cut/manual cut caches and refreshes
  the timeline clips used by export/package
- keep the operation non-destructive until the user explicitly confirms

Design note:

- This is tempo post-processing, not model pace prompting.
- Suggested Tempo must not run a parallel timing model. It consumes the current
  Auto Cut debug schema and therefore follows the same no-punctuation,
  soft-punctuation, hard-punctuation, word-matching, and RMS/ZCR rules as the
  mounted timeline.
- Stale Auto Cut debug caches are ignored when the debug schema changes, so old
  placement rules cannot affect tempo suggestions.
- Current observations suggest Audacity-style tempo processing can sound more
  stable than asking Qwen to change pace inside the model.
- This should coexist with the current manual/full-narration workflow rather
  than replace it immediately.


## VRAM Restart Policy

The controlled server restart used for VRAM release is currently kept in the
SRT2Voice frontend code but disabled by default.

Frontend flag:

- `AUTO_RESTART_SERVER_FOR_VRAM_RELEASE = false`

Reason:

- Whisper Turbo significantly reduces the Auto Cut VRAM footprint compared with
  Whisper Large
- automatic restart is useful as an emergency escape hatch, but it interrupts
  the user flow and can make the UI feel briefly empty
- keep the code path available, but do not restart automatically unless this
  flag is deliberately re-enabled

Local/backend VRAM cleanup already exists and should be preserved:

- global generation unloads the active backend and calls `gc.collect()`
- CUDA cleanup calls `torch.cuda.empty_cache()`
- when available, CUDA cleanup also calls `torch.cuda.ipc_collect()`
- SRT2Voice has its own cleanup hook after full narration and auto-cut work
- SRT2Voice full narration uses `unload_after=True`
- when `unload_after=True`, the backend is now also removed from the global
  TTS backend registry so stale Python references do not keep CUDA tensors alive
- entering SRT2Voice calls `POST /dubbing/release-memory` to unload already
  loaded TTS/STT backends before the SRT2Voice workflow starts
- after Auto Cut, Whisper/STT is unloaded and CUDA cache is cleared

Files to check before changing VRAM behavior:

- [backend/services/generation.py](backend/services/generation.py)
- [backend/services/dubbing.py](backend/services/dubbing.py)
- [backend/backends/base.py](backend/backends/base.py)

Future conformity task:

- The current SRT2Voice load/unload behavior works well in practice and should
  be kept for now.
- Before release or upstream discussion, re-check it against the official
  Voicebox v0.5.0 model-management contract from Jamie Pine's repository.
- Prefer aligning the SRT2Voice memory release path with the official
  `ModelConfig` / `/models/status` / `/models/unload` flow, especially
  `unload_model_by_config(config)`, instead of keeping a broad custom unload
  path forever.
- Regression-check the rest of Voicebox after that refactor: regular voice
  generation, Stories, model status, model load/unload buttons, CUDA status,
  LuxTTS, Kokoro, TADA, Chatterbox, Qwen Base, Qwen CustomVoice, and Qwen
  VoiceDesign.


## Auto Cut Boundary Rule

Status: validated in manual testing.

The current SRT2Voice Auto Cut rule is:

- Whisper word timestamps provide the first/last matched words for adjacent SRT
  segments.
- Punctuation selects the strategy, but the waveform validates the cut.
- Hard punctuation (`.`, `!`, `?`, ellipsis) uses RMS + ZCR acoustic analysis
  as the primary boundary, so sentence-final breathing and tails are preserved.
- Soft punctuation (`,`, `;`, `:`) and no-punctuation continuations use a
  hybrid softpoint:
  the mathematical midpoint between the matched previous word end and next word
  start is the default when the gap is short or unstable.
- If RMS + ZCR finds a reliable low-energy gap between the previous word's true
  energy tail and the next word's true acoustic attack, the cut is placed at the
  center of that acoustic gap instead.
- This protects both sides of the boundary: the previous segment keeps long
  nasals/fricatives/weak endings, and the next segment keeps aspirated or early
  attacks.
- No artificial silence is inserted. If the locutor naturally has a tiny
  continuous-word gap, SRT2Voice keeps it tiny instead of inventing a pause.
- Timeline placement keeps the next SRT segment as the timing anchor. For
  no-punctuation continuations and soft punctuation, the previous clip is
  shifted so its end touches the next anchored clip.
- Apply a very small anti-click fade/crossfade at exported/playback cut edges
  (target 5-10 ms) only to smooth the cut. It must not introduce artificial
  silence, timing drift, or helper timecode changes.
- If a robust acoustic gap is not found, fallback is the semantic midpoint
  between adjacent matched words, not the next-word attack alone.
- Helpers stay immutable visual SRT references. Auto Cut must not rewrite SRT
  helper timecodes or text.
- Regenerating the full narration invalidates derived cuts and debug files so
  stale cut artifacts cannot ghost into the next pass.

Debug files to inspect:

- `%APPDATA%\\sh.voicebox.app\generations\dubbing_cuts\<project_id>\word_matching_debug.json`
- `%APPDATA%\\sh.voicebox.app\generations\dubbing_cuts\<project_id>\alignment_debug.json`


## v0.5 Engine Restoration Notes

Voicebox v0.5 engines outside SRT2Voice must stay available:

- LuxTTS
- Kokoro
- TADA 1B
- TADA 3B Multilingual

Windows packaging notes from the current fork:

- the CUDA backend runtime lives at
  `%APPDATA%\\sh.voicebox.app\backends\cuda`
- the rebuilt CUDA backend source lives at
  `backend/dist/voicebox-server-cuda`
- `phonemizer-fork` is required for Kokoro/Misaki on Windows; the standard
  `phonemizer` package can break with `EspeakWrapper.set_data_path`
- NumPy must remain compatible with Qwen/Numba; the current safe version is
  `numpy 2.0.0`
- TADA intentionally uses `backend/utils/dac_shim.py` instead of installing the
  full `descript-audio-codec` dependency chain

Current smoke checks after restoring those engines:

- `luxtts -> LuxTTSBackend`
- `kokoro -> KokoroTTSBackend`
- `tada -> HumeTadaBackend`
- `chatterbox -> ChatterboxTTSBackend`
- `chatterbox_turbo -> ChatterboxTurboTTSBackend`
- `qwen -> PyTorchTTSBackend`
- `qwen_custom_voice -> QwenCustomVoiceBackend`
- `qwen_voice_design -> QwenVoiceDesignBackend`

Deployment checkpoint:

- CUDA backend rebuilt successfully
- runtime CUDA backend was backed up before replacement
- active runtime health check returned `200`
- CUDA was detected as `backend_variant=cuda`
- GPU was detected as `NVIDIA GeForce RTX 5090 Laptop GPU`

Follow-up debug note:

- The v0.5 engine rebranch is considered functionally restored in principle,
  but not yet release-clean.
- Before release, run deeper generation tests for LuxTTS, Kokoro, TADA 1B, and
  TADA 3B Multilingual from the real Voicebox UI, not only import/registry
  smoke checks.
- Specifically watch for packaging/runtime edge cases around PyInstaller,
  model cache resolution, phonemizer/Misaki data files, and TADA codec shims.
- Do not change the SRT2Voice pipeline while debugging those engines unless a
  shared backend bug is proven.


## Cloned Voice Prompt Cache Recovery

Observed case:

- An old cloned voice profile can start a Qwen generation and then appear to
  hang until the user kills the server/GPU process.
- The resulting database error may be `Server was shut down during generation`.
  That message only records the manual kill; it does not identify the original
  cause.
- If a freshly recreated clone from the same source audio works, the source WAV
  is probably not the primary problem.

Likely suspects:

- stale cloned profile metadata
- reference text mismatch between the stored clone text and the audio
- bad cached voice prompt for that specific profile/audio/text pair
- old profile created before later cache/backend changes

Future recovery actions:

- Add `Rebuild voice prompt cache` for a single cloned profile.
- Add `Clear voice prompt cache for this voice`.
- Keep these actions profile-scoped, not global, to avoid disrupting working
  voices.
- Do not treat delivery instructions as the likely cause unless the same
  failure reproduces across multiple healthy cloned profiles.
