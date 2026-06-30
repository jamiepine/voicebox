# Voice I/O

**Status:** Shipping — phases 1, 2, 4, 7 (macOS) complete · 3 partial · 5, 6, 7 (Windows/Linux), 8 pending
**Touches:** backend, Tauri shell, frontend, a new native shim crate
**Last reviewed:** 2026-04-21

## Progress

### Shipped

**Phase 1 — Groundwork.** Audio tab retired from the sidebar; its device / channel
config lives under Settings. Captures tab is live at `/captures` with no feature
flag.

**Phase 2 — Local LLM backend.** `LLMBackend` protocol alongside the existing
TTS/STT backends. `qwen_llm_backend.py`, `services/llm.py`, `routes/llm.py`, and
a shared model-download / cache pipeline. Qwen3 0.6B / 1.7B / 4B registered and
user-selectable via `capture_settings.llm_model`.

**Phase 4 — Captures tab.** List + detail view, source badges (dictation /
recording / file), retranscribe, refine (flags + model resolved from a
server-side `capture_settings` singleton), delete, and the Play-as-voice
dropdown over every profile.

### Partial

**Phase 3 — In-app voice input.** `CapturesTab` dictates end-to-end via
`useCaptureRecordingSession`, which the Phase 7 floating pill also consumes.
Outstanding: a universal mic button on other text inputs (Generate form,
profile descriptions, story titles, etc.), and the streaming
`/transcribe/stream` WebSocket — today's flow is a single `POST /captures`
with the complete audio blob.

**Phase 7 — External dictation shell (macOS).** Both halves shipped on macOS.

Hotkey half:

- `tauri/src-tauri/src/chord_engine.rs` — pure state machine. Unit tests green.
- `tauri/src-tauri/src/hotkey_monitor.rs` — `rdev`-based global listener on a
  background thread, with `set_is_main_thread(false)` applied to sidestep the
  macOS 14+ TSM crash ([Narsil/rdev#165](https://github.com/Narsil/rdev/issues/165)).
  Right-hand-only defaults preserve left-hand Cmd+Option+I devtools.
- Default bindings hardcoded: `Cmd+Option` (push-to-talk) and
  `Cmd+Option+Space` (toggle-to-talk). The PTT → Toggle upgrade transition is
  preserved — adding Space mid-hold promotes the session without interrupting
  audio.
- `DictateWindow` — transparent, always-on-top, borderless 420×64 webview
  pre-created hidden at app setup. Shows on chord-start, hides on
  capture-cycle completion. Error state on the pill auto-dismisses and
  copies-to-clipboard on click.

Paste half (macOS):

- `clipboard.rs` — `NSPasteboard` snapshot that walks `pasteboardItems` and
  copies every `(uti, bytes)` pair so multi-type content (images, styled
  text, file refs) survives the round-trip. `save_clipboard`,
  `write_text`, `restore_clipboard`, `current_change_count`.
- `synthetic_keys.rs` — `CGEventPost` at the HID tap with the full four-event
  Cmd+V sequence (Cmd down → V down w/ flag → V up w/ flag → Cmd up).
- `focus_capture.rs` — `AXUIElementCreateSystemWide` +
  `AXUIElementCopyAttributeValue(kAXFocusedUIElement)` +
  `AXUIElementGetPid`, with the AX attribute key CFStrings built at
  runtime because they're CFSTR macros, not linkable symbols.
  `NSRunningApplication.activateWithOptions:` for re-activation.
- `accessibility.rs` — `AXIsProcessTrusted` gate.
- `paste_final_text` command — activate → 120 ms settle → save clip →
  write text → ⌘V → 400 ms → restore. Skips when focus was in Voicebox
  itself.
- Focus rides the `dictate:start` event payload; `DictateWindow` holds the
  snapshot in a ref and consume-once-nulls on paste so a late-arriving
  refine from an earlier session can't misfire.
- Dictation recording no longer hard-caps at 29 s — the limit still
  applies to voice-profile reference clips.

Outstanding: Windows `SendInput` / UIAutomation / `SetForegroundWindow`
equivalents, Linux `uinput` / AT-SPI equivalents (and the Wayland story),
first-run Accessibility prompt UI with deep-link to System Settings,
direct-injection path for focus-was-inside-Voicebox (step 6 — dictating
into our own Generate tab currently falls back to the capture list).

### Not started

- **Phase 5 — Agent voice output + persona loop.** No `/speak` endpoint, no
  `voicebox.speak` MCP tool, no per-agent voice binding, no persona metadata
  on profiles.
- **Phase 6 — STT engine expansion.** Only Whisper (`mlx_backend.py`).
  Parakeet v3, Qwen3-ASR, Kyutai — all unregistered.
- **Phase 8 — Pipeline routing, sinks, long-form.** No preset primitive, no
  MCP sink, no webhook sink, no dual-stream recorder, no summary transform.

### Additionally landed (not explicit in the original plan)

These fell out of the Phase 3/4/7 work but deserve their own mention:

- **Server-authoritative settings.** Singleton `capture_settings` and
  `generation_settings` tables. The client sends nothing but the audio; STT
  model, refine flags, refine LLM, and the auto-refine flag are all resolved
  server-side, so sibling Tauri webviews can't go stale.
- **Backend audio normalisation.** `POST /captures` transcodes anything
  librosa can decode (webm/opus, m4a, etc.) to WAV before handing it to
  whisper, side-stepping miniaudio's format gaps inside mlx-audio.
- **Short-recording guard.** Sub-300 ms blobs short-circuit client-side so a
  fumbled chord tap never uploads an empty webm.
- **Refinement prompt.** Rewritten with firmer anti-chatbot framing and
  inline examples covering multi-sentence preservation and self-correction.

### Near-term outstanding

Called out in recent sessions but not yet in a phase:

- **Configurable chord bindings.** Pass 2 of the hotkey work — persist
  `push_to_talk_chord` / `toggle_to_talk_chord` in `capture_settings`,
  surface a chord-picker UI in `CapturesPage`, and wire a Tauri
  `update_chord_bindings` command so `HotkeyMonitor::update_bindings` picks
  up user changes live.
- **Generate-tab empty-state explainer.** The parallel aside to the Captures
  explainer described in *Product surface → Parallel explainer on the
  Generate tab*. Lands alongside Phase 3's universal mic button so both tabs
  feel symmetric.

## Overview

Voicebox ships the output half of a voice I/O loop: clone a voice, generate
speech, apply effects, compose multi-voice projects. The input half — speech to
text, dictation, routing — exists today as a single Whisper model wired into the
Recording & Transcription panel. This doc proposes making voice *input* a
first-class pillar: more STT engines, a dictation shell (global hotkey, audio
capture, paste, streaming), a local LLM backend, and a user-configurable
pipeline from captured audio to whatever the user wants to do with it.

Positioning is the key move. **Voicebox becomes the local voice I/O layer for
humans and AI agents** — a local alternative to cloud dictation tools, with the
differentiator that we also do TTS and voice cloning. The same app that
captures your voice can generate a response in any voice profile you've
cloned. "Anything voice is Voicebox."

### Positioning shift

Before this plan, Voicebox was **"the open-source AI voice cloning studio."**
Cloning was the headline capability.

After this plan, Voicebox is **"the open-source AI voice studio."** Cloning is
one capability in a broader category that now spans input (STT, dictation),
intelligence (local LLM, refinement, persona), output (TTS, cloning, effects,
Stories), and routing. The word "cloning" drops out of the top-line descriptor
because it's become a feature rather than the thesis.

### Competitive frame

Voicebox ends up covering the territory of two separately-funded, separately
branded cloud incumbents that operate on opposite sides of the same voice I/O
loop:

- **ElevenLabs** (~$3B+): voice cloning and TTS — the "agents speak" side
- **WisprFlow** (~$70M raised): voice dictation for agents and power users —
  the "users talk" side

Both are cloud-only. Voicebox becomes the only local alternative to either,
running in one app, with a single model directory and LLM shared between input
and output. That bridging — dictation → LLM → TTS with a cloned voice in the
middle — is the thing no single incumbent can match, because neither has the
other half.

### Launch-time copy tasks

These are not engineering tasks but should ride the Phase 4 ship so marketing
and positioning stay in sync with the product.

- **README.md** — drop "cloning" from the top-line descriptor. Add a section
  that explicitly frames Voicebox as "the open-source local alternative to
  WisprFlow and ElevenLabs." Competitive framing belongs in the README and on
  the landing page — not in-app (reads as defensive).
- **voicebox.sh landing page** — same positioning shift.
- **GitHub About / repo topics** — swap "voice-cloning" or similar tags for
  broader "voice-io," "local-tts," "local-stt," etc.
- **Release notes** — the Phase 4 launch note is the "we're now voice I/O" moment.

## Why now

- Cross-platform local dictation is an empty category. The tools people love
  (Superwhisper, MacWhisper, Aiko) are macOS-only. WisprFlow and
  Willow are cloud. Our Windows install base is the wedge — first-class Windows
  support for a local dictation product is genuinely differentiated.
- The `STTBackend` protocol already exists. The multi-engine registry pattern
  shipped with TTS makes adding Parakeet v3 and Qwen3-ASR a days-not-weeks
  effort on the backend side.
- The **persona loop** — speak to an agent, have it reply in a cloned voice —
  is a feature only we can ship. Nobody with a dictation product has TTS; nobody
  with a TTS product has good dictation. The full duplex is ours.
- Agent harnesses already pipe Voicebox TTS into their stacks. Giving those
  users STT from the same app closes the loop and makes Voicebox the default
  voice I/O layer for the agentic dev-tool crowd.
- **Typing a 2,000-character TTS script is user-hostile.** The most immediate
  internal win is dictating directly into Voicebox's own generation form —
  speak the script, generate the voice. This dogfoods the whole STT pipeline
  without touching a single OS-level API.
- **Voice-to-voice models are landing.** Moshi (Kyutai), GLM-4-Voice, Qwen2.5
  Omni, Mini-Omni, Sesame CSM, Spirit LM (Meta) — end-to-end speech LLMs that
  take audio in and emit audio out are a near-term reality. The pipeline we're
  building today is the scaffolding they slot into tomorrow.

## Non-goals

- Cloud fallback or "bring your own API key" STT/LLM. Local is the product.
- A separate tray-only dictation app. We extend Voicebox, not fork it.
- Replacing the Stories editor with a notes layout. Long-form capture is a
  preset on top of the pipeline, not a new product surface.
- Real-time translation UI. It can exist as a transform later, but it's not in
  this plan.
- Full agent orchestration. We provide the voice rails; the agent lives
  elsewhere and talks to us via the developer API.

## Architecture

### Three new backend concepts

**1. Expanded STT registry.** The existing `STTBackend` protocol abstracts
Whisper today. Add:

- **Parakeet v3** — 25 languages, very fast, the current quality leader for
  non-English local STT. Python path via `nemo_toolkit` or `transformers`.
- **Qwen3-ASR 0.6B int8** — 50+ languages, highest multilingual quality,
  cross-platform via `transformers`.
- **Kyutai ASR** *(optional)* — streaming-first, small, CPU-friendly. Fills the
  "CPU-only laptop" tier.

All register via `ModelConfig` and use the same download, cache, and model
management UI we already have for TTS. Zero special-casing.

**2. `LLMBackend` protocol.** Mirror of `TTSBackend` / `STTBackend`. First
implementations are Qwen3 0.6B / 1.7B / 4B running on the same PyTorch + MLX
infrastructure we already run. One runtime, one model cache, one GPU-memory
story.

Why not `llama.cpp` or `ollama`: we already have the dependency surface and the
model download UX. A second runtime fragments cache directories and model-status
UI. If CPU-only Windows latency becomes a problem we can revisit.

**3. Streaming transcribe transport.** Add `/transcribe/stream` as a WebSocket
endpoint alongside the existing HTTP `/transcribe`. Audio frames flow in,
partial transcripts stream back. Same FastAPI process, same loaded models. This
keeps dictation latency off the per-request JSON-encode critical path and lets
us ship real-time partial transcripts later without a protocol change.

### The pipeline abstraction

Every captured audio event flows through the same shape:
**Source → Transforms → Sink(s)**. Users configure presets that bind a source
to a transform chain to one or more sinks.

```
Source                      Transform                    Sink
──────────────────          ─────────────────            ─────────────────
Hold to speak        ──┐    STT model                    Clipboard + paste
Tap to toggle          │    Refinement LLM               Capture history
Long-form recorder     ├──▶ Persona LLM              ──▶ File on disk
File drop              │    Translation (later)          HTTP webhook
API call (WS / HTTP) ──┘                                 MCP server sink
                                                         TTS loopback (persona)
                                                         Platform sinks (later)
```

`Source → Transform → Sink` is internal, dataflow-style vocabulary (same shape
as Unix pipes, Apache Beam, Kafka) — not user-facing. The UI surface will use
Voicebox-native language (see open questions).

Concrete preset examples this shape enables:

- **Dictation** — hold-to-speak → Parakeet v3 → light refinement → clipboard + paste + history
- **Code prompt** — dedicated hotkey → Whisper Turbo → technical-vocab refinement → MCP sink for Claude Code
- **Agent voice reply** — hold-to-speak → STT → persona LLM → TTS with cloned profile → system audio out
- **Long-form capture** — dual-stream recorder → chunked STT → summary LLM → markdown file + history

Every user-facing feature collapses into (source + transform chain + sinks).
Meeting-style capture isn't a separate product; it's a preset. Competing tools
hardcode integrations (Trello, Granola); we make routing user-configurable.

### Native shim crate

The parts Tauri doesn't handle cleanly, gathered in one Rust crate with a
platform-agnostic API:

- **Global hotkey with modifier-only support.** Tauri's `global-shortcut`
  plugin requires full combos. We need "hold right-cmd" or "hold ctrl" as
  primitives. On macOS this means a CGEventTap on a background thread with
  polling fallback for dropped modifier events; on Windows a low-level keyboard
  hook; on Linux X11 + libinput, with Wayland as a known gap.
- **Focus introspection.** Query the frontmost app and its focused element via
  OS accessibility APIs — `AXUIElement` on macOS, UIAutomation on Windows,
  AT-SPI on Linux. Check the element's role to decide between a direct
  injection, a clipboard + paste, and a clipboard-only fallback with a
  notification. A blind paste that only "works when a text field happens to
  be focused" is the easy default; we should make the decision deliberately.
- **Simulated paste.** CGEvent on macOS, SendInput on Windows, uinput / ydotool
  on Linux. Wayland is the hard case and needs explicit handling.
- **Atomic clipboard save/restore.** Save *all* items and *all* MIME
  representations before writing our transcript, restore atomically after
  paste. Pasting a transcript shouldn't clobber a user's in-progress rich-media
  clipboard.
- **Frontmost-window context capture** *(later).* macOS Vision, Windows OCR,
  Linux tesseract. Optional feature to feed the refinement LLM disambiguation
  hints from the window being pasted into.

Main process owns this crate. Webview never sees platform differences.

### Target-aware delivery

The paste sink adapts to what's in focus. This is a single sink type with
branching behavior, not four separate sinks.

| Target | Delivery strategy |
|---|---|
| Focused text field inside Voicebox | Direct React state update via event. No clipboard involved. |
| Focused text field in another app | Accessibility-verified paste: save clipboard, write transcript, simulate paste, restore clipboard. |
| No text focus detected | Clipboard only, toast notification ("Transcript copied — no text field focused"). |
| Platform-specific special cases (terminal apps, specific editors) | Per-app overrides where the generic path misbehaves. |

### Where each concern lives

| Concern | Layer |
|---|---|
| STT / LLM / TTS inference | Python backend |
| Model downloads, progress, cache | Python backend |
| Pipeline runner (orchestrates transforms and sinks) | Python backend |
| Audio capture from mic / system audio | Rust (Tauri side) |
| Audio streaming over WebSocket to backend | Rust |
| Global hotkey capture | Rust (native shim crate) |
| Paste simulation, clipboard save/restore | Rust (native shim crate) |
| Pipeline preset UI, capture history, settings | React |

Model work in Python. OS work in Rust. User config in React.

## Product surface

### A new tab (and a sidebar reshuffle)

The current sidebar is `Generate · Stories · Voices · Effects · Audio · Models ·
Settings`. The existing Audio tab is output-device and channel routing
config — infrastructure, not a creative workspace — and the Settings page
already has a sub-tab pattern (`ServerSettings/`: Connection, Models, GPU,
Update) that fits it naturally.

**Move Audio to a Settings sub-tab. Reclaim the sidebar slot for voice input.**

The new tab shows recent captures (audio + transcript paired), active presets,
dictation settings, model pickers for STT and LLM. Exact name is an open
question.

**Sidebar placement:** Captures sits at position 3, directly under Stories and
above Voices. Creates an "input voice / output voice" adjacency — captured
speech is one slot away from the voices you can play it back through, which
mirrors the Phase 4 "Play as voice" feature's mental model. Full order:
Generate · Stories · Captures · Voices · Effects · Models · Settings.

### Parallel explainer on the Generate tab

The Captures settings page gets a "What's different" aside that introduces
Voicebox's dictation story. The Generate tab deserves a parallel — first-time
users need to be told what voice generation is *for* in a post-Voice-I/O
world, not just handed a text field.

Shape: an **empty-state card** rendered in the Generate tab when there's no
generation history yet, disappearing once the user has generated anything.
Teaches without claiming permanent real estate. Parallel bullets to the
Captures aside so the two tabs feel like two sides of one product:

- **Clone any voice in seconds** — a short sample is enough
- **Seven engines, 23 languages** — creative range, not a single model
- **Agent-ready** — REST + WebSocket API, one checkbox away from giving any
  AI agent a voice

This lands in Phase 4 alongside the Captures tab, for visual and thematic
symmetry. Not a persistent sidebar — the Generate tab is a workspace and
should reclaim its space once the user is producing work.

### Archival by default

Every capture saves the original audio alongside the final transcript in a
pattern that mirrors `data/generations/`. Optional retention setting. Free for
us — the storage and UI patterns exist today for generations.

### Developer API, day one

The WebSocket transcribe endpoint is a first-class public API, documented
alongside `/generate`. Pipeline presets are addressable by ID via
`/pipelines/{id}/run` so agent harnesses and shell scripts can invoke
user-configured flows. An MCP server sink ships built-in, so integrations with
Claude Code, Cursor, Cline, etc. are one checkbox rather than a custom build.

### Agent voice output

Dictation is one half of the loop — user speaks, agent listens. The other half
— agent speaks, user hears — is equally load-bearing and deserves a
first-class primitive rather than being buried as a TTS loopback sink or a
consumer read-aloud button.

The shape is a single new capability: any agent can call Voicebox to speak
arbitrary text in a user-configured voice. The same pill that surfaces during
dictation surfaces during agent speech, so the user always sees what's coming
out of their machine.

```
MCP tool:  voicebox.speak({ text, profile?, style? })
REST:      POST /speak { text, profile_id?, style? }
```

Both accept an optional voice profile (defaults to the user's configured
default), an optional delivery-style string for engines that support it, play
audio through system output, and surface the pill in a `speaking` state.

**Key design points:**

- **Pill is bidirectional.** States expand from `recording / transcribing /
  refining / rest` to include `speaking` — voice profile name, waveform in
  the profile's color, visible duration. Same floating surface for both
  directions so users have one mental model.
- **Visibility is mandatory.** Silent background TTS is a trust hazard. Every
  agent-initiated `speak()` surfaces the pill. No headless "TTS daemon" mode.
- **Per-source voice policy.** Settings let users bind specific MCP clients or
  API keys to specific voice profiles — Claude Code in "Morgan," Cursor in
  "Scarlett" — so users can tell which agent is talking without looking.
- **Mute + rate limits.** One-toggle mute for all agent speech. Per-source
  rate limits prevent a runaway agent from monologuing.

This primitive is what makes "Voicebox as voice layer for every agent on your
machine" a concrete shipping capability rather than marketing language. MCP,
ACP, and A2A integrations all slot into it — none of those agent protocols
need to know anything about TTS models, GPU placement, or voice profiles.
They call `speak()`.

**Relationship to the persona loop.** The persona loop below is *one* use of
`speak()` — STT → LLM → `speak(llm_reply)`. Other uses skip STT entirely: a
long-running task announcing completion, a notification, an agent proactively
asking the user a question. The primitive is deliberately simpler than the
persona loop so it can serve both flows from the same API.

### Relationship to voice profile samples

A capture and a voice profile sample both hold `audio + text`, so there's an
obvious temptation to unify them. Don't. The metadata and lifecycle
differences are real:

| | Capture | Voice profile sample |
|---|---|---|
| Profile association | Standalone | Bound to one profile |
| Text field | Raw transcript + optional LLM-refined version | Exact `reference_text` only |
| LLM refinement | Often applied | Must not be applied — the reference text must match the audio verbatim or cloning breaks |
| Volume | Dozens per day | ~5 per profile, semi-permanent |
| Typical content | Whatever the user said | Often scripted phrases for cloning |

A unified table would mean nullable `profile_id`, nullable `refined_transcript`,
nullable `reference_text` — a fat row that means different things in different
states. Not worth the complexity.

**What to ship instead: a one-way promote action.** Capture → Sample, zero
data-model churn. Thin endpoint:

```
POST /profiles/{id}/samples/from-capture/{capture_id}
```

Reads the capture's audio path and raw transcript, calls the existing
`add_sample()` service with `reference_text` pre-filled from the transcript,
lets the user edit the reference text in a dialog before saving (transcripts
are usually 90% right but cloning wants 100%). The capture stays in the
Captures tab untouched — the sample is a copy, not a move.

UI hook: the Captures tab's Send-to menu gains a **"Use as voice sample…"**
option that opens a profile picker (with "+ New voice" for cold starts) and a
reference-text confirm dialog.

The inverse direction (sample → capture) we deliberately skip. Samples are
often scripted phrases used for cloning and they'd clutter the Captures list
without adding value; also a subtle privacy surprise for users who don't
expect their sample text browsable alongside real captures.

**Audio storage deduplication is a later optimization.** Today a promoted
capture duplicates the audio file on disk. That's fine. Content-addressable
storage (`data/audio/<sha256>.wav` with refcounting) can come in Phase 8 as
housekeeping — it'd let a capture and a sample share one underlying file, but
it's not user-visible and not necessary to ship the promote flow.

### The persona loop

One flow on top of the `speak()` primitive: STT → persona LLM →
`speak(llm_reply)`. Voice profiles gain optional metadata — a natural-language
personality description and default LLM behavior. The LLM runs text through
the profile's voice context, then `speak()` generates TTS with the cloned
profile. End-to-end voice-to-voice with a cloned identity transforming the
content, not just reading it.

Use cases this unlocks:

- Agents that respond to spoken input in a specific voice
- Interactive character experiences (games, narrative tools, accessibility)
- Speech assistance for people who can't speak in their original voice

The shape — STT + LLM + TTS — also stages us for end-to-end speech LLMs which
collapse all three into one transform. See *Voice-to-voice readiness* below.

### Voice-to-voice readiness

The STT → LLM → TTS chain that powers the persona loop is a staged approximation
of voice-to-voice. A real end-to-end speech LLM (Moshi, GLM-4-Voice, Qwen2.5
Omni, Mini-Omni, Sesame CSM) replaces the three middle boxes with a single
fused transform: audio in, audio out, no text in between. The pipeline shape
accommodates this natively — register the model as a single `LLMBackend` (or
a new `SpeechLLMBackend` if the protocol needs to differ), expose it as a
transform type, and the same sinks work unchanged.

Framing this plan as "voice-to-voice scaffolding, with today's models as the
staged fallback" is a strong pitch for agent-harness users who are already
tracking these models.

## Open questions

1. **Tab name.** Leaning **Captures** — neutral, extensible across dictation,
   long-form recordings, and uploaded audio without repainting the tab later.
   "Dictations" is narrower (office-productivity coded, doesn't fit meeting
   recordings). "Notes" is the wrong mental model — nobody opens Voicebox to
   write notes. "Transcriptions" is flat.
2. **Refinement vocabulary.** The LLM-post-STT step needs a user-facing name.
   "Refine," "polish," "rewrite," "smart edit" are candidates. "Refinement" in
   this doc as a placeholder only.
3. **Preset primitive.** What do we call a user-configured pipeline? "Intent"
   collides with the existing `instruct` field on TTS generation. "Flow" is
   Zapier-coded. "Route" is too networking. Needs its own pass.
4. **Persona metadata shape.** Does personality live directly on the voice
   profile, or as a separate persona construct that wraps profile + LLM config?
   The first is simpler; the second scales better if we later want multiple
   personas per voice.
5. **Long-form capture product surface.** Pure preset, or dedicated entry point
   in the new tab? Leaning preset, but long-form is the feature that most
   justifies its own landing page.
6. **Hotkey primitive naming.** Hold-vs-tap needs Voicebox-native phrasing in
   UI copy. Settings can still use industry-standard terms.

## Ordered phases

The v1 prototype deliberately skips the hardest parts of the long-term plan
(native OS shim, global hotkeys, paste injection, new STT models). Everything
in Phase 1–4 is in-process code using Whisper (which we already ship) and the
existing model infra. No CGEvent taps, no SendInput, no clipboard timing.
The usual OS-level sprawl of a dictation stack is exactly what we sidestep
by starting in-app.

### Phase 1 — Groundwork

- Move the Audio tab into a Settings sub-tab (`ServerSettings/` gains one
  more section). Audio is device/channel config, not a creative workspace.
- Reserve the sidebar slot for the new Captures tab (name TBD but leaning
  Captures — see open questions).
- Gate the Captures tab behind a feature flag so we can merge to `main` and
  iterate without shipping half-built UI to users.

### Phase 2 — Local LLM backend

`LLMBackend` protocol alongside `TTSBackend` / `STTBackend`. Register Qwen3
0.6B / 1.7B / 4B via `ModelConfig`. Reuses the HF download path, cache
directory, and model management UI. MLX (4-bit community quants) on Apple
Silicon, PyTorch (transformers AutoModelForCausalLM) elsewhere, same as our
TTS split.

No new runtime. No `llama.cpp`, no `ollama`, no fragmented model cache.

### Phase 3 — In-app voice input

A universal mic button on every Voicebox text input. Hold, speak, release —
text lands in the focused field via direct React state update. No OS APIs
involved; Voicebox owns the input.

Marquee use cases:

- **Generation form.** Dictate a 2,000-character TTS script instead of typing
  it. This alone justifies the feature.
- **Voice profile descriptions.** Describe a voice's personality by speaking,
  which then becomes the input for Phase 4's persona loop.
- **Story titles, preset names, any free-text field.** Free reuse.

Backend: add `/transcribe/stream` WebSocket endpoint. Audio frames in, partial
transcripts out. Reuses the existing Whisper model in memory. Optionally routes
through the LLM from Phase 2 for light refinement.

### Phase 4 — Captures tab

Graduates the tab out from behind the feature flag. Shows recent captures
(audio + transcript pairs), lets the user replay, re-transcribe with a
different model, edit the transcript, and send the output through the LLM.
Archival is automatic — every capture saves audio alongside transcript.

**Includes the "Play as voice profile" action.** This is the simplest version
of the persona loop and it lands here for free — no LLM involved, no new
backend endpoints, just a Captures-tab button that sends the transcript text
to the existing `/generate` endpoint with a user-selected voice profile and
plays the result. Category-defining differentiator from the v1 prototype
onward: Superwhisper and WisprFlow cannot do this because they have no TTS. Voicebox can, with one day of frontend wiring.

Keep it aggressively minimal on day one. A capture list, a detail view, a
model picker, a Play-as-voice dropdown. Refinement prompt editing, correction
dictionaries, per-source overrides — none of that ships here. They become
Tier-2 work when someone actually asks for them.

### Phase 5 — Agent voice output + persona loop

Two features that together make "Voicebox as the voice layer for every agent
on your machine" a shipping reality:

1. **`speak()` primitive.** New `POST /speak` endpoint and `voicebox.speak`
   MCP tool. Any agent calls Voicebox to speak arbitrary text in a
   user-configured voice; the pill surfaces in a `speaking` state. Settings
   UI for default voice, per-agent voice binding (Claude Code → Morgan,
   Cursor → Scarlett), and a global mute.
2. **Persona loop.** Extends `speak()` with an LLM step — STT → persona LLM
   → `speak(llm_reply)`. Voice profiles gain optional personality metadata
   and default LLM behavior. End-to-end voice-to-voice with a cloned
   identity transforming the content, not just reading it.

Phase 4 demoed the user-initiated direction of the loop (Play as voice). This
phase ships the *agent*-initiated direction, which is the category-defining
capability and the pitch that lands with agent-harness users. The persona
loop is one flow on top of the `speak()` primitive — notifications, proactive
agent questions, and task-completion announcements all use `speak()` directly
without the LLM in the middle.

Launchable headline moment for the "local voice I/O" positioning.

### Phase 6 — STT engine expansion

Parakeet v3 and Qwen3-ASR register as additional `STTBackend` implementations.
Optional: Kyutai ASR. Multilingual coverage upgrades (50+ languages). Whisper
stays as the sensible default.

Deferred to here because Whisper is already good enough for v1 and the model
picker UI exists. Adding rows to it doesn't change the product shape.

### Phase 7 — External dictation shell

Native shim crate (global hotkey with modifier-only support, focus
introspection via OS accessibility APIs, paste simulation, atomic clipboard
save/restore). Tauri-side audio capture streams to the same WebSocket endpoint
Phase 3 already ships. Paste sink with target-aware delivery.

This is the feel-good phase. It's also the riskiest: paste timing, hotkey
reliability, and cross-platform focus detection are all engineering problems
that have to be nailed or the product doesn't work. Phase 3's success derisks
the backend plumbing before we start it.

### Phase 8 — Pipeline routing, sinks, long-form

Multiple source types, user-configurable transform chains, multiple sinks per
preset. MCP server sink (the agent-harness play). HTTP webhook sink. File
sink. Developer-facing `/pipelines/{id}/run` endpoint. Preset editor UI in
the Captures tab.

Dual-stream recorder (mic + system audio) as a source type. Chunked STT
transform with overlap-based deduplication. Summary LLM transform. Long-form
capture becomes a preset, not a new tab.

Platform-specific sinks (Apple Notes on macOS, Obsidian, etc.) as opt-in
integrations behind the generic sink interface.

## Architectural prerequisites

Two pieces of existing `docs/PROJECT_STATUS.md` work become load-bearing here:

- **Platform support tiers** (#420, PR #465). Native shim capabilities vary by
  platform — Wayland paste is worse than X11, Windows system-audio capture has
  edge cases, frontmost-window OCR is platform-gated. Tier definitions let us
  ship confidently with honest user-facing expectations.
- **Platform gating on `ModelConfig`** (bottleneck #6 in PROJECT_STATUS).
  Parakeet's Core ML path is Apple-only; the PyTorch path is Windows/Linux.
  Same gating mechanism that currently blocks shipping VoxCPM.

Neither needs to complete before Phase 1, but both should complete before
Phase 4 when user-configurable pipelines surface the differences to end users.
