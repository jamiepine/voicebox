# Accessibility: screen reader and keyboard improvements

## Summary

Improvements to support screen reader and keyboard users across the main app surfaces: audio player, generation UI, voice selection, history, voices tab, model management, server tab, and stories.

**Tested with NVDA and Narrator on Windows.**

---

## What changed

### Audio player (after generating audio)

- **Play/Pause, Loop, Mute, Close** – `aria-label` added so each control is announced (e.g. "Play", "Pause", "Loop", "Mute", "Close player").
- **Playback position slider** – `aria-label="Playback position"` and `aria-valuetext` with current/total time (e.g. "0:30 of 2:15").
- **Volume** – Wrapped in a labelled group; volume slider has an associated screen-reader-only label and `aria-valuetext` for the level (e.g. "Volume level, 75%").

### Generation UI (text box and voice choice)

- **Generate speech** (submit) and **Fine-tune instructions** (sliders) – Icon buttons now have `aria-label` (and state for fine-tune, e.g. "Fine-tune instructions, on").

### Voice selection (cards on Generate screen)

- Each **voice card** is focusable (`tabIndex={0}`), has `role="button"`, and an `aria-label` (e.g. "Prashant, en. Select as voice for generation.") with `aria-pressed` when selected.
- **Enter/Space** on the card selects that voice; tab order is card → Export/Edit/Delete.

### History list (generated samples)

- Each **sample row** is focusable with `role="button"` and an `aria-label` (e.g. "Sample from [profile], [duration], [date]. Press Enter to play."); **Enter/Space** plays or restarts.
- **Transcript textarea** has `aria-label` (e.g. "Transcript for sample from [profile], [duration]") so when you focus on the text area, the sample is announced in context.

### Voices tab (table)

- Each **voice row** is focusable with `role="button"` and an `aria-label` (e.g. "[Name], [language], [N] generations, [N] samples. Press Enter to edit."); **Enter/Space** opens edit (except when focus is in a control).
- **Actions** dropdown trigger has `aria-label="Actions for [profile name]"`.

### Model management

- Each **model row** is a focusable region (`tabIndex={0}`, `role="group"`) with an `aria-label` (e.g. "[Model name], [status], [size]. Use Tab to reach Download or Delete.").
- **Download** and **Delete** (and Downloading) buttons have `aria-label` (e.g. "Download [name]", "Delete [name]").

### Server tab (panels)

- **Server Connection**, **Server Status**, and **App Updates** cards are landmarks: `role="region"`, `aria-label`, and `tabIndex={0}` so each panel is focusable and announced (e.g. "Server Connection", "Server Status", "App Updates").

### Stories list

- Each **story row** is a focusable control (`role="button"`, `tabIndex={0}`) with `aria-label` (e.g. "Story [name], [N] items, [date]. Press Enter to select."); **Enter/Space** selects the story. Actions button has `aria-label="Actions for [story name]"`.

### Other controls

- **Story list** – Actions (⋮) button: `aria-label="Actions for [story name]"`.
- **Story track editor** – Play/Pause, Stop, Split, Duplicate, Delete, Zoom in/out: `aria-label` on all icon buttons.
- **Voice profile samples** (SampleList, AudioSampleUpload, AudioSampleRecording, AudioSampleSystem) – Play/Pause and Stop: `aria-label` (e.g. "Play sample", "Pause", "Stop playback").
- **SampleList** mini sample player – Seek slider has `aria-label="Sample playback position"` and `aria-valuetext` for time.

---

## Testing

- **Screen readers:** Tested with **NVDA** and **Narrator** on Windows.
- **Keyboard:** Tab order and Enter/Space activation verified for focusable rows and buttons.

---

## Tech note

- React + TypeScript; Radix UI primitives; labels added via `aria-label`, `aria-labelledby`, `aria-valuetext`, and `role`/`tabIndex` where needed.
- No new dependencies.
