# Voicebox Conformity Check

This document tracks how the local Dubbing module aligns with the upstream
Voicebox architecture and where it intentionally diverges.

## Current Assessment

The Dubbing module is broadly conformant as an autonomous feature module, but
not yet fully idiomatic as an upstream Voicebox extension.

It follows the main Voicebox architecture:

- React frontend talking to the local FastAPI backend
- backend served as a Tauri sidecar on `localhost:17493`
- TTS generation delegated to the existing Voicebox generation service and TTS
  backends
- generated audio stored as regular Voicebox generation records
- generated Dubbing rows marked with `source="dubbing_segment"`
- no direct modification of global TTS behavior for Dubbing-only needs

## Conformant Areas

- Dubbing stays local to the app/backend and does not require a remote service.
- It reuses existing Qwen engines instead of introducing a separate inference
  stack.
- It persists generated audio in the same storage model as normal Voicebox
  generations.
- It keeps server startup compatible with the original Tauri sidecar model.
- It now treats Dubbing-specific generation behavior as isolated product logic
  rather than global Voicebox behavior.

## Divergences From Upstream Architecture

- Dubbing introduces its own routes and services instead of being implemented
  only through the generic `generate`, `history`, and `stories` modules.
- Dubbing has its own timeline logic instead of directly reusing the upstream
  Stories timeline data model.
- Dubbing adds SRT-specific concepts that do not exist upstream:
  `fit_status`, `delta_ms`, `pace_groups`, fixed subtitle windows, and
  timeline WAV export against subtitle timecodes.
- Sequential SRT batch generation and segment regeneration are product-specific
  workflows rather than generic Voicebox generation flows.
- Timeline WAV export is Dubbing-specific and does not currently reuse Stories
  export.

## Rationale For Divergence

Dubbing needs constraints that Stories does not model directly:

- importing SRT files as the source of truth
- preserving fixed subtitle start/end timecodes
- warning when generated audio exceeds a subtitle window
- letting users edit subtitle text after import
- allowing manual timeline correction while keeping SRT-derived metadata
- exporting a single WAV aligned to the original subtitle timeline

These requirements justify a dedicated Dubbing module for now.

## Upstream Integration Notes

If this fork is proposed upstream, present Dubbing as a separate feature module
rather than a replacement for Stories.

Recommended framing:

- Stories is a creative multi-voice timeline editor.
- Dubbing is an SRT/timecode-driven production workflow.
- Both can share UI patterns and audio utilities, but Dubbing needs its own
  persistence and validation rules.

Potential future alignment:

- reuse more Stories timeline components where practical
- keep Dubbing export logic isolated but document it beside Stories export
- keep all Dubbing-only generation heuristics out of global generation services
- register any new TTS engine, such as Qwen VoiceDesign, through the official
  `TTSBackend` / `ModelConfig` path
