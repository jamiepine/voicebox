# Quickstart: Validating MiniCPM5-1B LLM Engine Support

## Prerequisites

- Backend running locally (`just` recipe or however this repo's dev server is normally started — see repo root docs).
- For the MLX path: an Apple Silicon Mac, `mlx-lm>=0.31` installed (see [research.md](research.md) action item).
- For the PyTorch path: any platform the app already supports Qwen3 PyTorch inference on.

## Backend-only validation (before touching the UI)

1. **Registry**: `python -c "from backend.backends import get_llm_model_configs; print([c.model_name for c in get_llm_model_configs()])"` → must include both `qwen3-*` entries and `minicpm5-1b`.
2. **Factory**: confirm `get_llm_backend_for_engine("minicpm_llm")` returns `MLXMiniCPMLLMBackend` on Apple Silicon and `PyTorchMiniCPMLLMBackend` elsewhere (unit test, see [data-model.md](data-model.md) / research.md Decision 6 — mock `get_backend_type()`).
3. **Load + generate (real, one-time manual check, not CI)**: hit `POST /llm/generate` with `{"model_size": "minicpm5-1b", "prompt": "Say hello in one short sentence."}` against a running dev backend. Expect: model downloads (first time) or loads from cache, returns a short non-empty completion, response echoes the model identifier used.
4. **Cross-engine unload guard**: with a Qwen3 size already loaded (hit `/llm/generate` with `model_size: "qwen3-0.6b"` first), then call it again with `model_size: "minicpm5-1b"`. Check `/models/status` before and after: the Qwen3 entry's `loaded` flag must flip to `false` once MiniCPM5-1B's flips to `true` — this is the new guard from [data-model.md](data-model.md)'s "State / relationships" section, and it's the one behavior that's easy to silently skip.
5. **Migration**: run the app once against a copy of an old (pre-feature) SQLite DB file that has `llm_model = "1.7B"` in `capture_settings`. After startup, query the DB directly — `llm_model` must now read `"qwen3-1.7b"`. Run the app a second time against the same file — value must be unchanged (idempotency).

## End-to-end UI validation (User Stories 1–3 from spec.md)

1. Open Settings → the LLM/refine model picker (`CapturesPage.tsx` today) → confirm "MiniCPM5 1B" is listed alongside the three Qwen3 sizes.
2. Select it while it isn't downloaded yet → confirm a download starts and progress is shown, same as selecting an un-downloaded Qwen3 size.
3. Once downloaded, trigger Compose on a profile with a personality set, and Rewrite via `/generate` with `personality=true` → confirm output is produced and the app is now using MiniCPM5-1B (cross-check against `/models/status`'s `loaded` field, or backend logs).
4. Trigger a capture refine (`CapturesTab.tsx` flow) with MiniCPM5-1B selected → confirm the refined transcript is produced and the capture's stored `llm_model` reflects `"minicpm5-1b"`.
5. Open the model management page (`ModelManagement.tsx`) → confirm MiniCPM5-1B appears under the language-models grouping (verify the `startsWith('qwen3-')` filter fix — see [plan.md](plan.md) — actually includes it) with working download/delete/loaded-status controls.
6. Delete MiniCPM5-1B from the management page while it's the active selection → confirm it unloads cleanly and a subsequent Compose/Rewrite/Refine call surfaces the same "not downloaded" error class Qwen3 already produces in that situation (Edge Cases in spec.md).
7. **Regression check (User Story 3)**: repeat steps 1–4 with a Qwen3 size instead of MiniCPM5-1B, on a machine whose DB predates this feature, to confirm nothing about the existing Qwen3 flow changed shape or broke during the identifier-scheme migration.

## Non-Apple-Silicon check (Edge Cases in spec.md, SC-003)

Repeat "End-to-end UI validation" step 1–4 on a non-Apple-Silicon machine (or by forcing `get_backend_type()` to return `"pytorch"` in a local override) to confirm MiniCPM5-1B is fully usable via `PyTorchMiniCPMLLMBackend`, not just the MLX path.
