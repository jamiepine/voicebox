# Phase 0 Research: MiniCPM5-1B LLM Engine Support

All architectural unknowns were resolved in conversation with the user before this plan was written. This document records the resulting decisions plus the codebase findings that ground them, so no `NEEDS CLARIFICATION` markers remain in the Technical Context.

## Baseline update (2026-08-26, re-assessed after fast-forwarding to origin/main)

Two commits landed on `main` after this plan was first drafted and before implementation started: `5488b6c fix(backend): release memory when unloading MLX models` and `34b8433 fix(backend): suppress E402 for intentionally-late MLX imports`. Re-reviewed both against every artifact in this feature; net effect is one required addition, no other change to scope, decisions, or task ordering:

- `5488b6c` adds `empty_mlx_cache()` (`backend/backends/base.py`, wraps `mx.clear_cache()`) and now calls it from `MLXQwenLLMBackend.unload_model()` (`backend/backends/qwen_llm_backend.py`) — MLX keeps freed buffers in its own allocator pool instead of returning them to the OS, so a plain `del self.model` doesn't actually shrink process memory on Apple Silicon. **This directly changes Decision 1 below**: `MLXMiniCPMLLMBackend.unload_model()` must call `empty_mlx_cache()` too, or MiniCPM5-1B would regress the exact memory-release problem this project just fixed for Qwen3 — undermining this feature's own resource-efficiency premise. Reflected in tasks.md T002/T003/T004.
- The same commit adds `clear_voice_prompt_memory_cache()` and wires it into the TTS unload paths only — its own commit message states "Whisper and the LLM backends never produce voice prompts, so their unload paths are left alone," and the diff confirms the `if config.engine == "qwen_llm":` block in `unload_model_by_config` (`backend/backends/__init__.py`) is untouched. **No change** to Decision 3/data-model.md's plan for generalizing that block — it still just needs the `"qwen_llm"` literal check broadened to any `LLM_ENGINES` member, with no cache-clearing addition.
- `34b8433` is a lint-only follow-up (`# noqa: E402` on four already-late imports in `mlx_backend.py`) — irrelevant to this feature; `minicpm_llm_backend.py` mirrors `qwen_llm_backend.py`'s import order (no offline-patch-before-import pattern there), so it doesn't inherit this E402 situation.

No other file this feature touches changed on `main` in the interim. Everything else in this plan stands as designed.

## Decision 1: Model loading — mirror `qwen_llm_backend.py` exactly

**Decision**: New file `backend/backends/minicpm_llm_backend.py` with `PyTorchMiniCPMLLMBackend` and `MLXMiniCPMLLMBackend`, structurally identical to `backend/backends/qwen_llm_backend.py` (same `LLMBackend` protocol methods, same `_build_messages` helper, same `apply_chat_template(..., enable_thinking=False)` call).

**Rationale**:
- `openbmb/MiniCPM5-1B`'s `config.json` declares `"architectures": ["LlamaForCausalLM"]`, `"model_type": "llama"`, no `auto_map` — a plain HF Llama checkpoint, loadable via `AutoModelForCausalLM.from_pretrained` with no `trust_remote_code`, identical to how Qwen3 loads today.
- MiniCPM5-1B's chat template accepts the same `enable_thinking: bool` kwarg Qwen3's does (confirmed via OpenBMB's own deployment docs) — the existing `enable_thinking=False` call in `_generate_sync` needs zero changes to work for both model families.
- mlx-lm supports the architecture natively (standard Llama attention shape, GQA with `num_key_value_heads=2`) — no custom kernel or model-code fork required (source: `github.com/OpenBMB/MiniCPM/docs/deployment/mlx.md`).
- The existing repo convention for TTS is "one backend module per engine" (`chatterbox_backend.py`, `kokoro_backend.py`, `hume_backend.py`, `luxtts_backend.py` each mirror a shared shape rather than sharing one parametrized class) — a new `minicpm_llm_backend.py` file follows that established pattern rather than introducing a new shared-abstraction layer (YAGNI: don't generalize two data points into a framework).

**Alternatives considered**:
- Parametrizing `qwen_llm_backend.py` to take an arbitrary repo-map instead of adding a new file — rejected: diverges from the codebase's existing per-engine-file convention and would rename an established, working module for no functional gain.
- Using `mlx-community/MiniCPM5-1B-OptiQ-4bit` (community mixed-precision quant) — rejected per explicit user decision; only the official `openbmb/MiniCPM5-1B-MLX` repo is in scope. Revisit later as a separate decision once the two can be compared empirically (out of scope here).

## Decision 2: Model repos

| Backend | Repo |
|---|---|
| PyTorch | `openbmb/MiniCPM5-1B` |
| MLX (Apple Silicon) | `openbmb/MiniCPM5-1B-MLX` (official, pre-quantized 4-bit) |

**Constraint found**: MiniCPM5-1B has two EOS token ids (`eos_token_id: [1, 130073]`). The MLX path requires `mlx-lm>=0.31` — earlier versions only honor a single EOS id and would fail to stop generation cleanly. `backend/requirements` (or platform-specific requirements file) needs its `mlx-lm` pin checked/bumped as part of implementation if it currently pins below 0.31.

## Decision 3: Global "active LLM" identifier — switch to `ModelConfig.model_name`

**Finding**: The codebase already has a per-model, cross-engine-unique key (`ModelConfig.model_name`, e.g. `"qwen-tts-0.6B"`, `"chatterbox-tts"`) used everywhere in `backend/backends/__init__.py`'s TTS-side lookup helpers (`get_model_config`, `unload_model_by_config`, `check_model_loaded`, `get_model_load_func` all key off `model_name` already for TTS). The LLM side never needed this because only one LLM engine existed — `llm_model`/`model_size` was always implicitly `"qwen_llm"`.

**Decision**: Reuse `model_name` as the LLM selector too (`"qwen3-0.6b"`, `"qwen3-1.7b"`, `"qwen3-4b"`, `"minicpm5-1b"`), resolved to `(engine, model_size)` via the existing `get_model_config(model_name)` helper. No new concept, no new field.

**Rationale**: Matches an established in-repo pattern instead of inventing a parallel `llm_engine` field that would need to be kept in sync with `llm_model` everywhere it's read or written (KISS — fewer pieces of state that can drift out of sync).

**Alternatives considered**: A separate `llm_engine` column/field alongside the existing `llm_model` (size) field — rejected: doubles the surface area of every read/write site for no behavior difference, and the codebase already has a working single-key pattern for exactly this problem on the TTS side.

## Decision 4: Data migration — one-time idempotent rewrite, no compatibility shim

**Finding**: `backend/database/migrations.py`'s file header explicitly documents why this project has no Alembic (single-user desktop app, one SQLite file per install, idempotent startup checks have covered 12 schema changes so far) and prescribes the exact pattern: append a `_migrate_*` helper, call it from `run_migrations()`, make it idempotent, log when it does real work.

**Decision**: Add `_migrate_llm_model_names(engine, tables)` following that pattern — a plain `UPDATE ... SET llm_model = :new WHERE llm_model = :old` for the three legacy values (`"0.6B"→"qwen3-0.6b"`, `"1.7B"→"qwen3-1.7b"`, `"4B"→"qwen3-4b"`) against both `capture_settings.llm_model` and `captures.llm_model`. Naturally idempotent — after the first run the `WHERE llm_model = '0.6B'` clause matches zero rows.

**Rationale**: This is a single-user local SQLite file; there is no fleet of servers to coordinate and no rollback requirement beyond "don't lose the user's setting." A one-time rewrite plus removing the old format everywhere else is simpler and has less permanent surface area than threading "accept either format" logic through every read site forever.

**Alternatives considered**: Backward-compatibility shim (accept both old and new value shapes indefinitely in every lookup) — rejected: permanent complexity for a one-time transition, and this project's own migration file explicitly favors the "just rewrite the data" approach elsewhere.

## Decision 5: Validation — runtime membership check replaces static regex

**Finding**: `backend/models.py` currently validates `llm_model`/`model_size` fields with a compile-time regex (`pattern="^(0\\.6B|1\\.7B|4B)$"`). This can't express a value set that depends on `get_backend_type()` (mlx vs pytorch) and now spans two engines.

**Decision**: Replace the `Field(pattern=...)` constraints with a Pydantic field/model validator that checks membership in `{cfg.model_name for cfg in get_llm_model_configs()}` at request time.

**Rationale**: The valid set is already computed dynamically elsewhere (`routes/llm.py`'s existing `valid_sizes = {cfg.model_size for cfg in get_llm_model_configs()}` check) — this decision generalizes an existing runtime-check idiom already present in the codebase rather than inventing a new validation mechanism.

## Decision 6: Testing approach

**Finding**: There is no existing dedicated unit test for `qwen_llm_backend.py` or for the LLM entries in `backend/backends/__init__.py`'s config registry today. The closest LLM-adjacent tests (`test_personality_samples.py`, `test_refinement_samples.py`) are manual, human-scored eval scripts run against a live server (`if __name__ == "__main__"` / argparse entry points, not pytest-collected) — appropriate for judging non-deterministic text quality, not for CI-gated correctness. `test_model_status_pending_downloads.py` is the closest precedent for plain pytest-style unit tests over pure Python registry/state logic (no live model loading, no server).

**Decision**: New coverage for this feature follows the `test_model_status_pending_downloads.py` style — plain pytest functions against pure logic:
- `backend/backends/__init__.py`: `get_llm_model_configs()` returns entries for both engines; `get_llm_backend_for_engine("minicpm_llm")` returns the right class for each platform (mock `get_backend_type()`); `unload_model_by_config`/`check_model_loaded`/`get_model_load_func` work for a `minicpm_llm` config the same way they already do for a `qwen_llm` one.
- `backend/backends/minicpm_llm_backend.py`: load/generate/unload with `AutoModelForCausalLM`/`AutoTokenizer`/`mlx_lm.load` mocked out (no real model download in CI) — same level of coverage as would exist for `qwen_llm_backend.py` if it had a dedicated test file (it doesn't, so this is new coverage, not parity coverage, but follows the codebase's demonstrated mocking idiom used elsewhere for backend classes, e.g. `test_rocm_backends.py`).
- `backend/database/migrations.py`: one test creating a throwaway SQLite engine with legacy values, running `_migrate_llm_model_names`, asserting the rewrite and its idempotency on a second run.

**Non-goal**: No new frontend test infrastructure — the existing Qwen3 LLM picker code in `CapturesPage.tsx`/`CapturesTab.tsx` has no dedicated frontend unit tests today, so this feature doesn't introduce a new testing tier just for its own UI additions (match existing coverage level, per user's non-goals).

## Post-implementation finding: `openbmb/MiniCPM5-1B-MLX`'s tokenizer_class is incompatible with this project's pinned transformers

**Found via real end-to-end testing** (not caught by mocked unit tests, which correctly verify our own code's wiring but mock out `mlx_lm.load()` itself): a live download + load attempt against the actual `openbmb/MiniCPM5-1B-MLX` repo failed with:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

**Root cause, confirmed by inspection**: the repo's `tokenizer_config.json` declares `"tokenizer_class": "TokenizersBackend"` — a transformers-5.x-era class name. This project pins `transformers<=4.57.6` project-wide (`backend/requirements-mlx.txt`'s existing comment: mlx-audio's TTS/STT path needs that older API surface; bumping to 5.x would need separate, wider testing this feature doesn't own). `AutoTokenizer.from_pretrained` refuses to load when the declared class isn't importable in the installed transformers version — confirmed this is specific to the MLX conversion, not the model itself: `openbmb/MiniCPM5-1B` (the plain PyTorch repo) declares the ordinary `PreTrainedTokenizerFast` and loads fine; the **community** `mlx-community/MiniCPM5-1B-OptiQ-4bit` quant has the exact same `TokenizersBackend` issue too (checked directly), so this isn't an argument for switching quant sources — both MLX conversions of this model hit it. Qwen3's existing MLX repos (`mlx-community/Qwen3-*-4bit`) declare `Qwen2Tokenizer`, an ordinary compatible class, which is why this problem is new to MiniCPM5-1B specifically and wasn't hit before.

**Fix implemented**: `_ensure_compatible_mlx_snapshot()` in `backend/backends/minicpm_llm_backend.py`. Before calling `mlx_lm.load()`, `MLXMiniCPMLLMBackend._load_model_sync` now calls `huggingface_hub.snapshot_download(repo)` itself (idempotent — no extra download, HF's cache dedupes) to get a local path, then builds a small staging directory: every file symlinked to the original except `tokenizer_config.json`, which gets a patched copy with `tokenizer_class` rewritten to `"PreTrainedTokenizerFast"` (the tokenizer *data* — `tokenizer.json` — is a standard fast-tokenizer file; only the declared class name is stale for this transformers version, confirmed by loading it directly with `PreTrainedTokenizerFast.from_pretrained` successfully). The check only fires when the declared class genuinely isn't resolvable (`hasattr(transformers, declared_class)`), so it's a no-op for any correctly-labeled repo (Qwen3's included) and self-heals if OpenBMB ever re-saves the repo with a compatible class name.

**Verified end-to-end after the fix**: real download (608MB weight file, ~79 minutes on this sandbox's throttled bandwidth — not a code issue), then through the actual `MLXMiniCPMLLMBackend` class: load → `enable_thinking=False` chat template → `generate()` → `"Hello!"` → clean `unload_model()`. This is the strongest evidence in this feature that the approved design (official MLX repo, mirror-Qwen3 loading pattern) is genuinely viable once this one gap is patched — not just "should work in theory."

**Alternatives considered and rejected**:
- Bump `transformers` to >=5.0 project-wide — rejected: affects every other engine (Qwen TTS, Whisper, mlx-audio's own pin comment), needs its own dedicated testing pass, way outside this feature's scope, and isn't guaranteed compatible with the other pinned versions (`chatterbox-tts` also pins `transformers==5.2.0` in a way that already conflicts per this project's own `pip install` warnings — a pre-existing, unrelated tangle).
- Switch to the community quant — rejected, confirmed above: same bug, wouldn't fix anything.
- Give up on the MLX path for MiniCPM5-1B, ship PyTorch-only — rejected: would fail SC-003 ("no platform where it is listed but non-functional") on the very platform (Apple Silicon) this whole feature was originally scoped around, and a working fix was available.

## Constraint check: `mlx-lm` version pin

**Finding**: `backend/requirements-mlx.txt` does not pin `mlx-lm` directly — its only mention is a comment noting it arrives transitively as a dependency of `mlx-audio` (installed `--no-deps`), with no version floor. `backend/requirements.txt` doesn't mention it either.

**Action item for implementation**: add an explicit `mlx-lm>=0.31` line to `backend/requirements-mlx.txt` so the MiniCPM5-1B dual-EOS requirement is guaranteed rather than accidental. This is a one-line addition, not a redesign.
