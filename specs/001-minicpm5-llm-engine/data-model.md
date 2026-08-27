# Phase 1 Data Model: MiniCPM5-1B LLM Engine Support

## Entities

### LLM engine (code-level registry entry, not a DB table)

Existing concept, extended with one new member. `LLM_ENGINES: dict[str, str]` in `backend/backends/__init__.py`.

| Key | Display name | Status |
|---|---|---|
| `qwen_llm` | "Qwen3 LLM" | existing |
| `minicpm_llm` | "MiniCPM5 LLM" | **new** |

### LLM model config (code-level registry entry, not a DB table)

Existing `ModelConfig` dataclass ([backend/backends/__init__.py](../../backend/backends/__init__.py)), no field changes needed — MiniCPM5-1B is a new *instance*, not a new shape.

| Field | Qwen3 0.6B (existing, unchanged) | MiniCPM5-1B (new) |
|---|---|---|
| `model_name` | `"qwen3-0.6b"` | `"minicpm5-1b"` |
| `display_name` | `"Qwen3 0.6B"` | `"MiniCPM5 1B"` |
| `engine` | `"qwen_llm"` | `"minicpm_llm"` |
| `hf_repo_id` | `"Qwen/Qwen3-0.6B"` (pytorch) / `"mlx-community/Qwen3-0.6B-4bit"` (mlx) | `"openbmb/MiniCPM5-1B"` (pytorch) / `"openbmb/MiniCPM5-1B-MLX"` (mlx) |
| `model_size` | `"0.6B"` | `"1B"` |
| `size_mb` | 400 (mlx) / 1400 (pytorch) | ~900 (mlx, official 4-bit) / ~2100 (pytorch bf16) — confirm exact figure against the actual repo file listing during implementation, don't guess further than the research already gathered |
| `languages` | `common_languages` (en/zh/ja/ko/de/fr/ru/pt/es/it) | same `common_languages` — MiniCPM5-1B is bilingual EN/ZH per its model card; for this app's non-Chinese/English UI languages, output quality for those languages is unverified but this feature doesn't scope per-language gating (Qwen3 doesn't do this either) |

Produced by new `_get_minicpm_llm_configs() -> list[ModelConfig]`, mirroring `_get_qwen_llm_configs()`'s backend-type branching. Folded into `get_llm_model_configs()` alongside the existing Qwen3 list.

### Active LLM selection (persisted setting)

**Before**: `CaptureSettingsResponse.llm_model: str` — a bare Qwen3 size (`"0.6B"`/`"1.7B"`/`"4B"`), validated by a static regex, implicitly always engine `qwen_llm`.

**After**: `llm_model: str` — now holds a `ModelConfig.model_name` value (`"qwen3-0.6b"`, `"qwen3-1.7b"`, `"qwen3-4b"`, or `"minicpm5-1b"`), validated at request time by membership in `{cfg.model_name for cfg in get_llm_model_configs()}` rather than a compile-time regex. Field name and column name are unchanged — only the shape of the string value changes, and only internally (the DB column, Pydantic field, and TS field all keep the name `llm_model` / `model_size` per call site, per YAGNI — renaming a field that already means "the identifier of the active LLM" would touch more call sites for no behavioral gain).

**Validation rule change**: `Field(pattern="^(0\\.6B|1\\.7B|4B)$")` → a validator function checking dynamic membership, since the valid set now depends on `get_backend_type()` (platform) and spans two engines. Applies to (four fields — `LLMGenerateRequest.model_size` was found during implementation and missed in the original research pass):
- `backend/models.py: CaptureRefineRequest.model_size`
- `backend/models.py: LLMGenerateRequest.model_size`
- `backend/models.py: CaptureSettingsResponse.llm_model`
- `backend/models.py: CaptureSettingsUpdate.llm_model`

**Default value change**: `"0.6B"` → `"qwen3-0.6b"` in `CaptureSettingsResponse.llm_model` and `backend/database/models.py`'s `CaptureSettings.llm_model` column default.

### Capture model attribution (DB column, historical record)

`backend/database/models.py`: `Capture.llm_model` (nullable `String`) — same shape change as above (bare size → `model_name`), applied retroactively to existing rows by the migration below. No schema change (still a nullable `String` column).

### Migration: legacy value rewrite

One-time, idempotent, run on every startup (no-op after the first run per install):

| Table | Column | Old value | New value |
|---|---|---|---|
| `capture_settings` | `llm_model` | `"0.6B"` | `"qwen3-0.6b"` |
| `capture_settings` | `llm_model` | `"1.7B"` | `"qwen3-1.7b"` |
| `capture_settings` | `llm_model` | `"4B"` | `"qwen3-4b"` |
| `captures` | `llm_model` | `"0.6B"` | `"qwen3-0.6b"` |
| `captures` | `llm_model` | `"1.7B"` | `"qwen3-1.7b"` |
| `captures` | `llm_model` | `"4B"` | `"qwen3-4b"` |

`captures.llm_model` is nullable — `NULL` rows are left untouched (no LLM was ever recorded for them; unrelated to this migration).

## State / relationships

**Correction from an initial assumption — verified against the actual code, not just inferred**: switching between Qwen3 *sizes* today is memory-safe only because it happens **inside one Python object** — `PyTorchQwenLLMBackend`/`MLXQwenLLMBackend` is a singleton per engine, and its own `load_model()` checks `self._current_model_size != model_size` and calls `self.unload_model()` on itself before loading the new size. Nothing today unloads across *different* `_llm_backends[engine]` singletons — there is no settings-change hook that calls unload (checked `backend/routes/settings.py`'s `CaptureSettingsUpdate` handler and `app/src/lib/hooks/useSettings.ts`'s `update()`: neither triggers an unload).

This means introducing a second engine singleton (`minicpm_llm`) alongside the first (`qwen_llm`) — without an additional guard — would let a user switch their selection from a Qwen3 size to `minicpm5-1b`, have the app load MiniCPM5-1B into memory on next use, **while the previously-loaded Qwen3 backend stays resident** (double RAM/VRAM held simultaneously) until an explicit manual unload or app restart. That would violate FR-005 ("selecting one unloads the other") silently.

**New guard, added in this feature**: `backend/services/llm.py`'s `get_llm_model(engine)` unloads every *other* registered LLM engine's backend (if loaded) before returning the requested one:

```python
def get_llm_model(engine: str = "qwen_llm") -> LLMBackend:
    from ..backends import LLM_ENGINES, get_llm_backend_for_engine
    for other_engine in LLM_ENGINES:
        if other_engine != engine:
            other = get_llm_backend_for_engine(other_engine)
            if other.is_loaded():
                other.unload_model()
    return get_llm_backend_for_engine(engine)
```

This is the single chokepoint every consumer already passes through (`personality.py`, `refinement.py`, `routes/llm.py` all call `get_llm_model()` before generating), so no call site beyond adding the `engine` argument needs to change to get cross-engine exclusivity. This is new logic, not a mirror of existing code — it did not need to exist with only one engine, and it is required to satisfy FR-005 once a second engine exists.
