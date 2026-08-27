# Contract: LLM model selection surface

Scope: the existing FastAPI endpoints and response shapes that expose "which LLM models exist" and "which one is active" — extended to include MiniCPM5-1B, no new endpoints.

## `GET /models/status` (existing endpoint, unchanged shape)

Backed by `get_all_model_configs()` → now includes one additional row from `_get_minicpm_llm_configs()`:

```json
{
  "model_name": "minicpm5-1b",
  "display_name": "MiniCPM5 1B",
  "engine": "minicpm_llm",
  "cached": false,
  "loaded": false,
  ...
}
```
No field added or removed — this is an additive row, same shape as every existing `qwen3-*` row.

## `POST /llm/generate` (existing endpoint — `backend/routes/llm.py`)

**Before**: `model_size` request field validated against `{cfg.model_size for cfg in get_llm_model_configs()}` (a flat set of size labels, implicitly all `qwen_llm`).

**After**: the field is renamed in meaning (not necessarily in wire name — confirm during implementation whether the existing field name `model_size` is kept for backward wire-compatibility with any external caller, or whether it's safe to rename since this is an internal desktop-app API with no external consumers) to hold a `model_name` value, validated against `{cfg.model_name for cfg in get_llm_model_configs()}`. Resolves to `(engine, model_size)` via `get_model_config(model_name)` before dispatch.

Error response for an unknown value is unchanged in shape (`400` with `detail` listing valid values) — just a larger valid set.

## `GET /capture-settings`, `PATCH /capture-settings` (existing endpoints — `backend/routes/settings.py`)

`CaptureSettingsResponse.llm_model` / `CaptureSettingsUpdate.llm_model`: **wire type unchanged (`string`)**, only the set of accepted/returned values changes (now includes `"minicpm5-1b"` alongside the three `qwen3-*` values). No breaking change to the JSON shape — a frontend that doesn't know about the new value would simply not render it as an option, same as any other additive enum-like string field.

## `POST /captures/{id}/refine` (existing endpoint — `backend/routes/captures.py`)

`CaptureRefineRequest.model_size` (field name kept as-is as it's a request field, not a stored value — same rename-vs-keep question as `/llm/generate` applies): accepts a `model_name` value now; the internal `c.model_size == saved.llm_model` lookup becomes `c.model_name == saved.llm_model` (see [data-model.md](../data-model.md)).

## `POST /models/{model_name}/unload`, download endpoints (existing, `backend/routes/models.py`)

No shape changes — `model_name` path/lookup parameter already accepts any value in `get_all_model_configs()`, and `"minicpm5-1b"` is just a new valid value flowing through the same existing generic code path (`get_model_config`, `unload_model_by_config`, `get_model_load_func`).

## Naming decision: keep existing field names on the wire

Several existing fields are literally named `model_size` (`CaptureRefineRequest.model_size`, the `/llm/generate` request field) but will hold a `model_name`-shaped value after this change. **Decision: keep the existing field names** (`model_size` stays `model_size` on the wire) — this is an internal desktop-app API with no external consumers to break, and renaming would touch every frontend/backend call site for a field that already means "the identifier of the model to use," just with a broadened value shape. Add a one-line code comment at each Pydantic field noting that the value is a `model_name` (e.g. `"minicpm5-1b"`), not a bare size, so the mismatch between name and content doesn't confuse a future reader.
