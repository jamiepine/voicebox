# Phase 1 Data Model: ModelScope Download Source

## Model Download Source (new)

Persisted as a JSON file (`get_data_dir()/model_source.json`), not a DB
table — see [research.md](research.md) §4 for why.

| Field | Type | Values | Notes |
|---|---|---|---|
| `source` | string enum | `"huggingface"` \| `"modelscope"` | Defaults to `"huggingface"` when the file is absent. |

Read fresh on every `resolve_model_source()` call — no caching, no
apply-at-startup step. A change is in effect for the very next download.

Exposed to the frontend via:
- `GET /settings/model-source` → `{ "source": "..." }`
- `PUT /settings/model-source` with body `{ "source": "..." }` → same shape
  as GET. Validates `source` is one of the two allowed values (422 on
  anything else).

## ModelConfig (existing, extended)

`backend/backends/__init__.py`, dataclass `ModelConfig`. One new optional
field:

| Field | Type | Notes |
|---|---|---|
| `ms_repo_id` | `Optional[str]` (default `None`) | ModelScope repository id for this model, when one exists. `None` means "no ModelScope mirror — this model always downloads from huggingface.co, even when the active source is ModelScope." |

No changes to existing fields. 13 of the ~17 `ModelConfig` entries (built by
`_get_qwen_model_configs`, `_get_qwen_custom_voice_configs`,
`_get_non_qwen_tts_configs`, `_get_whisper_configs`,
`_get_qwen_llm_configs`) get a value per the table in research.md §1; the 4
Chatterbox/TADA entries in `_get_non_qwen_tts_configs` leave it unset.

## Resolved model location (new, in-memory concept — not persisted)

Not a stored entity; the return value of the new resolver function used by
every backend's `_get_model_path()`. Conceptually:

```
resolve_model_source(hf_repo_id, ms_repo_id, model_name) -> str
```

Returns either:
- the original `hf_repo_id` string (HuggingFace active, or ModelScope
  active but `ms_repo_id` is `None`) — callers pass this straight
  into `from_pretrained()` / `is_model_cached()` exactly as today, or
- an absolute local directory path under
  `get_models_dir()/modelscope/<ms_repo_id with "/" replaced by "--">/`
  (ModelScope active and `ms_repo_id` is set) — already downloaded by the
  time this returns; callers pass this into `from_pretrained()` the same
  way they'd pass a repo id (both are valid inputs to `from_pretrained`).

## Status/deletion visibility (behavioral change, no new schema)

`GET /models/status` and `DELETE /models/{model_name}` gain a second lookup
path: alongside the existing HuggingFace-cache check
(`is_model_cached(hf_repo_id)` / deleting `HF_HUB_CACHE/models--org--repo`),
they also check/delete
`get_models_dir()/modelscope/<safe-ms-repo-id>/` when that directory has
weight files. A model is "downloaded" if *either* location has it (a model
should never legitimately be in both, since a given process only downloads
through whichever source is currently configured, but checking both keeps
status accurate if the user switched sources between runs).
