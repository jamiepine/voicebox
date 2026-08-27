# Contract: Model download source setting

## `GET /settings/model-source`

Returns the persisted setting. Applies live — there's no separate
"active" value to reconcile against (see research.md §4 for why the
original design's `active_source` field was removed).

**Response 200**
```json
{ "source": "huggingface | modelscope" }
```

## `PUT /settings/model-source`

**Request**
```json
{ "source": "huggingface | modelscope" }
```

**Response 200**: same shape as GET, reflecting the newly persisted
`source`. Takes effect for the next download triggered after this call.

**Response 422**: `source` is not one of the two allowed values.

## Existing endpoints — behavior change only, no schema change

### `GET /models/status`

`models.ModelStatus` response schema is unchanged. Behavior change: a model
downloaded into the ModelScope-managed local directory (see
[data-model.md](../data-model.md)) is now also detected as `downloaded` with
a populated `size_mb`, not just models found in the HuggingFace cache.

### `DELETE /models/{model_name}`

Behavior change: deletes from whichever location the model was actually
downloaded into (HuggingFace cache, or the ModelScope-managed local
directory) instead of assuming the HuggingFace cache path.

### `POST /models/download`

No schema or behavior change to the request/response shape. Internally, the
background download task consults the active download source and routes
accordingly (see [research.md](../research.md) §2); this is invisible to
the caller except that progress events for a ModelScope-routed download may
be coarser-grained (see FR-012 in [spec.md](../spec.md)).
