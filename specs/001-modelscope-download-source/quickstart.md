# Quickstart: Validating the ModelScope Download Source

## Prerequisites

- Backend venv set up (`just setup-python`), `modelscope` installed
  (added to `backend/requirements.txt` by this feature).
- A clean `data/` dir (or at least a model not yet downloaded) to observe a
  real download rather than a cache hit.

## Scenario 1 — default behavior unchanged (SC-004)

```bash
curl -s http://localhost:PORT/settings/model-source
# → {"source": "huggingface"}
```

Trigger a download of any model as usual (`POST /models/download`) and
confirm it behaves exactly as before this feature.

**Status: verified for real** against a running backend, 2026-08-27.

## Scenario 2 — switch to ModelScope, download a mirrored model (US1)

```bash
curl -s -X PUT http://localhost:PORT/settings/model-source \
  -H "Content-Type: application/json" -d '{"source": "modelscope"}'
# → {"source": "modelscope"}   # applies immediately, no restart needed

curl -s -X POST http://localhost:PORT/models/download \
  -H "Content-Type: application/json" -d '{"model_name": "kokoro"}'
```

Watch `GET /models/progress/kokoro` (SSE) until `status: complete`. Confirm:
- No request to `huggingface.co` appears in backend logs/network capture.
- `GET /models/status` reports `kokoro` as `downloaded: true` with a
  populated `size_mb`.
- Kokoro generation (`POST /generate` with `engine: kokoro`) works.

**Status: verified for real**, 2026-08-27 — real ~312 MB download from
`AI-ModelScope/Kokoro-82M`, no `huggingface.co` traffic, and the downloaded
weights were loaded and used to generate real audio (not just checked for a
"loaded" flag).

## Scenario 3 — unmirrored model still downloads when ModelScope is selected (US2)

With `source` still `modelscope`:

```bash
curl -s -X POST http://localhost:PORT/models/download \
  -H "Content-Type: application/json" -d '{"model_name": "chatterbox-tts"}'
```

Confirm the download completes successfully (directly from huggingface.co,
same as it would under the `huggingface` source) with no error surfaced —
same SSE/status flow as Scenario 2.

**Status: not re-run against this specific model** (needs the
`chatterbox-tts` package installed, which wasn't in the verification
environment). The underlying mechanism this depends on — that
`resolve_model_source()` returns `hf_repo_id` unchanged when `ms_repo_id`
is `None`, regardless of source — is covered by automated tests
(`test_model_source_resolution.py`), and is now just the plain HuggingFace
path (already exercised for real by Scenario 1), not a mirror redirect.

## Scenario 4 — status/delete recognize ModelScope downloads (US3)

After Scenario 2:

```bash
curl -s http://localhost:PORT/models/status | jq '.models[] | select(.model_name=="kokoro")'
curl -s -X DELETE http://localhost:PORT/models/kokoro
curl -s http://localhost:PORT/models/status | jq '.models[] | select(.model_name=="kokoro")'
# → downloaded: false, and the modelscope-managed directory for kokoro is gone from disk
```

**Status: covered by automated integration tests**
(`test_model_status_modelscope.py`) against a real ModelScope-downloaded
directory structure; not re-run manually against the live Scenario 2
download in this pass.
