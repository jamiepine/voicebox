# End-to-End Model Generation Test — Design

## Goal

A single script, runnable on macOS and Windows, that exercises every TTS model against the **frozen PyInstaller binary** (not the dev server), captures per-model pass/fail and error messages, and exits non-zero if any model fails. Generation is strictly sequential — one model loaded at a time.

## Test matrix (10 runs)

Derived from `backend/backends/__init__.py:185-316`. Each row maps to one `POST /generate` call.

| # | engine                | model_size | profile kind | notes |
|---|-----------------------|------------|--------------|-------|
| 1 | `qwen`                | `1.7B`     | cloned       | reference audio required |
| 2 | `qwen`                | `0.6B`     | cloned       | |
| 3 | `qwen_custom_voice`   | `1.7B`     | preset       | `preset_voice_id="Ryan"` |
| 4 | `qwen_custom_voice`   | `0.6B`     | preset       | `preset_voice_id="Ryan"` |
| 5 | `luxtts`              | —          | cloned       | English only |
| 6 | `chatterbox`          | —          | cloned       | |
| 7 | `chatterbox_turbo`    | —          | cloned       | English only |
| 8 | `tada`                | `1B`       | cloned       | tada-1b, English only |
| 9 | `tada`                | `3B`       | cloned       | tada-3b-ml, multilingual |
| 10| `kokoro`              | —          | preset       | `preset_voice_id="af_heart"` |

Cloned engines (1, 2, 5, 6, 7, 8, 9) share **one** profile created once with the reference WAV. Preset profiles are created separately, one for kokoro and one for qwen_custom_voice.

Language for every run: `en` (covers every engine's supported set).

## End-to-end flow

```
1. Resolve paths          → find binary, build if missing
2. Launch binary          → spawn with --port --data-dir --parent-pid
3. Wait for /health       → poll until status=="healthy" or 120s timeout
4. Create profiles        → 1 cloned + 2 preset, via /profiles (+ /samples)
5. For each (engine, model_size) in matrix:
     a. Check cache       → GET /models/status → cached? short timeout : long
     b. POST /generate    → get generation_id
     c. Stream /status    → consume SSE until completed/failed/timeout
     d. Record result     → {engine, model_size, status, duration, error, elapsed}
6. Write results          → JSON + Markdown table to ./results/
7. Shutdown binary        → SIGTERM, fall back to kill, verify port freed
8. Exit code              → 0 if all passed, 1 otherwise
```

## Binary resolution

Search order — **first hit wins**:

| Platform | Path | Build type |
|----------|------|------------|
| macOS    | `backend/dist/voicebox-server-cuda/voicebox-server-cuda` | onedir (CUDA, rarely on Mac) |
| macOS    | `backend/dist/voicebox-server`                          | onefile (CPU) |
| Windows  | `backend\dist\voicebox-server-cuda\voicebox-server-cuda.exe` | onedir (CUDA) |
| Windows  | `backend\dist\voicebox-server.exe`                      | onefile (CPU) |

If none exist, run `python backend/build_binary.py` and wait for it to finish (can take 5-20 min). Fail with a clear error if the build itself fails. `--skip-build` flag forces "error out if no binary" instead of building.

## Spawn command

Mirrors Tauri's launch in `tauri/src-tauri/src/main.rs:369-388`:

```
<binary> --host 127.0.0.1 --port <free-port> --data-dir <tempdir> --parent-pid <test-pid>
```

- **Port**: bind to `0` first in Python to grab a free port, then pass that number.
- **Data dir**: `tempfile.mkdtemp(prefix="voicebox-e2e-")`. Deleted after the run unless `--keep-data-dir`. Profiles and generated WAVs land here.
- **Parent PID**: current Python PID — ensures the backend dies if the test crashes (watchdog in `server.py:102-224`).
- **stdout/stderr**: tee to both a log file in `./results/server-<timestamp>.log` and a rolling in-memory buffer. On model failure, last 100 lines of the buffer are attached to that model's error record.

## Profile setup

One cloned profile shared across all cloning engines:

```http
POST /profiles
{
  "name": "e2e-cloned",
  "voice_type": "cloned",
  "language": "en"
}
```

Then:

```http
POST /profiles/{id}/samples   (multipart)
  file: <reference WAV>
  reference_text: <exact transcription>
```

Two preset profiles:

```http
POST /profiles
{ "name": "e2e-kokoro",  "voice_type": "preset", "language": "en",
  "preset_engine": "kokoro",            "preset_voice_id": "af_heart" }

POST /profiles
{ "name": "e2e-qwen-cv", "voice_type": "preset", "language": "en",
  "preset_engine": "qwen_custom_voice", "preset_voice_id": "Ryan" }
```

## Generation request (per matrix row)

```http
POST /generate
{
  "profile_id": "<appropriate profile>",
  "text": "The quick brown fox jumps over the lazy dog.",
  "language": "en",
  "engine": "<engine>",
  "model_size": "<size or omitted>",
  "seed": 42,
  "normalize": true
}
```

Response `id` feeds into the SSE status loop (`GET /generate/{id}/status`, `routes/generations.py:190-227`). Loop reads lines until a payload with `status in ("completed", "failed")` arrives, then breaks.

## Timeout strategy (split)

Check `GET /models/status` for the target model **before** generation:

| Cached? | Per-model timeout | Rationale |
|---------|-------------------|-----------|
| Yes     | **3 minutes**     | Inference only; generous for CPU builds |
| No      | **20 minutes**    | First-run HF download up to 8 GB (tada-3b-ml) |

On timeout: cancel the SSE stream, mark the row `timeout`, and continue to the next row. Don't abort the whole run on one timeout.

## Result format

`./results/e2e-<platform>-<arch>-<timestamp>.json`:

```json
{
  "platform": "darwin-arm64",
  "binary": "/abs/path/voicebox-server",
  "binary_size_mb": 612,
  "started_at": "2026-04-16T12:34:56Z",
  "finished_at": "...",
  "results": [
    {
      "engine": "qwen",
      "model_size": "1.7B",
      "status": "passed|failed|timeout",
      "generation_id": "...",
      "was_cached": true,
      "elapsed_seconds": 12.4,
      "audio_duration": 3.1,
      "audio_path": "/tmp/.../gen.wav",
      "error": null,
      "server_log_tail": null
    }
  ]
}
```

Companion `./results/e2e-<...>.md`:

```
# Voicebox E2E — darwin-arm64 — 2026-04-16 12:34

| Engine              | Size | Status | Elapsed | Error |
|---------------------|------|--------|---------|-------|
| qwen                | 1.7B | PASS   |  12.4s  |       |
| qwen                | 0.6B | FAIL   |   4.1s  | CUDA OOM: ... |
...
```

## CLI flags

```
python -m backend.tests.test_all_models_e2e [flags]

--binary PATH           Use this binary instead of auto-detecting
--skip-build            Error if no binary found (no auto-build)
--reference-wav PATH    Reference audio (default: backend/tests/fixtures/reference_voice.wav)
--reference-text STR    Transcription (default: read from fixtures/reference_voice.txt)
--only ENGINE[,...]     Run only these engines (e.g. kokoro,qwen)
--skip ENGINE[,...]     Skip these engines
--keep-data-dir         Don't delete tempdir after run
--timeout-cached SEC    Override 180
--timeout-download SEC  Override 1200
--port N                Override auto-picked port
--output-dir PATH       Default: backend/tests/results/
```

## File layout

```
backend/tests/
├── E2E_MODEL_TEST_DESIGN.md       (this file)
├── test_all_models_e2e.py         (main script, ~400-500 LoC)
├── fixtures/
│   ├── reference_voice.wav        (user-provided, ~5-15s clean speech)
│   └── reference_voice.txt        (exact transcription)
└── results/                       (gitignored)
    ├── e2e-darwin-arm64-<ts>.json
    ├── e2e-darwin-arm64-<ts>.md
    └── server-<ts>.log
```

The script uses only stdlib + `httpx` (or `requests`) + `sseclient-py` — all already in `backend/requirements.txt`. No pytest to keep it invocable as a single command on fresh checkouts.

## Safety & cleanup

- Always kill the spawned binary in a `try/finally`. On Windows, `taskkill /F /T` the whole tree (Tauri does the same).
- Verify the port is free on shutdown (Tauri port-reuse check in `main.rs:114-186` could otherwise pick up a ghost).
- Don't touch the user's HF cache by default — let the server use `HF_HUB_CACHE` / `VOICEBOX_MODELS_DIR`. Passing `--isolated-cache` would point both env vars at the tempdir for a true cold-start run (opt-in only; would re-download every time).

## Non-goals

- Not validating audio quality (no WER, no waveform comparison). Pass = "endpoint returned `completed` and produced a non-empty WAV".
- Not testing STT (Whisper), effects chains, channels, or streaming endpoints.
- Not running on CI today — human-invoked on dev machines. CI integration is a follow-up once the script is stable.
- No model unload between runs — models stay loaded; server manages its own eviction.
- No version-drift check on the binary.
- No `instruct` parameter exercised on qwen_custom_voice runs.
