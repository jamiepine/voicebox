# RunPod Serverless Deployment

Voicebox can run as a [RunPod Serverless](https://docs.runpod.io/serverless/quickstart) worker. Workers spin up on demand, process requests, and shut down automatically — you only pay while they're running.

## How it works

The serverless image starts a RunPod handler (`serverless_handler.py`) which:

1. Launches the existing FastAPI server in a background thread
2. Waits for `/health` to respond (up to 5 min on cold start for model downloads)
3. Proxies each RunPod job as an HTTP request to the local server
4. Returns the response — JSON as-is, audio files as base64

Model idle-unloading is disabled in serverless mode (`SERVERLESS=1`). The model stays loaded for the worker's lifetime. RunPod shuts down the entire worker after the configured idle timeout.

## Build the image

```bash
# From the voicebox/ directory
./scripts/serverless-build.sh

# Build + push to Docker Hub
./scripts/serverless-build.sh --push --tag youruser/voicebox-serverless:latest

# Build + push to GHCR
./scripts/serverless-build.sh --push --tag ghcr.io/youruser/voicebox-serverless:latest
```

Or manually:

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA=1 \
  --build-arg SERVERLESS=1 \
  -t voicebox-serverless \
  .
```

## RunPod endpoint settings

When creating your endpoint on [runpod.io](https://runpod.io):

| Setting           | Recommended                      |
| ----------------- | -------------------------------- |
| Container image   | your pushed image tag            |
| GPU               | RTX 4090 or similar (16GB+ VRAM) |
| Idle timeout      | **60 seconds**                   |
| Execution timeout | 600 seconds                      |
| Active workers    | 0 (pure on-demand)               |
| Max workers       | 1 (increase for production)      |
| FlashBoot         | enabled                          |

**On idle timeout:** The GPU stays allocated (and billed) for the full idle window regardless of VRAM usage. Keeping the model hot and using a short idle timeout (60s) is more cost-effective than unloading the model and using a long idle timeout.

## Authentication

Add your RunPod API key to the root `.env`:

```
RUNPOD_API_KEY=your_runpod_api_key_here
```

All requests to the RunPod endpoint require this key as a bearer token:

```
Authorization: Bearer $RUNPOD_API_KEY
```

## Sending requests

RunPod wraps requests in a job envelope. The handler accepts:

| Field     | Type   | Description                            |
| --------- | ------ | -------------------------------------- |
| `method`  | string | HTTP method (default: `"POST"`)        |
| `path`    | string | Required. API path, e.g. `"/generate"` |
| `body`    | object | JSON body for POST/PUT requests        |
| `params`  | object | Query string parameters                |
| `headers` | object | Additional HTTP headers                |

### Health check

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"method": "GET", "path": "/health"}}'
```

### Generate speech

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "method": "POST",
      "path": "/generate",
      "body": {
        "profile_id": "your-profile-id",
        "text": "Hello from RunPod."
      }
    }
  }'
```

### Download audio

Audio endpoints return base64-encoded content with `"is_binary": true`:

```json
{
  "output": {
    "status_code": 200,
    "is_binary": true,
    "body_base64": "UklGRi..."
  }
}
```

Decode it:

```bash
echo "$BODY_BASE64" | base64 -d > output.wav
```

### Async jobs (long generations)

For long texts, use `/run` instead of `/runsync` to avoid the 90s sync timeout:

```bash
# Submit
JOB=$(curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"path": "/generate", "body": {"profile_id": "...", "text": "..."}}}')

JOB_ID=$(echo $JOB | jq -r '.id')

# Poll
curl https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Local testing

The RunPod SDK includes a local test server:

```bash
cd voicebox/
SERVERLESS=1 python3 -m backend.serverless_handler --rp_serve_api
```

This starts a local HTTP server that simulates the RunPod job protocol at `http://localhost:8000`.

## Limitations

- **SSE streaming not supported** — RunPod jobs return a single response. Generation still works, just without real-time progress events. Use `/generate` (non-streaming) via the job body.
- **Ephemeral storage** — The SQLite database (profiles, history) is lost when the worker shuts down. Voice profiles need to be re-imported each cold start, or use a RunPod network volume for persistence.
- **Cold start time** — First start downloads model weights (~3–5 GB from HuggingFace). Subsequent starts with FlashBoot are much faster.
