# Simple Generate Example (Local API)

This example shows how to generate speech using the local Voicebox API.

1. Start the backend (follow local docs):

```bash
# recommended: install deps per README
just setup || echo "run setup per README"
just dev
```

2. Generate speech using curl (replace profile_id):

```bash
curl -X POST http://localhost:17493/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en"}' > sample.wav
```

3. Play the resulting file:

```bash
aplay sample.wav || ffplay -autoexit sample.wav
```

