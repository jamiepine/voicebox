# E2E Test Fixtures

## TTS reference voice

Place two files here before running `test_all_models_e2e.py`:

- `reference_voice.wav` — a clean speech sample, mono, 16–24 kHz, ~5–15 seconds.
- `reference_voice.txt` — the **exact** transcription of the WAV (single line, no trailing newline required).

These are used to create a cloned voice profile for every cloning-capable engine (qwen, luxtts, chatterbox, chatterbox_turbo, tada). Keep personal audio out of version control; the directory is not gitignored by default, so add local files to `.git/info/exclude`.

You can point the test at different files with:

```bash
python backend/tests/test_all_models_e2e.py \
  --reference-wav /path/to/your.wav \
  --reference-text "exact transcription here"
```

## Whisper long-audio fixtures

`generate_fixtures.sh` creates optional, untracked `generated/` media from a
local Mozilla Common Voice archive. Common Voice is used only as a public
source; do not commit downloaded audio or derived media.

```bash
bash backend/tests/fixtures/generate_fixtures.sh /path/to/common-voice/en
```

The generator requires `ffmpeg` and a TSV containing `path` and `sentence`
columns (defaults to `validated.tsv`). It writes `cv_10s.wav`, `cv_30s.wav`,
`cv_6m43s.{wav,flac,mp3}`, `references.tsv`, and `SHA256SUMS` under
`generated/` for local GPU evaluation.

The long-form GPU tests are intentionally not enabled until the generated
fixtures exist. This keeps CI deterministic and prevents private/user voice
material from entering the repository.

## Licensing

Verify the current Common Voice dataset terms before each download. Record
the source release and checksums in local evaluation results; no dataset
archive or generated media belongs in this repository.
