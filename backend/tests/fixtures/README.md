# E2E Test Fixtures

Place two files here before running `test_all_models_e2e.py`:

- `reference_voice.wav` — a clean speech sample, mono, 16–24 kHz, ~5–15 seconds.
- `reference_voice.txt` — the **exact** transcription of the WAV (single line, no trailing newline required).

These are used to create a cloned voice profile for every cloning-capable engine (qwen, luxtts, chatterbox, chatterbox_turbo, tada). Keep them out of version control if they contain personal audio — this directory is not gitignored by default, so add them to `.gitignore` locally if needed.

You can point the test at different files with:

```
python backend/tests/test_all_models_e2e.py \
  --reference-wav /path/to/your.wav \
  --reference-text "exact transcription here"
```
