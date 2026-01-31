# Backend Tests

Manual test scripts for debugging and validating backend functionality.

## Test Files

### `test_generation_progress.py`
Tests TTS generation with SSE progress monitoring to identify UX issues where users see download progress even when the model is already cached.

**Usage:**
```bash
cd backend
python tests/test_generation_progress.py
```

**Prerequisites:**
- Server must be running (`python main.py`)
- At least one voice profile must exist

### `test_real_download.py`
Tests real model download with SSE progress monitoring.

**Usage:**
```bash
cd backend
# Delete cache first to force fresh download
rm -rf ~/.cache/huggingface/hub/models--openai--whisper-base
python tests/test_real_download.py
```

**Prerequisites:**
- Server must be running (`python main.py`)

### `test_progress.py`
Unit tests for ProgressManager and HFProgressTracker functionality.

**Usage:**
```bash
cd backend
python tests/test_progress.py
```

### `test_check_progress_state.py`
Debugging script to inspect the internal state of ProgressManager and TaskManager.

**Usage:**
```bash
cd backend
python tests/test_check_progress_state.py
```

## Notes

These are manual test scripts, not automated unit tests. They're designed for:
- Debugging progress tracking issues
- Validating SSE event streams
- Monitoring real-time download behavior
- Inspecting internal state during development
