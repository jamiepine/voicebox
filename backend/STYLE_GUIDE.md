# Python Style Guide

Target: **Python 3.12+** | Formatter/Linter: **Ruff** | Config: `backend/pyproject.toml`

This guide codifies the conventions used across the backend, and prescribes the target style for code written during the refactor (Phases 3-6). Existing code should be migrated incrementally -- don't reformat entire files in unrelated PRs.

---

## Formatting

Enforced by `ruff format` (Black-compatible).

- **Line length**: 120 characters.
- **Indent**: 4 spaces. No tabs.
- **Trailing commas**: Required on multi-line function signatures, arguments, collections.
- **Quotes**: Double quotes (`"`) for strings. Single quotes are acceptable in f-string expressions and dict keys inside f-strings where avoiding escapes improves readability.

Run: `ruff format backend/`

---

## Imports

Enforced by ruff's `isort` rules (rule set `I`).

**Grouping** -- three blocks separated by a blank line:

```python
import asyncio                          # 1. stdlib
from pathlib import Path

import numpy as np                      # 2. third-party
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from backend.config import get_data_dir  # 3. local (absolute)
from .database import get_db            #    or relative
```

**Rules:**
- Within the `backend` package, use **relative imports** for sibling/child modules: `from .database import get_db`, `from ..utils.audio import load_audio`.
- Absolute imports are fine for top-level references from entry points (`main.py`, `server.py`).
- Never use wildcard imports (`from module import *`).
- One import per line for `from X import Y` when there are 4+ names; below that, comma-separated is fine.
- **Lazy imports** are acceptable for heavy dependencies (torch, transformers, mlx) inside functions to reduce startup time. Add a comment: `# lazy: heavy import`.

---

## Type Annotations

Python 3.12 means we use **built-in generics and union syntax natively**. No `from __future__ import annotations`, no `typing.List`/`typing.Dict`.

```python
# Yes
def process(items: list[str], config: dict[str, int] | None = None) -> tuple[int, str]: ...

# No
from typing import List, Dict, Optional, Tuple
def process(items: List[str], config: Optional[Dict[str, int]] = None) -> Tuple[int, str]: ...
```

**What to annotate:**
- All public function signatures (parameters + return type).
- Private functions: parameters at minimum; return type encouraged.
- Module-level variables: only when the type isn't obvious from the assignment.
- Route handlers: parameters are annotated via FastAPI's dependency injection. Add explicit `-> SomeResponse` return types when the route doesn't use `response_model`.

**Imports from `typing` that are still needed** (no built-in equivalent):
`Literal`, `TypeAlias`, `Protocol`, `runtime_checkable`, `Callable`, `Any`, `ClassVar`, `TypeVar`, `overload`, `TYPE_CHECKING`.

Use `collections.abc` for abstract types: `Sequence`, `Mapping`, `Iterable`, `Iterator`, `Generator`.

---

## Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Module | `snake_case` | `task_queue.py` |
| Class | `PascalCase` | `ProgressManager` |
| Function / method | `snake_case` | `create_profile` |
| Variable | `snake_case` | `sample_rate` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_SAMPLE_RATE` |
| Private | `_leading_underscore` | `_generation_queue` |
| Type alias | `PascalCase` | `EffectChain = list[dict[str, Any]]` |

**Specific conventions:**
- Database ORM models imported with `DB` prefix alias: `from .database import VoiceProfile as DBVoiceProfile`.
- Pydantic models use descriptive suffixes: `VoiceProfileCreate`, `VoiceProfileResponse`, `GenerationRequest`.
- Backend classes use engine-name prefix: `MLXTTSBackend`, `PyTorchSTTBackend`.

---

## Docstrings

**Google style**. Required on all public functions, classes, and modules.

```python
def combine_voice_prompts(
    profile_dir: Path,
    *,
    target_sr: int = 24000,
) -> tuple[np.ndarray, int]:
    """Load and concatenate all voice prompt files for a profile.

    Reads .wav/.mp3/.flac files from the profile directory, resamples to
    the target sample rate, normalizes, and concatenates into a single array.

    Args:
        profile_dir: Path to the voice profile directory containing audio files.
        target_sr: Target sample rate for the output. Defaults to 24000.

    Returns:
        Tuple of (concatenated audio array, sample rate).

    Raises:
        FileNotFoundError: If profile_dir does not exist.
        ValueError: If no valid audio files are found.
    """
```

**Short form** is fine for simple functions:

```python
def get_db_path() -> Path:
    """Get the path to the SQLite database file."""
```

**When to skip**: Private helpers under ~5 lines where the name and signature make intent obvious.

**Module docstrings**: A single sentence at the top of every file describing its purpose.

```python
"""Voice profile CRUD operations."""
```

---

## Comments

Comments explain **why**, not **what**. If the code needs a comment to explain what it does, the code should be rewritten to be clearer. The exceptions are non-obvious performance choices, external constraints, and concurrency/race-condition reasoning -- those always deserve a comment.

### No section dividers

Do not use ASCII dividers to create visual sections in files:

```python
# No -- any of these:
# ============================================
# GENERATION ENDPOINTS
# ============================================

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

# --- Load model --------------------------------------------------
```

If a file needs section dividers to be navigable, the file is too long. Split it into modules. Within a function, if you need labeled sections to follow the logic, extract those sections into named functions.

### Inline comments

Inline comments (end-of-line) are fine when they add information the code can't express:

```python
# Yes -- explains a non-obvious constraint or gives context:
audio, sr = load_audio(path, sr=24000)  # Qwen expects 24kHz mono
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
"tauri://localhost",         # Tauri webview (macOS)

# No -- restates the code:
# Check if profile name already exists
existing = db.query(DBVoiceProfile).filter_by(name=data.name).first()

# Delete from database
db.delete(sample)

# Update fields
profile.name = data.name
```

Delete comments that narrate what the next line of code obviously does. If the function name, variable name, or method call already communicates intent, the comment is noise.

### Block comments

Use block comments for **why** explanations -- constraints, workarounds, non-obvious decisions:

```python
# PyInstaller + multiprocessing: child processes re-execute the frozen binary
# with internal arguments. freeze_support() handles this and exits early.
multiprocessing.freeze_support()

# Mark any stale "generating" records as failed -- these are leftovers
# from a previous process that was killed mid-generation.
db.query(Generation).filter_by(status="generating").update({"status": "failed"})
```

Keep block comments tight. Two to three lines is normal. If you need a paragraph, it probably belongs in the docstring or a design doc.

### Linter/type-checker suppression

Always add a reason after `noqa` and `type: ignore`:

```python
import intel_extension_for_pytorch  # noqa: F401 -- side-effect import enables XPU
_queue: asyncio.Queue = None  # type: ignore[assignment]  # initialized at startup
```

Bare `# noqa` or `# type: ignore` with no explanation are not allowed.

### TODO / FIXME

Use sparingly. Every `TODO` must include a brief description of what needs doing. Don't use them as a substitute for tracking work properly:

```python
# TODO: replace with async SQLAlchemy once CRUD modules are migrated (Phase 5)
result = await asyncio.to_thread(profiles.get_profile, profile_id, db)
```

Never commit `HACK`, `XXX`, or `FIXME` -- fix the problem or file an issue.

### Commented-out code

Delete it. That's what git is for. If you need to document that something was intentionally removed, a short tombstone comment is acceptable:

```python
# Removed config.json-only check -- too lenient, doesn't confirm weights exist.
```

---

## Error Handling

The refactor is standardizing on a **two-layer pattern**:

### 1. Domain layer -- raise plain exceptions

CRUD modules and services raise `ValueError`, `FileNotFoundError`, or (post-refactor) custom exceptions defined in `backend/errors.py`:

```python
# backend/errors.py  (to be created in Phase 4)
class NotFoundError(Exception):
    """Raised when a requested resource does not exist."""

class ConflictError(Exception):
    """Raised on uniqueness constraint violations."""
```

```python
# In a service or CRUD module:
raise NotFoundError(f"Profile {profile_id} not found")
```

### 2. Route layer -- translate to HTTPException

Route handlers catch domain exceptions and convert:

```python
@router.post("/profiles")
async def create_profile(data: VoiceProfileCreate, db: Session = Depends(get_db)):
    try:
        return await profiles.create_profile(data, db)
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
```

**Background tasks** catch `Exception` broadly, log with `logger.exception()`, and update the task status to `"failed"`.

**Never**: silently swallow exceptions, use bare `except:`, or catch `BaseException`.

---

## Async

### Rules for the refactor

1. **Don't declare `async def` unless the function awaits something.** Several service modules still declare `async def` without awaiting -- these should be migrated to sync functions with `asyncio.to_thread()` at the route layer, or to real async SQLAlchemy.
2. **CPU-bound work** (audio processing, numpy operations) goes through `asyncio.to_thread()`:
   ```python
   audio, sr = await asyncio.to_thread(load_audio, source_path)
   ```
3. **GPU-bound TTS inference** is serialized through the generation queue (`services/task_queue.py`). Never call a backend's `generate()` directly from a route handler.
4. **Fire-and-forget tasks**: use `asyncio.create_task()` and track the task reference to prevent garbage collection:
   ```python
   task = asyncio.create_task(some_coro())
   _background_tasks.add(task)
   task.add_done_callback(_background_tasks.discard)
   ```

---

## Logging

Use the `logging` module. Not `print()`.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Loading model %s on %s", model_name, device)
logger.warning("Cache miss for %s, downloading", repo_id)
logger.exception("Generation %s failed")  # logs traceback automatically
```

**Rules:**
- Use `%s`-style placeholders in log calls (not f-strings). This avoids formatting the string if the log level is filtered out.
- Use `logger.exception()` inside `except` blocks -- it captures the traceback.
- Logger name should be `__name__` (yields `backend.utils.audio`, etc.).
- Existing `print()` calls should be migrated to logging as files are touched during the refactor.

---

## Constants

- Define at **module level** in the file where they're primarily used.
- Use `UPPER_SNAKE_CASE`.
- Shared/cross-cutting constants (sample rates, file size limits, CORS origins) go in `backend/config.py` after Phase 6 consolidation.
- Magic numbers in function bodies should be extracted to named constants:
  ```python
  # No
  if len(audio) > 24000 * 60 * 10:

  # Yes
  MAX_AUDIO_DURATION_SAMPLES = SAMPLE_RATE * 60 * 10
  if len(audio) > MAX_AUDIO_DURATION_SAMPLES:
  ```

---

## Function Signatures

- **Keyword-only arguments** (after `*`) for functions with 3+ parameters, especially when several share the same type:
  ```python
  def is_model_cached(
      hf_repo: str,
      *,
      weight_extensions: tuple[str, ...] = (".safetensors", ".bin"),
      required_files: list[str] | None = None,
  ) -> bool:
  ```
- Parameters on **separate lines** when the signature exceeds ~100 characters or has 3+ params.
- **Trailing comma** after the last parameter in multi-line signatures.
- Default values inline with the parameter.

---

## String Formatting

- **f-strings** for runtime string construction.
- **`%s`-style** for `logging` calls (lazy evaluation).
- **`.format()`**: avoid; f-strings are preferred.

---

## Testing

Framework: **pytest** with `pytest-asyncio`.

- Test files: `test_<module>.py` in `backend/tests/`.
- Use `conftest.py` for shared fixtures (db sessions, test client, mock backends).
- Group related tests in classes: `class TestProfileCRUD:`.
- Use `@pytest.mark.asyncio` for async tests.
- Use `@pytest.mark.parametrize` to reduce repetition.
- Manual integration scripts stay in `tests/` but are clearly marked (filename prefix `manual_` or documented in `tests/README.md`).

---

## Project Layout

```
backend/
  app.py              # FastAPI app factory, CORS, lifecycle events
  main.py             # Entry point (imports app, runs uvicorn)
  config.py           # Data directory paths
  models.py           # Pydantic request/response schemas
  server.py           # Tauri sidecar launcher, parent-pid watchdog
  routes/             # Thin HTTP handlers (validation, delegation, response formatting)
  services/           # Business logic, CRUD, orchestration
  backends/           # TTS/STT engine implementations
  database/           # ORM models, session management, migrations, seeds
  utils/              # Shared utilities (audio, effects, caching, progress)
  tests/              # pytest suite
```

---

## Ruff Adoption

`pyproject.toml` configures ruff for linting and formatting. Run:

```bash
# Lint (check)
ruff check backend/

# Lint (auto-fix)
ruff check backend/ --fix

# Format
ruff format backend/
```

Introduce ruff fixes file-by-file as you touch them. Don't run `--fix` across the entire codebase in one shot -- that creates unreviewable diffs.
