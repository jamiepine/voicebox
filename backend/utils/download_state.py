"""Track which HuggingFace snapshots Voicebox has verified as complete.

The HF cache only records which files landed, not which files a model needs,
so a download that dies after the main weights arrive looks finished. Each
repo cache gets a small ``.voicebox`` directory (ignored by
``huggingface_hub.scan_cache_dir``) holding:

- ``downloading-<revision>``: written before a download starts, removed only
  once the download is verified. While it exists the model is not installed.
- ``verified-<revision>.json``: the snapshot's file list, recorded after the
  model loaded successfully. Every listed file must exist for the model to
  count as installed.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_DIR_NAME = ".voicebox"
DEFAULT_REVISION = "main"


def repo_cache_dir(hf_repo: str) -> Path:
    from huggingface_hub import constants as hf_constants

    return Path(hf_constants.HF_HUB_CACHE) / ("models--" + hf_repo.replace("/", "--"))


def _marker_path(hf_repo: str, revision: str) -> Path:
    return repo_cache_dir(hf_repo) / STATE_DIR_NAME / f"downloading-{revision.replace('/', '--')}"


def _record_path(hf_repo: str, revision: str) -> Path:
    return repo_cache_dir(hf_repo) / STATE_DIR_NAME / f"verified-{revision.replace('/', '--')}.json"


def _resolve_commit(hf_repo: str, revision: str) -> str | None:
    ref = repo_cache_dir(hf_repo) / "refs" / revision
    if ref.is_file():
        return ref.read_text().strip() or None
    if (repo_cache_dir(hf_repo) / "snapshots" / revision).is_dir():
        return revision
    return None


def mark_download_started(hf_repo: str, revision: str = DEFAULT_REVISION) -> None:
    marker = _marker_path(hf_repo, revision)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def is_download_in_progress(hf_repo: str, revision: str = DEFAULT_REVISION) -> bool:
    return _marker_path(hf_repo, revision).exists()


def mark_download_verified(hf_repo: str, revision: str = DEFAULT_REVISION) -> bool:
    """Record the snapshot's file list and clear the in-progress marker.

    Returns False (leaving any marker in place) when there is no snapshot
    for *revision* to record.
    """
    commit = _resolve_commit(hf_repo, revision)
    snapshot = repo_cache_dir(hf_repo) / "snapshots" / commit if commit else None
    if snapshot is None or not snapshot.is_dir():
        logger.warning("No snapshot of %s@%s to verify", hf_repo, revision)
        return False

    files = sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file())
    record = _record_path(hf_repo, revision)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"commit": commit, "files": files}))
    _marker_path(hf_repo, revision).unlink(missing_ok=True)
    return True


def verified_snapshot_complete(hf_repo: str, revision: str = DEFAULT_REVISION) -> bool | None:
    """Whether every recorded file exists in the snapshot *revision* points at.

    Returns None when no record exists (e.g. a model downloaded by an older
    Voicebox), so callers can fall back to their own heuristic.
    """
    record = _record_path(hf_repo, revision)
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text())
        commit = _resolve_commit(hf_repo, revision) or data["commit"]
        files = data["files"]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Unreadable download record for %s, ignoring it", hf_repo)
        return None

    snapshot = repo_cache_dir(hf_repo) / "snapshots" / commit
    return bool(files) and all((snapshot / name).is_file() for name in files)
