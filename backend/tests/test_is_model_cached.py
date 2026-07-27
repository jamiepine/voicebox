"""
Unit tests for ``is_model_cached``'s handling of stale ``.incomplete`` blobs.

A retried or concurrent download can leave an orphaned ``.incomplete`` file
next to its now-completed counterpart (same blob hash, no suffix). Only a
genuinely in-progress download -- an ``.incomplete`` with no completed blob
alongside it -- should mark the model as not cached.

``is_model_cached`` is extracted and exec'd standalone (instead of importing
``backend.backends.base``) so this test doesn't pull in the module's sibling
imports (audio/progress/hf_progress/tasks), which in turn require the full
ML stack (torch/transformers/librosa/fastapi/...) this pure filesystem check
never touches.
"""

import ast
import logging
from pathlib import Path
from typing import Optional

_SOURCE = (Path(__file__).parent.parent / "backends" / "base.py").read_text()
_MODULE = ast.parse(_SOURCE)
_FUNC_SRC = next(
    ast.get_source_segment(_SOURCE, node)
    for node in _MODULE.body
    if isinstance(node, ast.FunctionDef) and node.name == "is_model_cached"
)

_namespace = {
    "Path": Path,
    "Optional": Optional,
    "logger": logging.getLogger("test_is_model_cached"),
}
exec(_FUNC_SRC, _namespace)  # noqa: S102
is_model_cached = _namespace["is_model_cached"]


def _make_repo_cache(tmp_path, repo="org/model"):
    repo_dir = tmp_path / ("models--" + repo.replace("/", "--"))
    blobs_dir = repo_dir / "blobs"
    snapshots_dir = repo_dir / "snapshots" / "abc123"
    blobs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    return repo_dir, blobs_dir, snapshots_dir


def test_orphaned_incomplete_blob_does_not_block_cache_hit(tmp_path, monkeypatch):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))

    repo = "org/model"
    _, blobs_dir, snapshots_dir = _make_repo_cache(tmp_path, repo)

    completed_blob = blobs_dir / "deadbeef"
    completed_blob.write_bytes(b"weights")
    (blobs_dir / "deadbeef.incomplete").write_bytes(b"stale partial")
    (snapshots_dir / "model.safetensors").symlink_to(completed_blob)

    assert is_model_cached(repo) is True


def test_genuinely_in_progress_download_is_not_cached(tmp_path, monkeypatch):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))

    repo = "org/model"
    _, blobs_dir, snapshots_dir = _make_repo_cache(tmp_path, repo)

    (blobs_dir / "feedface.incomplete").write_bytes(b"partial")
    (snapshots_dir / "config.json").write_text("{}")

    assert is_model_cached(repo) is False
