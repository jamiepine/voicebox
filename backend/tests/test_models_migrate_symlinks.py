"""Cross-filesystem model migration must not dereference HF cache symlinks.

A HuggingFace cache stores each file once in ``blobs/<hash>`` and links it
from ``snapshots/<rev>/<name>``. The cross-drive branch of ``/models/migrate``
(e.g. ``C:`` -> ``D:`` on Windows) used to follow those links, writing every
model twice at the destination and double-counting the progress total.
"""

import os
from pathlib import Path

import pytest

from backend.routes.models import _copy_with_progress, _get_dir_size


class _Progress:
    def __init__(self):
        self.calls = []

    def update_progress(self, model_name, current, total, filename=None, status=None):
        self.calls.append((current, total))


def _make_cache(root: Path, payload: bytes) -> Path:
    repo = root / "models--org--model"
    blob = repo / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    try:
        os.symlink(os.path.join("..", "..", "blobs", "abc123"), snap / "model.bin")
    except OSError:
        pytest.skip("symlinks not permitted on this system")
    return repo


def test_dir_size_counts_blob_once(tmp_path):
    payload = b"x" * 4096
    repo = _make_cache(tmp_path / "src", payload)
    assert _get_dir_size(repo) == len(payload) + len("rev1")


def test_copy_preserves_links_instead_of_duplicating(tmp_path):
    payload = b"y" * 8192
    repo = _make_cache(tmp_path / "src", payload)
    dest = tmp_path / "dst" / repo.name
    progress = _Progress()

    total = _get_dir_size(repo)
    copied = _copy_with_progress(repo, dest, progress, 0, total)

    linked = dest / "snapshots" / "rev1" / "model.bin"
    blob = dest / "blobs" / "abc123"
    assert linked.read_bytes() == payload
    # Either a real symlink or a hard link to the same blob — never a second copy.
    assert linked.is_symlink() or os.path.samefile(linked, blob)
    assert copied == total
    assert all(cur <= tot for cur, tot in progress.calls)


def test_copy_falls_back_to_hardlink_when_symlink_refused(tmp_path, monkeypatch):
    payload = b"z" * 1024
    repo = _make_cache(tmp_path / "src", payload)
    dest = tmp_path / "dst" / repo.name

    def refuse(*args, **kwargs):
        raise OSError("symlinks not permitted")

    monkeypatch.setattr(os, "symlink", refuse)
    _copy_with_progress(repo, dest, _Progress(), 0, _get_dir_size(repo))

    linked = dest / "snapshots" / "rev1" / "model.bin"
    assert not linked.is_symlink()
    assert linked.read_bytes() == payload
    assert os.path.samefile(linked, dest / "blobs" / "abc123")
