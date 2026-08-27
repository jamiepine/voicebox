"""Persisted model-download-source setting.

Stored as a flat JSON file (not a DB row) — see
specs/001-modelscope-download-source/research.md §4. Applies live: read
fresh by ``resolve_model_source()`` on every call, no apply-at-startup step.
"""

import pytest

import backend.config as config
from backend.utils import model_source


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point the setting file at a throwaway directory for every test."""
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    return


def test_defaults_to_huggingface_when_file_absent():
    assert model_source.get_model_source() == "huggingface"


def test_set_then_get_round_trips():
    model_source.set_model_source("modelscope")
    assert model_source.get_model_source() == "modelscope"


def test_set_persists_across_a_fresh_read(tmp_path):
    model_source.set_model_source("modelscope")
    # Simulate a new process reading the same file from scratch.
    assert (tmp_path / "model_source.json").exists()
    assert model_source.get_model_source() == "modelscope"


def test_set_rejects_invalid_value():
    with pytest.raises(ValueError):
        model_source.set_model_source("not-a-real-source")


def test_set_rejects_the_removed_hf_mirror_source():
    # hf_mirror was removed after real-world verification showed the mirror
    # doesn't work — see research.md §2. Must not be resurrectable via a
    # stale client/request.
    with pytest.raises(ValueError):
        model_source.set_model_source("hf_mirror")


def test_set_writes_atomically_no_tmp_file_left_behind(tmp_path):
    model_source.set_model_source("modelscope")
    leftover_tmp_files = list(tmp_path.glob("model_source.json.*.tmp"))
    assert leftover_tmp_files == []


def test_a_failed_write_does_not_corrupt_the_existing_file(tmp_path, monkeypatch):
    """set_model_source() must write to a temp file and rename it into place
    — not truncate the real file in place — so a reader mid-write (or a
    write that fails partway) never sees invalid JSON and silently falls
    back to the default, masking the user's actual setting."""
    model_source.set_model_source("modelscope")

    original_replace = __import__("os").replace

    def failing_replace(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("os.replace", failing_replace)
    with pytest.raises(OSError):
        model_source.set_model_source("huggingface")
    monkeypatch.setattr("os.replace", original_replace)

    # The original value must still be intact and readable.
    assert model_source.get_model_source() == "modelscope"
