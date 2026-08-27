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
    yield


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
