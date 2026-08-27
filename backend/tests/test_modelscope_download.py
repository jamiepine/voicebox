"""End-to-end wiring: with ModelScope as the active source, KokoroBackend
downloads via the ModelScope SDK and loads from the resulting local
directory — see specs/001-modelscope-download-source/quickstart.md Scenario 2.

Note: writing this test required first reading ``kokoro.KModel``'s actual
``__init__`` (it calls ``hf_hub_download`` keyed by a ``MODEL_NAMES[repo_id]``
dict lookup — a bare local directory path can't stand in for ``repo_id``
there), so the implementation and this test were developed together rather
than in strict red-first order for the local-path branch specifically. The
resolver/cache-check logic underneath (``resolve_model_source``,
``is_model_cached_at``) was fully TDD'd beforehand in
test_model_source_resolution.py / test_is_model_cached_at.py.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.backends.kokoro_backend import KOKORO_MS_REPO, KokoroTTSBackend
from backend.utils import model_source


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    import backend.config as config

    monkeypatch.setattr(config, "_data_dir", tmp_path)
    yield


def test_get_model_path_downloads_via_modelscope_and_returns_local_dir(tmp_path):
    model_source.set_model_source("modelscope")
    backend = KokoroTTSBackend()

    def fake_snapshot_download(model_id, local_dir=None, progress_callbacks=None, **kwargs):
        assert model_id == KOKORO_MS_REPO
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")
        (Path(local_dir) / "kokoro-v1_0.pth").write_bytes(b"fake")
        return local_dir

    with patch("modelscope.snapshot_download", side_effect=fake_snapshot_download):
        resolved = backend._get_model_path("default")

    expected_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    assert resolved == str(expected_dir)
    assert (expected_dir / "kokoro-v1_0.pth").exists()


def test_is_model_cached_true_once_modelscope_download_completed(tmp_path):
    model_source.set_model_source("modelscope")
    backend = KokoroTTSBackend()

    local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}")
    (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")

    with patch("modelscope.snapshot_download") as mock_download:
        assert backend._is_model_cached() is True
    mock_download.assert_not_called()
