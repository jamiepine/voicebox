"""End-to-end wiring: with ModelScope as the active source, KokoroBackend
downloads via the ModelScope SDK and loads from the resulting local
directory — see specs/001-modelscope-download-source/quickstart.md Scenario 2.

Also covers the post-review fix (2026-08-27): checking cache status must
never trigger a download. Before the fix, `_get_model_path()` did both the
path lookup AND the download, so `_is_model_cached()` — called before
`model_load_progress()` starts — silently downloaded the model outside any
progress/task-manager/error-handling coverage. Now `_get_model_path()` is
pure (backed by `resolve_model_source()`) and only `_load_model_sync()`
(inside `model_load_progress()`) triggers a download, via
`ensure_model_downloaded()`.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.backends.kokoro_backend import KOKORO_MS_REPO, KokoroTTSBackend
from backend.utils import model_source


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    import backend.config as config

    monkeypatch.setattr(config, "_data_dir", tmp_path)
    return


def test_get_model_path_never_downloads_even_when_not_cached(tmp_path):
    """_get_model_path() backs _is_model_cached() and must stay pure."""
    model_source.set_model_source("modelscope")
    backend = KokoroTTSBackend()

    with patch("modelscope.snapshot_download") as mock_download:
        resolved = backend._get_model_path("default")
        assert backend._is_model_cached() is False

    mock_download.assert_not_called()
    expected_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    assert resolved == str(expected_dir)
    assert not expected_dir.exists()


def test_get_model_path_downloads_via_modelscope_and_returns_local_dir(tmp_path):
    """The actual download only happens through ensure_model_downloaded(),
    exercised here directly (this is what _load_model_sync() must call)."""
    from backend.backends.base import ensure_model_downloaded

    model_source.set_model_source("modelscope")

    def fake_snapshot_download(model_id, local_dir=None, progress_callbacks=None, **kwargs):
        assert model_id == KOKORO_MS_REPO
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")
        (Path(local_dir) / "kokoro-v1_0.pth").write_bytes(b"fake")
        return local_dir

    with patch("modelscope.snapshot_download", side_effect=fake_snapshot_download):
        resolved = ensure_model_downloaded("hexgrad/Kokoro-82M", KOKORO_MS_REPO, "kokoro")

    expected_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    assert resolved == str(expected_dir)
    assert (expected_dir / "kokoro-v1_0.pth").exists()


def test_load_model_sync_downloads_via_ensure_model_downloaded(tmp_path):
    """_load_model_sync() must call the downloading resolver — not the pure
    one — so a fresh download is registered with the task manager and any
    failure is caught by model_load_progress()'s error handling."""
    model_source.set_model_source("modelscope")
    backend = KokoroTTSBackend()

    local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"

    def fake_ensure_downloaded(hf_repo_id, ms_repo_id, model_name):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")
        return str(local_dir)

    fake_kmodel_instance = MagicMock()
    fake_kmodel_instance.to.return_value = fake_kmodel_instance
    fake_kmodel_instance.eval.return_value = fake_kmodel_instance
    fake_kmodel_cls = MagicMock(return_value=fake_kmodel_instance)

    with (
        patch("backend.backends.base.ensure_model_downloaded", side_effect=fake_ensure_downloaded) as mock_ensure,
        patch("kokoro.KModel", fake_kmodel_cls),
    ):
        backend._load_model_sync()

    mock_ensure.assert_called_once_with("hexgrad/Kokoro-82M", KOKORO_MS_REPO, "kokoro")
    fake_kmodel_cls.assert_called_once()
    _, kwargs = fake_kmodel_cls.call_args
    assert kwargs["config"] == str(local_dir / "config.json")
    assert kwargs["model"] == str(local_dir / "kokoro-v1_0.pth")


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
