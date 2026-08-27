"""``resolve_model_source`` — picks HF repo id vs. a local ModelScope
download directory depending on the active download source, per
specs/001-modelscope-download-source/data-model.md.
"""

from unittest.mock import patch

import pytest

from backend.backends import base
from backend.utils import model_source


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    import backend.config as config

    # get_models_dir()/get_data_dir() read the module-global _data_dir, not
    # a function call — patch that directly so it's auto-restored after the
    # test (monkeypatch.setattr reverts on teardown).
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    yield


def test_returns_hf_repo_id_unchanged_for_huggingface_source():
    model_source.set_model_source("huggingface")
    result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")
    assert result == "hexgrad/Kokoro-82M"


def test_returns_hf_repo_id_unchanged_when_modelscope_selected_but_no_mirror():
    model_source.set_model_source("modelscope")
    result = base.resolve_model_source("ResembleAI/chatterbox", None, "chatterbox-tts")
    assert result == "ResembleAI/chatterbox"


def test_downloads_via_modelscope_and_returns_local_dir_when_mirror_available(tmp_path):
    model_source.set_model_source("modelscope")

    def fake_snapshot_download(model_id, local_dir=None, progress_callbacks=None, **kwargs):
        assert model_id == "AI-ModelScope/Kokoro-82M"
        # Simulate the SDK actually writing a weight file to local_dir.
        from pathlib import Path

        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "kokoro-v1_0.pth").write_bytes(b"fake")
        return local_dir

    with patch("modelscope.snapshot_download", side_effect=fake_snapshot_download) as mock_download:
        result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")

    mock_download.assert_called_once()
    assert result == str(tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M")


def test_skips_download_when_already_cached_locally(tmp_path):
    model_source.set_model_source("modelscope")
    local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    local_dir.mkdir(parents=True)
    (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")

    with patch("modelscope.snapshot_download") as mock_download:
        result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")

    mock_download.assert_not_called()
    assert result == str(local_dir)
