"""``resolve_model_source`` (pure path lookup) and ``ensure_model_downloaded``
(the side-effecting counterpart) — picks HF repo id vs. a local ModelScope
download directory depending on the active download source, per
specs/001-modelscope-download-source/data-model.md.

Split into two functions (2026-08-27, post-review) because the original
single function downloaded as a side effect of resolving a path — which
meant ``_is_model_cached()`` (a supposedly pure check, called before
``model_load_progress()`` even starts) silently triggered full downloads
that bypassed task-manager registration and error handling. See
specs/001-modelscope-download-source/research.md for the full account.
``resolve_model_source`` must never download; only ``ensure_model_downloaded``
may, and only when called from inside a ``model_load_progress()`` block.
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
    return


class TestResolveModelSourceIsPure:
    """resolve_model_source() must never download — it's called by
    _is_model_cached(), which every caller relies on being side-effect-free."""

    def test_returns_hf_repo_id_unchanged_for_huggingface_source(self):
        model_source.set_model_source("huggingface")
        result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")
        assert result == "hexgrad/Kokoro-82M"

    def test_returns_hf_repo_id_unchanged_when_modelscope_selected_but_no_mirror(self):
        model_source.set_model_source("modelscope")
        result = base.resolve_model_source("ResembleAI/chatterbox", None, "chatterbox-tts")
        assert result == "ResembleAI/chatterbox"

    def test_returns_local_dir_path_without_downloading_even_when_not_cached(self, tmp_path):
        model_source.set_model_source("modelscope")

        with patch("modelscope.snapshot_download") as mock_download:
            result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")

        mock_download.assert_not_called()
        assert result == str(tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M")

    def test_returns_local_dir_path_when_already_cached(self, tmp_path):
        model_source.set_model_source("modelscope")
        local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
        local_dir.mkdir(parents=True)
        (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")

        result = base.resolve_model_source("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")
        assert result == str(local_dir)


class TestEnsureModelDownloaded:
    """ensure_model_downloaded() is the only function allowed to trigger a
    ModelScope download — callers must wrap it in model_load_progress()."""

    def test_returns_hf_repo_id_unchanged_for_huggingface_source(self):
        model_source.set_model_source("huggingface")
        with patch("modelscope.snapshot_download") as mock_download:
            result = base.ensure_model_downloaded("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")
        mock_download.assert_not_called()
        assert result == "hexgrad/Kokoro-82M"

    def test_returns_hf_repo_id_unchanged_when_modelscope_selected_but_no_mirror(self):
        model_source.set_model_source("modelscope")
        with patch("modelscope.snapshot_download") as mock_download:
            result = base.ensure_model_downloaded("ResembleAI/chatterbox", None, "chatterbox-tts")
        mock_download.assert_not_called()
        assert result == "ResembleAI/chatterbox"

    def test_downloads_via_modelscope_and_returns_local_dir_when_mirror_available(self, tmp_path):
        model_source.set_model_source("modelscope")

        def fake_snapshot_download(model_id, local_dir=None, progress_callbacks=None, **kwargs):
            assert model_id == "AI-ModelScope/Kokoro-82M"
            from pathlib import Path

            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "kokoro-v1_0.pth").write_bytes(b"fake")
            return local_dir

        with patch("modelscope.snapshot_download", side_effect=fake_snapshot_download) as mock_download:
            result = base.ensure_model_downloaded("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")

        mock_download.assert_called_once()
        assert result == str(tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M")

    def test_skips_download_when_already_cached_locally(self, tmp_path):
        model_source.set_model_source("modelscope")
        local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
        local_dir.mkdir(parents=True)
        (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")

        with patch("modelscope.snapshot_download") as mock_download:
            result = base.ensure_model_downloaded("hexgrad/Kokoro-82M", "AI-ModelScope/Kokoro-82M", "kokoro")

        mock_download.assert_not_called()
        assert result == str(local_dir)
