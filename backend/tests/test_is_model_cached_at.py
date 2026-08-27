"""``is_model_cached_at`` — the source-aware cache check used by backends'
``_is_model_cached()``. Given a local filesystem path (absolute), checks the
directory directly; given a HuggingFace repo id string, delegates to the
existing ``is_model_cached``.
"""

from unittest.mock import patch

from backend.backends.base import is_model_cached_at


def test_local_path_with_weight_file_is_cached(tmp_path):
    local_dir = tmp_path / "some-model"
    local_dir.mkdir()
    (local_dir / "model.safetensors").write_bytes(b"fake")
    assert is_model_cached_at(str(local_dir)) is True


def test_local_path_without_weight_file_is_not_cached(tmp_path):
    local_dir = tmp_path / "some-model"
    local_dir.mkdir()
    (local_dir / "README.md").write_text("hi")
    assert is_model_cached_at(str(local_dir)) is False


def test_nonexistent_local_path_is_not_cached(tmp_path):
    assert is_model_cached_at(str(tmp_path / "does-not-exist")) is False


def test_local_path_honors_required_files(tmp_path):
    local_dir = tmp_path / "kokoro"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")
    assert is_model_cached_at(str(local_dir), required_files=["config.json", "kokoro-v1_0.pth"]) is False
    (local_dir / "kokoro-v1_0.pth").write_bytes(b"fake")
    assert is_model_cached_at(str(local_dir), required_files=["config.json", "kokoro-v1_0.pth"]) is True


def test_repo_id_string_delegates_to_is_model_cached():
    with patch("backend.backends.base.is_model_cached", return_value=True) as mock_check:
        assert is_model_cached_at("hexgrad/Kokoro-82M", required_files=["x"]) is True
    mock_check.assert_called_once_with(
        "hexgrad/Kokoro-82M", weight_extensions=(".safetensors", ".bin"), required_files=["x"]
    )
