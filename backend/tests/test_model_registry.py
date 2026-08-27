"""``ModelConfig.ms_repo_id`` — the optional ModelScope repository id.

Absence (``None``) is the signal that a model has no ModelScope mirror and
downloads from huggingface.co directly, even when ModelScope is the active
source (see specs/001-modelscope-download-source/data-model.md).
"""

from backend.backends import ModelConfig


def test_ms_repo_id_defaults_to_none():
    config = ModelConfig(
        model_name="kokoro",
        display_name="Kokoro 82M",
        engine="kokoro",
        hf_repo_id="hexgrad/Kokoro-82M",
    )
    assert config.ms_repo_id is None


def test_ms_repo_id_can_be_set():
    config = ModelConfig(
        model_name="kokoro",
        display_name="Kokoro 82M",
        engine="kokoro",
        hf_repo_id="hexgrad/Kokoro-82M",
        ms_repo_id="AI-ModelScope/Kokoro-82M",
    )
    assert config.ms_repo_id == "AI-ModelScope/Kokoro-82M"
