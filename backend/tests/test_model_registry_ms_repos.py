"""Every ``ModelConfig`` in the registry gets the ``ms_repo_id`` from
specs/001-modelscope-download-source/research.md §1 (verified live against
modelscope.cn before this feature was built) — or stays unset for the 4
models with no ModelScope mirror.
"""

from unittest.mock import patch

from backend.backends import get_all_model_configs

# model_name -> expected ms_repo_id (None means "no mirror, must stay unset")
EXPECTED_PYTORCH = {
    "qwen-tts-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen-tts-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "qwen-custom-voice-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen-custom-voice-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "luxtts": "hf/YatharthS-LuxTTS",
    "chatterbox-tts": None,
    "chatterbox-turbo": None,
    "tada-1b": None,
    "tada-3b-ml": None,
    "kokoro": "AI-ModelScope/Kokoro-82M",
    "whisper-base": "openai-mirror/whisper-base",
    "whisper-small": "openai-mirror/whisper-small",
    "whisper-medium": "openai-mirror/whisper-medium",
    "whisper-large": "openai-mirror/whisper-large-v3",
    "whisper-turbo": "openai-mirror/whisper-large-v3-turbo",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

EXPECTED_MLX_OVERRIDES = {
    "qwen-tts-1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "qwen-tts-0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
    "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
}


def _ms_repo_ids_by_name():
    return {cfg.model_name: cfg.ms_repo_id for cfg in get_all_model_configs()}


def test_pytorch_backend_ms_repo_ids():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        actual = _ms_repo_ids_by_name()
    # Exact key-set equality first: an extra/renamed registry entry with a
    # wrong ms_repo_id would otherwise be silently skipped by the loop below.
    assert set(actual) == set(EXPECTED_PYTORCH)
    for model_name, expected in EXPECTED_PYTORCH.items():
        assert actual[model_name] == expected, f"{model_name}: {actual[model_name]!r} != {expected!r}"


def test_mlx_backend_ms_repo_id_overrides():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        actual = _ms_repo_ids_by_name()
    expected = dict(EXPECTED_PYTORCH, **EXPECTED_MLX_OVERRIDES)
    assert set(actual) == set(expected)
    for model_name, expected_id in expected.items():
        assert actual[model_name] == expected_id, f"{model_name}: {actual[model_name]!r} != {expected_id!r}"


def test_exactly_four_models_have_no_modelscope_mirror():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        actual = _ms_repo_ids_by_name()
    unmirrored = {name for name, ms_id in actual.items() if ms_id is None}
    assert unmirrored == {"chatterbox-tts", "chatterbox-turbo", "tada-1b", "tada-3b-ml"}
