"""
Unit tests for the MiniCPM5-1B LLM backend classes.

Mirrors qwen_llm_backend.py's shape (same LLMBackend protocol, same
_build_messages/enable_thinking=False chat-template pattern) — MiniCPM5-1B
is a plain LlamaForCausalLM checkpoint whose chat template accepts the same
enable_thinking kwarg Qwen3's does (see specs/001-minicpm5-llm-engine/research.md
Decision 1). No real model weights or MLX runtime are used here — every
external call (transformers, mlx_lm) is mocked out.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("torch")

from backend.backends import minicpm_llm_backend


# ── PyTorch backend ─────────────────────────────────────────────────────


def test_pytorch_backend_reports_unloaded_by_default():
    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    assert backend.is_loaded() is False


def test_pytorch_get_model_path_resolves_official_repo():
    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    assert backend._get_model_path("1B") == "openbmb/MiniCPM5-1B"


def test_pytorch_get_model_path_rejects_unknown_size():
    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    with pytest.raises(ValueError):
        backend._get_model_path("7B")


@pytest.mark.asyncio
async def test_pytorch_load_model_uses_official_repo_and_marks_loaded():
    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as mock_tok,
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as mock_model,
        patch.object(minicpm_llm_backend, "is_model_cached", return_value=True),
        patch.object(minicpm_llm_backend, "model_load_progress") as mock_progress,
    ):
        mock_progress.return_value.__enter__.return_value = None
        mock_progress.return_value.__exit__.return_value = False
        await backend.load_model("1B")

    mock_tok.assert_called_once_with("openbmb/MiniCPM5-1B")
    assert mock_model.call_args[0][0] == "openbmb/MiniCPM5-1B"
    assert backend.is_loaded() is True
    assert backend._current_model_size == "1B"


@pytest.mark.asyncio
async def test_pytorch_generate_calls_chat_template_with_enable_thinking_false():
    import torch

    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    backend.model = MagicMock()
    backend._current_model_size = "1B"
    backend.device = "cpu"

    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "<rendered prompt>"
    fake_tokenizer.eos_token_id = 1
    fake_inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
    fake_inputs_obj = MagicMock()
    fake_inputs_obj.to.return_value = fake_inputs
    fake_tokenizer.return_value = fake_inputs_obj
    fake_tokenizer.decode.return_value = "hello there"
    backend.tokenizer = fake_tokenizer

    backend.model.generate.return_value = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3]])

    with patch.object(backend, "load_model", new=AsyncMock()):
        result = await backend.generate("hi", system="be nice")

    _, kwargs = fake_tokenizer.apply_chat_template.call_args
    assert kwargs["enable_thinking"] is False
    assert result == "hello there"


def test_pytorch_unload_model_clears_state():
    backend = minicpm_llm_backend.PyTorchMiniCPMLLMBackend()
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()
    backend._current_model_size = "1B"
    backend.device = "cpu"

    with patch.object(minicpm_llm_backend, "empty_device_cache") as mock_empty:
        backend.unload_model()

    assert backend.model is None
    assert backend.tokenizer is None
    assert backend._current_model_size is None
    mock_empty.assert_called_once_with("cpu")


# ── MLX backend ──────────────────────────────────────────────────────────


def test_mlx_backend_reports_unloaded_by_default():
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    assert backend.is_loaded() is False


def test_mlx_get_model_path_resolves_official_repo():
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    assert backend._get_model_path("1B") == "openbmb/MiniCPM5-1B-MLX"


@pytest.mark.asyncio
async def test_mlx_load_model_uses_official_repo_and_marks_loaded():
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load.return_value = (fake_model, fake_tokenizer)
    fake_hf_hub = MagicMock()
    fake_hf_hub.snapshot_download.return_value = "/fake/local/snapshot"

    with (
        patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm, "huggingface_hub": fake_hf_hub}),
        patch.object(minicpm_llm_backend, "is_model_cached", return_value=True),
        patch.object(minicpm_llm_backend, "model_load_progress") as mock_progress,
        patch.object(
            minicpm_llm_backend,
            "_ensure_compatible_mlx_snapshot",
            side_effect=lambda p: p,
        ),
    ):
        mock_progress.return_value.__enter__.return_value = None
        mock_progress.return_value.__exit__.return_value = False
        await backend.load_model("1B")

    fake_hf_hub.snapshot_download.assert_called_once_with("openbmb/MiniCPM5-1B-MLX")
    fake_mlx_lm.load.assert_called_once_with("/fake/local/snapshot")
    assert backend.is_loaded() is True
    assert backend._current_model_size == "1B"


@pytest.mark.asyncio
async def test_mlx_generate_calls_chat_template_with_enable_thinking_false():
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    backend.model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "<rendered prompt>"
    backend.tokenizer = fake_tokenizer

    fake_mlx_lm = MagicMock()
    fake_mlx_lm.generate.return_value = "hello from minicpm"
    fake_sample_utils = MagicMock()

    with (
        patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm, "mlx_lm.sample_utils": fake_sample_utils}),
        patch.object(backend, "load_model", new=AsyncMock()),
    ):
        result = await backend.generate("hi", system="be nice")

    _, kwargs = fake_tokenizer.apply_chat_template.call_args
    assert kwargs["enable_thinking"] is False
    assert result == "hello from minicpm"


def test_mlx_unload_model_clears_state_and_calls_empty_mlx_cache():
    """Mirrors test_mlx_unload_clears_cache.py's pattern for MLXQwenLLMBackend —
    empty_mlx_cache() must run so MLX's allocator actually releases the freed
    buffers, not just the Python reference (see research.md's baseline-update note)."""
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()
    backend._current_model_size = "1B"

    with patch.object(minicpm_llm_backend, "empty_mlx_cache") as mock_clear:
        backend.unload_model()

    assert backend.model is None
    assert backend.tokenizer is None
    assert backend._current_model_size is None
    mock_clear.assert_called_once()


def test_mlx_unload_when_already_unloaded_is_noop():
    backend = minicpm_llm_backend.MLXMiniCPMLLMBackend()
    assert backend.model is None

    with patch.object(minicpm_llm_backend, "empty_mlx_cache") as mock_clear:
        backend.unload_model()

    mock_clear.assert_not_called()
