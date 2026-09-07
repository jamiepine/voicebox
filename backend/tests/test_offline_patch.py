"""
Unit tests for ``patch_transformers_mistral_regex``.

Verifies that our wrapper around
``transformers.PreTrainedTokenizerBase._patch_mistral_regex`` catches
exceptions from the unconditional ``huggingface_hub.model_info()`` lookup
and returns the tokenizer unchanged — matching the success-path behavior
for non-Mistral repos (transformers 4.57.3, ``tokenization_utils_base.py:2503``).

NOTE: These tests mutate ``transformers.PreTrainedTokenizerBase`` globally;
run serially, not under ``pytest-xdist`` with per-worker process isolation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub.errors import OfflineModeIsEnabled
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import utils.hf_offline_patch as hf_offline_patch


@pytest.fixture(autouse=True)
def restore_mistral_regex():
    """Snapshot the current ``_patch_mistral_regex`` and restore after each test."""
    saved = PreTrainedTokenizerBase.__dict__.get("_patch_mistral_regex")
    saved_flag = hf_offline_patch._mistral_regex_patched
    try:
        yield
    finally:
        if saved is not None:
            PreTrainedTokenizerBase._patch_mistral_regex = saved
        hf_offline_patch._mistral_regex_patched = saved_flag


def _apply_patch():
    hf_offline_patch._mistral_regex_patched = False
    hf_offline_patch.patch_transformers_mistral_regex()


def test_suppresses_offline_mode_is_enabled(monkeypatch):
    _apply_patch()

    import huggingface_hub

    def raise_offline(*_args, **_kwargs):
        raise OfflineModeIsEnabled("offline")

    monkeypatch.setattr(huggingface_hub, "model_info", raise_offline)

    sentinel = object()
    result = PreTrainedTokenizerBase._patch_mistral_regex(
        sentinel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    assert result is sentinel


def test_suppresses_connection_errors(monkeypatch):
    _apply_patch()

    import huggingface_hub

    def raise_connection(*_args, **_kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "model_info", raise_connection)

    sentinel = object()
    result = PreTrainedTokenizerBase._patch_mistral_regex(
        sentinel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    assert result is sentinel


def test_passthrough_on_success(monkeypatch):
    """When model_info returns non-Mistral tags the original falls through and returns the tokenizer unchanged."""
    _apply_patch()

    import huggingface_hub

    class FakeInfo:
        tags = ["model-type:qwen", "language:en"]

    monkeypatch.setattr(huggingface_hub, "model_info", lambda *_a, **_kw: FakeInfo())

    sentinel = object()
    result = PreTrainedTokenizerBase._patch_mistral_regex(
        sentinel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    assert result is sentinel


def test_idempotent():
    _apply_patch()
    first = PreTrainedTokenizerBase._patch_mistral_regex
    hf_offline_patch.patch_transformers_mistral_regex()
    second = PreTrainedTokenizerBase._patch_mistral_regex
    assert first.__func__ is second.__func__


def test_missing_method_is_noop(monkeypatch):
    monkeypatch.delattr(PreTrainedTokenizerBase, "_patch_mistral_regex", raising=False)
    hf_offline_patch._mistral_regex_patched = False
    hf_offline_patch.patch_transformers_mistral_regex()
    assert hf_offline_patch._mistral_regex_patched is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
