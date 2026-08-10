"""Unit tests for per-engine chunk-size resolution in generation.

Regression: tada degenerates on long inputs (a 650-char paragraph yielded
a 1.0s clip containing only "y"). GenerationRequest defaults
max_chunk_chars=800, so the cap must apply even when the caller passes
the default value explicitly.
"""

import pytest

from backend.services.generation import (
    TADA_MAX_CHUNK_CHARS,
    effective_max_chunk_chars,
)


def test_tada_none_gets_cap():
    assert effective_max_chunk_chars("tada", None) == TADA_MAX_CHUNK_CHARS


def test_tada_default_800_is_capped():
    # The request default (800) must not bypass the tada cap.
    assert effective_max_chunk_chars("tada", 800) == TADA_MAX_CHUNK_CHARS


def test_tada_smaller_explicit_value_respected():
    assert effective_max_chunk_chars("tada", 100) == 100


def test_other_engines_untouched():
    assert effective_max_chunk_chars("qwen", 800) == 800
    assert effective_max_chunk_chars("kokoro", 800) == 800
    assert effective_max_chunk_chars("qwen", None) is None
