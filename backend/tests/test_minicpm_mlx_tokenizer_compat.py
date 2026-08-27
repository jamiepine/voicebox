"""
openbmb/MiniCPM5-1B-MLX's tokenizer_config.json declares
`"tokenizer_class": "TokenizersBackend"` — a transformers-5.x-era class name
that doesn't exist in this project's pinned transformers (<=4.57.6, capped
because mlx-audio's TTS/STT path needs the older API surface — see
backend/requirements-mlx.txt). AutoTokenizer.from_pretrained refuses to load
with a ValueError when the declared class isn't importable, which breaks
mlx_lm.load() outright for this specific repo (confirmed via a real,
end-to-end download+load attempt against the live repo — see
specs/001-minicpm5-llm-engine/tasks.md's T026 note).

Confirmed via direct testing that the underlying tokenizer.json is a
standard fast-tokenizer file compatible with plain PreTrainedTokenizerFast —
the checkpoint's declared class name is just stale/wrong for this
transformers version, not the tokenizer data itself. This module builds a
local staging directory (symlinks to the original files, except a patched
tokenizer_config.json) so mlx_lm.load() can load it without duplicating the
multi-hundred-MB weight file on disk.
"""

import json

import pytest

pytest.importorskip("torch")

from backend.backends import minicpm_llm_backend


def test_leaves_a_resolvable_tokenizer_class_untouched(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (src / "model.safetensors").write_bytes(b"fake weights")

    staged = minicpm_llm_backend._ensure_compatible_mlx_snapshot(str(src))

    from pathlib import Path

    staged_cfg = json.loads((Path(staged) / "tokenizer_config.json").read_text())
    assert staged_cfg["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_rewrites_an_unresolvable_tokenizer_class_to_a_safe_fallback(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
    (src / "model.safetensors").write_bytes(b"fake weights")

    staged = minicpm_llm_backend._ensure_compatible_mlx_snapshot(str(src))

    from pathlib import Path

    staged_path = Path(staged)
    assert staged_path != src
    staged_cfg = json.loads((staged_path / "tokenizer_config.json").read_text())
    assert staged_cfg["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_does_not_duplicate_large_files_on_disk(tmp_path):
    """Weights must be symlinked, not copied — this repo's weight file is
    600MB+; copying it on every load would be a real disk/time cost."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
    (src / "model.safetensors").write_bytes(b"fake weights")

    staged = minicpm_llm_backend._ensure_compatible_mlx_snapshot(str(src))

    from pathlib import Path

    staged_weights = Path(staged) / "model.safetensors"
    assert staged_weights.is_symlink()
    assert staged_weights.resolve() == (src / "model.safetensors").resolve()
