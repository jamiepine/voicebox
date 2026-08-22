#!/usr/bin/env python3
"""Vendor the Higgs Audio V2 tokenizer from transformers into backend/vendor/.

OmniVoice needs ``transformers.HiggsAudioV2TokenizerModel``, which only exists
in transformers >= 5.3. Voicebox is pinned to <= 4.57.6 because several engine
workarounds depend on 4.57.x internals (see ``backend/utils/hf_offline_patch.py``
and ``backend/pyi_rth_torch_compiler_disable.py``).

Rather than move the whole app to transformers 5, we vendor the two files that
define the tokenizer and rewrite their imports. Only a handful of symbols differ
between the two versions; ``backend/vendor/_compat.py`` supplies them.  This
mirrors the ``--no-deps`` treatment already applied to mlx-lm / mlx-audio in the
justfile, which declare transformers >= 5 for the same reason.

Re-run to re-sync against a newer upstream revision:

    python scripts/sync-higgs-tokenizer.py [--ref main]
"""

import argparse
import re
import urllib.request
from pathlib import Path

UPSTREAM = (
    "https://raw.githubusercontent.com/huggingface/transformers/{ref}"
    "/src/transformers/models/higgs_audio_v2_tokenizer/{name}.py"
)
FILES = ["configuration_higgs_audio_v2_tokenizer", "modeling_higgs_audio_v2_tokenizer"]
DEST = Path(__file__).resolve().parent.parent / "backend" / "vendor" / "higgs_audio_v2_tokenizer"

# Relative imports rewritten to absolute. Anything resolving to a symbol that
# 4.57.6 does not have points at _compat instead of transformers.
REWRITES = [
    (r"^from \.\.\. import initialization as init$", "from .._compat import initialization as init"),
    (r"^from \.\.\.audio_utils import conv1d_output_length$", "from .._compat import conv1d_output_length"),
    (r"^from \.\.\.configuration_utils import PreTrainedConfig$", "from .._compat import PreTrainedConfig"),
    (r"^from huggingface_hub\.dataclasses import strict$", "from .._compat import strict"),
    (r"^from \.\.\.modeling_utils import ", "from transformers.modeling_utils import "),
    (r"^from \.\.\.processing_utils import ", "from transformers.processing_utils import "),
    (r"^from \.\.\.utils\.import_utils import ", "from transformers.utils.import_utils import "),
    (r"^from \.\.\.utils import ", "from transformers.utils import "),
    (r"^from \.\.auto import ", "from transformers.models.auto import "),
]

# Symbols that exist in both versions but behave differently, so a targeted
# rewrite of the import line would break whenever upstream reorders it. Rebind
# them after the import block instead: the later binding wins.
OVERRIDES = [
    (
        "auto_docstring",
        "4.57.6's auto_docstring rejects classes absent from its registry",
    ),
]

HEADER = (
    "# Vendored from huggingface/transformers @ {ref}\n"
    "#   src/transformers/models/higgs_audio_v2_tokenizer/{name}.py\n"
    "# Apache License 2.0 - Copyright 2025 Boson AI and The HuggingFace Team.\n"
    "#\n"
    "# Do NOT edit by hand. Regenerate with: python scripts/sync-higgs-tokenizer.py\n"
    "# Only the imports are touched; the body is verbatim.\n\n"
)


def rewrite_imports(source: str) -> tuple[list[str], int]:
    lines, applied = [], 0
    for line in source.splitlines():
        for pattern, replacement in REWRITES:
            new = re.sub(pattern, replacement, line)
            if new != line:
                line, applied = new, applied + 1
                break
        lines.append(line)
    return lines, applied


def append_overrides(lines: list[str]) -> list[str]:
    """Rebind divergent symbols right after the module's import block."""
    last_import = max(
        (i for i, line in enumerate(lines) if re.match(r"^(from|import)\s", line)),
        default=None,
    )
    if last_import is None:
        return lines

    block = [""]
    for symbol, reason in OVERRIDES:
        if not any(re.search(rf"\b{symbol}\b", line) for line in lines[: last_import + 1]):
            continue
        block.append(f"# Override: {reason}.")
        block.append(f"from .._compat import {symbol}  # noqa: F811")

    if len(block) == 1:
        return lines
    return lines[: last_import + 1] + block + lines[last_import + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="main", help="transformers git ref to vendor from")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = UPSTREAM.format(ref=args.ref, name=name)
        with urllib.request.urlopen(url) as response:
            source = response.read().decode("utf-8")

        lines, applied = rewrite_imports(source)
        lines = append_overrides(lines)
        body = "\n".join(lines) + "\n"
        (DEST / f"{name}.py").write_text(HEADER.format(ref=args.ref, name=name) + body)
        print(f"{name}.py: {len(lines)} lines, {applied} imports rewritten")

    # A surviving relative import would resolve against our vendor package
    # instead of transformers and break in a confusing way. Fail loudly.
    for name in FILES:
        for lineno, line in enumerate((DEST / f"{name}.py").read_text().splitlines(), 1):
            if re.match(r"^from \.\.\.|^from \.\.[a-z]", line) and "_compat" not in line:
                raise SystemExit(f"unrewritten relative import at {name}.py:{lineno}: {line}")
    print("no unrewritten relative imports")


if __name__ == "__main__":
    main()
