"""
Regression guard for #1264: the frozen Windows build must bundle onnx.libs.

onnx's Windows wheel keeps the msvcp140-<hash>.dll that onnx_cpp2py_export.pyd
needs in site-packages/onnx.libs/. backend/pyi_hooks/hook-onnx.py collects it,
and both build_binary.py and the committed spec must point PyInstaller at that
hooks directory.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from build_binary import build_server  # noqa: E402


def test_onnx_hook_collects_delvewheel_libs():
    tree = ast.parse((BACKEND_DIR / "pyi_hooks" / "hook-onnx.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "collect_delvewheel_libs_directory"
    ]
    assert [ast.literal_eval(call.args[0]) for call in calls] == ["onnx"]


def test_cuda_build_uses_pyi_hooks_dir():
    with (
        patch("build_binary.PyInstaller.__main__.run") as mock_run,
        patch("build_binary.platform.system", return_value="Linux"),
        patch("build_binary.os.chdir"),
    ):
        build_server(cuda=True)
    args = mock_run.call_args[0][0]
    assert args[args.index("--additional-hooks-dir") + 1] == "pyi_hooks"


def test_committed_spec_uses_pyi_hooks_dir():
    tree = ast.parse((BACKEND_DIR / "voicebox-server.spec").read_text())
    hookspath = next(
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Analysis"
        for kw in node.keywords
        if kw.arg == "hookspath"
    )
    assert "pyi_hooks" in ast.literal_eval(hookspath)
