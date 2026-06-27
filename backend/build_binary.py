"""
PyInstaller build script for creating standalone Python server binary.

Usage:
    python build_binary.py           # Build default (CPU) server binary
    python build_binary.py --cuda    # Build CUDA-enabled server binary
"""

import PyInstaller.__main__
import argparse
import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def is_apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build_server(cuda=False):
    """Build Python server as standalone binary.

    Args:
        cuda: If True, build with CUDA support and name the binary
              voicebox-server-cuda instead of voicebox-server.
    """
    backend_dir = Path(__file__).parent

    binary_name = "voicebox-server-cuda" if cuda else "voicebox-server"

    # PyInstaller arguments
    # CUDA builds use --onedir so we can split the output into two archives:
    #   1. Server core (~200-400MB) — versioned with the app
    #   2. CUDA libs (~2GB) — versioned independently (only redownloaded on
    #      CUDA toolkit / torch major version changes)
    # CPU builds remain --onefile for simplicity.
    pack_mode = "--onedir" if cuda else "--onefile"
    args = [
        "server.py",  # Use server.py as entry point instead of main.py
        pack_mode,
        "--name",
        binary_name,
    ]

    # Hide console window on Windows only. On macOS/Linux the sidecar needs
    # stdout/stderr for Tauri to capture logs.
    if platform.system() == "Windows":
        args.append("--noconsole")

    # numpy 2.x / torch ABI mismatch fix: install memmove fallback for
    # torch.from_numpy() before the app starts. Runtime hooks run after
    # FrozenImporter is registered so frozen torch/numpy are importable.
    # Paths are passed relative to backend_dir because os.chdir(backend_dir)
    # runs before PyInstaller. Absolute paths would get baked into the
    # generated .spec, breaking reproducible builds on other machines / CI.
    args.extend(
        [
            "--runtime-hook",
            "pyi_rth_numpy_compat.py",
            # Stub torch.compiler.disable before transformers imports
            # flex_attention, which otherwise triggers torch._dynamo →
            # torch._numpy._ufuncs and crashes at module load under
            # PyInstaller. See pyi_rth_torch_compiler_disable.py.
            "--runtime-hook",
            "pyi_rth_torch_compiler_disable.py",
            # Per-module collection overrides (e.g. forcing scipy.stats._distn_infrastructure
            # to bundle .py source alongside .pyc so the runtime hook can source-patch it).
            "--additional-hooks-dir",
            "pyi_hooks",
        ]
    )

    # Add local qwen_tts path if specified (for editable installs)
    qwen_tts_path = os.getenv("QWEN_TTS_PATH")
    if qwen_tts_path and Path(qwen_tts_path).exists():
        args.extend(["--paths", str(qwen_tts_path)])
        logger.info("Using local qwen_tts source from: %s", qwen_tts_path)

    # Add common hidden imports
    args.extend(
        [
            "--hidden-import",
            "backend",
            "--hidden-import",
            "backend.main",
            "--hidden-import",
            "backend.config",
            "--hidden-import",
            "backend.database",
            "--hidden-import",
            "backend.models",
            "--hidden-import",
            "backend.services.profiles",
            "--hidden-import",
            "backend.services.history",
            "--hidden-import",
            "backend.services.tts",
            "--hidden-import",
            "backend.services.transcribe",
            "--hidden-import",
            "backend.utils.platform_detect",
            "--hidden-import",
            "backend.backends",
            "--hidden-import",
            "backend.backends.pytorch_backend",
            "--hidden-import",
            "backend.backends.qwen_custom_voice_backend",
            "--hidden-import",
            "backend.utils.audio",
            "--hidden-import",
            "backend.utils.cache",
            "--hidden-import",
            "backend.utils.progress",
            "--hidden-import",
            "backend.utils.hf_progress",
            "--hidden-import",
            "backend.services.cuda",
            "--hidden-import",
            "backend.services.effects",
            "--hidden-import",
            "backend.utils.effects",
            "--hidden-import",
            "backend.services.versions",
            "--hidden-import",
            "pedalboard",
            "--hidden-import",
            "chatterbox",
            "--hidden-import",
            "chatterbox.tts_turbo",
            "--hidden-import",
            "chatterbox.mtl_tts",
            "--hidden-import",
            "backend.backends.chatterbox_backend",
            "--hidden-import",
            "backend.backends.chatterbox_turbo_backend",
            # chatterbox multilingual uses spacy_pkuseg for Chinese word
            # segmentation, which ships pickled dict files (dicts/default.pkl)
            # and native .so extensions that --hidden-import alone won't bundle.
            "--collect-all",
            "spacy_pkuseg",
            "--hidden-import",
            "backend.backends.luxtts_backend",
            "--hidden-import",
            "zipvoice",
            "--hidden-import",
            "zipvoice.luxvoice",
            "--collect-all",
            "zipvoice",
            "--collect-all",
            "linacodec",
            "--hidden-import",
            "torch",
            "--hidden-import",
            "transformers",
            # Parakeet STT modeling — AutoModelForTDT resolves this
            # dynamically, so collect it explicitly so the frozen build can
            # load parakeet-tdt variants.
            "--collect-submodules",
            "transformers.models.parakeet",
            "--hidden-import",
            "fastapi",
            "--hidden-import",
            "uvicorn",
            "--hidden-import",
            "sqlalchemy",
            # librosa uses lazy_loader which generates .pyi stub files at
            # install time and reads them at runtime to discover submodules.
            # --hidden-import alone doesn't bundle the stubs, causing
            # "Cannot load imports from non-existent stub" at runtime.
            "--collect-all",
            "lazy_loader",
            "--collect-all",
            "librosa",
            "--hidden-import",
            "soundfile",
            "--hidden-import",
            "qwen_tts",
            "--hidden-import",
            "qwen_tts.inference",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_model",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_tokenizer",
            "--hidden-import",
            "qwen_tts.core",
            "--hidden-import",
            "qwen_tts.cli",
            "--copy-metadata",
            "qwen-tts",
            "--copy-metadata",
            "requests",
            "--copy-metadata",
            "transformers",
            "--copy-metadata",
            "huggingface-hub",
            "--copy-metadata",
            "tokenizers",
            "--copy-metadata",
            "safetensors",
            "--copy-metadata",
            "tqdm",
            "--hidden-import",
            "requests",
            # qwen_tts uses inspect.getsource() at runtime to locate
            # modeling_qwen3_tts.py — needs physical .py source files bundled
            "--collect-all",
            "qwen_tts",
            # Fix for pkg_resources and jaraco namespace packages
            "--hidden-import",
            "pkg_resources.extern",
            "--collect-submodules",
            "jaraco",
            # inflect uses typeguard @typechecked which calls inspect.getsource()
            # at import time — needs .py source files, not just .pyc bytecode
            "--collect-all",
            "inflect",
            # perth ships pretrained watermark model files (hparams.yaml, .pth.tar)
            # in perth/perth_net/pretrained/ — needed by chatterbox at runtime
            "--collect-all",
            "perth",
            # piper_phonemize ships espeak-ng-data/ (phoneme tables, language dicts)
            # needed by LuxTTS for text-to-phoneme conversion
            "--collect-all",
            "piper_phonemize",
            # HumeAI TADA — speech-language model using Llama + flow matching
            "--hidden-import",
            "backend.backends.hume_backend",
            "--hidden-import",
            "tada",
            "--hidden-import",
            "tada.modules",
            "--hidden-import",
            "tada.modules.tada",
            "--hidden-import",
            "tada.modules.encoder",
            "--hidden-import",
            "tada.modules.decoder",
            "--hidden-import",
            "tada.modules.aligner",
            "--hidden-import",
            "tada.modules.acoustic_spkr_verf",
            "--hidden-import",
            "tada.nn",
            "--hidden-import",
            "tada.nn.vibevoice",
            "--hidden-import",
            "tada.utils",
            "--hidden-import",
            "tada.utils.gray_code",
            "--hidden-import",
            "tada.utils.text",
            # DAC shim — provides dac.nn.layers.Snake1d without the real
            # descript-audio-codec package (which pulls onnx/tensorboard via
            # descript-audiotools). The shim is in backend/utils/dac_shim.py.
            "--hidden-import",
            "backend.utils.dac_shim",
            "--hidden-import",
            "torchaudio",
            "--collect-submodules",
            "tada",
            # Kokoro 82M — lightweight TTS engine using misaki G2P
            # collect-all is required because transformers introspects .py source
            # files at runtime (e.g. _can_set_attn_implementation opens the class
            # file); hidden-import alone only bundles bytecode.
            "--hidden-import",
            "backend.backends.kokoro_backend",
            "--collect-all",
            "kokoro",
            # misaki ships G2P data files (dictionaries, phoneme tables)
            # that must be bundled for espeak/en/ja/zh G2P to work
            "--collect-all",
            "misaki",
            # language_tags ships JSON data files (index.json etc.) loaded at
            # runtime via: misaki → phonemizer → segments → csvw → language_tags
            "--collect-all",
            "language_tags",
            # espeakng_loader ships the entire espeak-ng-data directory (369 files)
            # loaded at import time by misaki.espeak via get_data_path()
            "--collect-all",
            "espeakng_loader",
            # spacy en_core_web_sm model — misaki.en tries to spacy.cli.download()
            # at runtime if not found, which calls pip as a subprocess and crashes
            # the frozen binary. Bundle the model so spacy.util.is_package() passes.
            "--collect-all",
            "en_core_web_sm",
            "--copy-metadata",
            "en_core_web_sm",
            "--hidden-import",
            "en_core_web_sm",
            # unidic-lite ships the MeCab dictionary used by fugashi (pulled in
            # by misaki[ja]). The dict lives in unidic_lite/dicdir/ and is
            # discovered via the package's DICDIR constant, so the data files
            # must be collected or Japanese Kokoro voices crash at runtime.
            "--collect-all",
            "unidic_lite",
            "--hidden-import",
            "loguru",
            # MCP server — Streamable-HTTP endpoint and the 4 voicebox.* tools.
            # FastMCP pulls in a chain of deps (mcp, cyclopts, openapi-pydantic,
            # etc.) that don't auto-discover cleanly under PyInstaller, so we
            # collect them whole. Small compared to torch.
            "--hidden-import",
            "backend.mcp_server",
            "--hidden-import",
            "backend.mcp_server.server",
            "--hidden-import",
            "backend.mcp_server.tools",
            "--hidden-import",
            "backend.mcp_server.context",
            "--hidden-import",
            "backend.mcp_server.resolve",
            "--hidden-import",
            "backend.mcp_server.events",
            "--collect-all",
            "fastmcp",
            "--collect-all",
            "mcp",
            "--hidden-import",
            "sse_starlette",
        ]
    )

    # Add CUDA-specific hidden imports
    if cuda:
        logger.info("Building with CUDA support")
        args.extend(
            [
                "--hidden-import",
                "torch.cuda",
                "--hidden-import",
                "torch.backends.cudnn",
            ]
        )
    else:
        # Exclude NVIDIA CUDA packages from CPU-only builds to keep binary small.
        # When building from a venv with CUDA torch installed, PyInstaller would
        # bundle ~3GB of NVIDIA shared libraries. We exclude both the Python
        # modules and the binary DLLs.
        nvidia_packages = [
            "nvidia",
            "nvidia.cublas",
            "nvidia.cuda_cupti",
            "nvidia.cuda_nvrtc",
            "nvidia.cuda_runtime",
            "nvidia.cudnn",
            "nvidia.cufft",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
            "nvidia.nccl",
            "nvidia.nvjitlink",
            "nvidia.nvtx",
        ]
        for pkg in nvidia_packages:
            args.extend(["--exclude-module", pkg])

    # Add MLX-specific imports if building on Apple Silicon (never for CUDA builds)
    if is_apple_silicon() and not cuda:
        logger.info("Building for Apple Silicon - including MLX dependencies")
        args.extend(
            [
                "--hidden-import",
                "backend.backends.mlx_backend",
                "--hidden-import",
                "mlx",
                "--hidden-import",
                "mlx.core",
                "--hidden-import",
                "mlx.nn",
                "--hidden-import",
                "mlx_audio",
                "--hidden-import",
                "mlx_audio.tts",
                "--hidden-import",
                "mlx_audio.stt",
                "--hidden-import",
                "mlx_lm",
                "--hidden-import",
                "backend.backends.qwen_llm_backend",
                "--collect-submodules",
                "mlx",
                "--collect-submodules",
                "mlx_audio",
                "--collect-submodules",
                "mlx_lm",
                # Use --collect-all so PyInstaller bundles both data files AND
                # native shared libraries (.dylib, .metallib) for MLX.
                # Previously only --collect-data was used, which caused MLX to
                # raise OSError at runtime inside the bundled binary because
                # the Metal shader libraries were missing.
                "--collect-all",
                "mlx",
                "--collect-all",
                "mlx_audio",
                # mlx_lm ships chat_templates/ JSON files and loads tool_parsers
                # submodules dynamically via importlib at tokenizer load time,
                # which --hidden-import alone can't resolve.
                "--collect-all",
                "mlx_lm",
            ]
        )
    elif not cuda:
        logger.info("Building for non-Apple Silicon platform - PyTorch only")

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")

    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    # Change to backend directory
    os.chdir(backend_dir)

    # For CPU builds on Windows, ensure we're using CPU-only torch.
    # If CUDA torch is installed (local dev), swap to CPU torch before building,
    # then restore CUDA torch after. This prevents PyInstaller from bundling
    # ~3GB of CUDA DLLs into the CPU binary.
    restore_cuda = False
    if not cuda and platform.system() == "Windows":
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"], capture_output=True, text=True
        )
        has_cuda_torch = bool(result.stdout.strip())
        if has_cuda_torch:
            logger.info("CUDA torch detected — installing CPU torch for CPU build...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                    "--force-reinstall",
                    "-q",
                ],
                check=True,
            )
            restore_cuda = True

    # Run PyInstaller
    try:
        PyInstaller.__main__.run(args)
    finally:
        # Restore CUDA torch if we swapped it out (even on build failure)
        if restore_cuda:
            logger.info("Restoring CUDA torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                    "--force-reinstall",
                    "-q",
                ],
                check=True,
            )

    logger.info("Binary built in %s", backend_dir / "dist" / binary_name)


def build_shim():
    """Build the voicebox-mcp stdio shim as a tiny standalone binary.

    This is the bridge for MCP clients that only speak stdio — it proxies
    JSON-RPC to the main voicebox-server's /mcp endpoint. Keep it small: no
    torch, no ML deps, just httpx + asyncio.
    """
    backend_dir = Path(__file__).parent

    args = [
        "mcp_shim/__main__.py",
        "--onefile",
        "--name",
        "voicebox-mcp",
        # Stdio-only — no console hiding needed on Windows since the parent
        # MCP client is spawning this as a child process and wants stdio.
        "--hidden-import",
        "backend.mcp_shim",
        "--hidden-import",
        "backend.mcp_shim.__main__",
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "httpx._transports.default",
        "--hidden-import",
        "anyio",
        # Exclude everything heavy that httpx/asyncio don't actually need so
        # the binary stays tiny (~15 MB instead of ~400 MB).
        "--exclude-module",
        "torch",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "mlx",
        "--exclude-module",
        "mlx_audio",
        "--exclude-module",
        "mlx_lm",
        "--exclude-module",
        "qwen_tts",
        "--exclude-module",
        "chatterbox",
        "--exclude-module",
        "zipvoice",
        "--exclude-module",
        "tada",
        "--exclude-module",
        "kokoro",
        "--exclude-module",
        "misaki",
        "--exclude-module",
        "spacy",
        "--exclude-module",
        "librosa",
        "--exclude-module",
        "numba",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "pedalboard",
        "--exclude-module",
        "fastapi",
        "--exclude-module",
        "uvicorn",
        "--exclude-module",
        "sqlalchemy",
        "--exclude-module",
        "fastmcp",
        "--exclude-module",
        "mcp",
    ]

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")
    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    os.chdir(backend_dir)
    PyInstaller.__main__.run(args)
    logger.info("Shim built: %s", backend_dir / "dist" / "voicebox-mcp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build voicebox binaries")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Build CUDA-enabled binary (voicebox-server-cuda)",
    )
    parser.add_argument(
        "--shim",
        action="store_true",
        help="Build the voicebox-mcp stdio shim binary instead of the server",
    )
    cli_args = parser.parse_args()
    if cli_args.shim:
        build_shim()
    else:
        build_server(cuda=cli_args.cuda)
