"""
PyInstaller build script for creating standalone Python server binary.
"""

import PyInstaller.__main__
import os
import platform
from pathlib import Path


def is_apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build_server():
    """Build Python server as standalone binary."""
    backend_dir = Path(__file__).parent

    # PyInstaller arguments
    args = [
        'server.py',  # Use server.py as entry point instead of main.py
        '--onefile',
        '--name', 'voicebox-server',
    ]

    # Add local qwen_tts path if specified (for editable installs)
    qwen_tts_path = os.getenv('QWEN_TTS_PATH')
    if qwen_tts_path and Path(qwen_tts_path).exists():
        args.extend(['--paths', str(qwen_tts_path)])
        print(f"Using local qwen_tts source from: {qwen_tts_path}")

    # Add common hidden imports
    args.extend([
        '--hidden-import', 'backend',
        '--hidden-import', 'backend.main',
        '--hidden-import', 'backend.config',
        '--hidden-import', 'backend.database',
        '--hidden-import', 'backend.models',
        '--hidden-import', 'backend.profiles',
        '--hidden-import', 'backend.history',
        '--hidden-import', 'backend.tts',
        '--hidden-import', 'backend.transcribe',
        '--hidden-import', 'backend.platform_detect',
        '--hidden-import', 'backend.backends',
        '--hidden-import', 'backend.backends.pytorch_backend',
        '--hidden-import', 'backend.utils.audio',
        '--hidden-import', 'backend.utils.cache',
        '--hidden-import', 'backend.utils.progress',
        '--hidden-import', 'backend.utils.hf_progress',
        '--hidden-import', 'backend.utils.validation',
        '--hidden-import', 'torch',
        '--hidden-import', 'transformers',
        '--hidden-import', 'fastapi',
        '--hidden-import', 'uvicorn',
        '--hidden-import', 'sqlalchemy',
        '--hidden-import', 'librosa',
        '--hidden-import', 'soundfile',
        '--hidden-import', 'qwen_tts',
        '--hidden-import', 'qwen_tts.inference',
        '--hidden-import', 'qwen_tts.inference.qwen3_tts_model',
        '--hidden-import', 'qwen_tts.inference.qwen3_tts_tokenizer',
        '--hidden-import', 'qwen_tts.core',
        '--hidden-import', 'qwen_tts.cli',
        '--copy-metadata', 'qwen-tts',
        '--collect-submodules', 'qwen_tts',
        '--collect-data', 'qwen_tts',
        # Fix for pkg_resources and jaraco namespace packages
        '--hidden-import', 'pkg_resources.extern',
        '--collect-submodules', 'jaraco',
    ])

    # Add MLX-specific imports if building on Apple Silicon
    if is_apple_silicon():
        print("Building for Apple Silicon - including MLX dependencies")
        args.extend([
            '--hidden-import', 'backend.backends.mlx_backend',
            '--hidden-import', 'mlx',
            '--hidden-import', 'mlx.core',
            '--hidden-import', 'mlx.nn',
            '--hidden-import', 'mlx_audio',
            '--hidden-import', 'mlx_audio.tts',
            '--hidden-import', 'mlx_audio.stt',
            '--collect-submodules', 'mlx',
            '--collect-submodules', 'mlx_audio',
            # Collect MLX data files including Metal shader libraries (.metallib)
            '--collect-data', 'mlx',
            '--collect-data', 'mlx_audio',
        ])
    else:
        print("Building for non-Apple Silicon platform - PyTorch only")

    args.extend([
        '--noconfirm',
        '--clean',
    ])

    # Change to backend directory
    os.chdir(backend_dir)
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print(f"Binary built in {backend_dir / 'dist' / 'voicebox-server'}")


if __name__ == '__main__':
    build_server()
