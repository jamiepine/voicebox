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

    # Add common hidden imports (always included)
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
        '--hidden-import', 'backend.providers',
        '--hidden-import', 'backend.providers.base',
        '--hidden-import', 'backend.providers.bundled',
        '--hidden-import', 'backend.providers.types',
        '--hidden-import', 'backend.utils.audio',
        '--hidden-import', 'backend.utils.cache',
        '--hidden-import', 'backend.utils.progress',
        '--hidden-import', 'backend.utils.hf_progress',
        '--hidden-import', 'backend.utils.validation',
        '--hidden-import', 'numpy',
        '--hidden-import', 'numpy.core',
        '--hidden-import', 'numpy.core._multiarray_umath',
        '--hidden-import', 'scipy',
        '--hidden-import', 'scipy.signal',
        '--hidden-import', 'fastapi',
        '--hidden-import', 'uvicorn',
        '--hidden-import', 'sqlalchemy',
        '--hidden-import', 'librosa',
        '--hidden-import', 'soundfile',
        # Fix for pkg_resources and jaraco namespace packages
        '--hidden-import', 'pkg_resources.extern',
        '--collect-submodules', 'jaraco',
        # Asyncio and threading support for PyInstaller
        '--hidden-import', 'asyncio',
        '--hidden-import', 'asyncio.subprocess',
        '--hidden-import', 'concurrent.futures',
        '--hidden-import', 'concurrent.futures.thread',
    ])

    # Platform-specific TTS backend handling
    system = platform.system()

    if is_apple_silicon():
        print("Building for Apple Silicon - including MLX dependencies (bundled)")
        args.extend([
            '--hidden-import', 'backend.backends',
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
    elif system == "Windows" or (system == "Darwin" and not is_apple_silicon()):
        # Windows and Intel macOS: Bundle PyTorch CPU provider
        print(f"Building for {system} - including PyTorch CPU provider (bundled)")
        args.extend([
            '--hidden-import', 'backend.backends',
            '--hidden-import', 'backend.backends.pytorch_backend',
            '--hidden-import', 'torch',
            '--hidden-import', 'transformers',
            '--hidden-import', 'qwen_tts',
            '--hidden-import', 'qwen_tts.inference',
            '--hidden-import', 'qwen_tts.inference.qwen3_tts_model',
            '--hidden-import', 'qwen_tts.inference.qwen3_tts_tokenizer',
            '--hidden-import', 'qwen_tts.core',
            '--hidden-import', 'qwen_tts.cli',
            '--copy-metadata', 'qwen-tts',
            '--collect-submodules', 'qwen_tts',
            '--collect-data', 'qwen_tts',
        ])
    else:
        # Linux: No bundled provider - users download providers separately
        print("Building for Linux - no bundled provider (users download separately)")
        args.extend([
            '--hidden-import', 'backend.backends',
            '--hidden-import', 'backend.backends.pytorch_backend',
        ])

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
