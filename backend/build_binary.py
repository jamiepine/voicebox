"""
PyInstaller build script for creating standalone Python server binary.
"""

import PyInstaller.__main__
import os
from pathlib import Path


def build_server():
    """Build Python server as standalone binary."""
    backend_dir = Path(__file__).parent

    # Check for local editable qwen_tts install
    local_qwen_path = Path.home() / 'Projects' / 'voice' / 'Qwen3-TTS'

    # PyInstaller arguments
    args = [
        'server.py',  # Use server.py as entry point instead of main.py
        '--onefile',
        '--name', 'voicebox-server',
    ]

    # Add local qwen_tts path if it exists (for editable installs)
    if local_qwen_path.exists():
        args.extend(['--paths', str(local_qwen_path)])
        print(f"Using local qwen_tts source from: {local_qwen_path}")

    # Exclude unnecessary modules to reduce size
    args.extend([
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'IPython',
        '--exclude-module', 'notebook',
        '--exclude-module', 'pytest',
        # Don't exclude setuptools - pkg_resources needs it
        '--exclude-module', 'torch.distributions',
        '--exclude-module', 'torch.testing',
        '--exclude-module', 'tensorboard',
        '--exclude-module', 'torch.utils.tensorboard',
        '--exclude-module', 'scipy',
        '--exclude-module', 'PIL',
        '--exclude-module', 'tkinter',
        '--exclude-module', 'unittest',
        '--exclude-module', 'test',
        '--exclude-module', 'sklearn',
        '--exclude-module', 'pandas',
        '--exclude-module', 'torchaudio',
    ])

    # Add hidden imports
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
        '--hidden-import', 'jaraco',
        '--hidden-import', 'jaraco.text',
        '--hidden-import', 'jaraco.functools',
        '--hidden-import', 'jaraco.context',
        '--collect-submodules', 'jaraco',
        '--copy-metadata', 'jaraco.text',
        '--copy-metadata', 'jaraco.functools',
        '--copy-metadata', 'jaraco.context',
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
