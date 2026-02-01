"""
PyInstaller build script for PyTorch CPU provider.
"""

import PyInstaller.__main__
import os
import platform
from pathlib import Path


def build_provider():
    """Build PyTorch CPU provider as standalone binary."""
    provider_dir = Path(__file__).parent
    backend_dir = provider_dir.parent.parent / "backend"
    
    # PyInstaller arguments
    args = [
        'main.py',
        '--onefile',
        '--name', 'tts-provider-pytorch-cpu',
    ]
    
    # Add backend to path
    args.extend([
        '--paths', str(backend_dir.parent),
    ])
    
    # Add hidden imports
    args.extend([
        '--hidden-import', 'backend',
        '--hidden-import', 'backend.backends',
        '--hidden-import', 'backend.backends.pytorch_backend',
        '--hidden-import', 'backend.config',
        '--hidden-import', 'backend.utils.audio',
        '--hidden-import', 'backend.utils.cache',
        '--hidden-import', 'backend.utils.progress',
        '--hidden-import', 'backend.utils.hf_progress',
        '--hidden-import', 'backend.utils.tasks',
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
        '--hidden-import', 'pkg_resources.extern',
        '--collect-submodules', 'jaraco',
        '--hidden-import', 'fastapi',
        '--hidden-import', 'uvicorn',
        '--hidden-import', 'soundfile',
        '--hidden-import', 'numpy',
        '--hidden-import', 'librosa',
    ])
    
    # Exclude large unused modules to reduce binary size
    args.extend([
        '--exclude-module', 'torch.utils.tensorboard',
        '--exclude-module', 'tensorboard',
        '--exclude-module', 'triton',
        '--exclude-module', 'torch._dynamo',
        '--exclude-module', 'torch._inductor',
        '--exclude-module', 'torch.testing',
        '--exclude-module', 'torch.utils.benchmark',
        '--exclude-module', 'IPython',
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'PIL',
        '--exclude-module', 'cv2',
        '--exclude-module', 'torchvision',
        '--exclude-module', 'torchaudio',
    ])
    
    args.extend([
        '--noconfirm',
        '--clean',
    ])
    
    # Change to provider directory
    os.chdir(provider_dir)
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    binary_name = 'tts-provider-pytorch-cpu'
    if platform.system() == "Windows":
        binary_name += '.exe'
    
    print(f"Binary built in {provider_dir / 'dist' / binary_name}")


if __name__ == '__main__':
    build_provider()
