# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = []
hiddenimports = ['backend', 'backend.backends', 'backend.backends.pytorch_backend', 'backend.config', 'backend.utils.audio', 'backend.utils.cache', 'backend.utils.progress', 'backend.utils.hf_progress', 'backend.utils.tasks', 'torch', 'transformers', 'qwen_tts', 'qwen_tts.inference', 'qwen_tts.inference.qwen3_tts_model', 'qwen_tts.inference.qwen3_tts_tokenizer', 'qwen_tts.core', 'qwen_tts.cli', 'pkg_resources.extern', 'fastapi', 'uvicorn', 'soundfile', 'numpy', 'librosa']
datas += collect_data_files('qwen_tts')
datas += copy_metadata('qwen-tts')
hiddenimports += collect_submodules('qwen_tts')
hiddenimports += collect_submodules('jaraco')


a = Analysis(
    ['main.py'],
    pathex=['/Users/jamespine/Projects/voicebox'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.utils.tensorboard', 'tensorboard', 'triton', 'torch._dynamo', 'torch._inductor', 'torch.utils.benchmark', 'IPython', 'matplotlib', 'PIL', 'cv2', 'torchvision', 'torchaudio'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='tts-provider-pytorch-cpu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
