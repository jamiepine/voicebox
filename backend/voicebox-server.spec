# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = []
hiddenimports = ['backend', 'backend.main', 'backend.config', 'backend.database', 'backend.models', 'backend.profiles', 'backend.history', 'backend.tts', 'backend.transcribe', 'backend.platform_detect', 'backend.backends', 'backend.backends.pytorch_backend', 'backend.utils.audio', 'backend.utils.cache', 'backend.utils.progress', 'backend.utils.hf_progress', 'backend.utils.validation', 'torch', 'transformers', 'fastapi', 'uvicorn', 'sqlalchemy', 'librosa', 'soundfile', 'qwen_tts', 'qwen_tts.inference', 'qwen_tts.inference.qwen3_tts_model', 'qwen_tts.inference.qwen3_tts_tokenizer', 'qwen_tts.core', 'qwen_tts.cli', 'pkg_resources.extern', 'backend.backends.mlx_backend', 'mlx', 'mlx.core', 'mlx.nn', 'mlx_audio', 'mlx_audio.tts', 'mlx_audio.stt']
datas += collect_data_files('qwen_tts')
# Use collect_all (not collect_data_files) so native .dylib and .metallib
# files are bundled as binaries, not data. Without this, MLX raises OSError
# when loading Metal shaders inside the PyInstaller bundle.
from PyInstaller.utils.hooks import collect_all as _collect_all
_mlx_datas, _mlx_bins, _mlx_hidden = _collect_all('mlx')
_mlxa_datas, _mlxa_bins, _mlxa_hidden = _collect_all('mlx_audio')
datas += _mlx_datas + _mlxa_datas
datas += copy_metadata('qwen-tts')
hiddenimports += collect_submodules('qwen_tts')
hiddenimports += collect_submodules('jaraco')
hiddenimports += collect_submodules('mlx')
hiddenimports += collect_submodules('mlx_audio')


a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=_mlx_bins + _mlxa_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='voicebox-server',
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
