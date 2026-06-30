# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []
hiddenimports = ['backend', 'backend.main', 'backend.config', 'backend.database', 'backend.models', 'backend.services.profiles', 'backend.services.history', 'backend.services.tts', 'backend.services.transcribe', 'backend.utils.platform_detect', 'backend.backends', 'backend.backends.pytorch_backend', 'backend.backends.qwen_custom_voice_backend', 'backend.utils.audio', 'backend.utils.cache', 'backend.utils.progress', 'backend.utils.hf_progress', 'backend.services.cuda', 'backend.services.effects', 'backend.utils.effects', 'backend.services.versions', 'pedalboard', 'chatterbox', 'chatterbox.tts_turbo', 'chatterbox.mtl_tts', 'backend.backends.chatterbox_backend', 'backend.backends.chatterbox_turbo_backend', 'backend.backends.luxtts_backend', 'zipvoice', 'zipvoice.luxvoice', 'torch', 'transformers', 'fastapi', 'uvicorn', 'sqlalchemy', 'soundfile', 'qwen_tts', 'qwen_tts.inference', 'qwen_tts.inference.qwen3_tts_model', 'qwen_tts.inference.qwen3_tts_tokenizer', 'qwen_tts.core', 'qwen_tts.cli', 'requests', 'pkg_resources.extern', 'backend.backends.hume_backend', 'tada', 'tada.modules', 'tada.modules.tada', 'tada.modules.encoder', 'tada.modules.decoder', 'tada.modules.aligner', 'tada.modules.acoustic_spkr_verf', 'tada.nn', 'tada.nn.vibevoice', 'tada.utils', 'tada.utils.gray_code', 'tada.utils.text', 'backend.utils.dac_shim', 'torchaudio', 'backend.backends.kokoro_backend', 'en_core_web_sm', 'loguru', 'backend.mcp_server', 'backend.mcp_server.server', 'backend.mcp_server.tools', 'backend.mcp_server.context', 'backend.mcp_server.resolve', 'backend.mcp_server.events', 'sse_starlette', 'backend.backends.mlx_backend', 'mlx', 'mlx.core', 'mlx.nn', 'mlx_audio', 'mlx_audio.tts', 'mlx_audio.stt', 'mlx_lm', 'backend.backends.qwen_llm_backend']
datas += copy_metadata('qwen-tts')
datas += copy_metadata('requests')
datas += copy_metadata('transformers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('tokenizers')
datas += copy_metadata('safetensors')
datas += copy_metadata('tqdm')
datas += copy_metadata('en_core_web_sm')
hiddenimports += collect_submodules('jaraco')
hiddenimports += collect_submodules('tada')
hiddenimports += collect_submodules('mlx')
hiddenimports += collect_submodules('mlx_audio')
hiddenimports += collect_submodules('mlx_lm')
tmp_ret = collect_all('spacy_pkuseg')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('zipvoice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('linacodec')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lazy_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('librosa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('qwen_tts')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('inflect')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('perth')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('piper_phonemize')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('kokoro')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('misaki')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('language_tags')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('espeakng_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('en_core_web_sm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('unidic_lite')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fastmcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_audio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_lm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyi_hooks'],
    hooksconfig={},
    runtime_hooks=['pyi_rth_numpy_compat.py', 'pyi_rth_torch_compiler_disable.py'],
    excludes=['nvidia', 'nvidia.cublas', 'nvidia.cuda_cupti', 'nvidia.cuda_nvrtc', 'nvidia.cuda_runtime', 'nvidia.cudnn', 'nvidia.cufft', 'nvidia.curand', 'nvidia.cusolver', 'nvidia.cusparse', 'nvidia.nccl', 'nvidia.nvjitlink', 'nvidia.nvtx'],
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
