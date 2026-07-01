"""
PyInstaller runtime hook: numpy 2.x / torch ABI mismatch fix.

Problem
-------
torch is compiled against numpy 1.x headers. numpy 2.x changed the version
number returned by PyArray_GetNDArrayCVersion() (0x01000009 → 0x02000000),
so torch's is_numpy_available() returns False and every torch.from_numpy()
call raises:

    RuntimeError: Numpy is not available

This surfaces as:

    ValueError: Unable to create tensor, you should probably activate
    padding with 'padding=True'

during TTS generation (EncodecFeatureExtractor → BatchFeature.convert_to_tensors).

Fix
---
Runtime hooks execute after PyInstaller's FrozenImporter is registered, so
frozen torch/numpy are importable here. We start a background thread that
waits for torch to finish loading then wraps torch.from_numpy with a ctypes
memmove fallback that bypasses the C-level numpy ABI check entirely.

This approach works with any numpy version and is safer than binary-patching
libtorch_python.dylib (which risks PyArray_Descr struct layout mismatches).
"""

import sys
import threading


def _install_torch_numpy_fallback(torch, np, ctypes):
    """Install a torch.from_numpy fallback that works when torch rejects numpy."""
    if getattr(torch, "_vb_from_numpy_patched", False):
        return False

    orig_from_numpy = torch.from_numpy

    # Explicit numpy -> torch dtype map. Silent fallback to float32 on
    # unknown dtypes would reinterpret the memcpy'd bytes as fp32 and
    # silently corrupt data (e.g. fp16 tensors from some TTS engines),
    # so we raise instead.
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    def _safe_from_numpy(arr, *args, _orig=orig_from_numpy, _c=ctypes, _np=np, _t=torch, _map=dtype_map, **kwargs):
        try:
            return _orig(arr, *args, **kwargs)
        except RuntimeError as err:
            a = _np.ascontiguousarray(arr)
            key = str(a.dtype)
            if key not in _map:
                raise TypeError(
                    f"pyi_rth_numpy_compat: unsupported numpy dtype "
                    f"{key!r} in torch.from_numpy fallback; add an "
                    f"explicit mapping rather than silently copying "
                    f"bytes into the wrong dtype."
                ) from err
            out = _t.empty(list(a.shape), dtype=_map[key])
            _c.memmove(out.data_ptr(), a.ctypes.data, a.nbytes)
            return out

    torch.from_numpy = _safe_from_numpy
    torch._vb_from_numpy_patched = True
    return True


def _patch_torch_from_numpy():
    import time

    for _ in range(7200):  # poll up to 360 s at 50 ms intervals
        time.sleep(0.05)
        torch = sys.modules.get("torch")
        if torch is None or not hasattr(torch, "from_numpy"):
            continue
        if getattr(torch, "_vb_from_numpy_patched", False):
            return
        try:
            import ctypes
            import numpy as np

            _install_torch_numpy_fallback(torch, np, ctypes)
        except Exception:
            pass
        return


threading.Thread(target=_patch_torch_from_numpy, daemon=True).start()
