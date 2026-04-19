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

            _orig = torch.from_numpy

            # Explicit numpy → torch dtype map. Silent fallback to float32 on
            # unknown dtypes would reinterpret the memcpy'd bytes as fp32 and
            # silently corrupt data (e.g. fp16 tensors from some TTS engines),
            # so we raise instead.
            dtype_map = {
                "float16": _t.float16,
                "float32": _t.float32,
                "float64": _t.float64,
                "int8": _t.int8,
                "int16": _t.int16,
                "int32": _t.int32,
                "int64": _t.int64,
                "uint8": _t.uint8,
                "bool": _t.bool,
                "complex64": _t.complex64,
                "complex128": _t.complex128,
            }

            def _safe_from_numpy(
                arr, _orig=_orig, _c=ctypes, _np=np, _t=torch, _map=dtype_map
            ):
                try:
                    return _orig(arr)
                except RuntimeError:
                    a = _np.ascontiguousarray(arr)
                    key = str(a.dtype)
                    if key not in _map:
                        raise TypeError(
                            f"pyi_rth_numpy_compat: unsupported numpy dtype "
                            f"{key!r} in torch.from_numpy fallback; add an "
                            f"explicit mapping rather than silently copying "
                            f"bytes into the wrong dtype."
                        )
                    out = _t.empty(list(a.shape), dtype=_map[key])
                    _c.memmove(out.data_ptr(), a.ctypes.data, a.nbytes)
                    return out

            torch.from_numpy = _safe_from_numpy
            torch._vb_from_numpy_patched = True
        except Exception:
            pass
        return


threading.Thread(target=_patch_torch_from_numpy, daemon=True).start()
