"""Regression tests for the PyInstaller numpy/torch compatibility hook."""

import ctypes

import pytest

from backend.pyi_rth_numpy_compat import _install_torch_numpy_fallback


class _FakeArray:
    def __init__(self, values, dtype="float32"):
        self.dtype = dtype
        self.shape = (len(values),)
        self._buffer = (ctypes.c_float * len(values))(*values)
        self.nbytes = ctypes.sizeof(self._buffer)
        self.ctypes = type("_Ctypes", (), {"data": ctypes.addressof(self._buffer)})()

    def values(self):
        return list(self._buffer)


class _FakeNP:
    @staticmethod
    def ascontiguousarray(arr):
        return arr


class _FakeTensor:
    def __init__(self, shape):
        self._buffer = (ctypes.c_float * shape[0])()

    def data_ptr(self):
        return ctypes.addressof(self._buffer)

    def values(self):
        return list(self._buffer)


class _FakeTorch:
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    bool = "bool"
    complex64 = "complex64"
    complex128 = "complex128"

    def __init__(self):
        self._vb_from_numpy_patched = False
        self.empty_calls = []
        self.last_tensor = None

    def from_numpy(self, _arr):
        raise RuntimeError("Numpy is not available")

    def empty(self, shape, dtype):
        self.empty_calls.append((shape, dtype))
        self.last_tensor = _FakeTensor(shape)
        return self.last_tensor


def test_install_torch_numpy_fallback_copies_bytes_after_runtime_error():
    fake_torch = _FakeTorch()
    arr = _FakeArray([1.0, 2.0, 3.0, 4.0])

    installed = _install_torch_numpy_fallback(fake_torch, _FakeNP, ctypes)
    out = fake_torch.from_numpy(arr)

    assert installed is True
    assert fake_torch._vb_from_numpy_patched is True
    assert out is fake_torch.last_tensor
    assert fake_torch.empty_calls == [([4], "float32")]
    assert out.values() == arr.values()


def test_install_torch_numpy_fallback_is_idempotent():
    fake_torch = _FakeTorch()

    assert _install_torch_numpy_fallback(fake_torch, _FakeNP, ctypes) is True
    assert _install_torch_numpy_fallback(fake_torch, _FakeNP, ctypes) is False


def test_install_torch_numpy_fallback_rejects_unsupported_dtype():
    fake_torch = _FakeTorch()
    arr = _FakeArray([1.0], dtype="bfloat16")

    _install_torch_numpy_fallback(fake_torch, _FakeNP, ctypes)

    with pytest.raises(TypeError, match="unsupported numpy dtype 'bfloat16'") as exc_info:
        fake_torch.from_numpy(arr)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Numpy is not available"
