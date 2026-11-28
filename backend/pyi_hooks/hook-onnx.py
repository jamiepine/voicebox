"""
Collect the onnx.libs directory from onnx's delvewheel-repaired Windows wheel.

onnx_cpp2py_export.pyd links against a hash-named msvcp140-<hash>.dll that
the wheel ships in site-packages/onnx.libs/, outside the onnx package, and
onnx/__init__.py registers that folder with os.add_dll_directory(). PyInstaller
has no onnx hook, so the folder was never bundled and the DLL only loaded if
pandas (whose stock hook collects pandas.libs) had already put an identical
copy in the process. See #1264.
"""

from PyInstaller.utils.hooks import collect_delvewheel_libs_directory

datas, binaries = collect_delvewheel_libs_directory("onnx")
