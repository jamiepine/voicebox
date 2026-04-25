"""
Runtime hook: repair rocm_sdk.find_libraries under PyInstaller.

rocm_sdk 7.2.x ships a find_libraries() with a latent bug: when the
backend package (_rocm_sdk_core / _rocm_sdk_libraries_{target}) cannot
be imported, the except clause records the miss but falls through to
`py_root = Path(py_module.__file__).parent`, where py_module was never
assigned. This surfaces as UnboundLocalError instead of the intended
ModuleNotFoundError, masking the real cause.

Frozen apps trip this because rocm_sdk imports the backend packages
dynamically via importlib, which PyInstaller's static analyzer cannot
see. We re-collect those packages in build_binary.py; this hook is
defense-in-depth: it replaces find_libraries with a corrected version
so any future missing-package case surfaces a readable error.
"""


def _patch_rocm_sdk():
    try:
        import rocm_sdk
        from rocm_sdk import _dist_info
    except ModuleNotFoundError as e:
        if e.name not in {"rocm_sdk", "rocm_sdk._dist_info"}:
            raise
        return

    import importlib
    import platform
    from pathlib import Path

    def find_libraries(*shortnames):
        paths = []
        missing_extras = set()
        is_windows = platform.system() == "Windows"
        for shortname in shortnames:
            try:
                lib_entry = _dist_info.ALL_LIBRARIES[shortname]
            except KeyError:
                raise ModuleNotFoundError(f"Unknown rocm library '{shortname}'") from None

            if is_windows and not lib_entry.dll_pattern:
                continue

            package = lib_entry.package
            target_family = None
            if package.is_target_specific:
                target_family = _dist_info.determine_target_family()
            py_package_name = package.get_py_package_name(target_family)
            try:
                py_module = importlib.import_module(py_package_name)
            except ModuleNotFoundError as e:
                if e.name != py_package_name:
                    raise
                missing_extras.add(package.logical_name)
                continue

            py_root = Path(py_module.__file__).parent
            if is_windows:
                relpath = py_root / lib_entry.windows_relpath
                entry_pattern = lib_entry.dll_pattern
            else:
                relpath = py_root / lib_entry.posix_relpath
                entry_pattern = lib_entry.so_pattern
            matching_paths = sorted(relpath.glob(entry_pattern))
            if len(matching_paths) == 0:
                raise FileNotFoundError(
                    f"Could not find rocm library '{shortname}' at path "
                    f"'{relpath},' no match for pattern '{entry_pattern}'"
                )
            paths.append(matching_paths[0])

        if missing_extras:
            raise ModuleNotFoundError(
                f"Missing required rocm backend packages: "
                f"{', '.join(sorted(missing_extras))}. The frozen build did "
                f"not bundle _rocm_sdk_core / _rocm_sdk_libraries_<target>. "
                f"Check build_binary.py --collect-all flags."
            )
        return paths

    rocm_sdk.find_libraries = find_libraries


_patch_rocm_sdk()
