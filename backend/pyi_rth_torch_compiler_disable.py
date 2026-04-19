"""
PyInstaller runtime hook: stub torch._dynamo to a no-op module.

Problem
-------
transformers triggers torch._dynamo import at module-load time (not just
when torch.compile is called) via class-body decorators:

    transformers/modeling_utils.py:1984
        @torch._dynamo.allow_in_graph
        class PreTrainedModel(...)

    transformers/integrations/flex_attention.py:61
        @torch.compiler.disable(recursive=False)
        class WrappedFlexAttention...

The attribute access triggers torch.__getattr__ -> importlib.import_module
-> torch._dynamo -> torch._dynamo.utils imports torch._numpy ->
torch._numpy._ndarray imports torch._numpy._ufuncs, which crashes under
PyInstaller with:

    File "torch/_numpy/_ufuncs.py", line 235, in <module>
        vars()[name] = deco_binary_ufunc(ufunc)
    NameError: name 'name' is not defined

(The module-level `for name in _binary: vars()[name] = ...` pattern works
in a regular venv but fails in the PyInstaller bundle. Root cause is in
PyInstaller's importer / bytecode pipeline and not easily fixed upstream.)

Surfaces as Kokoro failing to load when `from transformers import AlbertModel`
trips the decorator chain.

Fix
---
voicebox never uses torch.compile / torch._dynamo for inference, so we
replace torch._dynamo with a no-op stub module before transformers is
imported. Any attribute access on the stub returns a pass-through callable,
so `@torch._dynamo.allow_in_graph`, `torch._dynamo.is_compiling()`,
`torch._dynamo.mark_static_address(...)`, etc. all work.

This hook is pure sys.modules manipulation — we deliberately do NOT import
torch here. Runtime hooks run before the app starts and before
pyi_rth_numpy_compat has had a chance to patch torch.from_numpy (it runs
in a background thread, waiting for torch to appear in sys.modules).
Eager-importing torch at hook time would trip the numpy ABI issue and
kill the server process at startup.

torch.compiler.disable does not need a separate stub: its implementation
is effectively `import torch._dynamo; return torch._dynamo.disable(...)`,
and since our stub is in sys.modules, that call resolves to our no-op
_NoopDecorator pass-through.
"""

import os
import sys
import tempfile
import types


# Diagnostics — log hook activity to a file alongside the bundle so we can
# see what's happening when the server is run as a sidecar (no stdout for
# runtime hook prints). Safe no-op if the file can't be written.
_DIAG_PATH = os.path.join(tempfile.gettempdir(), "voicebox_rt_hook.log")


def _diag(msg: str) -> None:
    try:
        with open(_DIAG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


_HOOK_VERSION = "v6-masking-utils-finder"
_diag(f"=== runtime hook load @ pid={os.getpid()} version={_HOOK_VERSION} ===")


class _NoopDecorator:
    """Multi-role no-op: decorator, falsey predicate, and context manager.

    Returned from calls like `torch._dynamo.disable()` (decorator),
    `torch._dynamo.is_compiling()` (predicate used in `if not ...`), and
    `with torch._dynamo._trace_wrapped_higher_order_op.TransformGetItemToIndex():`
    (context manager used to scope an fx graph transformation).

    By implementing __call__, __bool__, __enter__, __exit__, and __iter__ we
    cover every use pattern we've seen transformers/torch use on a stubbed
    object. Anything we haven't covered will raise a clearer error than a
    silent wrong-result.
    """

    __slots__ = ()

    def __call__(self, fn=None, *args, **kwargs):
        return fn

    def __bool__(self) -> bool:
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False  # don't suppress exceptions

    def __iter__(self):
        return iter(())


_noop_decorator_singleton = _NoopDecorator()


def _noop_callable(*args, **kwargs):
    # Direct-decorator use: @torch._dynamo.foo (no parens) — fn is positional
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    # Side-effect call with non-callable arg(s), e.g. mark_static_address(tensor)
    return _noop_decorator_singleton


class _NoopDynamoModule(types.ModuleType):
    """Permissive stub: every attribute is a pass-through callable.

    Covers attributes transformers hits at import time (allow_in_graph) and
    runtime (is_compiling, mark_static_address, reset, disable, ...).

    Dunder attributes (__file__, __spec__, __loader__, ...) raise
    AttributeError so probes like inspect.getmodule() — which does
    `hasattr(m, '__file__')` then `os.path.normpath(m.__file__)` — see the
    module as having no source file and fall through to its normal
    handling, instead of receiving a function and blowing up.
    """

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _noop_callable


class _DynamoLoader:
    """Loader used by _DynamoMetaPathFinder to materialise stub submodules."""

    def create_module(self, spec):
        return _NoopDynamoModule(spec.name)

    def exec_module(self, module):
        # Mark every stub submodule as a package so deeper submodule imports
        # (`from torch._dynamo.X.Y import Z`) keep working.
        module.__path__ = []


class _DynamoMetaPathFinder:
    """Resolve any `torch._dynamo.X[.Y...]` import to a no-op stub module.

    Without this, `from torch._dynamo._trace_wrapped_higher_order_op import X`
    fails even with torch._dynamo pre-populated in sys.modules — Python's
    import machinery checks the parent's __path__ and then looks up the
    child, and we need to provide both.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch._dynamo":
            return None  # handled by the pre-populated sys.modules entry
        if not fullname.startswith("torch._dynamo."):
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, _DynamoLoader(), is_package=True)


class _TransformersStubFinder:
    """Replace specific transformers submodules with no-op stubs.

    Two modules are targeted:

    1. transformers.utils.auto_docstring
       The real @auto_docstring decorator loads
       transformers.models.auto.modeling_auto just to build example docstrings,
       which drags in GenerationMixin -> candidate_generator -> sklearn.metrics
       -> scipy.stats._distn_infrastructure and trips (2) below. Docstrings
       aren't functional for inference, so a pass-through decorator is safe.

    2. transformers.generation.candidate_generator
       Imported at module scope by transformers.generation.utils. It does
       `from sklearn.metrics import roc_curve` at module load, which triggers:

            File "scipy/stats/_distn_infrastructure.py", line 369, in <module>
            NameError: name 'obj' is not defined

       This is a PyInstaller-specific module-load bug (same class as the
       torch._numpy._ufuncs crash) where a module-level `for obj in [s for s
       in dir() if ...]` loop evaluates to empty in the bundle, leaving `obj`
       unbound before `del obj`.

       The exports (AssistedCandidateGenerator, EarlyExitCandidateGenerator,
       etc.) are speculative-decoding helpers voicebox's TTS engines do not
       use; a no-op stub module satisfies the imports.
    """

    _STUBBED_MODULES = frozenset(
        {
            "transformers.utils.auto_docstring",
            "transformers.generation.candidate_generator",
        }
    )

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._STUBBED_MODULES:
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, _NoopStubLoader(), is_package=False)


class _NoopStubLoader:
    def create_module(self, spec):
        return _NoopDynamoModule(spec.name)

    def exec_module(self, module):
        # _NoopDynamoModule.__getattr__ already answers every non-dunder
        # attribute with a pass-through callable, which satisfies
        # `from stubbed_module import X` for any X.
        pass


def _patch_scipy_distn_source(source: str) -> str:
    """Replace the unsafe `del obj` with a no-op that survives when obj is unbound.

    Returns the input unchanged if the target line isn't found (e.g. scipy
    version has changed).
    """
    target = "\ndel obj\n"
    replacement = "\nglobals().pop('obj', None)\n"
    if target in source:
        return source.replace(target, replacement, 1)
    return source


def _patch_masking_utils_source(source: str) -> str:
    """Force torch<2.6 code path in transformers.masking_utils.

    The torch>=2.6 path uses `with TransformGetItemToIndex():` to allow
    `.item()` calls inside vmap. That context manager is implemented via
    torch._dynamo graph transforms, which our stub doesn't reproduce — it's
    a no-op. The inner `_vmap_for_bhqkv` then crashes with:

        RuntimeError: vmap: It looks like you're calling .item() on a Tensor.

    Forcing the torch<2.6 flag off selects sdpa_mask_older_torch which uses
    a different vmap pattern that does not hit .item() and does not need
    TransformGetItemToIndex.
    """
    target = 'is_torch_greater_or_equal("2.6", accept_dev=True)'
    # Find the specific line that assigns _is_torch_greater_or_equal_than_2_6
    if "_is_torch_greater_or_equal_than_2_6 = " + target in source:
        return source.replace(
            "_is_torch_greater_or_equal_than_2_6 = " + target,
            "_is_torch_greater_or_equal_than_2_6 = False",
            1,
        )
    return source


class _SourcePatchingFinder:
    """Generic delegate-and-wrap meta-path finder that patches a module's
    source before exec'ing.

    Subclasses declare `target` (module fullname) and `patch` (str->str).
    Requires the target module's .py source to be bundled (use a PyInstaller
    hook setting module_collection_mode = "pyz+py").
    """

    target: str
    patch_fn: callable = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                real_spec = find(fullname, path, target)
            except Exception:
                continue
            if real_spec is None or real_spec.loader is None:
                continue
            real_spec.loader = _SourcePatchLoader(real_spec.loader, self.patch_fn)
            return real_spec
        return None


class _SourcePatchLoader:
    """Delegate loader that reads source via get_source, applies a patch, and
    compile/exec's the patched text into module.__dict__.
    """

    def __init__(self, inner, patch_fn):
        self._inner = inner
        self._patch_fn = patch_fn

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        source = None
        try:
            source = self._inner.get_source(module.__name__)
        except Exception as e:
            _diag(f"[source-patch] get_source({module.__name__}) failed: {e!r}")

        if not source:
            _diag(
                f"[source-patch] no source for {module.__name__}; "
                "falling back to inner exec_module (patch NOT applied)"
            )
            self._inner.exec_module(module)
            return

        patched = self._patch_fn(source)
        _diag(
            f"[source-patch] {module.__name__}: "
            f"patched={patched is not source}, len={len(patched)}"
        )
        spec = module.__spec__
        if spec is not None and spec.submodule_search_locations is not None:
            module.__path__ = spec.submodule_search_locations
        filename = getattr(self._inner, "path", module.__name__)
        exec(compile(patched, filename, "exec"), module.__dict__)
        _diag(f"[source-patch] {module.__name__} OK")


class _MaskingUtilsFinder(_SourcePatchingFinder):
    target = "transformers.masking_utils"
    patch_fn = staticmethod(_patch_masking_utils_source)


class _ScipyDistnPatchingFinder:
    """Delegate-and-wrap finder for scipy.stats._distn_infrastructure.

    That module ends with:

        for obj in [s for s in dir() if s.startswith('_doc_')]:
            exec('del ' + obj)
        del obj

    In the PyInstaller bundle the list comprehension evaluates to empty
    (module-level dir() under the frozen importer returns a different scope
    than CPython's normal module-exec path — same class of bug as the
    torch._numpy._ufuncs crash). The for loop body doesn't run, `obj` is
    never bound, and the trailing `del obj` raises NameError at module load.

    This kills every downstream module: librosa (needed by nearly every TTS
    engine for mel filters) -> scipy.signal -> scipy.stats -> here.

    Workaround: delegate to the real loader, but pre-bind `obj = None` in the
    module namespace before its bytecode runs. If the for loop executes, each
    iteration overwrites the sentinel via STORE_NAME (normal behaviour). If it
    doesn't, `del obj` removes the sentinel and module load succeeds. The
    `_doc_*` cleanup this line was meant to do is purely cosmetic — those vars
    stay in the module namespace but nothing references them after this point.
    """

    _TARGET = "scipy.stats._distn_infrastructure"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._TARGET:
            return None
        _diag(f"[scipy-finder] match: {fullname}, path={path!r}")
        # Delegate to the other finders to locate the real spec
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                real_spec = find(fullname, path, target)
            except Exception as e:
                _diag(f"[scipy-finder] inner finder {type(finder).__name__} raised: {e}")
                continue
            if real_spec is None:
                continue
            if real_spec.loader is None:
                _diag(f"[scipy-finder] {type(finder).__name__} returned spec with loader=None")
                continue
            _diag(
                f"[scipy-finder] wrapped loader from "
                f"{type(finder).__name__} -> {type(real_spec.loader).__name__}"
            )
            real_spec.loader = _ScipyDistnPrebindLoader(real_spec.loader)
            return real_spec
        _diag("[scipy-finder] NO inner finder returned a spec")
        return None


class _ScipyDistnPrebindLoader:
    """Thin wrapper that pre-binds `obj = None` before delegating to the
    real PyInstaller loader.

    Every other attribute/method delegates to the inner loader — PyiFrozenLoader
    is a rich FileLoader/ExecutionLoader with get_code/get_source/get_filename/
    is_package/get_resource_reader/etc., any of which Python's import machinery
    or 3rd-party code may call on spec.loader. Forwarding via __getattr__
    avoids breaking any of those paths (and preserves @_check_name contracts
    because the decorated methods run on the inner instance where self.name
    matches spec.name).
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        # __getattr__ fires only for attrs not already on self, so delegate
        # everything that isn't create_module/exec_module (or __getattr__/init).
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        # Compile scipy's module source with the problematic line patched.
        #
        # The real module ends with:
        #     for obj in [s for s in dir() if s.startswith('_doc_')]:
        #         exec('del ' + obj)
        #     del obj
        #
        # Under PyInstaller's frozen importer, `del obj` raises NameError
        # even when we pre-populate module.__dict__['obj'] — the pre-compiled
        # .pyc bytecode interacts with the frame setup differently than a
        # fresh compile() from source. Easiest robust fix: read the source
        # and replace `del obj` with a safe variant before compiling.
        #
        # Requires the .py source to be bundled alongside the .pyc — see
        # backend/pyi_hooks/hook-scipy.stats._distn_infrastructure.py.
        source = None
        try:
            source = self._inner.get_source(module.__name__)
        except Exception as e:
            _diag(f"[scipy-loader] get_source failed: {e!r}")

        if source:
            patched = _patch_scipy_distn_source(source)
            _diag(
                f"[scipy-loader] source-patch path: patched={patched is not source}, "
                f"len={len(patched)}"
            )
            spec = module.__spec__
            if spec is not None and spec.submodule_search_locations is not None:
                module.__path__ = spec.submodule_search_locations
            filename = getattr(self._inner, "path", module.__name__)
            bytecode = compile(patched, filename, "exec")
            try:
                exec(bytecode, module.__dict__)
            except Exception as e:
                _diag(f"[scipy-loader] patched exec raised {type(e).__name__}: {e!r}")
                raise
            _diag(f"[scipy-loader] exec_module {module.__name__} OK (source-patched)")
            return

        # No source available — fall back to the pre-bind approach. This is
        # best-effort; if the frozen .pyc really does see a different `obj`
        # slot, this will still crash, but we've done all we can without
        # source.
        _diag("[scipy-loader] no source available; falling back to pre-bind")
        module.__dict__["obj"] = None
        self._inner.exec_module(module)


def _install_dynamo_stub() -> None:
    stub = _NoopDynamoModule("torch._dynamo")
    # Mark as a package so `from torch._dynamo.X import Y` imports work
    # (Python's import machinery checks parent.__path__ before looking up
    # the child).
    stub.__path__ = []
    # torch._dynamo.config is accessed as a nested attribute namespace
    # (e.g. `torch._dynamo.config.capture_scalar_outputs = True`), so use
    # a permissive module so any attr read returns a no-op and sets succeed.
    stub.config = _NoopDynamoModule("torch._dynamo.config")
    stub.config.__path__ = []
    sys.modules["torch._dynamo"] = stub
    sys.modules["torch._dynamo.config"] = stub.config

    # Finders:
    #  - torch._dynamo.* submodules -> no-op stubs
    #  - transformers.utils.auto_docstring and
    #    transformers.generation.candidate_generator -> no-op stubs (both
    #    paths reach sklearn -> scipy.stats which trips a separate crash)
    #  - scipy.stats._distn_infrastructure -> real load with `obj` pre-bound,
    #    so librosa -> scipy.signal -> scipy.stats loads cleanly
    for _FinderCls in (
        _DynamoMetaPathFinder,
        _TransformersStubFinder,
        _ScipyDistnPatchingFinder,
        _MaskingUtilsFinder,
    ):
        try:
            sys.meta_path.insert(0, _FinderCls())
            _diag(f"installed finder: {_FinderCls.__name__}")
        except Exception as e:
            _diag(f"FAILED to install {_FinderCls.__name__}: {e!r}")
    _diag(
        "final sys.meta_path head: "
        + ", ".join(type(f).__name__ for f in sys.meta_path[:6])
    )

    # If torch is already imported, also set the attribute on the package so
    # `torch._dynamo` resolves to our stub without triggering torch.__getattr__
    # (which would lazy-import the real module and crash).
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod._dynamo = stub


try:
    _install_dynamo_stub()
except Exception as _e:
    # Best effort. If this fails the original NameError will surface when
    # transformers imports — no worse than not patching at all.
    _diag(f"_install_dynamo_stub FAILED: {_e!r}")

# NOTE: we deliberately do NOT import torch or torch.compiler here.
# Runtime hooks run before the app starts and before pyi_rth_numpy_compat
# has had a chance to patch torch.from_numpy (it runs in a background
# thread, waiting for torch to appear in sys.modules). Importing torch
# eagerly at hook time would trip the numpy ABI issue and kill the
# server process at startup.
#
# torch.compiler.disable does not need an explicit stub: its
# implementation is effectively `import torch._dynamo; return
# torch._dynamo.disable(fn, recursive, reason=reason)`, and since our
# stub is installed in sys.modules, that call resolves to our no-op
# _NoopDecorator pass-through.
