"""Compatibility shims for code vendored from transformers 5.x.

Voicebox pins ``transformers<=4.57.6``. The Higgs Audio V2 tokenizer vendored
into this package was written against 5.x and reaches for three symbols that
4.57.6 does not export. All three are thin and stable, so we reimplement them
here rather than move the whole app to transformers 5 and break the seven
engines whose workarounds depend on 4.57.x internals (see
``backend/utils/hf_offline_patch.py`` and
``backend/pyi_rth_torch_compiler_disable.py``).

Everything else the vendored code imports (``PreTrainedAudioTokenizerBase``,
``Unpack``, ``TransformersKwargs``, ``auto_docstring``, ``can_return_tuple``,
``ModelOutput``, ``requires``, ``AutoModel``) already exists in 4.57.6.
"""

import copy
import types

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

_NO_DEFAULT = object()


def _annotated_defaults(cls) -> dict:
    """Collect the dataclass-style field defaults a vendored config declares.

    Walks the MRO up to (but not including) :class:`PreTrainedConfig`, so only
    fields introduced by vendored classes are returned — never the hundred-odd
    attributes 4.57.6's PretrainedConfig manages itself.
    """
    defaults: dict = {}
    for klass in reversed(cls.__mro__):
        if klass in (PreTrainedConfig, PretrainedConfig, object):
            continue
        for name in getattr(klass, "__annotations__", {}):
            if name.startswith("_"):
                continue
            value = klass.__dict__.get(name, _NO_DEFAULT)
            if value is not _NO_DEFAULT:
                defaults[name] = value
    return defaults


class PreTrainedConfig(PretrainedConfig):
    """4.57.6 stand-in for the transformers 5.x config base class.

    5.x turned configs into dataclasses: fields are class-level annotations
    with defaults, and a generated ``__init__`` ends by calling
    ``__post_init__``, where subclasses promote nested dicts into real config
    objects. 4.57.6's ``PretrainedConfig`` is an ordinary class that never
    calls ``__post_init__``, which leaves ``sub_configs`` as plain dicts and
    blows up later in ``to_diff_dict()``.

    This reproduces the part of the 5.x protocol the vendored code relies on:
    apply annotated defaults, bind the incoming values, then hand control to
    ``__post_init__`` — which forwards the leftovers to the 4.57.6
    ``__init__`` through the base implementation below.

    Not a general-purpose backport. It covers exactly what
    ``HiggsAudioV2TokenizerConfig`` needs.
    """

    def __init__(self, **kwargs):
        for name, default in _annotated_defaults(type(self)).items():
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            else:
                setattr(self, name, copy.deepcopy(default))

        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)


def strict(cls=None, **kwargs):
    """No-op stand-in for ``huggingface_hub.dataclasses.strict``.

    The real decorator adds runtime type validation to dataclass fields, and
    refuses to wrap anything that is not a dataclass. ``PreTrainedConfig`` is a
    dataclass in transformers 5.x but a plain class in 4.57.6, so applying it
    here raises at class-definition time. The vendored config declares its
    fields through a normal ``__init__``, so the validation has nothing to
    check and dropping it changes no behaviour we depend on.
    """
    if cls is None:
        return lambda inner: inner
    return cls


def conv1d_output_length(module: nn.Conv1d, input_length: int) -> int:
    """Output length of a 1D convolution.

    Verbatim from ``transformers.audio_utils.conv1d_output_length`` (5.x),
    which does not exist in 4.57.6.
    """
    return int(
        (
            input_length
            + 2 * module.padding[0]
            - module.dilation[0] * (module.kernel_size[0] - 1)
            - 1
        )
        / module.stride[0]
        + 1
    )


def auto_docstring(obj=None, **kwargs):
    """No-op stand-in for ``transformers.utils.auto_docstring``.

    4.57.6's version only accepts classes whose name matches an entry in its
    internal registry (``...Model``, ``...ForCausalLM``, and so on) and raises
    on anything else; 5.x dropped that restriction. It composes docstrings and
    nothing else, so returning the object untouched costs only documentation.
    """
    if obj is None:
        return lambda inner: inner
    return obj


def _build_initialization_module() -> types.ModuleType:
    """Stand in for ``transformers.initialization`` (5.x only).

    Upstream that module is a set of thin wrappers over ``torch.nn.init`` that
    return the tensor and honour a global skip-init flag. The vendored
    tokenizer only touches it from ``_init_weights``, which runs solely for
    weights absent from the checkpoint, so plain delegation is enough.
    """
    module = types.ModuleType("transformers.initialization")

    for name in (
        "uniform_",
        "normal_",
        "constant_",
        "ones_",
        "zeros_",
        "eye_",
        "dirac_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "trunc_normal_",
        "orthogonal_",
        "sparse_",
    ):
        torch_fn = getattr(nn.init, name)

        def _wrap(tensor, *args, _fn=torch_fn, **kwargs):
            with torch.no_grad():
                _fn(tensor, *args, **kwargs)
            return tensor

        setattr(module, name, _wrap)

    def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tensor.copy_(other)
        return tensor

    module.copy_ = copy_
    return module


initialization = _build_initialization_module()

__all__ = ["PreTrainedConfig", "conv1d_output_length", "initialization", "strict", "auto_docstring"]
