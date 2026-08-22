"""Let transformers-5-era model assets load on the transformers 4.57.6 we pin.

Voicebox cannot move to transformers 5: the offline-mode fix in
``backend/utils/hf_offline_patch.py`` and the frozen-build rewrite in
``backend/pyi_rth_torch_compiler_disable.py`` both read 4.57.x internals, and
seven shipping engines depend on them. OmniVoice, however, was authored against
transformers 5.3+ and trips over two gaps:

1. Its audio codec is ``transformers.HiggsAudioV2TokenizerModel``, a class that
   does not exist before 5.3. The weights ship inside the OmniVoice repo, so
   only the modelling code is missing -- we vendor it under
   ``backend/vendor/higgs_audio_v2_tokenizer`` and graft it on.
2. Its ``tokenizer_config.json`` writes ``extra_special_tokens`` as a list,
   the 5.x form. 4.57.6 expects a mapping and raises ``AttributeError``.

Both fixes are additive: they add a name that was absent, and accept an input
shape that previously crashed. Neither changes behaviour for the other engines.

Same spirit as ``backend/utils/dac_shim.py``, which fakes ``dac.nn.layers`` so
TADA can skip the descript-audio-codec dependency tree. The direction differs:
that one replaces a package we do not want, these back-port from a version we
cannot upgrade to.

``install()`` must run before ``omnivoice`` is imported. It is idempotent, and
each half no-ops when the running transformers already handles the case.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Apply every transformers 5 -> 4.57.6 shim OmniVoice needs."""
    global _installed

    if _installed:
        return

    install_higgs_audio_tokenizer()
    patch_extra_special_tokens()
    _installed = True


# ---------------------------------------------------------------------------
# 1. HiggsAudioV2TokenizerModel
# ---------------------------------------------------------------------------


def install_higgs_audio_tokenizer() -> bool:
    """Graft the vendored Higgs Audio V2 codec onto ``transformers``.

    Returns:
        True if the vendored copy is now in place, False if the running
        transformers already ships the real one.
    """
    import transformers

    if hasattr(transformers, "HiggsAudioV2TokenizerModel"):
        logger.debug(
            "transformers %s ships HiggsAudioV2TokenizerModel, skipping shim",
            transformers.__version__,
        )
        return False

    from ..vendor import higgs_audio_v2_tokenizer as vendored
    from ..vendor.higgs_audio_v2_tokenizer import (
        configuration_higgs_audio_v2_tokenizer as vendored_config,
    )
    from ..vendor.higgs_audio_v2_tokenizer import (
        modeling_higgs_audio_v2_tokenizer as vendored_modeling,
    )

    # The first import of transformers.processing_utils -- which the line above
    # triggers -- re-runs transformers/__init__.py and rebinds
    # sys.modules["transformers"] to a fresh _LazyModule. Anything set on the
    # old object is lost, so re-fetch here, after the vendored import.
    transformers = sys.modules["transformers"]

    # Register under the module path upstream uses, so anything resolving the
    # class by qualified name (pickling, AutoModel, remote-code loaders) finds
    # it where it expects to.
    base = "transformers.models.higgs_audio_v2_tokenizer"
    sys.modules.setdefault(base, vendored)
    sys.modules.setdefault(f"{base}.configuration_higgs_audio_v2_tokenizer", vendored_config)
    sys.modules.setdefault(f"{base}.modeling_higgs_audio_v2_tokenizer", vendored_modeling)
    setattr(transformers.models, "higgs_audio_v2_tokenizer", vendored)

    # transformers is a _LazyModule whose __getattr__ only fires when normal
    # attribute lookup fails, so a plain setattr wins for
    # `from transformers import HiggsAudioV2TokenizerModel`.
    for name in vendored.__all__:
        setattr(transformers, name, getattr(vendored, name))

    _register_with_auto_classes(vendored)

    logger.info(
        "installed vendored HiggsAudioV2Tokenizer (transformers %s lacks it)",
        transformers.__version__,
    )
    return True


def _register_with_auto_classes(vendored) -> None:
    """Wire the vendored config/model into AutoConfig and AutoModel.

    Not needed for OmniVoice's own load path, which instantiates the concrete
    class, but it keeps ``AutoModel.from_config`` working for anything that
    reaches the codec generically. Registration raises if the model type is
    already known, which is fine to ignore.
    """
    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register("higgs_audio_v2_tokenizer", vendored.HiggsAudioV2TokenizerConfig)
        AutoModel.register(
            vendored.HiggsAudioV2TokenizerConfig, vendored.HiggsAudioV2TokenizerModel
        )
    except ValueError as exc:
        logger.debug("auto-class registration skipped: %s", exc)


# ---------------------------------------------------------------------------
# 2. extra_special_tokens as a list
# ---------------------------------------------------------------------------


def patch_extra_special_tokens() -> bool:
    """Accept the 5.x list form of ``extra_special_tokens``.

    4.57.6's ``_set_model_specific_special_tokens`` calls ``.keys()`` on its
    argument, so a list from a 5.x-era tokenizer_config.json dies with
    ``AttributeError: 'list' object has no attribute 'keys'``. 5.x accepts both
    shapes: a list means "these are special, but under no named attribute",
    which 4.57.6 has no way to express, so we derive a name per token.

    The tokens themselves are already marked ``special: true`` inside
    tokenizer.json, so segmentation is correct either way — this only restores
    the attribute exposure and, more to the point, stops the crash.

    Returns:
        True if the wrapper was installed, False if it was already there or
        the running transformers needs no help.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    original = getattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None)
    if original is None:
        logger.debug("transformers has no _set_model_specific_special_tokens, skipping patch")
        return False

    if getattr(original, "_voicebox_patched", False):
        return False

    def patched(self, special_tokens):
        if isinstance(special_tokens, (list, tuple, set)):
            special_tokens = {_attribute_name(token): token for token in special_tokens}
        return original(self, special_tokens)

    patched._voicebox_patched = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched

    logger.debug("installed extra_special_tokens list-form wrapper")
    return True


def _attribute_name(token: str) -> str:
    """Derive an attribute name for an unnamed extra special token.

    ``"<|lang_start|>"`` becomes ``"lang_start_token"``, matching the
    ``*_token`` convention of the built-in SPECIAL_TOKENS_ATTRIBUTES.
    """
    stem = str(token).strip()
    for delimiter in ("<|", "|>", "<", ">", "[", "]"):
        stem = stem.replace(delimiter, "")
    stem = "".join(char if char.isalnum() else "_" for char in stem).strip("_").lower()
    return f"{stem or 'extra'}_token"
