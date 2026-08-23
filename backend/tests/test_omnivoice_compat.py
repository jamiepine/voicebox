"""
Unit tests for the OmniVoice engine's transformers 5 -> 4.57.6 compatibility layer.

OmniVoice is built against transformers >= 5.3; Voicebox is pinned to <= 4.57.6
because several engine workarounds read 4.57.x internals. ``backend/vendor/``
carries the one class that gap costs us (the Higgs Audio V2 codec) and
``backend/utils/transformers5_compat.py`` grafts it on.

None of this downloads a model or touches a GPU. The full round trip — load,
clone, generate, unload — is covered by the manual E2E suite described in
``backend/tests/E2E_MODEL_TEST_DESIGN.md``.

NOTE: ``patch_extra_special_tokens`` mutates ``transformers.PreTrainedTokenizerBase``
globally; run serially, not under ``pytest-xdist`` with per-worker isolation.
"""

import sys
import time
import threading

import pytest
import torch
from torch import nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import backend.utils.transformers5_compat as compat
from backend.vendor import _compat


# ── conv1d_output_length ──────────────────────────────────────────────
# Absent from 4.57.6's audio_utils; the vendored codec computes encoder
# frame counts with it.


@pytest.mark.parametrize(
    "kwargs,length,expected",
    [
        ({"kernel_size": 1, "stride": 1}, 100, 100),
        ({"kernel_size": 3, "stride": 1, "padding": 1}, 100, 100),
        ({"kernel_size": 4, "stride": 2, "padding": 1}, 100, 50),
        ({"kernel_size": 3, "stride": 1, "dilation": 2, "padding": 2}, 100, 100),
    ],
)
def test_conv1d_output_length_matches_torch(kwargs, length, expected):
    layer = nn.Conv1d(1, 1, **kwargs)
    assert _compat.conv1d_output_length(layer, length) == expected
    # Cross-check against what the layer actually produces.
    actual = layer(torch.zeros(1, 1, length)).shape[-1]
    assert _compat.conv1d_output_length(layer, length) == actual


# ── initialization module ─────────────────────────────────────────────
# 5.x-only. Only reached from _init_weights, i.e. for weights missing from
# the checkpoint.


def test_initialization_functions_mutate_in_place_and_return_tensor():
    tensor = torch.empty(4, 4)
    result = _compat.initialization.zeros_(tensor)
    assert result is tensor
    assert torch.all(result == 0)

    assert torch.all(_compat.initialization.ones_(torch.empty(3)) == 1)
    assert _compat.initialization.normal_(torch.empty(8), mean=0.0, std=0.02) is not None


def test_initialization_copy_writes_through():
    target = torch.zeros(2)
    source = torch.tensor([1.0, 2.0])
    assert torch.equal(_compat.initialization.copy_(target, source), source)


# ── PreTrainedConfig dataclass adapter ────────────────────────────────
# 5.x configs are dataclasses whose generated __init__ ends in __post_init__.
# 4.57.6's PretrainedConfig never calls it, which leaves sub-configs as raw
# dicts and blows up later in to_diff_dict().


class _SampleConfig(_compat.PreTrainedConfig):
    model_type = "sample_compat_config"

    plain_field: int = 7
    listy_field: list = None
    nested: dict = None

    def __post_init__(self, **kwargs):
        if isinstance(self.nested, dict):
            self.nested = dict(self.nested, promoted=True)
        super().__post_init__(**kwargs)


def test_annotated_defaults_are_applied():
    config = _SampleConfig()
    assert config.plain_field == 7
    assert config.nested is None


def test_kwargs_override_defaults():
    config = _SampleConfig(plain_field=42)
    assert config.plain_field == 42


def test_post_init_runs():
    config = _SampleConfig(nested={"a": 1})
    assert config.nested == {"a": 1, "promoted": True}


def test_mutable_defaults_are_not_shared_between_instances():
    class _ListConfig(_compat.PreTrainedConfig):
        model_type = "list_compat_config"
        items: list = []

    first, second = _ListConfig(), _ListConfig()
    first.items.append("x")
    assert second.items == []


def test_leftover_kwargs_reach_pretrained_config():
    config = _SampleConfig(plain_field=1, output_hidden_states=True)
    assert config.output_hidden_states is True


# ── auto_docstring / strict no-ops ────────────────────────────────────
# Both raise on 4.57.6 for the vendored class: auto_docstring rejects names
# outside its registry, strict requires a dataclass.


def test_auto_docstring_returns_target_unchanged():
    class Target:
        pass

    assert _compat.auto_docstring(Target) is Target
    assert _compat.auto_docstring(custom_intro="x")(Target) is Target


def test_strict_returns_target_unchanged():
    class Target:
        pass

    assert _compat.strict(Target) is Target
    assert _compat.strict()(Target) is Target


# ── extra_special_tokens list form ────────────────────────────────────
# OmniVoice's tokenizer_config.json writes a list (5.x form); 4.57.6 calls
# .keys() on it.


@pytest.fixture
def restore_special_tokens_hook():
    """Undo the test double, including when the class never owned the attribute.

    ``_set_model_specific_special_tokens`` is defined on ``SpecialTokensMixin``,
    not on ``PreTrainedTokenizerBase``, so it is absent from the latter's own
    ``__dict__``. Restoring only when a previous value existed would leave the
    stub installed for the rest of the session and every later tokenizer test
    would run against it.
    """
    sentinel = object()
    saved = PreTrainedTokenizerBase.__dict__.get(
        "_set_model_specific_special_tokens", sentinel
    )
    yield
    if saved is sentinel:
        # Nothing was shadowed -- drop ours so the inherited one is found
        # again. A class __dict__ is a read-only mappingproxy, so this has to
        # go through del on the class itself.
        try:
            del PreTrainedTokenizerBase._set_model_specific_special_tokens
        except AttributeError:
            pass
    else:
        PreTrainedTokenizerBase._set_model_specific_special_tokens = saved


def test_patch_coerces_list_to_mapping(restore_special_tokens_hook):
    seen = {}

    def fake(self, special_tokens):
        seen.update(special_tokens)

    PreTrainedTokenizerBase._set_model_specific_special_tokens = fake
    assert compat.patch_extra_special_tokens() is True

    PreTrainedTokenizerBase._set_model_specific_special_tokens(
        object(), ["<|lang_start|>", "<|text_end|>"]
    )
    assert seen == {
        "lang_start_token": "<|lang_start|>",
        "text_end_token": "<|text_end|>",
    }


def test_patch_leaves_mapping_form_alone(restore_special_tokens_hook):
    seen = {}

    def fake(self, special_tokens):
        seen.update(special_tokens)

    PreTrainedTokenizerBase._set_model_specific_special_tokens = fake
    compat.patch_extra_special_tokens()

    PreTrainedTokenizerBase._set_model_specific_special_tokens(
        object(), {"boundary_token": "<|b|>"}
    )
    assert seen == {"boundary_token": "<|b|>"}


def test_patch_is_idempotent(restore_special_tokens_hook):
    PreTrainedTokenizerBase._set_model_specific_special_tokens = lambda self, tokens: None
    assert compat.patch_extra_special_tokens() is True
    assert compat.patch_extra_special_tokens() is False


def test_teardown_leaves_no_stub_behind(restore_special_tokens_hook):
    """The fixture must not leak the double into the rest of the session.

    Regression: the original teardown only restored when a previous value had
    been shadowed. ``_set_model_specific_special_tokens`` lives on
    ``SpecialTokensMixin``, so ``PreTrainedTokenizerBase`` never owns it and the
    stub survived every test that used this fixture.
    """
    assert "_set_model_specific_special_tokens" not in PreTrainedTokenizerBase.__dict__

    PreTrainedTokenizerBase._set_model_specific_special_tokens = lambda self, tokens: None
    compat.patch_extra_special_tokens()
    assert "_set_model_specific_special_tokens" in PreTrainedTokenizerBase.__dict__


def test_inherited_method_is_reachable_after_teardown():
    """Runs after the fixture above has torn down; the real method must be back."""
    assert "_set_model_specific_special_tokens" not in PreTrainedTokenizerBase.__dict__
    assert callable(PreTrainedTokenizerBase._set_model_specific_special_tokens)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("<|denoise|>", "denoise_token"),
        ("<|instruct_start|>", "instruct_start_token"),
        ("[SEP]", "sep_token"),
        ("<mask>", "mask_token"),
        ("<|weird-one|>", "weird_one_token"),
    ],
)
def test_attribute_name_derivation(token, expected):
    assert compat._attribute_name(token) == expected


# ── engine wiring ─────────────────────────────────────────────────────


def test_omnivoice_is_registered():
    from backend.backends import TTS_ENGINES, get_model_config

    assert TTS_ENGINES["omnivoice"] == "OmniVoice"

    config = get_model_config("omnivoice")
    assert config is not None
    assert config.engine == "omnivoice"
    assert config.hf_repo_id == "k2-fsa/OmniVoice"
    assert config.supports_instruct is True


def test_omnivoice_accepts_cloned_profiles():
    """Regression: the service-layer allow-list is separate from the frontend one."""
    from backend.services.profiles import CLONING_ENGINES

    assert "omnivoice" in CLONING_ENGINES


def test_declared_languages_are_resolvable_by_omnivoice():
    """Every advertised code must map to something OmniVoice actually knows.

    OmniVoice speaks ISO 639-3. Most two-letter codes pass straight through,
    but Arabic does not: it enumerates varieties instead of the macrolanguage.
    """
    # omnivoice imports HiggsAudioV2TokenizerModel at module scope, so the
    # shim has to be in place before the import -- which is itself the point.
    compat.install()
    try:
        from omnivoice.utils.lang_map import LANG_NAME_TO_ID
    except ImportError as exc:  # pragma: no cover - omnivoice not installed
        pytest.skip(f"omnivoice unavailable: {exc}")

    from backend.backends import get_model_config
    from backend.backends.omnivoice_backend import LANGUAGE_CODE_OVERRIDES

    known = set(LANG_NAME_TO_ID.values())
    unresolved = [
        code
        for code in get_model_config("omnivoice").languages
        if LANGUAGE_CODE_OVERRIDES.get(code, code) not in known
    ]
    assert unresolved == []


def test_clear_codec_layer_cache_is_safe_when_codec_never_imported():
    from backend.backends.omnivoice_backend import _clear_codec_layer_cache

    name = "backend.vendor.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer"
    saved = sys.modules.pop(name, None)
    try:
        _clear_codec_layer_cache()  # must not raise
    finally:
        if saved is not None:
            sys.modules[name] = saved


def test_ensure_asr_loads_once_under_concurrency():
    """Overlapping clone requests must load Whisper exactly once.

    Regression: ``_ensure_asr`` runs inside the worker thread that
    ``asyncio.to_thread`` spawns, so two clone requests without reference text
    could both clear the ``_asr_loaded`` check before either set it. Whisper
    large-v3-turbo is ~1.6 GB, so a duplicate load is real waste.

    The barrier and the sleep inside the fake load are what make the race
    deterministic: without the lock every worker gets past the check.
    """
    from backend.backends.omnivoice_backend import OmniVoiceBackend

    workers = 8
    calls = []
    calls_lock = threading.Lock()
    at_the_gate = threading.Barrier(workers)

    class FakeModel:
        def load_asr_model(self):
            with calls_lock:
                calls.append(1)
            time.sleep(0.05)

    backend = OmniVoiceBackend()
    backend.model = FakeModel()

    def worker():
        at_the_gate.wait()
        backend._ensure_asr()

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert backend._asr_loaded is True


def test_ensure_asr_is_a_noop_once_loaded():
    from backend.backends.omnivoice_backend import OmniVoiceBackend

    class ExplodingModel:
        def load_asr_model(self):
            raise AssertionError("must not reload once _asr_loaded is set")

    backend = OmniVoiceBackend()
    backend.model = ExplodingModel()
    backend._asr_loaded = True

    backend._ensure_asr()  # must not raise
