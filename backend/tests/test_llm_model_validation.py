"""
Dynamic model_name validation for the LLM-identifying fields in backend/models.py.

Before this feature, `CaptureRefineRequest.model_size`, `LLMGenerateRequest.model_size`,
`CaptureSettingsResponse.llm_model`, and `CaptureSettingsUpdate.llm_model` were
all constrained by a static regex (`^(0\\.6B|1\\.7B|4B)$`) — the valid set is now
platform-dependent (mlx vs pytorch) and spans two engines, so a compile-time
regex can no longer express it. These fields now validate against
`{cfg.model_name for cfg in get_llm_model_configs()}` at request time.

Note: GenerationRequest.model_size (TTS model selection, pattern includes "3B"
for TADA) is a different field entirely and is untouched by this feature.
"""

import pytest

pytest.importorskip("torch")

from backend import backends as backends_module
from backend.backends import ModelConfig
import backend.models as models


def _fake_llm_configs():
    return [
        ModelConfig(model_name="qwen3-0.6b", display_name="Qwen3 0.6B", engine="qwen_llm", hf_repo_id="x", model_size="0.6B"),
        ModelConfig(model_name="minicpm5-1b", display_name="MiniCPM5 1B", engine="minicpm_llm", hf_repo_id="y", model_size="1B"),
    ]


@pytest.fixture(autouse=True)
def _patch_llm_configs(monkeypatch):
    monkeypatch.setattr(models, "get_llm_model_configs", _fake_llm_configs)
    yield


@pytest.mark.parametrize(
    "model_cls,field",
    [
        (models.CaptureRefineRequest, "model_size"),
        (models.LLMGenerateRequest, "model_size"),
        (models.CaptureSettingsResponse, "llm_model"),
        (models.CaptureSettingsUpdate, "llm_model"),
    ],
)
def test_accepts_minicpm5_model_name(model_cls, field):
    kwargs = {field: "minicpm5-1b"}
    if model_cls is models.LLMGenerateRequest:
        kwargs["prompt"] = "hi"
    instance = model_cls(**kwargs)
    assert getattr(instance, field) == "minicpm5-1b"


@pytest.mark.parametrize(
    "model_cls,field",
    [
        (models.CaptureRefineRequest, "model_size"),
        (models.LLMGenerateRequest, "model_size"),
        (models.CaptureSettingsResponse, "llm_model"),
        (models.CaptureSettingsUpdate, "llm_model"),
    ],
)
def test_rejects_unknown_model_name(model_cls, field):
    kwargs = {field: "not-a-real-model"}
    if model_cls is models.LLMGenerateRequest:
        kwargs["prompt"] = "hi"
    with pytest.raises(Exception):
        model_cls(**kwargs)


def test_validation_is_dynamic_not_a_leftover_static_list(monkeypatch):
    """With minicpm5-1b absent from the registry, it must be rejected too —
    proving the check is a live lookup, not a hardcoded value list."""
    monkeypatch.setattr(
        models,
        "get_llm_model_configs",
        lambda: [c for c in _fake_llm_configs() if c.engine != "minicpm_llm"],
    )
    with pytest.raises(Exception):
        models.LLMGenerateRequest(prompt="hi", model_size="minicpm5-1b")
