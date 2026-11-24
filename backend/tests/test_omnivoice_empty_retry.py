"""Retry behaviour of OmniVoiceBackend when the model returns no usable audio.

At 32 diffusion steps some sentences make OmniVoice return an empty or
near-silent waveform (remove_silence can trim everything). The backend retries
with more steps instead of failing in post-processing.

Uses a fake model and a patched ``manual_seed``; no weights are loaded.
"""

from unittest.mock import patch

import numpy as np
import pytest

from backend.backends import omnivoice_backend as ov

SAMPLE_RATE = 24000
MODULE = "backend.backends.omnivoice_backend"


def _speech() -> np.ndarray:
    return np.full(SAMPLE_RATE, 0.2, dtype=np.float32)


class FakeModel:
    """Returns queued outcomes in order: an exception instance or an array."""

    sampling_rate = SAMPLE_RATE

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return [outcome]


def _make_backend(model: FakeModel) -> ov.OmniVoiceBackend:
    backend = ov.OmniVoiceBackend()
    # load_model() returns early when model is set, so no download happens.
    backend.model = model
    return backend


def _generate(backend, **overrides):
    kwargs = {
        "text": "Silakan download aplikasinya dari website resmi, lalu install seperti biasa.",
        "voice_prompt": {},
        "language": "ind",
        "seed": 42,
    }
    kwargs.update(overrides)
    return backend.generate(**kwargs)


@pytest.fixture
def seeds():
    with patch(f"{MODULE}.manual_seed") as mock_seed:
        yield mock_seed


@pytest.mark.asyncio
async def test_zero_size_error_then_success_retries_with_seed_plus_one(seeds):
    model = FakeModel([ValueError("zero-size array to reduction operation maximum which has no identity"), _speech()])
    backend = _make_backend(model)

    audio, sample_rate = await _generate(backend)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(audio, _speech())
    assert len(model.calls) == 2
    assert [c.args[0] if c.args else None for c in seeds.call_args_list] == [42, 43]
    assert [c["num_step"] for c in model.calls] == [ov.NUM_DIFFUSION_STEPS, ov.NUM_DIFFUSION_STEPS]


@pytest.mark.asyncio
async def test_all_attempts_fail_raises_runtime_error_with_last_step_64(seeds):
    empty_error = ValueError("zero-size array to reduction operation maximum which has no identity")
    model = FakeModel([empty_error, empty_error, empty_error])
    backend = _make_backend(model)

    with pytest.raises(RuntimeError, match="no usable audio after 3 attempts") as excinfo:
        await _generate(backend)

    assert "lang=ind" in str(excinfo.value)
    assert len(model.calls) == 3
    assert model.calls[-1]["num_step"] == ov.NUM_DIFFUSION_STEPS


@pytest.mark.asyncio
async def test_first_valid_output_is_a_single_call_at_default_steps(seeds):
    model = FakeModel([_speech()])
    backend = _make_backend(model)

    await _generate(backend)

    assert len(model.calls) == 1
    assert model.calls[0]["num_step"] == ov.NUM_DIFFUSION_STEPS
    seeds.assert_called_once_with(42, backend.device)


@pytest.mark.asyncio
async def test_empty_array_without_exception_is_retried(seeds):
    model = FakeModel([np.zeros(0, dtype=np.float32), _speech()])
    backend = _make_backend(model)

    audio, _ = await _generate(backend)

    assert audio.size > 0
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_instruct_value_error_is_not_retried(seeds):
    model = FakeModel([ValueError("Unknown instruct attribute: 'fluffy'")])
    backend = _make_backend(model)

    with pytest.raises(ValueError, match="accepts a fixed set of attributes") as excinfo:
        await _generate(backend, instruct="fluffy")

    assert "Unknown instruct attribute" in str(excinfo.value)
    assert len(model.calls) == 1


def test_is_degenerate_guards_empty_nan_and_silence_before_peak():
    assert ov._is_degenerate(np.zeros(0, dtype=np.float32)) is True
    assert ov._is_degenerate(np.array([0.0, np.nan], dtype=np.float32)) is True
    assert ov._is_degenerate(np.array([0.0, np.inf], dtype=np.float32)) is True
    assert ov._is_degenerate(np.zeros(100, dtype=np.float32)) is True
    assert ov._is_degenerate(_speech()) is False
