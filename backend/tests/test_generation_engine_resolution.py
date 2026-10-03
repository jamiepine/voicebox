"""Tests for generation request engine selection."""

import pytest
from pydantic import ValidationError

from backend import models


def _request(**kwargs) -> models.GenerationRequest:
    return models.GenerationRequest(profile_id="profile-1", text="hello", **kwargs)


def test_omitted_engine_does_not_override_profile_default():
    request = _request()

    assert request.engine is None


def test_explicit_engine_is_preserved():
    request = _request(engine="chatterbox")

    assert request.engine == "chatterbox"


def test_invalid_explicit_engine_is_rejected():
    with pytest.raises(ValidationError):
        _request(engine="invalid")
