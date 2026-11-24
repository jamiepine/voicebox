"""Indonesian ("id") is only accepted on engines that declare it."""

import pytest

from backend.services.profiles import validate_generation_language


@pytest.mark.parametrize("engine", ["kokoro", "qwen", "luxtts", "tada"])
def test_indonesian_rejected_on_engines_without_it(engine):
    with pytest.raises(ValueError, match="does not support language 'id'"):
        validate_generation_language(engine, "id")


def test_indonesian_accepted_on_omnivoice():
    validate_generation_language("omnivoice", "id")


@pytest.mark.parametrize("engine", ["kokoro", "qwen", "omnivoice"])
def test_other_languages_untouched(engine):
    validate_generation_language(engine, "en")
    validate_generation_language(engine, None)


def test_unknown_engine_left_to_existing_checks():
    validate_generation_language("not-an-engine", "id")
