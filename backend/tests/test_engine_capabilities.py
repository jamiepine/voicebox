"""
Tests for engine capability reporting.

``ModelConfig.supports_instruct`` was declared and then read nowhere, while the
desktop app kept its own hard-coded set of instruct-capable engines. Two
sources of truth for the same fact drift the moment an engine is added. These
tests pin the registry as the authority and pin the one behaviour that has no
UI to warn about it: an API caller passing ``instruct`` to an engine that
throws it away.

Usage:
    python -m pytest backend/tests/test_engine_capabilities.py -v
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-engine-caps-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.backends import (  # noqa: E402
    TTS_ENGINES,
    engine_languages,
    engine_model_sizes,
    engine_supports_instruct,
    get_tts_model_configs,
)
from backend.routes.generations import _warn_if_instruct_ignored  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── The flag is finally read ─────────────────────────────────────────


def test_base_qwen_does_not_support_instruct():
    """The whole reason the flag exists: base Qwen3-TTS accepts instruct and
    silently ignores it, so it must not be advertised as supporting it."""
    assert engine_supports_instruct("qwen") is False


def test_custom_voice_supports_instruct():
    assert engine_supports_instruct("qwen_custom_voice") is True


def test_unknown_engine_is_not_supported():
    """An engine with no registry entry must not read as capable — otherwise a
    typo in an engine name silently 'supports' everything."""
    assert engine_supports_instruct("no-such-engine") is False


def test_helper_agrees_with_the_registry():
    """The helper must not develop its own opinion: for every engine, it has to
    match what the configs declare."""
    for engine in TTS_ENGINES:
        configs = [c for c in get_tts_model_configs() if c.engine == engine]
        expected = bool(configs) and all(c.supports_instruct for c in configs)
        assert engine_supports_instruct(engine) is expected, engine


def test_mixed_variants_resolve_conservatively(monkeypatch):
    """If one model size honours instruct and another doesn't, the engine must
    report False — a request names an engine, and the size can change under it,
    so only the pessimistic answer is true for every variant."""
    configs = list(get_tts_model_configs())
    target = next(c for c in configs if c.engine == "qwen_custom_voice")

    from dataclasses import replace

    downgraded = replace(target, supports_instruct=False, model_name=target.model_name + "-x")
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [*configs, downgraded],
    )
    assert engine_supports_instruct("qwen_custom_voice") is False


# ── Derived metadata ─────────────────────────────────────────────────


def test_languages_are_unioned_across_variants():
    langs = engine_languages("qwen")
    assert "en" in langs
    assert "es" in langs
    assert len(langs) == len(set(langs)), "duplicates across variants"


def test_model_sizes_reported_for_multi_size_engines():
    assert set(engine_model_sizes("qwen")) == {"1.7B", "0.6B"}


def test_unknown_engine_has_no_metadata():
    assert engine_languages("no-such-engine") == []
    assert engine_model_sizes("no-such-engine") == []


# ── The endpoint ─────────────────────────────────────────────────────


def test_engines_endpoint_lists_every_tts_engine(client):
    r = client.get("/engines")
    assert r.status_code == 200, r.text
    engines = {e["engine"] for e in r.json()["engines"]}
    assert engines == set(TTS_ENGINES)


def test_engines_endpoint_matches_the_helpers(client):
    """The endpoint is what clients gate features on, so it must not diverge
    from the helpers the backend itself uses."""
    for entry in client.get("/engines").json()["engines"]:
        assert entry["supports_instruct"] == engine_supports_instruct(entry["engine"])
        assert entry["languages"] == engine_languages(entry["engine"])
        assert entry["model_sizes"] == engine_model_sizes(entry["engine"])


def test_engines_endpoint_reports_qwen_correctly(client):
    entry = next(e for e in client.get("/engines").json()["engines"] if e["engine"] == "qwen")
    assert entry["supports_instruct"] is False
    assert entry["has_model_sizes"] is True
    assert entry["display_name"] == "Qwen TTS"


# ── No more silent drops ─────────────────────────────────────────────


def test_warns_when_instruct_would_be_discarded(caplog):
    with caplog.at_level(logging.WARNING, logger="backend.routes.generations"):
        _warn_if_instruct_ignored("speak angrily", "qwen")
    assert any("does not honour" in r.getMessage() for r in caplog.records)


def test_no_warning_when_the_engine_honours_instruct(caplog):
    with caplog.at_level(logging.WARNING, logger="backend.routes.generations"):
        _warn_if_instruct_ignored("speak angrily", "qwen_custom_voice")
    assert not caplog.records


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_warning_without_an_instruction(caplog, value):
    """Every generate call passes instruct=None. Warning on that would bury the
    real case in noise."""
    with caplog.at_level(logging.WARNING, logger="backend.routes.generations"):
        _warn_if_instruct_ignored(value, "qwen")
    assert not caplog.records


def test_warning_names_a_working_alternative(caplog):
    """A warning that only says 'no' costs the caller another round of
    guessing; it should say what does work."""
    with caplog.at_level(logging.WARNING, logger="backend.routes.generations"):
        _warn_if_instruct_ignored("speak angrily", "qwen")
    rendered = caplog.records[0].getMessage()
    assert "qwen_custom_voice" in rendered
