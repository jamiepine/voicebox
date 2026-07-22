"""
Integration tests for the OpenAI-compatible LLM backend.

These tests hit a **real** endpoint — no HTTP mocks — because the whole
point of the backend is to survive whatever quirks a real server's
response shape has. They are skipped automatically when no reachable
endpoint is configured, so ``pytest`` on a fresh clone still returns
green.

To run them, boot any OpenAI-compatible server (llama.cpp,
vLLM, LM Studio, LocalAI, ALICE-LLM's ``server`` binary, or the OpenAI
API itself), then:

    export VOICEBOX_TEST_OPENAI_COMPAT_URL=http://localhost:8090/v1
    export VOICEBOX_TEST_OPENAI_COMPAT_MODEL=<the model the server serves>
    # optional if the endpoint requires bearer auth
    export VOICEBOX_TEST_OPENAI_COMPAT_API_KEY=<token>
    pytest backend/tests/test_openai_compat_integration.py -v
"""

import os
import socket
from urllib.parse import urlparse

import pytest

from backend.backends import (
    get_llm_backend,
    reset_backends,
    set_llm_config,
)
from backend.backends.openai_compat_backend import OpenAICompatLLMBackend


ENDPOINT_ENV = "VOICEBOX_TEST_OPENAI_COMPAT_URL"
MODEL_ENV = "VOICEBOX_TEST_OPENAI_COMPAT_MODEL"
API_KEY_ENV = "VOICEBOX_TEST_OPENAI_COMPAT_API_KEY"


def _endpoint() -> str | None:
    return os.environ.get(ENDPOINT_ENV) or None


def _model() -> str | None:
    return os.environ.get(MODEL_ENV) or None


def _api_key() -> str | None:
    return os.environ.get(API_KEY_ENV) or None


def _endpoint_reachable(url: str) -> bool:
    """Cheap TCP probe so we skip cleanly when nothing is listening.

    A full HTTP GET would work too, but that couples the skip to whichever
    path (``/v1``, ``/health``, ``/models``…) the local server happens to
    expose. TCP handshake to the ``host:port`` from the URL is enough to
    tell us "run the test" vs "user hasn't started their server".
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def _endpoint_ready() -> bool:
    url = _endpoint()
    model = _model()
    if not url or not model:
        return False
    return _endpoint_reachable(url)


pytestmark = pytest.mark.skipif(
    not _endpoint_ready(),
    reason=(
        f"OpenAI-compatible endpoint not configured or unreachable. Set "
        f"{ENDPOINT_ENV} and {MODEL_ENV} and start the server to run "
        "these integration tests."
    ),
)


@pytest.fixture(autouse=True)
def _reset_backends_between_tests():
    reset_backends()
    yield
    reset_backends()


async def test_backend_generates_non_empty_response():
    """One real round trip against the configured endpoint."""
    backend = OpenAICompatLLMBackend(
        endpoint=_endpoint(),
        model=_model(),
        api_key=_api_key(),
    )
    reply = await backend.generate(
        prompt="Reply with exactly the word: pong",
        system="You are a terse test harness. Follow the instruction literally.",
        max_tokens=16,
        temperature=0.0,
    )
    assert isinstance(reply, str)
    assert reply.strip(), "Endpoint returned an empty string"


async def test_backend_respects_examples_history():
    """Few-shot examples reach the endpoint as proper user/assistant turns."""
    backend = OpenAICompatLLMBackend(
        endpoint=_endpoint(),
        model=_model(),
        api_key=_api_key(),
    )
    reply = await backend.generate(
        prompt="hola",
        system=(
            "You are a translation bot. Reply with the single-word English "
            "translation of the user's message and nothing else."
        ),
        max_tokens=16,
        temperature=0.0,
        examples=[
            ("bonjour", "hello"),
            ("guten tag", "hello"),
        ],
    )
    assert reply.strip(), "Endpoint returned an empty translation"


async def test_dispatch_returns_openai_compat_when_config_set():
    """``get_llm_backend()`` routes to the custom backend once config is on."""
    set_llm_config(
        endpoint=_endpoint(),
        model=_model(),
        api_key=_api_key(),
    )
    backend = get_llm_backend()
    assert isinstance(backend, OpenAICompatLLMBackend)
    assert backend.endpoint == _endpoint().rstrip("/")
    assert backend.model == _model()


async def test_dispatch_reverts_when_config_cleared():
    """Clearing the endpoint falls back to the on-device Qwen path."""
    set_llm_config(
        endpoint=_endpoint(),
        model=_model(),
        api_key=_api_key(),
    )
    assert isinstance(get_llm_backend(), OpenAICompatLLMBackend)

    set_llm_config(endpoint=None, model=None, api_key=None)
    fallback = get_llm_backend()
    assert not isinstance(fallback, OpenAICompatLLMBackend)


async def test_dispatch_rebuilds_when_endpoint_changes():
    """A URL swap drops the cached backend so the next call rebuilds."""
    set_llm_config(
        endpoint=_endpoint(),
        model=_model(),
        api_key=_api_key(),
    )
    first = get_llm_backend()

    # Point at a syntactically different URL. We're testing the cache
    # invalidation path — the second URL doesn't need to be reachable.
    set_llm_config(
        endpoint=_endpoint() + "/alt",
        model=_model(),
        api_key=_api_key(),
    )
    second = get_llm_backend()

    assert first is not second
    assert isinstance(second, OpenAICompatLLMBackend)
    assert second.endpoint.endswith("/alt")
