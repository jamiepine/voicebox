"""Tests for backend.utils.redact."""

import pytest

from backend.utils.redact import REDACTED, redact_url, redact_urls_in_text


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://llm.example.com/v1", "https://llm.example.com/v1"),
        ("https://user:pass@llm.example.com/v1", "https://llm.example.com/v1"),
        ("https://token@llm.example.com:8443/v1", "https://llm.example.com:8443/v1"),
        ("https://us%40er:p@ss@llm.example.com/v1", "https://llm.example.com/v1"),
        ("http://user:pw@[::1]:8080/v1?model=x", "http://[::1]:8080/v1?model=x"),
        ("user:pw@localhost:8080/v1?key=abc", "localhost:8080/v1?key=REDACTED"),
        ("localhost:8080/v1", "localhost:8080/v1"),
        ("/v1/chat?token=abc", "/v1/chat?token=REDACTED"),
    ],
)
def test_strips_userinfo_and_keeps_the_rest(url, expected):
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    "name",
    [
        "api-key",
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "auth",
        "authorization",
        "sig",
        "signature",
        "secret",
        "client_secret",
        "password",
        "x-api-key",
        "X-Amz-Signature",
        "X-Amz-Credential",
        "X-Amz-Security-Token",
        "subscription_key",
        "Api-Key",
        "TOKEN",
    ],
)
def test_redacts_secret_query_params(name):
    url = f"https://h.example/v1?model=gpt&{name}=s3cret&stream=false"
    assert redact_url(url) == f"https://h.example/v1?model=gpt&{name}={REDACTED}&stream=false"


def test_leaves_ordinary_params_alone():
    url = "https://h.example/v1?model=gpt-4o&temperature=0.2&flag&empty="
    assert redact_url(url) == url


def test_redacts_every_repeated_param():
    assert (
        redact_url("https://h/v1?token=a&x=1&token=b&TOKEN=c")
        == "https://h/v1?token=REDACTED&x=1&token=REDACTED&TOKEN=REDACTED"
    )


def test_semicolon_separated_params():
    assert redact_url("https://h/v1?a=1;sig=2") == "https://h/v1?a=1;sig=REDACTED"


def test_url_encoded_param_name():
    assert redact_url("https://h/v1?api%5Fkey=abc") == "https://h/v1?api%5Fkey=REDACTED"


def test_fragment_params_are_redacted_and_plain_fragments_kept():
    assert redact_url("https://h/cb#access_token=abc&state=xyz") == "https://h/cb#access_token=REDACTED&state=xyz"
    assert redact_url("https://h/docs#section-2") == "https://h/docs#section-2"


@pytest.mark.parametrize(
    "url",
    [
        "http://user:s3cret@[::1/v1?token=s3cret",
        "http://user:s3cret@[bad::host]]:x/v1?api_key=s3cret",
    ],
)
def test_malformed_input_falls_back_without_raising(url):
    redacted = redact_url(url)
    assert "s3cret" not in redacted
    assert redacted.startswith("http://")


@pytest.mark.parametrize("value", ["", None, 42, b"https://u:p@h"])
def test_non_string_and_empty_input_pass_through(value):
    assert redact_url(value) == value


def test_redacts_urls_embedded_in_text():
    text = (
        "Client error '401 Unauthorized' for url "
        "'https://bob:s3cret@llm.example/v1/chat/completions?api_key=s3cret'\n"
        "See https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401"
    )
    redacted = redact_urls_in_text(text)
    assert "s3cret" not in redacted
    assert "'https://llm.example/v1/chat/completions?api_key=REDACTED'" in redacted
    assert "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401" in redacted


def test_text_without_urls_is_unchanged():
    assert redact_urls_in_text("[Errno 111] Connection refused") == "[Errno 111] Connection refused"
