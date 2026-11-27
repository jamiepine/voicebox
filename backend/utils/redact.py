"""Strip credentials from URLs before they reach logs or error messages.

User-configured endpoints (the custom LLM endpoint, proxies, mirrors) can
carry secrets in the userinfo (``https://user:pass@host``) or in the query
string (``?api_key=...``, pre-signed ``X-Amz-Signature``, ``?token=...``).
These helpers keep the scheme, host, port and path readable and replace
everything secret-looking with ``REDACTED``.
"""

import re
from urllib.parse import unquote_plus, urlsplit, urlunsplit

REDACTED = "REDACTED"

_SECRET_PARAM_NAMES = frozenset({"key", "sig", "auth", "authorization", "code", "pwd", "jwt", "bearer"})
_SECRET_PARAM_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "signature",
    "credential",
    "apikey",
    "api_key",
    "access_key",
    "private_key",
    "session",
)

_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>`]+")
_FALLBACK_USERINFO = re.compile(r"(//)[^/?#\s]*@")
_FALLBACK_QUERY = re.compile(r"([?#])\S*")


def _is_secret_param(name: str) -> bool:
    normalized = unquote_plus(name).strip().lower().replace("-", "_")
    if normalized in _SECRET_PARAM_NAMES or normalized.endswith("_key"):
        return True
    return any(part in normalized for part in _SECRET_PARAM_PARTS)


def _redact_params(params: str) -> str:
    parts = []
    for part in re.split(r"([&;])", params):
        name, sep, _value = part.partition("=")
        if sep and _is_secret_param(name):
            part = f"{name}={REDACTED}"
        parts.append(part)
    return "".join(parts)


def _redact_url_fallback(url: str) -> str:
    url = _FALLBACK_USERINFO.sub(r"\1", url)
    return _FALLBACK_QUERY.sub(rf"\1{REDACTED}", url, count=1)


def redact_url(url: str) -> str:
    """Return ``url`` with userinfo removed and secret query values redacted.

    Never raises: input that does not parse falls back to dropping the
    userinfo and replacing the whole query and fragment with ``REDACTED``.
    Non-string input is returned unchanged.
    """
    if not isinstance(url, str) or not url:
        return url
    try:
        schemeless = "://" not in url and not url.startswith(("/", "?", "#"))
        parts = urlsplit(f"//{url}" if schemeless else url)
        netloc = parts.netloc.rpartition("@")[2]
        query = _redact_params(parts.query)
        fragment = _redact_params(parts.fragment) if "=" in parts.fragment else parts.fragment
        redacted = urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
        if schemeless and redacted.startswith("//"):
            redacted = redacted[2:]
        return redacted
    except Exception:
        try:
            return _redact_url_fallback(url)
        except Exception:
            return REDACTED


def redact_urls_in_text(text: str) -> str:
    """Apply ``redact_url`` to every ``scheme://`` URL inside free text.

    For exception messages (httpx, for one) that embed the request URL.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        return _URL_IN_TEXT.sub(lambda m: redact_url(m.group(0)), text)
    except Exception:
        return REDACTED
