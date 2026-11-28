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

_SCHEME = r"[a-zA-Z][a-zA-Z0-9+.\-]*://"
_URL_IN_TEXT = re.compile(rf"{_SCHEME}(?:(?!{_SCHEME})[^\s<>`])+")
_PARAM_BEFORE_URL = re.compile(r"[?&;#]([^\s?&;#=]+)=[^\s?&;#=]*\Z")
_URL_TRAILING_PUNCTUATION = "'\".,;:!)]}"
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
        name, _sep, value = part.partition("=")
        if value and _is_secret_param(name):
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
        path = _redact_params(parts.path) if "=" in parts.path else parts.path
        query = _redact_params(parts.query)
        fragment = _redact_params(parts.fragment) if "=" in parts.fragment else parts.fragment
        redacted = urlunsplit((parts.scheme, netloc, path, query, fragment))
        if schemeless and redacted.startswith("//"):
            redacted = redacted[2:]
        return redacted
    except Exception:
        try:
            return _redact_url_fallback(url)
        except Exception:
            return REDACTED


def _redact_url_match(match: re.Match) -> str:
    url = match.group(0)
    trimmed = url.rstrip(_URL_TRAILING_PUNCTUATION)
    param = _PARAM_BEFORE_URL.search(match.string, 0, match.start())
    if param and _is_secret_param(param.group(1)):
        return REDACTED + url[len(trimmed) :]
    return redact_url(trimmed) + url[len(trimmed) :]


def redact_urls_in_text(text: str) -> str:
    """Apply ``redact_url`` to every ``scheme://`` URL inside free text.

    For exception messages (httpx, for one) that embed the request URL.
    Quotes inside a URL stay part of it, so redact the whole text before
    truncating it: a cut inside the userinfo hides the ``@`` that marks it.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        return _URL_IN_TEXT.sub(_redact_url_match, text)
    except Exception:
        return REDACTED
