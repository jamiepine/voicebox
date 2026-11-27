"""Host / Origin checks and baseline response headers for the local API.

The server listens on loopback by default, and any web page the user opens
can make their browser send requests to it. ``RequestGuardMiddleware``
blocks the two ways a page can do that:

* DNS rebinding: a page on ``attacker.example`` re-points its hostname at
  127.0.0.1, so its requests reach this server as same-origin. The request
  still carries ``Host: attacker.example``, so connections that arrive on a
  loopback address must name a loopback host (or one listed in
  ``VOICEBOX_ALLOWED_HOSTS``).
* Cross-site requests: CORS only hides the response from the page, while
  a "simple" POST still runs. Requests with an ``Origin`` header must come
  from one of the app's own origins, the server's own origin, or
  ``VOICEBOX_CORS_ORIGINS``. Requests without ``Origin`` (curl, SDKs, MCP
  clients) are not browser cross-site requests and pass.

Every response also gets ``X-Content-Type-Options: nosniff``.
"""

import ipaddress
import json
import logging
import os
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:17493",
    "http://127.0.0.1:17493",
    "tauri://localhost",  # Tauri webview (macOS)
    "https://tauri.localhost",  # Tauri webview (Windows/Linux)
    "http://tauri.localhost",  # Tauri webview (Windows, some builds)
]

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def get_cors_origins() -> list[str]:
    """App origins plus ``VOICEBOX_CORS_ORIGINS``, shared by CORS and the Origin check."""
    return DEFAULT_CORS_ORIGINS + _env_list("VOICEBOX_CORS_ORIGINS")


def _hostname(host: str) -> str:
    """Strip the port from a Host header value (``[::1]:17493`` -> ``[::1]``)."""
    host = host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def _arrived_on_loopback(scope: Scope) -> bool:
    server = scope.get("server")
    if not server:
        return True
    try:
        address = ipaddress.ip_address(server[0])
    except ValueError:
        return True
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


class RequestGuardMiddleware:
    """Pure ASGI middleware, so streaming and SSE responses pass through untouched."""

    def __init__(self, app: ASGIApp, allowed_origins: list[str], allowed_hosts: list[str]) -> None:
        self.app = app
        self.allow_any_origin = "*" in allowed_origins
        self.allowed_origins = {origin.rstrip("/").lower() for origin in allowed_origins}
        self.allow_any_host = "*" in allowed_hosts
        self.check_host_everywhere = bool(allowed_hosts) and not self.allow_any_host
        self.allowed_hosts = LOOPBACK_HOSTS | {_hostname(host) for host in allowed_hosts}

    def _rejection(self, scope: Scope) -> str | None:
        headers = {}
        for key, value in scope.get("headers", []):
            if key in (b"host", b"origin"):
                headers[key] = value.decode("latin-1")
        host = headers.get(b"host", "")
        origin = headers.get(b"origin")

        check_host = not self.allow_any_host and (self.check_host_everywhere or _arrived_on_loopback(scope))
        if check_host and _hostname(host) not in self.allowed_hosts:
            return f"Host {host!r} is not allowed"

        if origin is None or self.allow_any_origin:
            return None
        if origin.lower() in self.allowed_origins:
            return None
        parsed = urlsplit(origin)
        if parsed.scheme in ("http", "https") and host and parsed.netloc.lower() == host.strip().lower():
            return None
        return f"Origin {origin!r} is not allowed"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        reason = self._rejection(scope)
        if reason is not None:
            logger.warning(
                "Rejected %s %s: %s (see VOICEBOX_ALLOWED_HOSTS / VOICEBOX_CORS_ORIGINS)",
                scope.get("method", "WEBSOCKET"),
                scope.get("path", ""),
                reason,
            )
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            body = json.dumps({"detail": reason}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"x-content-type-options", b"nosniff"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        async def send_with_nosniff(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(key.lower() == b"x-content-type-options" for key, _ in headers):
                    headers.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_nosniff)


def configure_request_security(application: FastAPI) -> None:
    """Install CORS and the Host/Origin guard, with the guard outside CORS."""
    origins = get_cors_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(
        RequestGuardMiddleware,
        allowed_origins=origins,
        allowed_hosts=_env_list("VOICEBOX_ALLOWED_HOSTS"),
    )
