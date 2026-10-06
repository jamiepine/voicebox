import os
import logging
import ipaddress
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str | None:
    """Extract the client IP address.

    X-Forwarded-For is only honoured when the direct TCP peer is itself
    loopback (i.e. a trusted local reverse proxy). Otherwise a remote
    attacker could forge the header (e.g. "X-Forwarded-For: 127.0.0.1")
    to impersonate a loopback caller and bypass all security checks.
    """
    direct_host = request.client.host if request.client else None

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded and is_loopback(direct_host):
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return client_ip

    return direct_host


def is_loopback(host: str | None) -> bool:
    """Check if the given host IP is a loopback address.
    
    Fails closed (returns False) if host is None or missing.
    """
    if not host:
        return False  # Fail closed for missing/unknown client address
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError, Exception):
        return False


# Endpoints considered safe to expose to remote callers when no API key is
# configured: read-only, non-destructive, and carrying no sensitive data.
# Everything else (including every mutating request) requires either a
# loopback caller or a valid VOICEBOX_API_KEY bearer token. This is an
# allowlist, not a denylist, so newly added routes are protected by default.
SAFE_REMOTE_GET_PATHS = frozenset({
    "/",
    "/health",
    "/health/filesystem",
})


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware to secure the REST API for non-loopback callers.

    1. Loopback callers (localhost/127.0.0.1/::1) are always allowed.
    2. Non-loopback callers:
       - If VOICEBOX_API_KEY is configured in the environment, require matching Bearer token.
       - If VOICEBOX_API_KEY is NOT configured, only GET requests to SAFE_REMOTE_GET_PATHS
         are allowed; every other request gets 403 Forbidden.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            if request.method.upper() == "OPTIONS":
                return await call_next(request)

            client_host = get_client_ip(request)

            # Loopback is always trusted
            if is_loopback(client_host):
                return await call_next(request)

            # Check for configured API key
            api_key = os.environ.get("VOICEBOX_API_KEY")

            # Extract authorization token (scheme name is case-insensitive per RFC 7235)
            auth_header = request.headers.get("Authorization")
            token = None
            if auth_header and auth_header[:7].lower() == "bearer ":
                token = auth_header[7:].strip()

            if api_key:
                # Enforce API key verification for all remote requests
                if not token or token != api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: Invalid or missing API token"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            else:
                # No API key configured: only allow a small explicit allowlist
                # of safe read-only endpoints for remote callers.
                method = request.method.upper()
                path = request.url.path

                if method != "GET" or path not in SAFE_REMOTE_GET_PATHS:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "Access denied: this endpoint is restricted to loopback callers. Set VOICEBOX_API_KEY to enable remote access."
                        }
                    )

            return await call_next(request)
        except Exception as e:
            logger.exception("Error in SecurityMiddleware: %s", e)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error in security middleware"}
            )
