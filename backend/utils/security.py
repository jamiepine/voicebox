import os
import logging
import ipaddress
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def is_loopback(host: str | None) -> bool:
    """Check if the given host IP is a loopback address."""
    if not host:
        return True  # Internal calls or tests default to True
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError, Exception):
        return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware to secure the REST API for non-loopback callers.
    
    1. Loopback callers (localhost/127.0.0.1/::1) are always allowed.
    2. Non-loopback callers:
       - If VOICEBOX_API_KEY is configured in the environment, require matching Bearer token.
       - If VOICEBOX_API_KEY is NOT configured, block all administrative or destructive requests (DELETE, shutdown, etc.) with 403 Forbidden.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            if request.method.upper() == "OPTIONS":
                return await call_next(request)

            client_host = request.client.host if request.client else None

            # Loopback is always trusted
            if is_loopback(client_host):
                return await call_next(request)

            # Check for configured API key
            api_key = os.environ.get("VOICEBOX_API_KEY")

            # Extract authorization token
            auth_header = request.headers.get("Authorization")
            token = None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()

            if api_key:
                # Enforce API key verification for all remote requests
                if not token or token != api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: Invalid or missing API token"}
                    )
            else:
                # Destructive/Administrative endpoints are blocked for remote callers by default
                method = request.method.upper()
                path = request.url.path.lower()
                
                is_destructive = False

                # Any DELETE request is considered destructive
                if method == "DELETE":
                    is_destructive = True

                # Check for specific administrative/destructive paths
                admin_paths = (
                    "/shutdown",
                    "/watchdog/disable",
                    "/cache/clear",
                    "/tasks/clear",
                    "/models/unload",
                    "/models/download",
                    "/models/migrate",
                )

                if any(path.startswith(p) for p in admin_paths) or any(f"{p}/" in path for p in admin_paths):
                    is_destructive = True

                if is_destructive:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "Access denied: Administrative/destructive endpoints are restricted to loopback callers. Set VOICEBOX_API_KEY to enable remote access."
                        }
                    )

            return await call_next(request)
        except Exception as e:
            logger.exception("Error in SecurityMiddleware: %s", e)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error in security middleware"}
            )
