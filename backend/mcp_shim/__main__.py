"""voicebox-mcp — stdio ↔ Streamable-HTTP MCP proxy.

Some MCP clients only speak stdio. They spawn this binary, we pipe each
JSON-RPC message to ``http://127.0.0.1:<port>/mcp/``, and stream the
server's response back. The Voicebox server does all the real work.

Environment variables:
  VOICEBOX_PORT       Voicebox server port (default 17493).
  VOICEBOX_HOST       Host (default 127.0.0.1).
  VOICEBOX_CLIENT_ID  Forwarded as X-Voicebox-Client-Id on every request.

Stdout is JSON-RPC only. Diagnostics go to stderr.
Exit 0 on clean EOF, 1 on transport error, 2 if backend never answers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx


CLIENT_ID_HEADER = "X-Voicebox-Client-Id"
SESSION_HEADER = "mcp-session-id"
HEALTH_TIMEOUT_S = 30.0
DEFAULT_PORT = 17493


def _err(msg: str) -> None:
    print(f"voicebox-mcp: {msg}", file=sys.stderr, flush=True)


def _base_url() -> tuple[str, str]:
    host = os.environ.get("VOICEBOX_HOST", "127.0.0.1")
    port = int(os.environ.get("VOICEBOX_PORT", str(DEFAULT_PORT)))
    return f"http://{host}:{port}/mcp/", f"http://{host}:{port}/health"


async def _wait_for_backend(client: httpx.AsyncClient, health_url: str) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + HEALTH_TIMEOUT_S
    while loop.time() < deadline:
        try:
            r = await client.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def _read_stdin_line() -> str | None:
    """Async-read a single line from stdin. Returns None on EOF."""
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    if not line:
        return None
    return line


def _write_stdout(obj: Any) -> None:
    """Write a JSON object to stdout as one line, flushed."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


async def _handle_request(
    client: httpx.AsyncClient,
    url: str,
    raw: str,
    headers: dict[str, str],
    session_id: list[str | None],
) -> None:
    """Forward one JSON-RPC payload to the server and relay the response."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        _err(f"invalid JSON on stdin: {exc}")
        return

    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **headers,
    }
    if session_id[0]:
        req_headers[SESSION_HEADER] = session_id[0]

    # Notifications (no "id") don't expect a response body. Server returns
    # 202 Accepted and we stay quiet.
    is_notification = isinstance(message, dict) and "id" not in message

    async with client.stream(
        "POST", url, headers=req_headers, content=raw.encode("utf-8")
    ) as response:
        # Capture session id on initialize.
        if session_id[0] is None:
            sid = response.headers.get(SESSION_HEADER)
            if sid:
                session_id[0] = sid

        if response.status_code == 202:
            return  # notification acknowledged
        if response.status_code >= 400:
            body = await response.aread()
            _err(
                f"server {response.status_code}: "
                f"{body.decode('utf-8', errors='replace')[:400]}"
            )
            if is_notification:
                return
            _write_stdout(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {
                        "code": -32000,
                        "message": (
                            f"Voicebox MCP proxy got HTTP {response.status_code}"
                        ),
                    },
                }
            )
            return

        ctype = response.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            # SSE frames: lines prefixed "data: ..." contain the JSON-RPC msg.
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        _write_stdout(json.loads(payload))
                    except json.JSONDecodeError:
                        _err(f"malformed SSE payload: {payload[:200]}")
        else:
            body = await response.aread()
            try:
                _write_stdout(json.loads(body))
            except json.JSONDecodeError:
                _err(
                    f"non-JSON response ({ctype}): "
                    f"{body.decode('utf-8', errors='replace')[:200]}"
                )


async def _run() -> int:
    url, health_url = _base_url()
    forward_headers: dict[str, str] = {}
    client_id = os.environ.get("VOICEBOX_CLIENT_ID")
    if client_id:
        forward_headers[CLIENT_ID_HEADER] = client_id

    session_id: list[str | None] = [None]

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        if not await _wait_for_backend(client, health_url):
            _err(
                f"timed out waiting for Voicebox at {health_url} — is the app open?"
            )
            return 2

        try:
            while True:
                line = await _read_stdin_line()
                if line is None:
                    return 0
                line = line.strip()
                if not line:
                    continue
                await _handle_request(
                    client, url, line, forward_headers, session_id
                )
        except (KeyboardInterrupt, SystemExit):
            return 0
        except Exception as exc:
            _err(f"proxy failed: {exc!r}")
            return 1


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
