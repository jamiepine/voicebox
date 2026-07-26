"""Ollama bootstrap and chat client.

Handles three things: making sure an Ollama server is reachable, making sure
the requested model is present (pulling it if not), and running tool-calling
chat turns against it.

Installing the Ollama *binary* is deliberately opt-in. Pulling a model is a
download into a cache directory; piping a vendor install script into a shell
is a different category of action, so it only happens when the operator sets
VOXGUARD_AUTO_INSTALL_OLLAMA=1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
import subprocess
from typing import Any, Callable

import aiohttp

log = logging.getLogger(__name__)

INSTALL_HINTS = {
    "Linux": "curl -fsSL https://ollama.com/install.sh | sh",
    "Darwin": "brew install ollama   (or download from https://ollama.com/download)",
    "Windows": "winget install Ollama.Ollama   (or download from https://ollama.com/download)",
}


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str, *, auto_install: bool = False) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.auto_install = auto_install
        self._session: aiohttp.ClientSession | None = None
        self._server_proc: subprocess.Popen | None = None
        self._ready_models: set[str] = set()
        self._pull_lock = asyncio.Lock()

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=900, sock_connect=10)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()

    # -- bootstrap ----------------------------------------------------------

    async def is_up(self) -> bool:
        try:
            sess = await self._sess()
            async with sess.get(
                f"{self.host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                return r.status == 200
        except Exception:
            return False

    async def ensure_server(self) -> None:
        """Make sure something is listening on the Ollama host."""
        if await self.is_up():
            return

        binary = shutil.which("ollama")
        if binary is None and self.auto_install:
            binary = await self._install()

        if binary is None:
            hint = INSTALL_HINTS.get(platform.system(), "https://ollama.com/download")
            raise OllamaError(
                f"Ollama isn't running at {self.host} and the `ollama` binary wasn't found.\n"
                f"Install it with:  {hint}\n"
                "Or set VOXGUARD_AUTO_INSTALL_OLLAMA=1 to let the bot install it on startup."
            )

        log.info("Starting `ollama serve`...")
        self._server_proc = subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        for _ in range(30):
            await asyncio.sleep(1)
            if await self.is_up():
                log.info("Ollama is up at %s", self.host)
                return
        raise OllamaError(f"Started `ollama serve` but {self.host} never became reachable.")

    async def _install(self) -> str | None:
        system = platform.system()
        if system == "Linux":
            cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        elif system == "Darwin" and shutil.which("brew"):
            cmd = "brew install ollama"
        elif system == "Windows" and shutil.which("winget"):
            cmd = "winget install --silent --accept-package-agreements Ollama.Ollama"
        else:
            log.error("No automatic install path for %s. Install Ollama manually.", system)
            return None

        log.warning("Installing Ollama via: %s", cmd)
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            log.error("Ollama install failed:\n%s", (out or b"").decode(errors="replace")[-2000:])
            return None
        return shutil.which("ollama")

    async def ensure_model(
        self, model: str | None = None, on_progress: Callable[[str], Any] | None = None
    ) -> str:
        """Pull the model if it isn't already local. Returns the model name."""
        model = model or self.model
        if model in self._ready_models:
            return model

        async with self._pull_lock:
            if model in self._ready_models:
                return model
            await self.ensure_server()

            sess = await self._sess()
            async with sess.get(f"{self.host}/api/tags") as r:
                tags = await r.json() if r.status == 200 else {"models": []}
            have = {m.get("name", "") for m in tags.get("models", [])}
            # `llama3.1:8b` and a bare `llama3.1` refer to the same pull.
            if model in have or f"{model}:latest" in have:
                self._ready_models.add(model)
                return model

            log.info("Pulling Ollama model '%s' (first run — this may take a while)...", model)
            last_pct = -1
            async with sess.post(
                f"{self.host}/api/pull", json={"model": model, "stream": True}
            ) as r:
                if r.status != 200:
                    raise OllamaError(f"pull failed ({r.status}): {(await r.text())[:300]}")
                async for raw in r.content:
                    if not raw.strip():
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if error := event.get("error"):
                        raise OllamaError(f"pull failed: {error}")
                    total, done = event.get("total"), event.get("completed")
                    if total and done:
                        pct = int(done / total * 100)
                        if pct >= last_pct + 10:
                            last_pct = pct
                            message = f"Pulling {model}: {pct}%"
                            log.info(message)
                            if on_progress:
                                await _maybe_await(on_progress(message))

            self._ready_models.add(model)
            log.info("Model '%s' ready.", model)
            return model

    # -- inference ----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        num_predict: int = 400,
        json_mode: bool = False,
    ) -> dict:
        """Single chat turn. Returns the assistant message dict."""
        model = await self.ensure_model(model or self.model)
        sess = await self._sess()
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if tools:
            body["tools"] = tools
        if json_mode:
            body["format"] = "json"

        async with sess.post(f"{self.host}/api/chat", json=body) as r:
            if r.status != 200:
                raise OllamaError(f"chat failed ({r.status}): {(await r.text())[:300]}")
            payload = await r.json()

        message = payload.get("message") or {}
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        return message


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value
