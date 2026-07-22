"""
OpenAI-compatible LLM backend.

Delegates chat completion to any HTTP endpoint that speaks the OpenAI
``/v1/chat/completions`` protocol — llama.cpp server, vLLM, LocalAI,
LM Studio, Ollama's OpenAI-compat shim, or the OpenAI API itself.
Skips the on-device download / model-load path because the remote
server is expected to serve the model.

The user configures ``custom_llm_endpoint`` / ``custom_llm_model`` /
``custom_llm_api_key`` in capture settings; when the endpoint is set,
``get_llm_backend()`` returns an instance of this class instead of the
built-in Qwen3 backend.
"""

import logging
from typing import Optional

import httpx

from . import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


class OpenAICompatLLMBackend:
    """LLM backend that delegates to an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ):
        # Strip a trailing slash so callers can pass either
        # ``http://host:port/v1`` or ``http://host:port/v1/``.
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        # Empty string means "no auth" — the app UI stores an empty field
        # the same way an unset field looks in the DB.
        self.api_key = api_key or None
        self.timeout = timeout
        # ``model_size`` / ``_current_model_size`` are read by the shared
        # download-progress / unload plumbing on other backends. Report the
        # remote model name so the routes layer can echo it back to clients.
        self.model_size = model
        self._current_model_size = model

    def is_loaded(self) -> bool:
        # A remote endpoint is always "loaded" from this process's point of
        # view — no local weights to page in.
        return True

    async def load_model(self, model_size: Optional[str] = None) -> None:  # noqa: ARG002
        """No-op — remote endpoint holds the model."""
        return

    def unload_model(self) -> None:
        """No-op — nothing to unload locally."""
        return

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,  # noqa: ARG002 — kept for protocol parity
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        messages = _build_messages(prompt, system, examples)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.endpoint}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Truncate the body — some servers echo the whole prompt in
                # error responses, and log files fill up fast otherwise.
                body_preview = exc.response.text[:500]
                logger.error(
                    "OpenAI-compat endpoint %s returned %d: %s",
                    url,
                    exc.response.status_code,
                    body_preview,
                )
                raise
            except httpx.RequestError as exc:
                logger.error("OpenAI-compat endpoint %s request failed: %s", url, exc)
                raise

            data = response.json()

        return _extract_content(data, url)


def _build_messages(
    prompt: str,
    system: Optional[str],
    examples: Optional[list[tuple[str, str]]],
) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if examples:
        for user_text, assistant_text in examples:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_content(data: dict, url: str) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in response from {url}: {data!r}")

    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        # Fallback: some servers still return the legacy text-completion
        # shape (``choices[0].text``) even from the chat endpoint.
        content = first.get("text")
    if content is None:
        raise ValueError(f"No content in response from {url}: {data!r}")
    return content.strip()
