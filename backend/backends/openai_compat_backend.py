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

import json
import logging
from typing import AsyncIterator, Optional

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
        """Configure the backend for a specific remote endpoint.

        Args:
            endpoint: Full base URL up to and including the ``/v1``
                segment (trailing slash tolerated).
            model: Model name to send in every request's ``model`` field.
            api_key: Optional bearer token; empty string treated the same
                as ``None`` so a blank UI field disables auth.
            timeout: Per-request timeout in seconds. Defaults to two
                minutes so slow remote hosts don't cut off long
                generations.
        """
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
        """Return ``True`` unconditionally — the remote server owns model state.

        Voicebox's shared plumbing polls ``is_loaded`` to decide whether
        to show a "downloading model…" progress bar; since a remote
        endpoint has nothing to page in from this process's point of
        view, the answer is always yes.
        """
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
        """Post one non-streaming chat completion and return the assistant text.

        Mirrors ``LLMBackend.generate`` — the same call signature the
        built-in Qwen backends use — so refinement and personality
        services can swap between local and remote transparently.
        ``model_size`` is accepted for protocol parity but ignored:
        remote model selection is pinned at construction time.

        Raises:
            httpx.HTTPStatusError: The remote returned a non-2xx status.
            httpx.RequestError: Transport failure (timeout, DNS, TLS).
            ValueError: The response was well-formed HTTP but carried no
                content in either the chat or legacy completion shapes.
        """
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

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,  # noqa: ARG002 — kept for protocol parity
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Stream the assistant reply as text deltas via OpenAI SSE.

        Each yielded string is the ``delta.content`` field of one chunk.
        Consumers accumulate these to reconstruct the full reply; the
        streaming TTS pipeline in ``chunked_tts`` uses the accumulator
        to fire per-sentence TTS as soon as a sentence boundary lands.
        """
        messages = _build_messages(prompt, system, examples)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.endpoint}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        delta = _parse_sse_delta(line)
                        if delta is _SSE_DONE:
                            break
                        if delta:
                            yield delta
            except httpx.HTTPStatusError as exc:
                body_preview = ""
                try:
                    body_preview = (await exc.response.aread()).decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                logger.error(
                    "OpenAI-compat streaming endpoint %s returned %d: %s",
                    url,
                    exc.response.status_code,
                    body_preview,
                )
                raise
            except httpx.RequestError as exc:
                logger.error("OpenAI-compat streaming endpoint %s request failed: %s", url, exc)
                raise

    def supports_streaming(self) -> bool:
        return True


# Sentinel returned by ``_parse_sse_delta`` when the terminal ``[DONE]``
# marker arrives — an in-band value distinct from "no content to yield".
_SSE_DONE = object()


def _parse_sse_delta(line: str) -> object:
    """Extract ``delta.content`` from a single SSE line.

    Returns:
      - ``_SSE_DONE`` when the terminal ``data: [DONE]`` marker arrives.
      - An empty string on lines with no useful payload (comments,
        keep-alives, role-only deltas, blank frames) — callers should skip.
      - The delta content string otherwise.
    """
    if not line or not line.startswith("data:"):
        # Comments start with ``:`` (keep-alive heartbeats) or the line
        # is a blank separator between events; nothing to yield.
        return ""

    payload = line[5:].strip()
    if payload == "[DONE]":
        return _SSE_DONE
    if not payload:
        return ""

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Ignoring malformed SSE frame: %s", payload[:120])
        return ""

    choices = chunk.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _build_messages(
    prompt: str,
    system: Optional[str],
    examples: Optional[list[tuple[str, str]]],
) -> list[dict]:
    """Assemble the ``messages`` array for a chat completion request.

    Optional system prompt and few-shot ``(user, assistant)`` pairs are
    laid out in the order the OpenAI protocol expects: system first,
    then example turns, then the fresh user prompt. Small models pattern-
    match on inline examples in the system prompt but generalise from
    structured turns, so refinement passes examples through this path.
    """
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
    """Pull the assistant text out of a parsed chat-completion response.

    Prefers the modern ``choices[0].message.content`` field but falls
    back to the legacy ``choices[0].text`` layout for servers that still
    return the text-completion shape from the chat endpoint. Raises
    ``ValueError`` when neither field is present so the failure surfaces
    with the offending URL and payload for debugging.
    """
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
    if not isinstance(content, str):
        # Vision-capable and tool-use servers sometimes return the content as
        # an array of typed parts instead of a bare string. Refinement /
        # personality are text-only paths, so surface a clear error instead
        # of tripping ``AttributeError`` on ``.strip()`` and burying the
        # shape mismatch.
        raise ValueError(
            f"Unexpected content type {type(content).__name__} in response from {url}: {data!r}"
        )
    return content.strip()
