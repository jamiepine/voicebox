"""OpenAI-compatible LLM backend for BYO-LLM conversational mode."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OpenAICompatLLMBackend:
    """Calls any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None, model: str = "llama3"):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or ""
        self.model = model

    def is_loaded(self) -> bool:
        return True  # no local model to load

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model_size: Optional[str] = None,
        examples: Optional[list] = None,
    ) -> str:
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if examples:
            for user_text, assistant_text in examples:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"{self.endpoint}/v1/chat/completions"
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_with_history(
        self,
        history: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate with full conversation history (list of {role, content} dicts)."""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"{self.endpoint}/v1/chat/completions"
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
