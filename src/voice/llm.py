"""Thin client for an OpenAI-compatible local LLM server (LM Studio).

The pipeline is synchronous end-to-end (ASR and pyannote are sync), so this
wrapper uses httpx's blocking client. One retry on transport errors; callers
that need stricter validation (e.g. JSON parsing) handle retries themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "mlx-community/gemma-3-12b-it-qat-4bit"
DEFAULT_TIMEOUT_S = 300.0


class LLMError(RuntimeError):
    """Raised when the LLM server is unreachable or returns malformed data."""


@dataclass
class LLMClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT_S
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def health_check(self) -> list[str]:
        """Return available model IDs. Raises LLMError if the server is down."""
        assert self._client is not None
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"LM Studio server is not reachable at {self.base_url}. "
                "Open LM Studio → Develop → Start Server, then retry."
            ) from exc
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str:
        """Send a chat-completions request and return the assistant text.

        `response_format={"type": "json_object"}` asks the server for strict
        JSON output (LM Studio supports this OpenAI-compatible field).
        """
        assert self._client is not None
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = f" — {exc.response.text[:500]}"
            except Exception:  # noqa: BLE001
                pass
            raise LLMError(f"chat/completions {exc.response.status_code}{detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"chat/completions failed: {exc}") from exc

        body = resp.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {body!r}") from exc

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> Any:
        """Convenience: request JSON-mode output and parse the result.

        Retries once on `JSONDecodeError`, asking the model to fix its reply.
        """
        last_err: Exception | None = None
        attempt_msgs = list(messages)
        for attempt in range(retries + 1):
            text = self.chat(
                attempt_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": False,
                        "schema": {"type": "object"},
                    },
                },
            )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_err = exc
                if attempt >= retries:
                    break
                attempt_msgs = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Respond again with valid JSON only, no commentary."
                        ),
                    },
                ]
        raise LLMError(f"LLM returned invalid JSON after {retries + 1} attempt(s): {last_err}")
