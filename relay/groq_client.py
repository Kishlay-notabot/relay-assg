"""Real Groq client (OpenAI-compatible), with TTFT measured on the stream.

One call = one attempt. Retry/backoff policy belongs above this layer so the
pipeline can count and account for retries separately.
"""

from __future__ import annotations

import json
import os
import time

import requests

from relay.llm import CompletionResult, LLMError, LLMTimeout, RateLimited

DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqClient:
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE,
                 model: str = DEFAULT_MODEL, timeout_s: int = 30,
                 reasoning_effort: str | None = "low"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[dict], max_tokens: int = 256) -> CompletionResult:
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "stream": True}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        t0 = time.perf_counter()
        first_at = None
        text_parts: list[str] = []
        tokens_in = tokens_out = 0

        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=headers, stream=True, timeout=self.timeout_s)
        except requests.exceptions.Timeout as e:
            raise LLMTimeout(f"request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise LLMError(f"transport error: {e}") from e

        if resp.status_code == 429:
            raise RateLimited(
                f"429: {resp.text[:200]}",
                retry_after_s=_float_or_none(resp.headers.get("retry-after")),
            )
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if first_at is None:
                first_at = time.perf_counter()
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    text_parts.append(piece)
            usage = chunk.get("x_groq", {}).get("usage") or chunk.get("usage")
            if usage:
                tokens_in = usage.get("prompt_tokens", tokens_in)
                tokens_out = usage.get("completion_tokens", tokens_out)

        total_ms = round((time.perf_counter() - t0) * 1000)
        return CompletionResult(
            text="".join(text_parts).strip(),
            first_token_ms=round((first_at - t0) * 1000) if first_at else total_ms,
            total_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
