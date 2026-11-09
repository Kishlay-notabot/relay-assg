"""Model-client seam: real Groq and stand-in implement the same interface.

Nothing here makes a network call by itself. Timings are measured by the
client so the pipeline can attribute latency to waiting vs working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompletionResult:
    text: str
    first_token_ms: int
    total_ms: int
    tokens_in: int
    tokens_out: int
    status: str = "ok"


class LLMError(Exception):
    """Base for model-call failures the pipeline can branch on."""


class RateLimited(LLMError):
    """Provider (or stand-in) refused because a limit was reached."""

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTimeout(LLMError):
    """No useful response within the caller's deadline."""


class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 256) -> CompletionResult:
        ...
