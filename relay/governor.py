"""Client-side budget governor.

Per-minute limits delay (the request waits for the window to slide).
Per-day limits degrade or refuse: when the daily budget is nearly gone the
pipeline skips verification (weaker answer); when it is gone we refuse rather
than send a request the provider will reject.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GovernorConfig:
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 1000
    tpd: int = 200_000
    degrade_at_day_fraction: float = 0.9


@dataclass
class Decision:
    action: str          # allow | degrade | wait | refuse
    wait_s: float = 0.0
    reason: str = ""
    exhausted: bool = False


@dataclass
class Governor:
    config: GovernorConfig = field(default_factory=GovernorConfig)
    request_times: deque = field(default_factory=deque)
    token_window: deque = field(default_factory=deque)  # (time, tokens)
    day_requests: int = 0
    day_tokens: int = 0
    degraded: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _slide(self, now: float) -> None:
        while self.request_times and now - self.request_times[0] >= 60:
            self.request_times.popleft()
        while self.token_window and now - self.token_window[0][0] >= 60:
            self.token_window.popleft()

    def acquire(self, est_tokens: int) -> Decision:
        cfg = self.config
        with self._lock:
            now = time.monotonic()
            self._slide(now)

            if self.day_requests + 1 > cfg.rpd:
                return Decision("refuse", reason="daily request budget exhausted", exhausted=True)
            if self.day_tokens + est_tokens > cfg.tpd:
                return Decision("refuse", reason="daily token budget exhausted", exhausted=True)

            day_pressure = (
                self.day_requests + 1 >= cfg.rpd * cfg.degrade_at_day_fraction
                or self.day_tokens + est_tokens >= cfg.tpd * cfg.degrade_at_day_fraction
            )
            minute_tokens = sum(t for _, t in self.token_window)

            if len(self.request_times) >= cfg.rpm:
                wait = 60 - (now - self.request_times[0])
                return Decision("wait", wait_s=max(wait, 0.1),
                                reason="per-minute request limit reached")
            if minute_tokens + est_tokens > cfg.tpm:
                wait = 60 - (now - self.token_window[0][0]) if self.token_window else 60
                return Decision("wait", wait_s=max(wait, 0.1),
                                reason="per-minute token limit reached")

            self.request_times.append(now)
            self.token_window.append((now, est_tokens))
            self.day_requests += 1
            if day_pressure:
                self.degraded = True
                return Decision("degrade", reason="daily budget nearly exhausted")
            return Decision("allow")

    def record(self, actual_tokens: int, estimated_tokens: int = 0) -> None:
        with self._lock:
            self.day_tokens += actual_tokens
            if self.token_window and estimated_tokens:
                t, est = self.token_window[-1]
                self.token_window[-1] = (t, est + (actual_tokens - estimated_tokens))

    def reset_day(self) -> None:
        with self._lock:
            self.day_requests = 0
            self.day_tokens = 0
            self.degraded = False
