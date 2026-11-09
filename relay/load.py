"""Load harness: retry wrapper, load shapes, results.csv writer.

Every row is one request, with the exact columns the assignment requires so
that all table figures can be recomputed from the raw file.
"""

from __future__ import annotations

import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from relay.llm import CompletionResult, LLMError, LLMTimeout, RateLimited
from relay.trace import append_trace, build_trace

CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade",
]

# The brief allows only ok / rate_limited / timeout / error in `status`.
# Refusals and unverified answers are successful *transports*; their quality is
# captured in answer_grade instead.
_STATUS_MAP = {
    "ok": "ok",
    "refused": "ok",
    "unverified": "ok",
    "rate_limited": "rate_limited",
    "timeout": "timeout",
    "error": "error",
}


def csv_status(pipeline_status: str) -> str:
    return _STATUS_MAP.get(pipeline_status, "error")


class RetryingClient:
    """Adds bounded, counted retries around one model client (per request)."""

    def __init__(self, inner, max_retries: int = 1, base_delay_s: float = 0.5,
                 max_delay_s: float = 8.0):
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.retries = 0
        self.waited_ms = 0
        self.worked_ms = 0

    def complete(self, messages: list[dict], max_tokens: int = 256) -> CompletionResult:
        for attempt in range(self.max_retries + 1):
            try:
                result = self.inner.complete(messages, max_tokens=max_tokens)
                self.worked_ms += result.total_ms
                return result
            except RateLimited as exc:
                if attempt == self.max_retries:
                    raise
                delay = exc.retry_after_s or min(
                    self.base_delay_s * (2 ** attempt), self.max_delay_s)
            except LLMTimeout:
                if attempt == self.max_retries:
                    raise
                delay = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
            except LLMError:
                raise  # do not retry non-transient failures
            self.retries += 1
            self.waited_ms += round(delay * 1000)
            time.sleep(delay)
        raise LLMError("unreachable")


def steady(duration_s: float, rate_per_min: float) -> list[float]:
    """Arrival offsets for a constant offered rate."""
    n = max(1, int(duration_s * rate_per_min / 60))
    interval = 60.0 / rate_per_min
    return [round(i * interval, 4) for i in range(n)]


def burst(count: int) -> list[float]:
    return [0.0] * count


def overload(duration_s: float, rate_per_min: float) -> list[float]:
    return steady(duration_s, rate_per_min)


@dataclass
class Scenario:
    name: str
    source: str          # "real" or "stand-in"
    delays: list[float]  # arrival offset (s) per request
    workers: int = 4
    per_request_delay_s: float = 0.0
    retrieval_setting: str = ""


@dataclass
class RunSummary:
    scenario: str
    sent: int
    peak_queue: int
    rows: list[dict] = field(default_factory=list)


def write_rows(path: Path, rows: list[dict]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_scenario(pipeline_factory, questions: list[str], scenario: Scenario,
                 out_path: Path, trace_path: Path | None = None) -> RunSummary:
    """Drive one load shape. pipeline_factory() must return a fresh Pipeline
    (with its own client) per request so retry counters stay per-request."""
    lock = threading.Lock()
    counters = {"submitted": 0, "running": 0, "done": 0, "peak_queue": 0}
    rows: list[dict] = []

    def worker(index: int, question: str) -> None:
        with lock:
            counters["running"] += 1
        try:
            if scenario.per_request_delay_s:
                time.sleep(scenario.per_request_delay_s)
            started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            pipeline = pipeline_factory()
            answer = pipeline.run(question)
            client = pipeline.client
            row = {
                "request_id": f"{scenario.name}-{index + 1:03d}",
                "scenario": scenario.name,
                "source": scenario.source,
                "start_time": started,
                "first_token_ms": answer.first_token_ms,
                "total_ms": answer.total_ms,
                "status": csv_status(answer.status),
                "retries": getattr(client, "retries", 0) + answer.retries,
                "tokens_in": answer.tokens_in,
                "tokens_out": answer.tokens_out,
                "retrieval_setting": scenario.retrieval_setting,
                "answer_grade": "",
            }
            if trace_path is not None:
                append_trace(trace_path, build_trace(row["request_id"], scenario.name,
                                                     answer, client))
        except Exception:
            row = {
                "request_id": f"{scenario.name}-{index + 1:03d}",
                "scenario": scenario.name, "source": scenario.source,
                "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "first_token_ms": 0, "total_ms": 0, "status": "error",
                "retries": 0, "tokens_in": 0, "tokens_out": 0,
                "retrieval_setting": scenario.retrieval_setting, "answer_grade": "",
            }
        finally:
            with lock:
                counters["running"] -= 1
                counters["done"] += 1
                rows.append(row)

    executor = ThreadPoolExecutor(max_workers=scenario.workers)
    start = time.perf_counter()
    futures = []
    for i, delay in enumerate(scenario.delays):
        wait = start + delay - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        with lock:
            counters["submitted"] += 1
            waiting = counters["submitted"] - counters["running"] - counters["done"]
            counters["peak_queue"] = max(counters["peak_queue"], waiting)
        futures.append(executor.submit(worker, i, questions[i % len(questions)]))
    for f in futures:
        f.result()
    executor.shutdown(wait=True)

    rows.sort(key=lambda r: r["request_id"])
    write_rows(out_path, rows)
    return RunSummary(scenario.name, len(scenario.delays), counters["peak_queue"], rows)
