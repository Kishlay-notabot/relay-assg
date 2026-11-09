"""Per-request latency decomposition for "where the time goes".

results.csv keeps the fixed 13 columns; this sidecar (raw/traces.jsonl) keeps
the evidence behind the breakdown, so the write-up can point at real rows.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_trace(request_id: str, scenario: str, answer, client) -> dict:
    retrieval_ms = sum(s.total_ms for s in answer.steps if s.name == "retrieve")
    model_ms = sum(s.total_ms for s in answer.steps if s.name != "retrieve")
    pre = answer.first_token_ms
    post = max(0, answer.total_ms - pre)
    return {
        "request_id": request_id,
        "scenario": scenario,
        "status": answer.status,
        "degraded": answer.degraded,
        "total_ms": answer.total_ms,
        "first_token_ms": pre,
        "pre_first_token_ms": pre,
        "post_first_token_ms": post,
        "retrieval_ms": retrieval_ms,
        "model_working_ms": model_ms,
        "retry_wait_ms": getattr(client, "waited_ms", 0),
        "budget_wait_ms": getattr(answer, "governor_wait_ms", 0),
        "retries": answer.retries + getattr(client, "retries", 0),
        "tokens_in": answer.tokens_in,
        "tokens_out": answer.tokens_out,
        "steps": [{"name": s.name, "first_token_ms": s.first_token_ms,
                   "total_ms": s.total_ms} for s in answer.steps],
    }


def append_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(trace) + "\n")


def load_traces(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path)]


def slowest(traces: list[dict], n: int = 1) -> list[dict]:
    return sorted(traces, key=lambda t: t["total_ms"], reverse=True)[:n]
