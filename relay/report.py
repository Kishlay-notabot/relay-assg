"""Table 2a / 2b summaries computed from results.csv (single source of truth).

Conventions (documented so reviewers can recompute):
- duration = last start_time - first start_time + that request's total_ms
- peak queue = max requests outstanding at once, derived from
  [start_time, start_time + total_ms] intervals
- timed out or given up = status in {timeout, error}
- tokens per answer = mean (tokens_in + tokens_out) over the scenario's rows
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime

DEFAULT_COLUMNS = None


def percentile(values: list[int], p: float) -> int | None:
    """Nearest-rank percentile. Accepts a fraction (0.95) or a percent (95)."""
    if not values:
        return None
    if p > 1:
        p = p / 100
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, math.ceil(p * len(xs)) - 1))
    return xs[k]


def _start(row) -> datetime:
    return datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))


def peak_concurrency(rows: list[dict]) -> int:
    events = []
    for row in rows:
        try:
            start = _start(row).timestamp()
            total = int(row["total_ms"]) / 1000
        except (ValueError, KeyError):
            continue
        events.append((start, 1))
        events.append((start + total, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # end (-1) before start (+1)
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def outcomes(rows: list[dict], price_in_per_m: float = 0.0,
             price_out_per_m: float = 0.0) -> dict:
    n = len(rows) or 1
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    tokens = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in rows]
    mean_tokens = sum(tokens) / n
    cost_per_answer = (
        mean_tokens * 0  # placeholder replaced below by split cost
    )
    mean_in = sum(int(r["tokens_in"]) for r in rows) / n
    mean_out = sum(int(r["tokens_out"]) for r in rows) / n
    cost_per_answer = (mean_in * price_in_per_m + mean_out * price_out_per_m) / 1_000_000

    return {
        "succeeded_pct": round(100 * counts["ok"] / n, 1),
        "rate_limited_pct": round(100 * counts["rate_limited"] / n, 1),
        "timed_out_pct": round(100 * (counts["timeout"] + counts["error"]) / n, 1),
        "retries_per_request": round(sum(int(r["retries"]) for r in rows) / n, 2),
        "peak_queue": peak_concurrency(rows),
        "tokens_per_answer": round(mean_tokens, 1),
        "cost_per_1000_answers": round(1000 * cost_per_answer, 4),
    }


def summarise(rows: list[dict]) -> list[dict]:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("scenario"):
            by_scenario[row["scenario"]].append(row)

    table = []
    for scenario in sorted(by_scenario):
        rs = sorted(by_scenario[scenario], key=_start)
        starts = [_start(r) for r in rs]
        duration_s = round(
            (starts[-1] - starts[0]).total_seconds() + int(rs[-1]["total_ms"]) / 1000, 2)
        ok_rows = [r for r in rs if r["status"] == "ok"]
        firsts = [int(r["first_token_ms"]) for r in ok_rows]
        totals = [int(r["total_ms"]) for r in ok_rows]
        entry = {
            "scenario": scenario,
            "source": ",".join(sorted({r["source"] for r in rs})),
            "requests": len(rs),
            "duration_s": duration_s,
            "offered_rate_per_min": round(len(rs) / (duration_s / 60), 1) if duration_s else 0.0,
            "first_token_p50_ms": percentile(firsts, 0.50),
            "first_token_p95_ms": percentile(firsts, 0.95),
            "first_token_p99_ms": percentile(firsts, 0.99),
            "total_p50_ms": percentile(totals, 0.50),
            "total_p95_ms": percentile(totals, 0.95),
            "total_p99_ms": percentile(totals, 0.99),
        }
        entry.update(outcomes(rs))
        table.append(entry)
    return table


def load_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def retrieval_table(rows: list[dict]) -> list[dict]:
    """Table 3: quality and cost per retrieval setting, from graded rows."""
    by_setting: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("retrieval_setting"):
            by_setting[row["retrieval_setting"]].append(row)

    table = []
    for setting in sorted(by_setting):
        rs = by_setting[setting]
        n = len(rs) or 1
        grades = [r.get("answer_grade", "") for r in rs]
        ok_rows = [r for r in rs if r["status"] == "ok"]
        firsts = [int(r["first_token_ms"]) for r in ok_rows]
        totals = [int(r["total_ms"]) for r in ok_rows]
        tokens = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in rs]
        table.append({
            "setting": setting,
            "questions": len(rs),
            "correct_pct": round(100 * grades.count("correct") / n, 1),
            "wrong_pct": round(100 * grades.count("wrong") / n, 1),
            "refused_correctly_pct": round(100 * grades.count("refused_correctly") / n, 1),
            "tokens_per_answer": round(sum(tokens) / n, 1),
            "first_token_p50_ms": percentile(firsts, 0.50),
            "first_token_p95_ms": percentile(firsts, 0.95),
            "total_p50_ms": percentile(totals, 0.50),
            "total_p95_ms": percentile(totals, 0.95),
        })
    return table
