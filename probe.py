#!/usr/bin/env python3
"""
Groq free-tier probe for openai/gpt-oss-20b.

Measures, with a tiny budget of real requests:
  1. Latency spread  - time to first token (TTFT) and total time over N tiny
                       streaming requests (sequential, so no induced 429s).
  2. Header harvest  - x-ratelimit-remaining-* and x-ratelimit-reset-*
                       from every response, giving effective limits and
                       recovery windows without deliberately tripping them.
  3. 429 behaviour   - a slow ramp of tiny requests until the first 429,
                       then reads `retry-after` and reset headers, waits it
                       out, and confirms recovery with one more request.

Results: probe_results.csv (one row per request, raw) and a summary printed
to stdout. Every number in the assignment's Table 1 / Table 2 should be
computable from the CSV.

Run:  source ../lang/bin/activate && python probe.py
"""

import csv
import json
import os
import time
from datetime import datetime, timezone

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
OUT_CSV = "probe_results.csv"

LATENCY_REQUESTS = 20          # tiny sequential requests for latency spread
RAMP_MAX_REQUESTS = 45         # hard cap on the 429 ramp
RAMP_GAP_S = 1.2               # seconds between ramp requests (~50/min offered)
REQUEST_TIMEOUT_S = 30

CSV_FIELDS = [
    "request_id", "phase", "start_time", "first_token_ms", "total_ms",
    "status", "http_code", "retries", "tokens_in", "tokens_out",
    "remaining_requests", "remaining_tokens",
    "reset_requests", "reset_tokens", "retry_after", "error_detail",
]


def load_env(path=".env"):
    if not os.path.exists(path):
        raise SystemExit("Missing .env file")
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def tiny_request(api_key, phase, request_id, record, max_tokens=5):
    """One tiny streaming chat request. Records timings + headers into `record`."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    rec = {
        "request_id": request_id,
        "phase": phase,
        "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "first_token_ms": "",
        "total_ms": "",
        "status": "error",
        "http_code": "",
        "retries": 0,
        "tokens_in": "",
        "tokens_out": 0,
        "remaining_requests": "",
        "remaining_tokens": "",
        "reset_requests": "",
        "reset_tokens": "",
        "retry_after": "",
        "error_detail": "",
    }

    t0 = time.perf_counter()
    first_token_at = None
    try:
        r = requests.post(API_URL, headers=headers, json=body,
                          stream=True, timeout=REQUEST_TIMEOUT_S)
        rec["http_code"] = r.status_code
        for h in ("x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
                  "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
                  "retry-after"):
            v = r.headers.get(h)
            if v is not None:
                rec[h.removeprefix("x-ratelimit-").replace("-", "_")] = v

        if r.status_code != 200:
            rec["status"] = "rate_limited" if r.status_code == 429 else "error"
            rec["total_ms"] = round((time.perf_counter() - t0) * 1000)
            try:
                rec["error_detail"] = r.text[:300]
            except Exception:
                pass
            record(rec)
            return rec

        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            payload = raw[len(b"data: "):]
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                rec["first_token_ms"] = round((first_token_at - t0) * 1000)
            for choice in chunk.get("choices", []):
                rec["tokens_out"] += 1  # chunk count; full usage arrives in the last chunk
            usage = chunk.get("x_groq", {}).get("usage") or chunk.get("usage")
            if usage:
                if usage.get("prompt_tokens"):
                    rec["tokens_in"] = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    rec["tokens_out"] = usage["completion_tokens"]

        rec["total_ms"] = round((time.perf_counter() - t0) * 1000)
        rec["status"] = "ok"
    except requests.exceptions.Timeout:
        rec["status"] = "timeout"
        rec["total_ms"] = round((time.perf_counter() - t0) * 1000)
        rec["error_detail"] = f"timeout after {REQUEST_TIMEOUT_S}s"
    except Exception as e:
        rec["status"] = "error"
        rec["http_code"] = type(e).__name__
        rec["total_ms"] = round((time.perf_counter() - t0) * 1000)
        rec["error_detail"] = repr(e)[:300]
    record(rec)
    return rec


class CsvWriter:
    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS)
        self.w.writeheader()

    def record(self, rec):
        self.w.writerow(rec)
        self.f.flush()

    def close(self):
        self.f.close()


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return "not measured"
    k = max(0, min(len(xs) - 1, round(p * (len(xs) - 1))))
    return xs[k]


def main():
    load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set")

    csvw = CsvWriter(OUT_CSV)
    n = 0

    def record(rec):
        nonlocal n
        n += 1
        csvw.record(rec)
        s = f"#{rec['request_id']:>3} {rec['phase']:<10} {rec['status']:<12} http={rec['http_code'] or '-':<4}" \
            f" ttft={rec['first_token_ms'] or '-':>6} total={rec['total_ms'] or '-':>6}ms"
        if rec["error_detail"]:
            s += f"\n    !! {rec['error_detail']}"
        if rec["remaining_tokens"]:
            s += f" | tok_left={rec['remaining_tokens']} req_left={rec['remaining_requests']}"
        if rec["retry_after"]:
            s += f" | retry-after={rec['retry_after']}s"
        print(s)

    print(f"== Phase 1: latency spread ({LATENCY_REQUESTS} tiny sequential requests to {MODEL}) ==")
    for _ in range(LATENCY_REQUESTS):
        tiny_request(api_key, "latency", n + 1, record)
        time.sleep(0.4)

    print(f"\n== Phase 2: slow ramp to first 429 (gap {RAMP_GAP_S}s, max {RAMP_MAX_REQUESTS}) ==")
    hit_429 = None
    for _ in range(RAMP_MAX_REQUESTS):
        rec = tiny_request(api_key, "ramp", n + 1, record)
        if rec["status"] == "rate_limited":
            hit_429 = rec
            break
        time.sleep(RAMP_GAP_S)

    if hit_429:
        wait = float(hit_429["retry_after"] or "5") + 1.0
        print(f"\n== Phase 3: waiting {wait:.1f}s (retry-after + 1s slack), then confirming recovery ==")
        time.sleep(wait)
        tiny_request(api_key, "recovery", n + 1, record)
    else:
        print("\n== No 429 reached during ramp (rate may have been raised or gap too large) ==")

    csvw.close()

    rows = list(csv.DictReader(open(OUT_CSV)))
    ok = [r for r in rows if r["status"] == "ok"]
    ttft = [int(r["first_token_ms"]) for r in ok if r["first_token_ms"]]
    tot = [int(r["total_ms"]) for r in ok if r["total_ms"]]

    print(f"\n== Summary ({len(ok)} ok / {len(rows)} total) – written to {OUT_CSV} ==")
    for name, xs in (("first_token_ms", ttft), ("total_ms", tot)):
        if xs:
            print(f"{name:<16} p50={pct(xs, .5):>6} p95={pct(xs, .95):>6} p99={pct(xs, .99):>6} ms")
    rr = [int(r["remaining_requests"]) for r in ok if r["remaining_requests"]]
    rt = [int(r["remaining_tokens"]) for r in ok if r["remaining_tokens"]]
    if rr and rt:
        drop_req = max(rr) - min(rr)
        drop_tok = max(rt) - min(rt)
        print(f"consumed over the run: {drop_req} requests, {drop_tok} tokens")
        print(f"per request: ~{round(drop_tok / max(1, drop_req))} tokens")
    hardest = [r for r in rows if r["status"] == "rate_limited"]
    if hardest:
        h = hardest[0]
        print(f"429 details: retry_after={h['retry_after']}s "
              f"reset_requests={h['reset_requests']} reset_tokens={h['reset_tokens']}")


if __name__ == "__main__":
    main()
