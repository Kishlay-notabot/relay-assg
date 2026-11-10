#!/usr/bin/env python3
"""Run a load shape through the pipeline and append to results.csv.

Stand-in (default, free):
    source ../lang/bin/activate
    python run_load.py --scenario steady --duration 20 --rate 60 \
        --chunks chunks_1200.jsonl --workers 8

Real Groq (costs quota; refuses to run without --yes-spend-real):
    python run_load.py --source real --scenario burst --count 5 --yes-spend-real

Scenarios: steady, burst, overload. Each writes one row per request.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from relay.config import load_env
from relay.groq_client import GroqClient
from relay.ingest import load_chunks
from relay.load import RetryingClient, Scenario, burst, overload, run_scenario, steady
from relay.pipeline import Pipeline, PipelineConfig

QUESTIONS = [
    "How many attention heads does the Transformer use?",
    "What is scaled dot-product attention?",
    "How does Moshi handle full-duplex dialogue?",
    "What is dynamic chunking in hierarchical sequence modeling?",
    "Why do Transformers need positional encodings?",
    "What is the training objective of the Moshi model?",
]


def build_delays(args) -> list[float]:
    if args.scenario == "steady":
        return steady(args.duration, args.rate)
    if args.scenario == "burst":
        return burst(args.count)
    return overload(args.duration, args.rate)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    p.add_argument("--scenario", choices=["steady", "burst", "overload"], default="steady")
    p.add_argument("--name", default=None, help="scenario label written to results.csv")
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--rate", type=float, default=60, help="offered requests per minute")
    p.add_argument("--count", type=int, default=10, help="requests in a burst")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--rerank", action="store_true", default=True)
    p.add_argument("--chunks", type=Path, default=Path("chunks_1200.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    p.add_argument("--traces", type=Path, default=Path("raw/traces.jsonl"))
    p.add_argument("--setting", default="s1200-k3-rerank")
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--governor", action="store_true",
                   help="enforce client-side per-minute/daily limits")
    p.add_argument("--g-rpm", type=int, default=30)
    p.add_argument("--g-tpm", type=int, default=8000)
    p.add_argument("--g-rpd", type=int, default=1000)
    p.add_argument("--g-tpd", type=int, default=200000)
    # stand-in shaping
    p.add_argument("--ttft-ms", type=int, default=300)
    p.add_argument("--chunk-ms", type=int, default=10)
    p.add_argument("--rpm", type=int, default=30)
    p.add_argument("--tpm", type=int, default=8000)
    p.add_argument("--failure-rate", type=float, default=0.0)
    p.add_argument("--slowdown-ms", type=int, default=150)
    # real safety
    p.add_argument("--yes-spend-real", action="store_true")
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--reasoning-effort", default="none",
                   choices=["none", "low", "medium", "high", ""],
                   help="none is Qwen-only; reasoning models may need low/medium/high")
    p.add_argument("--base-url", default=None)
    args = p.parse_args()
    load_env()

    chunks = load_chunks(args.chunks)
    delays = build_delays(args)
    scenario = Scenario(name=args.name or args.scenario, source=args.source, delays=delays,
                        workers=args.workers, retrieval_setting=args.setting)

    governor = None
    if args.governor:
        from relay.governor import Governor, GovernorConfig
        governor = Governor(GovernorConfig(rpm=args.g_rpm, tpm=args.g_tpm,
                                           rpd=args.g_rpd, tpd=args.g_tpd))

    server = None
    if args.source == "real":
        if not args.yes_spend_real:
            raise SystemExit("real runs cost daily quota; pass --yes-spend-real to confirm")
        base_url = args.base_url or GroqClient().base_url
        model = args.model
    else:
        from relay.standin import StandinConfig, make_server
        cfg = StandinConfig(ttft_ms=args.ttft_ms, chunk_ms=args.chunk_ms,
                            rpm=args.rpm, tpm=args.tpm,
                            failure_rate=args.failure_rate,
                            slowdown_ms_per_inflight=args.slowdown_ms)
        server = make_server(cfg)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        model = "stand-in"

    def factory():
        client = GroqClient(api_key="stand-in" if server else None,
                            base_url=base_url, model=model,
                            reasoning_effort=args.reasoning_effort or None)
        return Pipeline(RetryingClient(client, max_retries=args.max_retries),
                        chunks, PipelineConfig(k=args.k, min_score=args.min_score,
                                               rerank=args.rerank),
                        governor=governor)

    try:
        summary = run_scenario(factory, QUESTIONS, scenario, args.out, args.traces)
    finally:
        if server:
            server.shutdown(); server.server_close()

    statuses = {}
    for row in summary.rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    total_tokens = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in summary.rows)
    print(f"{summary.scenario}: sent={summary.sent} peak_queue={summary.peak_queue} "
          f"statuses={statuses} tokens={total_tokens} -> {args.out}")


if __name__ == "__main__":
    main()
