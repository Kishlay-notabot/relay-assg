#!/usr/bin/env python3
"""Compare retrieval settings (Table 3) end to end.

Three settings: piece size, pieces fetched, re-rank on/off. Each rebuilds the
index from the PDFs (free), runs the question set, grades, and writes the
graded rows to results.csv.

    python run_retrieval.py --source stand-in --limit 10
    python run_retrieval.py --source real --yes-spend-real --limit 8
"""

from __future__ import annotations

import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path

from relay.config import load_env
from relay.grading import grade, load_questions
from relay.groq_client import GroqClient
from relay.ingest import build_chunks
from relay.load import RetryingClient, csv_status, write_rows
from relay.pipeline import Pipeline, PipelineConfig
from relay.report import retrieval_table

PRESETS = {
    "s600-k3-norerank": dict(size=600, overlap=100, k=3, rerank=False),
    "s1200-k4-rerank": dict(size=1200, overlap=200, k=4, rerank=True),
    "s1200-k5-rerank": dict(size=1200, overlap=200, k=5, rerank=True),
    "s2400-k2-rerank": dict(size=2400, overlap=300, k=2, rerank=True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    p.add_argument("--questions", default="questions.json")
    p.add_argument("--papers-dir", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    p.add_argument("--presets", nargs="*", default=list(PRESETS))
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--ids", nargs="*", default=None, help="only these question ids")
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--governor", action="store_true",
                   help="pace calls client-side (recommended for real runs)")
    p.add_argument("--ttft-ms", type=int, default=200)
    p.add_argument("--chunk-ms", type=int, default=5)
    p.add_argument("--rpm", type=int, default=1000)
    p.add_argument("--yes-spend-real", action="store_true")
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--reasoning-effort", default="none",
                   choices=["none", "low", "medium", "high", ""],
                   help="none is Qwen-only; reasoning models may need low/medium/high")
    args = p.parse_args()
    load_env()

    specs = load_questions(args.questions)[: args.limit]
    if args.ids:
        specs = [s for s in load_questions(args.questions) if s.id in set(args.ids)]

    governor = None
    if args.governor:
        from relay.governor import Governor, GovernorConfig
        governor = Governor(GovernorConfig())
    server = None
    if args.source == "real":
        if not args.yes_spend_real:
            raise SystemExit("real runs cost daily quota; pass --yes-spend-real")
        base_url, api_key, model = GroqClient().base_url, None, args.model
    else:
        from relay.standin import StandinConfig, make_server
        server = make_server(StandinConfig(ttft_ms=args.ttft_ms, chunk_ms=args.chunk_ms,
                                           rpm=args.rpm))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        api_key, model = "stand-in", "stand-in"

    rows = []
    try:
        for name in args.presets:
            spec = PRESETS[name]
            chunks = build_chunks(args.papers_dir, spec["size"], spec["overlap"])
            config = PipelineConfig(k=spec["k"], rerank=spec["rerank"],
                                    min_score=args.min_score)
            for q in specs:
                client = RetryingClient(
                    GroqClient(api_key=api_key, base_url=base_url, model=model,
                               reasoning_effort=args.reasoning_effort or None),
                    max_retries=args.max_retries)
                answer = Pipeline(client, chunks, config, governor=governor).run(q.question)
                letter = grade(q, answer.status, answer.text, answer.citations)
                rows.append({
                    "request_id": f"{name}-{q.id}",
                    "scenario": "retrieval",
                    "source": args.source,
                    "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "first_token_ms": answer.first_token_ms,
                    "total_ms": answer.total_ms,
                    "status": csv_status(answer.status),
                    "retries": client.retries + answer.retries,
                    "tokens_in": answer.tokens_in,
                    "tokens_out": answer.tokens_out,
                    "retrieval_setting": name,
                    "answer_grade": letter,
                })
            print(f"{name}: {len(specs)} questions done")
    finally:
        if server:
            server.shutdown(); server.server_close()

    write_rows(args.out, rows)
    print("\nTable 3: Retrieval settings\n")
    print("| Setting | Questions | Correct+cited % | Wrong % | Refused correctly % | "
          "Tokens/answer | First token p50/p95 | Total p50/p95 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in retrieval_table(rows):
        print(f"| {r['setting']} | {r['questions']} | {r['correct_pct']} | "
              f"{r['wrong_pct']} | {r['refused_correctly_pct']} | "
              f"{r['tokens_per_answer']} | {r['first_token_p50_ms']}/{r['first_token_p95_ms']} | "
              f"{r['total_p50_ms']}/{r['total_p95_ms']} |")


if __name__ == "__main__":
    main()
