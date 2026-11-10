#!/usr/bin/env python3
"""Run the hand-written question set and grade the answers.

Stand-in demo (free):
    python eval_quality.py --source stand-in --out demo_quality.csv

Real Groq (costs quota):
    python eval_quality.py --source real --yes-spend-real --out quality_real.csv
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from relay.config import load_env
from relay.grading import grade, load_questions, quality_summary
from relay.groq_client import GroqClient
from relay.ingest import load_chunks
from relay.load import RetryingClient, csv_status, write_rows
from relay.pipeline import Pipeline, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    p.add_argument("--questions", default="questions.json")
    p.add_argument("--chunks", type=Path, default=Path("chunks_1200.jsonl"))
    p.add_argument("--out", type=Path, default=Path("quality.csv"))
    p.add_argument("--answers", type=Path, default=Path("raw/answers.jsonl"),
                   help="raw answer text for auditing/regrading")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--setting", default="s1200-k3-rerank")
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--governor", action="store_true",
                   help="pace calls client-side (recommended for real runs)")
    p.add_argument("--g-rpm", type=int, default=30)
    p.add_argument("--g-tpm", type=int, default=8000)
    p.add_argument("--g-rpd", type=int, default=1000)
    p.add_argument("--g-tpd", type=int, default=200000)
    p.add_argument("--est-tokens", type=int, default=2000,
                   help="tokens the governor budgets per model call")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None, help="only these question ids")
    p.add_argument("--ttft-ms", type=int, default=300)
    p.add_argument("--chunk-ms", type=int, default=10)
    p.add_argument("--rpm", type=int, default=1000)
    p.add_argument("--yes-spend-real", action="store_true")
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--reasoning-effort", default="none",
                   choices=["none", "low", "medium", "high", ""],
                   help="none is Qwen-only; reasoning models may need low/medium/high")
    args = p.parse_args()
    load_env()

    specs = load_questions(args.questions)
    if args.ids:
        specs = [s for s in specs if s.id in set(args.ids)]
    if args.limit:
        specs = specs[: args.limit]
    chunks = load_chunks(args.chunks)

    governor = None
    if args.governor:
        from relay.governor import Governor, GovernorConfig
        governor = Governor(GovernorConfig(rpm=args.g_rpm, tpm=args.g_tpm,
                                           rpd=args.g_rpd, tpd=args.g_tpd))

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
        base_url, api_key, model = f"http://127.0.0.1:{server.server_address[1]}/v1", "stand-in", "stand-in"

    def make_pipeline():
        client = GroqClient(api_key=api_key, base_url=base_url, model=model,
                            reasoning_effort=args.reasoning_effort or None)
        return Pipeline(RetryingClient(client, max_retries=args.max_retries), chunks,
                        PipelineConfig(k=args.k, min_score=args.min_score,
                                       est_tokens_per_call=args.est_tokens),
                        governor=governor)

    rows, grades = [], []
    try:
        for i, spec in enumerate(specs, start=1):
            started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            answer = make_pipeline().run(spec.question)
            letter = grade(spec, answer.status, answer.text, answer.citations)
            grades.append(letter)
            rows.append({
                "request_id": f"quality-{spec.id}",
                "scenario": f"quality-{args.source}",
                "source": args.source,
                "start_time": started,
                "first_token_ms": answer.first_token_ms,
                "total_ms": answer.total_ms,
                "status": csv_status(answer.status),
                "retries": answer.retries,
                "tokens_in": answer.tokens_in,
                "tokens_out": answer.tokens_out,
                "retrieval_setting": args.setting,
                "answer_grade": letter,
            })
            print(f"{spec.id:<20} {letter:<18} {answer.status:<10} "
                  f"{answer.tokens_in + answer.tokens_out:>5} tok  {answer.text[:60]!r}")
            args.answers.parent.mkdir(parents=True, exist_ok=True)
            with open(args.answers, "a") as f:
                f.write(json.dumps({
                    "id": spec.id, "question": spec.question,
                    "status": answer.status, "text": answer.text,
                    "citations": answer.citations, "grade": letter,
                    "tokens_in": answer.tokens_in, "tokens_out": answer.tokens_out,
                    "first_token_ms": answer.first_token_ms,
                    "total_ms": answer.total_ms, "degraded": answer.degraded,
                    "verification": answer.verification,
                }) + "\n")
    finally:
        if server:
            server.shutdown(); server.server_close()

    write_rows(args.out, rows)
    summary = quality_summary(grades)
    tokens = [r["tokens_in"] + r["tokens_out"] for r in rows]
    print("\n" + " ".join(f"{k}={v}" for k, v in summary.items()))
    if tokens:
        print(f"mean tokens/answer={sum(tokens)/len(tokens):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
