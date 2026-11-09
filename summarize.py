#!/usr/bin/env python3
"""Print Table 2a / 2b markdown from results.csv.

    python summarize.py results.csv
    python summarize.py demo_results.csv --price-in 0.075 --price-out 0.30
"""

import argparse

from relay.report import load_rows, summarise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default="results.csv")
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    args = p.parse_args()

    rows = summarise(load_rows(args.csv))
    # summarise() uses free-tier cost; recompute paid cost if prices given.
    if args.price_in or args.price_out:
        raw = load_rows(args.csv)
        for entry in rows:
            scenario_rows = [r for r in raw if r["scenario"] == entry["scenario"]]
            from relay.report import outcomes
            entry.update(outcomes(scenario_rows, args.price_in, args.price_out))

    print("Table 2a: Latency\n")
    print("| Scenario | Real/stand-in | Requests | Duration (s) | Offered rate/min | "
          "First token p50/p95/p99 | Total p50/p95/p99 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {r['source']} | {r['requests']} | {r['duration_s']} | "
              f"{r['offered_rate_per_min']} | "
              f"{r['first_token_p50_ms']}/{r['first_token_p95_ms']}/{r['first_token_p99_ms']} | "
              f"{r['total_p50_ms']}/{r['total_p95_ms']}/{r['total_p99_ms']} |")

    print("\nTable 2b: Outcomes\n")
    print("| Scenario | Succeeded % | Rate-limited % | Timed out/given up % | "
          "Retries/request | Peak queue | Tokens/answer | Cost per 1,000 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {r['succeeded_pct']} | {r['rate_limited_pct']} | "
              f"{r['timed_out_pct']} | {r['retries_per_request']} | {r['peak_queue']} | "
              f"{r['tokens_per_answer']} | ${r['cost_per_1000_answers']} |")


if __name__ == "__main__":
    main()
