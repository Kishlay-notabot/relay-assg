#!/usr/bin/env python3
"""Show the latency split for the slowest request(s), for write-up heading 4.

    python trace_report.py raw/traces.jsonl --n 3
"""

import argparse

from relay.trace import load_traces, slowest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="raw/traces.jsonl")
    p.add_argument("--n", type=int, default=1)
    args = p.parse_args()

    traces = slowest(load_traces(args.path), args.n)
    if not traces:
        raise SystemExit("no traces found")
    for t in traces:
        print(f"\n{t['request_id']}  ({t['scenario']}, {t['status']})")
        print(f"  total            {t['total_ms']:>6} ms")
        print(f"  before token     {t['pre_first_token_ms']:>6} ms")
        print(f"  after token      {t['post_first_token_ms']:>6} ms   "
              f"({100 * t['post_first_token_ms'] / max(t['total_ms'], 1):.0f}%)")
        print(f"  retrieval        {t['retrieval_ms']:>6} ms")
        print(f"  model working    {t['model_working_ms']:>6} ms")
        print(f"  retry waiting    {t['retry_wait_ms']:>6} ms  ({t['retries']} retries)")
        if t.get("budget_wait_ms"):
            print(f"  budget waiting   {t['budget_wait_ms']:>6} ms  (governor)")
        print("  steps: " + " -> ".join(
            f"{s['name']} {s['total_ms']}ms" for s in t["steps"]))


if __name__ == "__main__":
    main()
