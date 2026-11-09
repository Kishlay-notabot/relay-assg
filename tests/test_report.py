"""Tests for Table 2a / 2b generation from results.csv rows."""

import unittest
from datetime import datetime, timedelta, timezone

from relay.report import (
    outcomes,
    peak_concurrency,
    percentile,
    retrieval_table,
    summarise,
)


def make_row(i, scenario, start, total_ms, status="ok", retries=0,
             tin=1000, tout=100, source="stand-in"):
    return {
        "request_id": f"{scenario}-{i:03d}",
        "scenario": scenario,
        "source": source,
        "start_time": start,
        "first_token_ms": total_ms // 2,
        "total_ms": total_ms,
        "status": status,
        "retries": retries,
        "tokens_in": tin,
        "tokens_out": tout,
        "retrieval_setting": "s1200-k3",
        "answer_grade": "",
    }


T0 = datetime(2026, 11, 9, 15, 0, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="milliseconds")


class PercentileTest(unittest.TestCase):
    def test_percentiles(self):
        xs = list(range(1, 101))
        self.assertEqual(percentile(xs, 50), 50)
        self.assertEqual(percentile(xs, 95), 95)
        self.assertEqual(percentile(xs, 99), 99)

    def test_empty_is_none(self):
        self.assertIsNone(percentile([], 50))


class ConcurrencyTest(unittest.TestCase):
    def test_peak_concurrency_from_intervals(self):
        rows = [
            make_row(1, "burst", iso(T0), 1000),
            make_row(2, "burst", iso(T0 + timedelta(milliseconds=100)), 500),
            make_row(3, "burst", iso(T0 + timedelta(milliseconds=200)), 100),
        ]
        self.assertEqual(peak_concurrency(rows), 3)

    def test_sequential_requests_have_peak_one(self):
        rows = [
            make_row(1, "steady", iso(T0), 100),
            make_row(2, "steady", iso(T0 + timedelta(milliseconds=200)), 100),
        ]
        self.assertEqual(peak_concurrency(rows), 1)


class OutcomesTest(unittest.TestCase):
    def test_percentages_and_retries_per_request(self):
        rows = [
            make_row(1, "x", iso(T0), 100, status="ok", retries=0),
            make_row(2, "x", iso(T0), 100, status="rate_limited", retries=1),
            make_row(3, "x", iso(T0), 100, status="timeout", retries=2),
            make_row(4, "x", iso(T0), 100, status="error", retries=0),
        ]
        out = outcomes(rows)

        self.assertEqual(out["succeeded_pct"], 25.0)
        self.assertEqual(out["rate_limited_pct"], 25.0)
        self.assertEqual(out["timed_out_pct"], 50.0)  # timeout + error
        self.assertEqual(out["retries_per_request"], 0.75)
        self.assertEqual(out["tokens_per_answer"], 1100.0)


class RetrievalTableTest(unittest.TestCase):
    def test_groups_by_setting_and_computes_quality_and_latency(self):
        rows = [
            make_row(1, "q", iso(T0), 100, status="ok") | {"retrieval_setting": "a", "answer_grade": "correct"},
            make_row(2, "q", iso(T0 + timedelta(seconds=1)), 300, status="ok") | {"retrieval_setting": "a", "answer_grade": "wrong"},
            make_row(3, "q", iso(T0 + timedelta(seconds=2)), 200, status="ok") | {"retrieval_setting": "b", "answer_grade": "refused_correctly"},
        ]

        table = retrieval_table(rows)

        self.assertEqual([r["setting"] for r in table], ["a", "b"])
        a = table[0]
        self.assertEqual(a["questions"], 2)
        self.assertEqual(a["correct_pct"], 50.0)
        self.assertEqual(a["wrong_pct"], 50.0)
        self.assertEqual(a["first_token_p50_ms"], 50)  # only ok rows, nearest-rank
        self.assertEqual(a["total_p95_ms"], 300)


class SummariseTest(unittest.TestCase):
    def test_table_2a_has_one_row_per_scenario_with_percentiles(self):
        rows = []
        for i in range(10):
            rows.append(make_row(i, "steady", iso(T0 + timedelta(seconds=i)), 100 + i * 10))
            rows.append(make_row(i, "burst", iso(T0 + timedelta(seconds=i)), 500 + i * 10))

        table = summarise(rows)

        self.assertEqual([r["scenario"] for r in table], ["burst", "steady"])
        steady = next(r for r in table if r["scenario"] == "steady")
        self.assertEqual(steady["requests"], 10)
        self.assertEqual(steady["duration_s"], 9.19)  # span 9s + last 190ms
        self.assertEqual(steady["total_p50_ms"], 140)
        self.assertEqual(steady["total_p95_ms"], 190)

    def test_latency_percentiles_ignore_failed_requests(self):
        rows = [
            make_row(1, "x", iso(T0), 100, status="ok"),
            make_row(2, "x", iso(T0 + timedelta(seconds=1)), 300, status="ok"),
            make_row(3, "x", iso(T0 + timedelta(seconds=2)), 0, status="rate_limited"),
            make_row(4, "x", iso(T0 + timedelta(seconds=3)), 0, status="error"),
        ]

        entry = summarise(rows)[0]

        self.assertEqual(entry["total_p50_ms"], 100)  # nearest-rank over ok rows
        self.assertEqual(entry["total_p95_ms"], 300)
        self.assertEqual(entry["succeeded_pct"], 50.0)

    def test_cost_per_1000_answers(self):
        rows = [make_row(1, "x", iso(T0), 100, tin=1000, tout=200)]
        out = outcomes(rows, price_in_per_m=0.075, price_out_per_m=0.30)
        # 1000 in @ $0.075/M + 200 out @ $0.30/M -> $0.000135 per answer
        self.assertAlmostEqual(out["cost_per_1000_answers"], 0.135, places=3)

    def test_cost_is_zero_on_free_tier(self):
        rows = [make_row(1, "x", iso(T0), 100)]
        self.assertEqual(outcomes(rows)["cost_per_1000_answers"], 0.0)


if __name__ == "__main__":
    unittest.main()
