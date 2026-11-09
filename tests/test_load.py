"""Tests for retry wrapper, load shapes, and results.csv writing."""

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path

from relay.ingest import load_chunks
from relay.llm import CompletionResult, LLMTimeout, RateLimited
from relay.load import (
    CSV_COLUMNS,
    RetryingClient,
    Scenario,
    burst,
    csv_status,
    overload,
    run_scenario,
    steady,
    write_rows,
)
from relay.pipeline import Pipeline, PipelineConfig
from relay.retrieval import Chunk


def ok_result(text="answer [1706.03762v7 p.9]"):
    return CompletionResult(text=text, first_token_ms=10, total_ms=20,
                            tokens_in=100, tokens_out=10)


class Scripted:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, max_tokens=256):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RetryTest(unittest.TestCase):
    def test_retries_rate_limit_then_succeeds_and_counts_retry(self):
        inner = Scripted([RateLimited("rpm", retry_after_s=0.01), ok_result()])
        client = RetryingClient(inner, max_retries=2, base_delay_s=0.01)

        result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.text, ok_result().text)
        self.assertEqual(client.retries, 1)
        self.assertGreaterEqual(client.waited_ms, 10)

    def test_gives_up_after_max_retries(self):
        inner = Scripted([RateLimited("rpm", retry_after_s=0.01)] * 3)
        client = RetryingClient(inner, max_retries=2, base_delay_s=0.01)

        with self.assertRaises(RateLimited):
            client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(client.retries, 2)

    def test_retries_timeout_but_not_generic_llm_error(self):
        inner = Scripted([LLMTimeout("slow"), ok_result()])
        client = RetryingClient(inner, max_retries=1, base_delay_s=0.01)
        client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(client.retries, 1)


class ScheduleTest(unittest.TestCase):
    def test_steady_spaces_requests_at_rate(self):
        delays = steady(duration_s=3, rate_per_min=60)
        self.assertEqual(len(delays), 3)
        self.assertAlmostEqual(delays[1], 1.0, places=2)

    def test_burst_sends_everything_at_once(self):
        self.assertEqual(burst(count=5), [0.0] * 5)

    def test_overload_is_steady_at_a_rate_above_capacity(self):
        self.assertEqual(len(overload(duration_s=4, rate_per_min=90)), 6)


class CsvTest(unittest.TestCase):
    def test_csv_has_exact_required_columns(self):
        rows = [{
            "request_id": "r1", "scenario": "steady", "source": "stand-in",
            "start_time": "2026-11-09T00:00:00Z", "first_token_ms": 10,
            "total_ms": 20, "status": "ok", "retries": 0, "tokens_in": 100,
            "tokens_out": 10, "retrieval_setting": "s1200-k3-rerank",
            "answer_grade": "",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_rows(path, rows)
            with open(path) as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                first = next(reader)
        self.assertEqual(header, CSV_COLUMNS)
        self.assertEqual(first["request_id"], "r1")

    def test_status_mapping(self):
        self.assertEqual(csv_status("ok"), "ok")
        self.assertEqual(csv_status("refused"), "ok")
        self.assertEqual(csv_status("unverified"), "ok")
        self.assertEqual(csv_status("rate_limited"), "rate_limited")
        self.assertEqual(csv_status("timeout"), "timeout")
        self.assertEqual(csv_status("error"), "error")


CHUNKS = [Chunk("1706.03762v7", 9, 0, "the transformer uses 8 attention heads")]


class RunnerTest(unittest.TestCase):
    def make_factory(self, inner):
        def factory():
            return Pipeline(RetryingClient(inner, max_retries=0),
                            CHUNKS, PipelineConfig(k=1, min_score=0.2))
        return factory

    def test_steady_run_writes_one_row_per_request(self):
        inner = Scripted([ok_result(), ok_result("YES")] * 3)
        scenario = Scenario(name="steady", source="stand-in", delays=[0, 0, 0], workers=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            summary = run_scenario(
                self.make_factory(inner), ["how many attention heads"],
                scenario, path,
            )
            rows = list(csv.DictReader(open(path)))

        self.assertEqual(summary.sent, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["scenario"] for r in rows}, {"steady"})
        self.assertEqual({r["status"] for r in rows}, {"ok"})
        self.assertTrue(all(r["first_token_ms"] for r in rows))

    def test_burst_reports_peak_queue_above_worker_count(self):
        inner = Scripted([ok_result(), ok_result("YES")] * 6)
        scenario = Scenario(name="burst", source="stand-in", delays=[0.0] * 6,
                            workers=2, per_request_delay_s=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_scenario(self.make_factory(inner),
                                   ["how many attention heads"],
                                   scenario, Path(tmp) / "results.csv")

        self.assertGreater(summary.peak_queue, 2)

    def test_rate_limited_result_is_written_with_status(self):
        inner = Scripted([RateLimited("rpm")])
        scenario = Scenario(name="overload", source="stand-in", delays=[0.0], workers=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            run_scenario(self.make_factory(inner), ["how many attention heads"],
                         scenario, path)
            row = next(csv.DictReader(open(path)))

        self.assertEqual(row["status"], "rate_limited")


class RealStandinIntegrationTest(unittest.TestCase):
    """One real HTTP loop through the local stand-in, no Groq."""

    def test_scenario_through_standin_writes_rows(self):
        from relay.groq_client import GroqClient
        from relay.standin import StandinConfig, make_server

        server = make_server(StandinConfig(ttft_ms=5, chunk_ms=1, reply="yes [1706.03762v7 p.9]"))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        chunks = load_chunks("chunks_1200.jsonl")
        port = server.server_address[1]

        def factory():
            client = GroqClient(api_key="x", base_url=f"http://127.0.0.1:{port}/v1")
            return Pipeline(RetryingClient(client, max_retries=1),
                            chunks, PipelineConfig(k=2))

        scenario = Scenario(name="steady", source="stand-in", delays=[0, 0], workers=2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary = run_scenario(factory, ["how many attention heads"],
                                       scenario, Path(tmp) / "results.csv")
                rows = list(csv.DictReader(open(Path(tmp) / "results.csv")))
        finally:
            server.shutdown(); server.server_close()

        self.assertEqual(summary.sent, 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(int(r["total_ms"]) > 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
