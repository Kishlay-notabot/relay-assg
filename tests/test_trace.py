"""Tests for latency-split tracing."""

import tempfile
import unittest
from pathlib import Path

from relay.pipeline import Answer, StepRecord
from relay.trace import append_trace, build_trace, load_traces, slowest


class FakeClient:
    waited_ms = 2000
    retries = 1
    worked_ms = 1200


class TraceTest(unittest.TestCase):
    def test_split_covers_total_and_splits_at_first_token(self):
        answer = Answer(text="x", status="ok", total_ms=5000, first_token_ms=3000,
                        retries=0,
                        steps=[StepRecord("retrieve", 0, 100),
                               StepRecord("draft", 2900, 3000),
                               StepRecord("verify", 300, 900)])

        trace = build_trace("r1", "steady", answer, FakeClient())

        self.assertEqual(trace["pre_first_token_ms"], 3000)
        self.assertEqual(trace["post_first_token_ms"], 2000)
        self.assertEqual(trace["retrieval_ms"], 100)
        self.assertEqual(trace["model_working_ms"], 3900)
        self.assertEqual(trace["retry_wait_ms"], 2000)
        self.assertEqual(trace["retries"], 1)
        self.assertEqual([s["name"] for s in trace["steps"]],
                         ["retrieve", "draft", "verify"])

    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw" / "traces.jsonl"
            append_trace(path, {"request_id": "a", "total_ms": 10})
            append_trace(path, {"request_id": "b", "total_ms": 99})

            traces = load_traces(path)

        self.assertEqual([t["request_id"] for t in traces], ["a", "b"])
        self.assertEqual(slowest(traces)[0]["request_id"], "b")


if __name__ == "__main__":
    unittest.main()
