"""Tests for the stand-in model service (runs entirely on localhost)."""

import json
import threading
import time
import unittest
import urllib.request
import urllib.error

from relay.standin import StandinConfig, make_server


class StandinFixture(unittest.TestCase):
    config = StandinConfig()

    def setUp(self):
        self.server = make_server(self.config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, payload=None, stream=True):
        body = json.dumps(payload or {
            "model": "standin",
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=10)


class StreamingTest(StandinFixture):
    def test_streams_reply_in_multiple_pieces(self):
        with self.post() as r:
            lines = [ln for ln in r.read().decode().splitlines() if ln.startswith("data: ")]

        chunks = [json.loads(ln[6:]) for ln in lines if ln != "data: [DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertTrue(text)
        self.assertGreater(len(chunks), 2)

    def test_reports_usage_on_final_chunk(self):
        with self.post() as r:
            lines = [ln for ln in r.read().decode().splitlines() if ln.startswith("data: ")]
        chunks = [json.loads(ln[6:]) for ln in lines if ln != "data: [DONE]"]
        usage = [c["usage"] for c in chunks if c.get("usage")]
        self.assertTrue(usage)
        self.assertGreater(usage[-1]["prompt_tokens"], 0)

    def test_first_token_arrives_before_last(self):
        from relay.groq_client import GroqClient

        cfg = StandinConfig(ttft_ms=60, chunk_ms=40, reply="one two three four")
        self.server.shutdown(); self.server.server_close()
        self.server = make_server(cfg)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        client = GroqClient(
            api_key="x", base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        )

        result = client.complete([{"role": "user", "content": "hi"}], max_tokens=10)

        self.assertGreaterEqual(result.first_token_ms, 40)
        self.assertGreater(result.total_ms, result.first_token_ms)
        self.assertIn("one", result.text)


class SmartModeTest(StandinFixture):
    def test_verification_prompt_gets_yes(self):
        from relay.groq_client import GroqClient

        client = GroqClient(api_key="x", base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1")
        result = client.complete([
            {"role": "system", "content": "You check an answer against the passages it cites. Reply 'YES' ..."},
            {"role": "user", "content": "ANSWER x"},
        ], max_tokens=10)

        self.assertTrue(result.text.lower().startswith("yes"))

    def test_draft_prompt_gets_a_cited_reply(self):
        from relay.groq_client import GroqClient

        client = GroqClient(api_key="x", base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1")
        result = client.complete([
            {"role": "system", "content": "Answer only from the PASSAGES below."},
            {"role": "user", "content": "PASSAGES ..."},
        ], max_tokens=40)

        self.assertIn("[1706.03762v7 p.9]", result.text)


class FailureTest(StandinFixture):
    config = StandinConfig(failure_rate=1.0)

    def test_failures_return_500(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post()
        self.assertEqual(ctx.exception.code, 500)


class RateLimitTest(StandinFixture):
    config = StandinConfig(rpm=2, retry_after_s=2)

    def test_third_request_in_a_window_is_429_with_retry_after(self):
        self.post().read()
        self.post().read()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post()
        self.assertEqual(ctx.exception.code, 429)
        self.assertEqual(ctx.exception.headers["retry-after"], "2")


class SlowdownTest(unittest.TestCase):
    def test_concurrent_requests_are_slower_under_load(self):
        server = make_server(StandinConfig(ttft_ms=30, chunk_ms=5, slowdown_ms_per_inflight=80))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

        def timed():
            body = json.dumps({"model": "s", "stream": True, "max_tokens": 2,
                               "messages": [{"role": "user", "content": "x"}]}).encode()
            t0 = time.perf_counter()
            with urllib.request.urlopen(urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}), timeout=10) as r:
                r.read()
            return (time.perf_counter() - t0) * 1000

        alone = timed()
        results = []
        threads = []
        for _ in range(3):
            t = threading.Thread(target=lambda: results.append(timed()))
            t.start(); threads.append(t)
        for t in threads:
            t.join()
        server.shutdown(); server.server_close()

        self.assertGreater(max(results), alone * 1.5)


if __name__ == "__main__":
    unittest.main()
