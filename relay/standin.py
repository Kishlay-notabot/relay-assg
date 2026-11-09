#!/usr/bin/env python3
"""Stand-in model service: an OpenAI-compatible chat endpoint we control.

It can be slow, fail, refuse (429), slow down under load, and stream its reply
piece by piece - the behaviours the assignment asks for. Calibrated defaults
come from the real Groq probe (writeup_notes.txt); change via flags.

Run:
    python -m relay.standin --port 8000 --ttft-ms 450 --chunk-ms 15
    # then point GroqClient at http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class StandinConfig:
    # Calibrated from the real probe: TTFT p50 ~449ms, total ~461ms.
    ttft_ms: int = 300
    chunk_ms: int = 15
    reply: str = (
        "The stand-in model answers from the retrieved passage "
        "[1706.03762v7 p.9]. This reply exists to exercise the pipeline."
    )
    failure_rate: float = 0.0
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 1000
    retry_after_s: int = 2
    slowdown_ms_per_inflight: int = 150
    seed: int = 7


@dataclass
class _State:
    lock: threading.Lock
    request_times: deque
    token_window: deque  # (timestamp, tokens)
    day_requests: int = 0
    inflight: int = 0


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # set by make_server
    cfg: StandinConfig
    state: _State

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send_json(self, code: int, payload: dict, extra_headers: dict | None = None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _rate_limited(self) -> bool:
        cfg, st = self.cfg, self.state
        now = time.time()
        with st.lock:
            while st.request_times and now - st.request_times[0] > 60:
                st.request_times.popleft()
            while st.token_window and now - st.token_window[0][0] > 60:
                st.token_window.popleft()
            tokens_used = sum(t for _, t in st.token_window)
            if len(st.request_times) >= cfg.rpm or tokens_used >= cfg.tpm or st.day_requests >= cfg.rpd:
                return True
            st.request_times.append(now)
            st.token_window.append((now, 40))
            st.day_requests += 1
            return False

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "bad json"}})
            return

        if self.cfg.failure_rate > 0 and random.random() < self.cfg.failure_rate:
            self._send_json(500, {"error": {"message": "stand-in injected failure"}})
            return
        if self._rate_limited():
            self._send_json(
                429,
                {"error": {"message": "Rate limit reached (stand-in)", "type": "rate_limit"}},
                {"retry-after": str(self.cfg.retry_after_s)},
            )
            return

        with self.state.lock:
            self.state.inflight += 1
            inflight = self.state.inflight
        try:
            self._serve(req, inflight)
        finally:
            with self.state.lock:
                self.state.inflight -= 1

    def _serve(self, req: dict, inflight: int):
        cfg = self.cfg
        messages = req.get("messages", [])
        reply = cfg.reply
        # Stand-in heuristic so it can play both pipeline roles: it approves a
        # draft when asked to verify, and cites a passage when asked to answer.
        if any("You check an answer" in m.get("content", "") for m in messages):
            reply = "YES"
        words = reply.split()
        max_tokens = int(req.get("max_tokens") or len(words))
        words = words[:max_tokens]
        prompt_tokens = sum(len(m.get("content", "")) for m in req.get("messages", [])) // 4 + 4
        completion_tokens = len(words)

        first_delay = (cfg.ttft_ms + cfg.slowdown_ms_per_inflight * (inflight - 1)) / 1000
        time.sleep(first_delay)
        model = req.get("model", "stand-in")

        if not req.get("stream"):
            time.sleep(cfg.chunk_ms * len(words) / 1000)
            self._send_json(200, {
                "id": "chatcmpl-standin",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("x-ratelimit-remaining-requests", str(max(0, cfg.rpd - self.state.day_requests)))
        self.send_header("x-ratelimit-remaining-tokens", str(max(0, cfg.tpm)))
        self.end_headers()
        self.close_connection = True
        for i, word in enumerate(words):
            delta = word if i == 0 else " " + word
            chunk = {"id": "chatcmpl-standin", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
            if i == len(words) - 1:
                chunk["usage"] = {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": completion_tokens,
                                  "total_tokens": prompt_tokens + completion_tokens}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            if i < len(words) - 1:
                time.sleep(cfg.chunk_ms / 1000)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def make_server(config: StandinConfig | None = None, host: str = "127.0.0.1", port: int = 0):
    cfg = config or StandinConfig()
    random.seed(cfg.seed)
    handler = type("BoundHandler", (_Handler,), {
        "cfg": cfg,
        "state": _State(lock=threading.Lock(), request_times=deque(), token_window=deque()),
    })
    return ThreadingHTTPServer((host, port), handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ttft-ms", type=int, default=StandinConfig.ttft_ms)
    p.add_argument("--chunk-ms", type=int, default=StandinConfig.chunk_ms)
    p.add_argument("--failure-rate", type=float, default=StandinConfig.failure_rate)
    p.add_argument("--rpm", type=int, default=StandinConfig.rpm)
    p.add_argument("--tpm", type=int, default=StandinConfig.tpm)
    p.add_argument("--rpd", type=int, default=StandinConfig.rpd)
    p.add_argument("--slowdown-ms", type=int, default=StandinConfig.slowdown_ms_per_inflight)
    args = p.parse_args()

    cfg = StandinConfig(ttft_ms=args.ttft_ms, chunk_ms=args.chunk_ms,
                        failure_rate=args.failure_rate, rpm=args.rpm, tpm=args.tpm,
                        rpd=args.rpd, slowdown_ms_per_inflight=args.slowdown_ms)
    server = make_server(cfg, args.host, args.port)
    print(f"stand-in listening on http://{args.host}:{server.server_address[1]}/v1  (rpm={cfg.rpm})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
