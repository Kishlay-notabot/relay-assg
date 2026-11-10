"""Tests for the pipeline skeleton. No network: FakeClient is injected."""

import unittest
import time

from relay.llm import CompletionResult, RateLimited
from relay.pipeline import (  # noqa: E402
    REFUSAL_TEXT,
    Answer,
    Pipeline,
    PipelineConfig,
)
from relay.retrieval import Chunk


class FakeClient:
    """Scripted client. Each call pops the next response (or raises it)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.max_tokens = []

    def complete(self, messages, max_tokens=256):
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        time.sleep(0.005)
        return CompletionResult(
            text=item,
            first_token_ms=2,
            total_ms=5,
            tokens_in=500,
            tokens_out=50,
            status="ok",
        )


CHUNKS = [
    Chunk("1706.03762v7", 9, 0, "table 3 shows the transformer uses 8 attention heads"),
    Chunk("2410.00037v1", 6, 0, "moshi is a full-duplex speech dialogue model"),
]
CONFIG = PipelineConfig(k=2, min_score=0.2, max_redrafts=1)
GROUNDED = "The Transformer uses 8 heads [1706.03762v7 p.9]."


class RefusalTest(unittest.TestCase):
    def test_unanswerable_question_refuses_without_calling_model(self):
        client = FakeClient([])
        answer = Pipeline(client, CHUNKS, CONFIG).run("what is the capital of france")

        self.assertEqual(answer.status, "refused")
        self.assertEqual(client.calls, [])
        self.assertIn("do not", answer.text.lower())

    def test_refusal_names_the_library(self):
        answer = Pipeline(FakeClient([]), CHUNKS, CONFIG).run("quantum gravity")
        self.assertTrue(answer.citations == [])
        self.assertIn("papers", answer.text.lower())


class HappyPathTest(unittest.TestCase):
    def test_grounded_answer_is_returned_with_citations(self):
        client = FakeClient([GROUNDED, "YES"])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "ok")
        self.assertIn("1706.03762v7 p.9", answer.citations)
        self.assertEqual(len(client.calls), 2)  # draft + verify
        self.assertEqual(answer.tokens_in, 1000)
        self.assertEqual(answer.tokens_out, 100)

    def test_draft_prompt_contains_retrieved_passages_and_their_citations(self):
        client = FakeClient([GROUNDED, "YES"])
        Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        draft_prompt = str(client.calls[0])
        self.assertIn("8 attention heads", draft_prompt)
        self.assertIn("1706.03762v7 p.9", draft_prompt)

    def test_verify_prompt_contains_draft_and_passages(self):
        client = FakeClient([GROUNDED, "YES"])
        Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        verify_prompt = str(client.calls[1])
        self.assertIn(GROUNDED, verify_prompt)
        self.assertIn("8 attention heads", verify_prompt)


class VerificationLoopTest(unittest.TestCase):
    def test_unsupported_draft_triggers_one_redraft_then_succeeds(self):
        client = FakeClient([
            "The Transformer uses 12 heads [1706.03762v7 p.9].",  # draft 1
            "NO: passage says 8",                               # verify 1
            GROUNDED,                                           # draft 2
            "YES",                                              # verify 2
        ])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "ok")
        self.assertEqual(answer.text, GROUNDED)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(answer.retries, 1)
        self.assertEqual(answer.redrafts, 1)

    def test_redrafts_are_capped_and_final_attempt_is_returned(self):
        client = FakeClient([
            "bad one [1706.03762v7 p.9].",
            "NO: unsupported",
            "bad two [1706.03762v7 p.9].",
            "NO: still unsupported",
        ])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "unverified")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(answer.redrafts, 1)

    def test_answer_without_citation_is_sent_back_for_redraft(self):
        client = FakeClient(["8 heads, no citation"])
        answer = Pipeline(client, CHUNKS, PipelineConfig(k=2, min_score=0.2, max_redrafts=0)).run(
            "how many attention heads"
        )
        # max_redrafts=0 -> no retry; but a citation-free draft must never be ok
        self.assertEqual(answer.status, "unverified")
        self.assertEqual(len(client.calls), 1)

    def test_draft_without_citation_is_redrafted_when_budget_allows(self):
        client = FakeClient(["8 heads, no citation", GROUNDED, "YES"])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "ok")
        self.assertEqual(len(client.calls), 3)
        self.assertIn("1706.03762v7 p.9", answer.citations)


    def test_model_refusal_maps_to_refused_status(self):
        client = FakeClient(["NOT_IN_PASSAGES"])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "refused")
        self.assertEqual(answer.text, REFUSAL_TEXT)
        self.assertEqual(len(client.calls), 1)  # no redraft, no verify

    def test_refusal_marker_appended_to_prose_is_still_a_refusal(self):
        client = FakeClient([
            "The passages do not cover this. NOT_IN_PASSAGES [1706.03762v7 p.9]"
        ])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "refused")
        self.assertEqual(answer.text, REFUSAL_TEXT)
        self.assertEqual(len(client.calls), 1)  # no verify call spent on a refusal

    def test_refusal_phrasing_with_citations_is_still_a_refusal(self):
        client = FakeClient([
            "The provided passages do not contain any information regarding quantum "
            "computing [2507.07955v2 p.32]."
        ])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "refused")
        self.assertEqual(answer.text, REFUSAL_TEXT)
        self.assertEqual(len(client.calls), 1)

    def test_verifier_gets_enough_tokens_for_reasoning_models(self):
        client = FakeClient([GROUNDED, "YES"])
        Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertGreaterEqual(client.max_tokens[1], 150)


class VerdictTest(unittest.TestCase):
    def test_verifier_verdict_is_recorded_on_answer(self):
        client = FakeClient([GROUNDED, "NO: passage says 8"])
        answer = Pipeline(client, CHUNKS, PipelineConfig(max_redrafts=0)).run(
            "how many attention heads")

        self.assertIn("NO", answer.verification)

    def test_empty_verdict_is_recorded(self):
        client = FakeClient([GROUNDED, ""])
        answer = Pipeline(client, CHUNKS, PipelineConfig(max_redrafts=0)).run(
            "how many attention heads")

        self.assertEqual(answer.verification, "")


class GovernorTest(unittest.TestCase):
    def test_exhausted_budget_refuses_before_calling_model(self):
        from relay.governor import Governor, GovernorConfig

        client = FakeClient([])
        g = Governor(GovernorConfig(rpd=0))
        answer = Pipeline(client, CHUNKS, CONFIG, governor=g).run("how many attention heads")

        self.assertEqual(answer.status, "rate_limited")
        self.assertEqual(client.calls, [])

    def test_degraded_mode_skips_verification(self):
        from relay.governor import Governor, GovernorConfig

        client = FakeClient([GROUNDED])
        g = Governor(GovernorConfig(rpm=30, tpm=8000, rpd=2,
                                    degrade_at_day_fraction=0.5))
        answer = Pipeline(client, CHUNKS, CONFIG, governor=g).run("how many attention heads")

        self.assertEqual(answer.status, "ok")
        self.assertTrue(answer.degraded)
        self.assertEqual(len(client.calls), 1)  # no verify call


    def test_governor_wait_time_is_attributed(self):
        from relay.governor import Decision

        class StubGovernor:
            degraded = False

            def acquire(self, est):
                return Decision("wait", wait_s=0.05, reason="rpm")

            def record(self, actual, est):
                pass

        answer = Pipeline(FakeClient([GROUNDED, "YES"]), CHUNKS, CONFIG,
                          governor=StubGovernor()).run("how many attention heads")

        self.assertGreaterEqual(answer.governor_wait_ms, 45)


class FailureTest(unittest.TestCase):
    def test_rate_limit_on_draft_surfaces_as_rate_limited(self):
        client = FakeClient([RateLimited("rpm")])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.status, "rate_limited")
        self.assertIn("busy", answer.text.lower())

    def test_answer_records_step_latencies(self):
        client = FakeClient([GROUNDED, "YES"])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        steps = [s.name for s in answer.steps]
        self.assertEqual(steps, ["retrieve", "draft", "verify"])
        self.assertTrue(all(s.total_ms >= 0 for s in answer.steps))

    def test_answer_records_end_to_end_and_first_token_timing(self):
        client = FakeClient([GROUNDED, "YES"])
        answer = Pipeline(client, CHUNKS, CONFIG).run("how many attention heads")

        self.assertEqual(answer.first_token_ms, 2)  # first model call's TTFT
        self.assertGreaterEqual(answer.total_ms, answer.first_token_ms)

    def test_refusal_has_zero_first_token_and_positive_total(self):
        answer = Pipeline(FakeClient([]), CHUNKS, CONFIG).run("quantum gravity")

        self.assertEqual(answer.first_token_ms, 0)
        self.assertGreaterEqual(answer.total_ms, 0)


if __name__ == "__main__":
    unittest.main()
