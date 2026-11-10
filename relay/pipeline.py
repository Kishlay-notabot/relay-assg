"""Pipeline skeleton: retrieve -> refusal branch -> draft -> verify -> redraft loop.

The model client is injected, so this whole module runs for free with any
stand-in or fake. Step timings are recorded per request for the write-up's
"where the time goes" section.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from relay.llm import CompletionResult, LLMError, LLMTimeout, RateLimited
from relay.retrieval import Chunk, is_answerable, retrieve

CITE_RE = re.compile(r"\b(\d{4}\.\d{4,5}v\d+)\s+p\.(\d+)")
REFUSAL_START_RE = re.compile(
    r"^\s*(the\s+)?(provided\s+)?(passages?|context|texts?|excerpts?)\s+"
    r"(do not|does not|don't|doesn't|did not)\s+"
    r"(contain|cover|address|mention|say|provide|include|discuss)",
    re.IGNORECASE,
)

REFUSAL_TEXT = (
    "The papers in this library do not say. I can only answer from the three "
    "included papers and none of them covers this question."
)
BUSY_TEXT = (
    "The assistant is busy right now and could not finish your answer. "
    "Please try again shortly."
)
UNVERIFIED_NOTE = " (This answer could not be fully verified against the passages.)"

DRAFT_SYSTEM = (
    "Answer only from the PASSAGES below. Cite every claim in the form "
    "[paper_id p.N] using the citation given with each passage. The passages are "
    "excerpts: if they contain evidence for the answer, use it and answer, even "
    "if they do not restate the question. Always write a complete answer - never "
    "reply with only a citation. If the passages truly do not address the "
    "question, reply exactly: NOT_IN_PASSAGES."
)

VERIFY_SYSTEM = (
    "You check an answer against the passages it cites. Reply 'YES' if every "
    "factual claim is supported by the passages and citations are present. "
    "Otherwise reply 'NO: <one short reason>'."
)


@dataclass
class PipelineConfig:
    k: int = 4
    min_score: float = 0.2
    rerank: bool = True
    max_redrafts: int = 2
    verify: bool = True
    est_tokens_per_call: int = 1200
    draft_max_tokens: int = 700
    verify_max_tokens: int = 400


@dataclass
class StepRecord:
    name: str
    first_token_ms: int = 0
    total_ms: int = 0
    ok: bool = True


@dataclass
class Answer:
    text: str
    status: str = "ok"
    citations: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    redrafts: int = 0
    hits: list[Chunk] = field(default_factory=list)
    first_token_ms: int = 0
    total_ms: int = 0
    degraded: bool = False
    governor_wait_ms: int = 0
    verification: str = ""


def extract_citations(text: str) -> list[str]:
    seen, out = set(), []
    for paper, page in CITE_RE.findall(text):
        cite = f"{paper} p.{page}"
        if cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


def is_refusal(text: str) -> bool:
    """True when the model says the library cannot answer.

    Models do not always emit the bare marker: Qwen tends to explain first and
    append NOT_IN_PASSAGES (or start with "the passages do not contain ...")
    while still citing nearby pages. Treating that as a draft wastes a verifier
    call and mislabels a correct refusal.
    """
    return "not_in_passages" in text.lower() or bool(REFUSAL_START_RE.match(text))


def _passages(hits: list[Chunk]) -> str:
    return "\n\n".join(f"[{h.paper_id} p.{h.page_number}] {h.text}" for h in hits)


class Pipeline:
    def __init__(self, client, chunks: list[Chunk], config: PipelineConfig | None = None,
                 governor=None):
        self.client = client
        self.chunks = chunks
        self.config = config or PipelineConfig()
        self.governor = governor

    # --- steps ---------------------------------------------------------
    def _retrieve(self, question: str, answer: Answer) -> None:
        t0 = time.perf_counter()
        answer.hits = retrieve(question, self.chunks, k=self.config.k, rerank=self.config.rerank)
        answer.steps.append(StepRecord("retrieve", 0, round((time.perf_counter() - t0) * 1000)))

    def _call(self, messages: list[dict], answer: Answer, step: str, max_tokens: int) -> CompletionResult:
        est = self.config.est_tokens_per_call
        if self.governor:
            decision = self.governor.acquire(est)
            if decision.action == "refuse":
                raise RateLimited(decision.reason)
            if decision.action == "wait":
                time.sleep(decision.wait_s)
                answer.governor_wait_ms += round(decision.wait_s * 1000)
            if decision.action == "degrade":
                answer.degraded = True
        try:
            result = self.client.complete(messages, max_tokens=max_tokens)
        except (RateLimited, LLMTimeout, LLMError):
            raise
        if self.governor:
            self.governor.record(result.tokens_in + result.tokens_out, est)
        if not answer.first_token_ms:
            elapsed_before = sum(s.total_ms for s in answer.steps)
            answer.first_token_ms = elapsed_before + result.first_token_ms
        answer.tokens_in += result.tokens_in
        answer.tokens_out += result.tokens_out
        answer.steps.append(StepRecord(step, result.first_token_ms, result.total_ms))
        return result

    def _draft(self, question: str, answer: Answer, feedback: str | None) -> str:
        user = f"PASSAGES\n{_passages(answer.hits)}\n\nQUESTION: {question}"
        if feedback:
            user += f"\n\nYour previous draft was rejected: {feedback}\nFix it."
        result = self._call(
            [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}],
            answer, "draft", max_tokens=self.config.draft_max_tokens,
        )
        return result.text.strip()

    def _verify(self, draft: str, answer: Answer) -> bool:
        if not self.config.verify:
            return True
        if self.governor and self.governor.degraded:
            answer.degraded = True
            return True  # weaker answer: skip the check under budget pressure
        result = self._call(
            [{"role": "system", "content": VERIFY_SYSTEM},
             {"role": "user", "content": f"PASSAGES\n{_passages(answer.hits)}\n\nANSWER\n{draft}"}],
            answer, "verify", max_tokens=self.config.verify_max_tokens,
        )
        answer.verification = result.text.strip()[:200]
        return result.text.strip().lower().startswith("yes")

    # --- orchestration -------------------------------------------------
    def run(self, question: str) -> Answer:
        t_start = time.perf_counter()
        answer = self._run(question)
        answer.total_ms = round((time.perf_counter() - t_start) * 1000)
        return answer

    def _run(self, question: str) -> Answer:
        answer = Answer(text="")
        self._retrieve(question, answer)

        # Branch 1: refusal. No model call at all, so a hopeless question
        # costs zero tokens.
        if not is_answerable(answer.hits, self.config.min_score):
            answer.status = "refused"
            answer.text = REFUSAL_TEXT
            return answer

        draft, feedback = "", None
        try:
            for attempt in range(self.config.max_redrafts + 1):
                if attempt:
                    answer.redrafts += 1
                    answer.retries += 1
                draft = self._draft(question, answer, feedback)
                if is_refusal(draft):
                    answer.status, answer.text = "refused", REFUSAL_TEXT
                    return answer
                if not extract_citations(draft):
                    feedback = "the answer must cite passages as [paper_id p.N]"
                    continue
                if self._verify(draft, answer):
                    answer.status = "ok"
                    answer.text, answer.citations = draft, extract_citations(draft)
                    return answer
                feedback = "at least one claim was not supported by the cited passages"
            answer.status = "unverified"
            answer.text = draft + UNVERIFIED_NOTE
            answer.citations = extract_citations(draft)
            return answer
        except RateLimited:
            answer.status, answer.text = "rate_limited", BUSY_TEXT
        except LLMTimeout:
            answer.status, answer.text = "timeout", BUSY_TEXT
        except LLMError:
            answer.status, answer.text = "error", BUSY_TEXT
        return answer
