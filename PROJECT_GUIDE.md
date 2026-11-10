# Project guide — The Relay

This document explains what the system does, how data moves through it, how the
write-up maps to the raw file, and where the implementation is weak. It is written
for the live conversation: every claim here should be defensible from the code or
from `results.csv`.

## 1. One-paragraph summary

`relay/` is a cited question-answering pipeline over three pinned arXiv papers. It
splits the PDFs into overlapping character chunks, retrieves the top-k with BM25
(optionally re-ranked by query-term coverage), refuses when the best score is below
a floor, drafts an answer with `[paper p.N]` citations, verifies that draft against
the same passages with a second model call, and allows up to two redrafts. A
client-side governor enforces request/token budgets per minute and per day. The
same model interface is implemented by a real Groq client and by a controllable
stand-in HTTP service. A load harness drives steady, burst, and overload traffic and
writes one row per request to `results.csv`; a deterministic grader scores answers
against 20 hand-written questions.

## 2. Repository map

| Path | Role |
|---|---|
| `relay/ingest.py` | PDF → pages → chunks; refuses non-pinned filenames |
| `relay/retrieval.py` | normalization, chunking, BM25, coverage rerank, refusal gate |
| `relay/llm.py` | `CompletionResult`, `LLMClient` protocol, error types |
| `relay/groq_client.py` | real streaming client; TTFT; 429/timeout mapping |
| `relay/standin.py` | OpenAI-compatible stand-in: slow, fail, 429, slowdown, stream |
| `relay/pipeline.py` | retrieve → gate → draft → verify → redraft loop |
| `relay/governor.py` | client-side RPM/TPM/RPD/TPD policy: wait / degrade / refuse |
| `relay/load.py` | retry wrapper, load shapes, CSV writer, per-request traces |
| `relay/trace.py` | waiting/working/retrying split for one request |
| `relay/grading.py` | deterministic grading + normalization |
| `relay/report.py` | Tables 2a/2b/3 computed from CSV rows |
| `eval_quality.py` | 20-question quality run (real or stand-in) |
| `run_load.py` | load-shape CLI; stand-in by default, real only with `--yes-spend-real` |
| `run_retrieval.py` | three-setting retrieval comparison (Table 3) |
| `probe.py` | real-service latency/limit probe → `probe_results.csv` |
| `summarize.py` | prints Tables 2a/2b from `results.csv` |
| `trace_report.py` | prints the slowest request's latency split |
| `questions.json` | 20 questions: 16 answerable, 4 unanswerable |
| `results.csv` | required raw file: 218 request rows |
| `quality_qwen.csv` | the 20 quality rows alone (subset of `results.csv`) |
| `probe_results.csv` | 33 probe rows (latency, ramp, recovery attempt) |

## 3. End-to-end data flow

### Offline (free, local)

```text
3 pinned PDFs
   │  relay/ingest.py: extract pages (pypdf), keep paper_id + page number
   ▼
pages ──chunk_pages(size=1200, overlap=200)──► 434 chunks
   │  lowercase, strip arXiv stamp + page-number lines, collapse whitespace
   ▼
chunks_1200.jsonl  (paper_id, page_number, chunk_index, text)
```

Chunking is character-based, not token-based. Overlap exists so a fact split
across a boundary survives in at least one piece; it is not table-aware.

### One online request

```text
question
  │
  ├─► retrieve (BM25 over all chunks) ──► top-k=5 (+ optional coverage rerank)
  │        │
  │        └─ is_answerable(max_score ≥ 0.2)?
  │              no ──► refusal text, 0 model calls, 0 tokens
  │
  ├─► governor.acquire(est_tokens)
  │        wait (minute limit) │ degrade (≥90% day) │ refuse (day gone)
  │
  ├─► draft call (max_tokens 700), passages + question (+ prior feedback)
  │        │
  │        ├─ refusal marker/phrasing? ──► refusal text, stop
  │        ├─ no citation? ──► feedback, redraft (up to 2)
  │        └─ verify call (max_tokens 400) against the same passages
  │                YES ──► answer + citations, stop
  │                NO  ──► feedback, redraft; after last attempt → unverified
  │
  └─► Answer: text, citations, status, per-step timings, tokens, retries,
      governor wait, degraded flag, verifier verdict
```

### Under load

```text
run_load.py
  ├─ build arrival offsets: steady(duration, rate) / burst(count) / overload(...)
  ├─ ThreadPoolExecutor(workers) submits each request at its offset
  ├─ pipeline_factory() creates a fresh Pipeline + RetryingClient per request
  │     └─ RetryingClient: ≤1 retry on 429/timeout, honours retry-after
  └─ worker writes one CSV row + one trace line
```

### Results pipeline

```text
results.csv (13 fixed columns)
  ├─ summarize.py → Tables 2a/2b (latency percentiles, outcomes, queue, tokens)
  ├─ relay.report.retrieval_table → Table 3 (quality + cost per setting)
  └─ trace_report.py + raw/traces.jsonl → latency split for one slow request
```

## 4. What is in `results.csv`

218 data rows = 180 load rows + 20 quality rows + 18 retrieval rows.

| Scenario | Rows | Source | What it is |
|---|---|---|---|
| steady | 20 | stand-in | 24 answers/min for 50 s |
| burst | 10 | stand-in | 10 simultaneous arrivals |
| sustained-overload | 90 | stand-in | 180/min offered for 30 s, 8 workers |
| per-minute-limit-halved | 30 | stand-in | stand-in RPM 15, offered 60/min |
| daily-budget-exhausted | 30 | stand-in | governor TPD 4,000 |
| quality-real | 20 | real | 20-question graded set |
| retrieval | 18 | real | 3 settings × 6 questions |

Column conventions:

- `status` is transport status only: `ok | rate_limited | timeout | error`
  (refusals and unverified answers are `ok`; their quality lives in `answer_grade`).
- `answer_grade` is `correct | wrong | refused_correctly | refused_wrongly` or empty
  when the request produced no gradeable answer.
- Latency percentiles use `ok` rows only, nearest-rank.
- Peak queue is derived from `[start_time, start_time + total_ms]` overlaps, not
  from an internal scheduler counter (the run log's live counter reported 17 for
  overload; the CSV-derived overlap peak is 9 — the CSV definition is the one in
  the table and is reproducible).
- Duration = last start − first start + that request's `total_ms`.
- Cost is $0.00 on the free tier.

Quality rows were reassembled from `raw/answers_qwen2.jsonl`. The answer log had
no timestamps, so their `start_time` was reconstructed backwards from the log's
last write time, subtracting each measured `total_ms`. `retries` is 0 for those
rows because the log did not record it. This is an approximation; it does not
affect quality or token figures.

## 5. How to read the write-up

**Section 1 — design.** The governing idea is that limits are scheduling
constraints. Everything else (stand-in, governor, traces) exists to make that
testable. The pipeline is deliberately four stages, not a framework.

**Section 2 — Table 1.** Limits come from two sources: the docs table and real
429s. TPM was confirmed by a 429 body (`Limit 8000, Used 7556`); RPM by the probe
ramp hitting `Limit 30, Used 30`. TPD is docs-only because Groq returns no TPD
header and exhausting a day to measure it is wasteful. In-flight is not measured.
The recovery attempt (wait `retry-after` + 1 s, then one request) timed out, so
recovery is only partly evidenced; the probe logs it honestly.

**Section 3 — Tables 2a/2b.** The five load shapes separate four failure modes:
burst queueing (TTFT rises from 552 ms steady to 1,198 ms, while throughput is
fine), sustained load (all succeed but per-request total is ~3.4 s), provider
throttling (76.7% rate-limited), and daily exhaustion (96.7% refused at zero
tokens). The real row is the 20-question quality run; its duration is wall-clock
across two batches, not an offered load.

Quality is reported two ways so the denominator is not hidden: over all 20
questions, 13 correct (65%), 3 correct refusals (15%), 2 unverified (10%), 2
failures (10%); over the 16 gradeable answers, 13 correct (81.2%) and 3 correct
refusals (18.8%). **A number to emphasize:** verification is a second model call
that starts after the first token, so roughly 1.8 s of a 4 s burst request happens
after the user has already seen text.

**Section 4 — where time goes.** `burst-008`: 4,026 ms total, 1,969 ms pre-token,
2,057 ms post-token. Retrieval is 97 ms; draft 2,122 ms; verify 1,807 ms. The
pre-token half is concurrency-induced latency from the stand-in's per-in-flight
slowdown. The post-token half is stream completion plus verification.

**Section 5 — limits.** Per-minute pressure → governor waits (delayed, not
dropped). Day pressure → degrade at 90% (skip verify: weaker answer), refuse at
100% (dropped instantly, no tokens). The halved-RPM row shows what happens if the
governor is absent: provider 429s, and one retry only rescues 7/30.

**Section 6 — retries.** One retry, 429/timeout only, `retry-after` honoured.
The counter-example is `per-minute-limit-halved-008`: 2 s wait, retry, 429 again,
2,979 ms spent for nothing. All 23 retried rows in that run were still
rate-limited.

**Retrieval section — Table 3.** 600-char chunks are fast and useless (16.7%
correct); 1,200/k5 and 2,400/k2 tie on quality (66.7%), but 1,200/k5 has a much
better median total. The 2,400 setting uses 38% fewer tokens, which matters under
TPM pressure, but is slower at the median on six questions. Choice: 1,200/k5.

**Naive-retriever traps.** (1) Appendix chat transcripts outrank real content for
keyword-y questions. (2) BM25 scores are not comparable across queries — the
France trap scores 11.08 on a transcript. (3) Layout-separated tables are split by
character chunks, so the caption can arrive without the number.

**Section 8 — uncertainties.** The largest is that limits are per model. Not
built: provider failover and caching. New knowledge: reasoning tokens consume
`max_tokens`, so a small verifier budget returns empty content.

**Assistant section.** The prompt history is `conversation.md`. The three decisive
prompts were the docs check that led to Qwen, the stop that interrupted an
unapproved run, and the rule that results come before new real tests. The thing it
got wrong was launching a second full evaluation mid-review. The overrule was the
model choice: it recommended staying put; the docs check justified switching.

## 6. Shortcomings (what I would call out before someone else does)

### Correctness and evaluation

1. **Verification is not independent.** The same model checks its own draft
   against the same passages. Errors shared by drafter and verifier pass. There
   is no claim-by-claim citation check: the grader checks that *a* citation to the
   right paper exists, not that every sentence is supported. *Fix:* split claims,
   entail each against the cited chunk, or use a different model for verification.
2. **The grader is substring-based.** `must_include`/`any_of` matches normalized
   substrings, so a number can match inside a different number and a fact can
   appear in a negated sentence. It is cheap and reproducible, but it is not a
   judge. *Fix:* numeric-aware assertions plus a small human-audited judge set.
3. **The refusal gate is one global threshold.** BM25 scores are not comparable
   across queries; 0.2 is not calibrated per setting or paper. The France trap
   passes the score floor and is only caught because the model declines.
   *Fix:* score relative to the query's background distribution, or calibrate a
   threshold per setting on labelled data.
4. **Quality sample is small and single-shot.** 20 questions, one attempt each,
   no confidence intervals; Table 3 uses six questions per setting, where one
   answer moves a percentage by 16.7 points. *Fix:* more questions and repeats,
   report intervals.
5. **Unverified answers are still sent** (with a caveat) rather than withheld.
   This is a deliberate latency/quality trade, but nothing measures how often a
   caveated answer is wrong.

### Retrieval

6. **Lexical only.** No stemming or synonyms; English stopwords; BM25 parameters
   fixed. A paraphrase ("heads" vs "attention heads") may miss.
7. **Character chunking splits tables and formulas.** Overlap mitigates but does
   not solve it. *Fix:* page-window expansion for numeric questions or a
   table-aware extractor.
8. **The reranker is a heuristic.** `BM25 × (0.5 + coverage)` mixes an unbounded
   score with a fraction; it is not tuned or validated. Duplicate chunks from the
   same page can occupy several of the five slots.
9. **BM25 is recomputed per query over every chunk**, tokenizing the whole corpus
   each time. Fine at 434 chunks; it will not scale to 100K without an inverted
   index or cached term statistics.
10. **No embeddings/hybrid retrieval**, so there is no semantic fallback when
    lexical overlap is low; the k=3 failure on "8 heads" is the visible cost.

### Budget, retries, and operations

11. **Governor accounting is estimate-based.** Each call is charged a flat
    1,200-token estimate, but real calls ranged roughly 600–3,000. Actual usage is
    recorded after the fact, so a burst can overshoot TPM before the correction.
12. **Governor state is per process and in memory.** A new run starts with a full
    daily budget and does not know what previous runs spent; there is no midnight
    rollover. `--governor` protects a run, not the account.
13. **Retry policy is minimal.** One retry, no jitter, no 5xx retry, no circuit
    breaker. `retry-after` is honoured but all clients retry in lockstep, which
    can re-trigger the limit. The counter-example cost 2 s per retry for nothing.
14. **No request deadline or cancellation.** Socket timeout is 30 s per call; a
    request with redrafts can take ~90 s worst case. The user waits through
    verification even after the first token is ready, and nothing enforces a
    user-facing SLO.
15. **No caching.** Repeated questions pay the full token cost again; there is no
    idempotency key or dedupe for concurrent identical questions.
16. **Concurrency is unbounded by admission control.** ThreadPoolExecutor size is a
    flag; there is no queue cap or backpressure. Peak queue is computed after the
    fact from timestamps.
17. **CSV writing is append-only and not atomic.** Two concurrent runs can
    interleave rows or write two headers. There is no run id, so reruns are only
    distinguishable by scenario name.
18. **Error detail is lost in `results.csv`.** The fixed 13 columns carry only
    `status`; the 429 body or timeout reason lives in traces/logs. The quality
    rows' `start_time` and `retries` are reconstructed/unknown (documented in §4).
19. **Traces exist only for load runs**, not the quality or retrieval runs, so
    their per-step latency split is unavailable.
20. **Configuration is duplicated across CLIs.** Defaults drift: the pipeline
    default k=4 while the final quality and retrieval runs use k=5 via flags;
    three runners each parse their own governor flags.
21. **Observability is `print`.** No structured logs, metrics, or request ids
    across retries. A 429 spike is invisible unless someone reads the CSV.
22. **Secrets handling is basic.** `.env` is parsed by hand, no quote handling,
    no validation; nothing redacts headers if a client error echoes a request.
23. **No prompt-size guard.** Raising k or chunk size can exceed the model context;
    there is no tokenizer-side check before the call.

### Stand-in fidelity

24. **Token accounting is fake** (40 tokens per request), so TPM-driven behaviour
    is not realistic; the stand-in throttles on RPM in practice.
25. **The reply is canned.** It cannot exercise the verifier's NO path or grader
    quality; quality numbers from the stand-in are meaningless.
26. **No jitter or tail latency**, no TLS/auth, no day rollover, and the
    thread-per-connection server is unbounded.
27. **Fixed failure modes.** Failures are uniform random; real failures cluster
    (one bad minute, one slow model shard).

### Process and testing

28. **Most tests use fakes, not the HTTP stand-in end to end.** There is no test
    that drives `run_load.py` through a real socket and asserts CSV contents.
29. **No regression set for prompts.** The prompt change that fixed over-refusal
    (k=5 run) was validated on the same 20 questions used to tune it, so the
    measured 65% is optimistic for unseen questions.
30. **No cost model for the paid tier** in the CLI defaults; cost columns are
    zero unless prices are passed manually.

## 7. Live-conversation preparation

Likely questions and the short answer:

- **Why BM25 and not embeddings?** Deterministic, free, debuggable; the failure
  modes are visible. The k=3 miss and France trap show the ceiling. Hybrid is the
  next step, not a rewrite.
- **Why verify with the same model?** It is a cheap second pass that catches
  uncited and unsupported drafts; it is a filter, not a judge. Independent
  verification is the first thing I would add.
- **Why is the daily budget per process?** Honest answer: it is a scheduling
  simulation, not an account ledger. Persisting remaining quota in a file/DB and
  reading `x-ratelimit-*` headers on every call is the fix.
- **What breaks first at 10× load?** BM25 recomputation per query (CPU), then
  the single-process governor, then CSV appends.
- **What would you change live?** Lower k to 3 and drop rerank under TPM pressure;
  add retry jitter; persist governor state; add a claim-level citation check;
  stream the draft to the user before verification completes.
- **Why is 65% not 81%?** 65% is over all 20 questions; 81.2% is over the 16 that
  produced a gradeable answer. Both are in the write-up so the denominator is
  explicit.
- **What is genuinely yours?** The question set and France trap, the decision to
  measure before switching models, the choice of Qwen-after-docs, the budget
  discipline that stopped two runs, and the refusal classifier derived from real
  drafts.

## 8. Reproduce

```sh
source ../lang/bin/activate
python -m unittest discover -s tests            # 95 tests
python summarize.py results.csv                 # Tables 2a/2b
python trace_report.py raw/traces.jsonl --n 1   # slowest request
python - <<'PY'                                 # Table 3 from results.csv
from relay.report import load_rows, retrieval_table
for r in retrieval_table([x for x in load_rows("results.csv") if x["scenario"]=="retrieval"]):
    print(r)
PY
```

Stand-in load rows regenerate for free. Real rows cost quota and require
`--yes-spend-real`; the raw answer text needed to regrade them lives in
`raw/answers_qwen2.jsonl` (gitignored, local).
