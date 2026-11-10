# The Relay

## 1. What I built and why

A Python pipeline that answers engineers' questions from three pinned arXiv papers:
**retrieve** (BM25 over 1,200-character pieces, top 5, optional coverage re-rank) →
**refusal gate** (weak best hit = refuse with no model call) → **draft with citations** →
**verify the draft against the same passages** → **up to two redrafts**. A real Groq client
and a controllable stand-in implement the same interface.

The decision that shaped it: **treat provider limits as scheduling, not error handling.** A
client-side governor owns RPM/TPM/RPD/TPD — it waits under minute pressure, skips verification
above 90% of the daily budget, and refuses before the call when the day is gone. The stand-in
lets me test what the free tier cannot survive. Every request lands in one raw file, so every
table number is recomputable.

## 2. The real service and my stand-in

**Table 1: limits found on the Groq free tier (`qwen/qwen3.8-27b`)**

| Limit | Value found | How I found it | Confidence |
|---|---|---|---|
| Requests per minute | 30 | Docs; real 429 on a ~50/min ramp: `Limit 30, Used 30` | High |
| Tokens per minute | 8,000 | Docs; real 429: `Limit 8000, Used 7556` | High |
| Requests per day | 1,000 | Docs; `x-ratelimit-remaining-requests` counts down per request | High |
| Tokens per day | 200,000 | Docs only (no TPD header) | Low–medium |
| Requests in flight | not measured | TPM throttled us before concurrency did | — |
| On a limit | HTTP 429 naming the limit, `retry-after: 2` | Captured body and headers | High |
| Time to recover | `retry-after` 2 s; one post-wait request timed out | Probe: wait then retry; n=1, inconclusive | Medium |

Headers are per-model. Probe latency (n=31 tiny requests): TTFT p50 155 ms, p95 489 ms.

The stand-in mirrors the OpenAI streaming shape: fixed TTFT, word-by-word streaming, injected
failures, 429 with `retry-after`, and 150 ms per in-flight request. Calibrated from the probe
(450 ms TTFT, 30 RPM, 8K TPM). **Where it differs most:** flat 40 tokens/request (real prompts
are 1.5–2K), no reasoning, fixed reply — its latency and failure data is meaningful, its quality
data is not.

## 3. Results

Real: the 20-question quality set and Table 3. All load shapes use the stand-in. Percentiles use
`ok` rows; n = requests sent.

**Table 2a: latency**

| Scenario | Real/stand-in | Requests | Duration (s) | Offered/min | First token p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 20 | 48.8 | 24.6 | 552/556/559 | 1258/1269/1269 |
| burst | stand-in | 10 | 3.2 | 185.8 | 1198/1969/1969 | 3263/4026/4026 |
| sustained overload | stand-in | 90 | 39.4 | 137.1 | 1601/1618/1631 | 3387/3402/3411 |
| per-minute limit halved | stand-in | 30 | 31.1 | 57.9 | 705/712/712 | 1416/1423/1423 |
| daily budget exhausted | stand-in | 30 | 29.1 | 61.9 | 517/517/517 | 1228/1228/1228 |
| real quality run* | real | 20 | 675.1 | 1.8 | 229/423/423 | 31330/73174/73174 |

**Table 2b: outcomes (same rows)**

| Scenario | Succeeded % | Rate-limited % | Timed out/given up % | Retries/req | Peak queue | Tokens/answer | Cost/1,000 |
|---|---|---|---|---|---|---|---|
| steady | 100.0 | 0.0 | 0.0 | 0.00 | 1 | 2,980.8 | $0.00 |
| burst | 100.0 | 0.0 | 0.0 | 0.00 | 10 | 2,967.3 | $0.00 |
| sustained overload | 100.0 | 0.0 | 0.0 | 0.00 | 9 | 2,980.3 | $0.00 |
| per-minute limit halved | 23.3 | 76.7 | 0.0 | 0.77 | 3 | 744.6 | $0.00 |
| daily budget exhausted | 3.3 | 96.7 | 0.0 | 0.00 | 2 | 148.5 | $0.00 |
| real quality run* | 90.0 | 5.0 | 5.0 | 0.00 | 2 | 4,998.2 | $0.00 |

\* Two sequential batches with governor waits, not a load shape; the duration is wall-clock.

**Quality, 20 questions I wrote (16 answerable across the three papers, 4 unanswerable,
including the France trap):** 13 correct and cited (65%), 3 correctly refused (15%), 2 unverified
drafts (10%), 2 failures — one rate limit, one timeout (10%). **Wrong answers: 0.** Of the 16 that
produced a gradeable answer, 13 were correct (81.2%) and 3 were correct refusals (18.8%); both
denominators are stated so neither flatters the run. Mean 4,998 tokens/answer. **A number I could
not have guessed:** verification is a second model call that begins *after* the first token, so
~1.8 s of a burst request is spent after the user has already seen text. Also, the cheapest
retrieval setting is fastest to first token (190 ms) and least correct (16.7%).

## 4. Where the time goes

Slowest request `burst-008` (sidecar traces): total **4,026 ms**; **1,969 ms before the first
token**, **2,057 ms after (51%)**. Retrieval 97 ms, model work 3,929 ms, retry wait 0. Steps:
draft 2,122 ms → verify 1,807 ms. The pre-token half is burst queueing plus the stand-in's
per-in-flight slowdown; the post-token half is the rest of the draft stream **plus verify**, which
must finish before the answer is sent. I know because the client timestamps the first chunk and
each step, and the trace records the split.

## 5. When a limit is hit

**Per-minute:** the governor sleeps until the window frees a slot — the person is *delayed*, not
dropped (shown as budget-wait time). Without it, the provider 429s; with one 2 s retry only 7/30
survived (23.3%), p95 TTFT 712 ms. **Per-day:** above 90% the governor skips verification — later
callers get a **weaker, unverified answer**; at 100% it refuses before the model call — they are
**dropped immediately** at zero token cost (96.7% of that scenario). The trade: verify early in
the day, speed and honesty late.

## 6. Retries

At most one retry, only on 429 or timeout; honour `retry-after`; exponential backoff (0.5 s base,
8 s cap); never retry other 4xx. **Where retrying made it worse:** `per-minute-limit-halved-008`
waited the full 2 s, retried, got 429 again, and gave up after 2,979 ms for zero useful work. All
23 retried rows in that run were still rate-limited, so each retry only added 2 s.

## Retrieval: quality versus cost

**Table 3: retrieval settings (6 questions each, real)**

| Setting (size, pieces, re-rank) | Questions | Correct+cited % | Wrong % | Correctly refused % | Tokens/answer | First token p50/p95 (ms) | Total p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| 600, k=3, no re-rank | 6 | 16.7 | 0.0 | 33.3 | 749.5 | 190/218 | 421/588 |
| 1,200, k=5, re-rank | 6 | 66.7 | 0.0 | 33.3 | 3,890.7 | 217/226 | 10,407/72,895 |
| 2,400, k=2, re-rank | 6 | 66.7 | 0.0 | 33.3 | 2,412.8 | 210/566 | 17,036/31,161 |

I chose **1,200 / k=5 / re-rank**: it matches the larger pieces on correctness with a much better
median total (10.4 s vs 17.0 s); 600-character pieces are fast and cheap but miss facts (16.7%).
The 2,400 setting uses 38% fewer tokens — attractive under TPM pressure — but is slower at the
median on a six-question sample. When a limit is close I cut k from 5 to 3, drop re-ranking, then
skip verification before refusing.

**How this library misleads a naive retriever.** (1) *Appendix transcripts outrank tables*: "How
many attention heads does the Transformer use?" surface-matches chatty appendix pages ahead of the
table discussion. (2) *BM25 scores are not comparable across queries*: "What is the capital of
France?" scores 11.08 off a Moshi transcript, above genuinely relevant hits — the word appears, the
answer does not. (3) *Numbers straddle chunk boundaries*: "h = 8 heads"/"d_model = 512" sit in a
layout-separated table, and a 1,200-character cut can keep the caption but drop the row — k=3 did
exactly this and the model honestly refused.

## What I'm unsure about, what I left out, what I learned

Least certain: that free-tier limits are **per model**, so quality and retrieval runs spend separate
daily budgets. If shared, my budget math is wrong and those runs would have failed sooner. Not
built: cross-provider failover, and a persistent answer cache. New to me: on a reasoning model,
hidden reasoning tokens are charged against `max_tokens` — a 60-token verifier cap returned empty
content, so every answer looked unverified.

## Your feedback on Vaani

_[To be written by me.]_

## How I worked with the assistant

The real prompt history is included as `conversation.md`. Three prompts that mattered: (1) *"can we
set reasoning budget to zero, check docs"* — the docs showed the previous model floors at `low`
while Qwen supports `none`, which is why the final model is Qwen. (2) *"stop everything what's
happening?"* — interrupted an in-flight paid run and reset the workflow to estimate tokens before
every real request. (3) *"tell me the result before running any new real test"* — made results
review a gate rather than a race.

One thing it got wrong that I caught: it launched a second full real evaluation while I was still
reading the first result set; I stopped it at 13 of 20 questions. The fix was process: quote the
token cost and get a go-ahead. One decision where I overruled it: it recommended keeping the
previous model; after the docs check I chose Qwen with reasoning disabled. Distinctly mine: the
question set (including the France trap), the decision to keep one model and drop the comparison,
and the budget discipline that stopped two runs.
