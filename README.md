# The Relay — cited QA over three papers

Working pipeline for the Relay assignment: retrieval-augmented, citation-checked
answers over three pinned arXiv papers, a controlled stand-in model service, load
shapes, deterministic grading, and tables computed from one raw file.

- Design, data flow, and known weaknesses: `PROJECT_GUIDE.md`
- Submission write-up: `writeup_final.md`
- Real prompt history: `conversation.md`
- Raw results (one row per request): `results.csv`

## Setup

```sh
source ../lang/bin/activate        # Python 3.12, has pypdf/requests
cp .env.example .env 2>/dev/null || true   # put GROQ_API_KEY=... in .env
python -m unittest discover -s tests -v
```

## Ingest (free, local)

```sh
python -m relay.ingest --size 1200 --overlap 200 --out chunks_1200.jsonl
```

Uses only the pinned versions: 1706.03762v7, 2410.00037v1, 2507.07955v2.

## Run the pipeline

```sh
# stand-in only, free
python run_load.py --scenario steady --name steady --duration 50 --rate 24 \
  --workers 8 --k 5 --setting s1200-k5-rerank
python run_load.py --scenario burst --name burst --count 10 --workers 10 --k 5

# real Groq (costs daily quota; explicit confirmation required)
python run_load.py --source real --scenario burst --count 5 --yes-spend-real

python summarize.py results.csv          # Tables 2a / 2b as markdown
```

Scenarios: `steady`, `burst`, `overload`; `--name` sets the scenario label in
`results.csv`. Rows append with the required 13 columns; per-request traces go to
`raw/traces.jsonl` (gitignored).

## Quality and retrieval comparison

```sh
python eval_quality.py --source stand-in --out quality.csv
python eval_quality.py --source real --yes-spend-real --governor --k 5 \
  --setting s1200-k5-rerank --out quality_qwen.csv
python run_retrieval.py --source real --yes-spend-real --governor \
  --presets s600-k3-norerank s1200-k5-rerank s2400-k2-rerank \
  --ids t-heads t-dmodel m-codec m-inner-monologue u-france u-weather
```

`questions.json` holds 20 hand-written questions (16 answerable across all three
papers, 4 unanswerable). Grading is deterministic (`relay/grading.py`): a fact
match plus a citation to the correct paper, or a correct refusal.

## Layout

```
relay/retrieval.py   chunking, BM25, rerank, refusal gate
relay/ingest.py      PDF -> chunks (pinned-version checks)
relay/llm.py         client seam + error types
relay/groq_client.py real Groq client (streaming, TTFT, 429 mapping)
relay/standin.py     controllable OpenAI-compatible stand-in
relay/pipeline.py    retrieve -> refuse? -> draft -> verify -> redraft loop
relay/load.py        retry wrapper, load shapes, results.csv writer
relay/governor.py    client-side minute/day budget policy
relay/grading.py     deterministic answer grading
relay/report.py      Table 2a/2b/3 from results.csv
relay/trace.py       waiting / working / retrying latency split
probe.py             real-service limits + latency probe
```

## Measured limits (free tier, qwen/qwen3.8-27b)

30 RPM (confirmed by real 429 on a ramp), 8K TPM, 1K RPD, 200K TPD (docs +
header countdown), `retry-after: 2s` on 429. Raw probe: `probe_results.csv`.
Measured 429 body:

```
Rate limit reached ... on requests per minute (RPM): Limit 30, Used 30,
Requested 1. ... retry-after: 2
```

## Safety rules

- `.env`, raw chunks, answer logs, and demo outputs are gitignored.
- Real runs require `--yes-spend-real`; everything else runs against the stand-in.
- The daily quota is per model and shared across scripts; `--governor` enforces
  the limits client-side (delay per minute, degrade/refuse per day).
