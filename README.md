# The Relay — cited QA over three papers

Working pipeline for the Relay assignment: retrieval-augmented, citation-checked
answers over three pinned arXiv papers, a controlled stand-in model service, load
shapes, grading, and tables computed from one raw file.

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
python run_load.py --scenario steady --duration 20 --rate 60 --workers 6

# real Groq (costs daily quota; explicit confirmation required)
python run_load.py --source real --scenario burst --count 5 --yes-spend-real

python summarize.py results.csv          # Tables 2a / 2b as markdown
```

Scenarios: `steady`, `burst`, `overload`; rows append to `results.csv` with the
required 12 columns, per-request traces go to `raw/traces.jsonl`.

## Quality and retrieval comparison

```sh
python eval_quality.py --source stand-in --out quality.csv
python eval_quality.py --source real --yes-spend-real --out quality_real.csv
python run_retrieval.py --source stand-in --limit 10
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

## Measured limits (free tier, openai/gpt-oss-20b)

30 RPM (confirmed by real 429), 8K TPM, 1K RPD, 200K TPD (docs + header
countdown), `retry-after: 2s` on 429. See `writeup_notes.txt` for confidence and
method.

## Safety rules

- `.env`, raw chunks, `results.csv`, and demo outputs are gitignored.
- Real runs require `--yes-spend-real`; everything else runs against the stand-in.
- The daily quota is shared across all scripts; `--governor` enforces the limits
  client-side (delay per minute, degrade/refuse per day).
