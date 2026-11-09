"""Paper ingestion helpers: pages -> chunks, lexical retrieval, refusal gate.

Everything here is deterministic and free: no model calls, no network.
The CLI that turns the three pinned arXiv PDFs into a chunk store lives in
``relay/ingest.py``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace

# arXiv stamps a side watermark on page 1: "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"
_ARXIV_STAMP = re.compile(r"arXiv:\S+\s+\[[^\]]+\]\s+[^\n]*")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal; matching on them makes every chunk
# look relevant and lets nonsense questions through the refusal gate.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did", "for",
    "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "s", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with",
}


@dataclass
class Page:
    paper_id: str
    page_number: int
    text: str


@dataclass
class Chunk:
    paper_id: str
    page_number: int
    chunk_index: int
    text: str
    score: float = 0.0

    def cite(self) -> str:
        return f"{self.paper_id} p.{self.page_number} c.{self.chunk_index}"


def normalize_text(text: str) -> str:
    text = _ARXIV_STAMP.sub("", text)
    text = _PAGE_NUMBER_LINE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Split pages into fixed-size character chunks with overlap, keeping citations."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    chunks: list[Chunk] = []
    step = size - overlap
    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + size]
            if not piece.strip():
                continue
            chunks.append(Chunk(page.paper_id, page.page_number, i, piece))
            if start + size >= len(text):
                break
    return chunks


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _bm25_scores(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_terms = tokenize(query)
    if not q_terms or not chunks:
        return [0.0] * len(chunks)

    doc_terms = [tokenize(c.text) for c in chunks]
    avgdl = sum(len(d) for d in doc_terms) / len(doc_terms) or 1.0
    n = len(chunks)
    df = Counter()
    for terms in doc_terms:
        for t in set(terms):
            df[t] += 1

    scores = []
    for terms in doc_terms:
        tf = Counter(terms)
        dl = len(terms) or 1.0
        score = 0.0
        for t in q_terms:
            if tf[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _coverage(query: str, chunk: Chunk) -> float:
    """Second-stage heuristic: fraction of distinct query terms present in chunk."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    present = q & set(tokenize(chunk.text))
    return len(present) / len(q)


def retrieve(query: str, chunks: list[Chunk], k: int, rerank: bool = False) -> list[Chunk]:
    """Return top-k chunks by BM25, optionally re-ranked by query-term coverage."""
    if not chunks or k <= 0:
        return []

    scores = _bm25_scores(query, chunks)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    top = ranked[: max(k, len(chunks) if rerank else k)]

    if rerank:
        top = sorted(
            top,
            key=lambda pair: (pair[1] * (0.5 + _coverage(query, pair[0]))),
            reverse=True,
        )

    return [replace(chunk, score=round(score, 4)) for chunk, score in top[:k]]


def is_answerable(hits: list[Chunk], min_score: float) -> bool:
    """Refusal gate: no hits, or nothing above the score floor, means refuse."""
    return bool(hits) and max(h.score for h in hits) >= min_score
