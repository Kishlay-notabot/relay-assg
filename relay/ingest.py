#!/usr/bin/env python3
"""Ingest the three pinned arXiv PDFs into a chunk store (chunks.jsonl).

Usage:
    source ../lang/bin/activate
    python -m relay.ingest                    # default: 1200-char pieces, 200 overlap
    python -m relay.ingest --size 600 --overlap 100 --out chunks_600.jsonl

No model calls or network: extraction is local, so this costs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from relay.retrieval import Chunk, Page, chunk_pages

PAPERS_DIR = Path(".")
PINNED_PAPERS = {
    "1706.03762v7": "1706.03762v7.pdf",
    "2410.00037v1": "2410.00037v1.pdf",
    "2507.07955v2": "2507.07955v2.pdf",
}


def paper_id_from_path(path: Path) -> str:
    """Accept only the exact pinned filenames (arXiv id + version)."""
    stem = path.stem
    if stem not in PINNED_PAPERS:
        expected = ", ".join(PINNED_PAPERS)
        raise ValueError(f"{path.name} is not a pinned paper; expected one of: {expected}")
    return stem


def extract_pages(pdf_path: Path, paper_id: str) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Page(paper_id=paper_id, page_number=i, text=text))
    return pages


def build_chunks(
    papers_dir: Path,
    size: int,
    overlap: int,
    extractor=extract_pages,
    papers: dict[str, str] = PINNED_PAPERS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paper_id in sorted(papers):
        pdf_path = papers_dir / papers[paper_id]
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing pinned paper: {pdf_path}")
        pages = extractor(pdf_path, paper_id)
        chunks.extend(chunk_pages(pages, size=size, overlap=overlap))
    return chunks


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "paper_id": c.paper_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(d["paper_id"], d["page_number"], d["chunk_index"], d["text"]))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    args = parser.parse_args()

    chunks = build_chunks(args.papers_dir, args.size, args.overlap)
    save_chunks(args.out, chunks)

    per_paper: dict[str, int] = {}
    for c in chunks:
        per_paper[c.paper_id] = per_paper.get(c.paper_id, 0) + 1
    print(f"wrote {len(chunks)} chunks to {args.out} (size={args.size}, overlap={args.overlap})")
    for pid, n in sorted(per_paper.items()):
        print(f"  {pid}: {n} chunks")
    if chunks:
        approx_tokens = sum(len(c.text) for c in chunks) / 4
        print(f"  corpus approx tokens: {approx_tokens:,.0f}")


if __name__ == "__main__":
    main()
