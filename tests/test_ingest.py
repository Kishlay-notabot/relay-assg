"""Tests for PDF ingestion: pinned version checks, chunk building, store IO."""

import tempfile
import unittest
from pathlib import Path

from relay.ingest import (
    PINNED_PAPERS,
    extract_pages,
    paper_id_from_path,
    save_chunks,
    load_chunks,
    build_chunks,
)
from relay.retrieval import Chunk, Page


class PaperIdTest(unittest.TestCase):
    def test_accepts_only_pinned_versions(self):
        self.assertEqual(paper_id_from_path(Path("1706.03762v7.pdf")), "1706.03762v7")
        self.assertEqual(paper_id_from_path(Path("2410.00037v1.pdf")), "2410.00037v1")
        self.assertEqual(paper_id_from_path(Path("2507.07955v2.pdf")), "2507.07955v2")

    def test_rejects_unpinned_version(self):
        with self.assertRaises(ValueError):
            paper_id_from_path(Path("1706.03762v6.pdf"))

    def test_pinned_papers_are_exactly_three(self):
        self.assertEqual(
            sorted(PINNED_PAPERS),
            ["1706.03762v7", "2410.00037v1", "2507.07955v2"],
        )


class BuildChunksTest(unittest.TestCase):
    def test_build_chunks_uses_extractor_and_adds_citations(self):
        def fake_extract(path, paper_id):
            return [Page(paper_id, 1, "alpha beta gamma " * 40)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in PINNED_PAPERS.values():
                (root / name).touch()
            chunks = build_chunks(
                root, size=200, overlap=20, extractor=fake_extract
            )

        self.assertTrue(chunks)
        self.assertEqual({c.paper_id for c in chunks}, set(PINNED_PAPERS))
        self.assertTrue(all(c.cite().startswith(f"{c.paper_id} p.1") for c in chunks))


class StoreTest(unittest.TestCase):
    def test_save_and_load_roundtrip_preserves_citation_metadata(self):
        chunks = [
            Chunk("1706.03762v7", 3, 1, "attention is all you need"),
            Chunk("2410.00037v1", 7, 0, "speech text dialogue"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            save_chunks(path, chunks)
            loaded = load_chunks(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].paper_id, "1706.03762v7")
        self.assertEqual(loaded[0].page_number, 3)
        self.assertEqual(loaded[0].chunk_index, 1)
        self.assertEqual(loaded[1].text, "speech text dialogue")


class ExtractPagesTest(unittest.TestCase):
    def test_extract_pages_from_real_pdf(self):
        pdf = Path("1706.03762v7.pdf")
        if not pdf.exists():
            self.skipTest("paper PDF not present")

        pages = extract_pages(pdf, "1706.03762v7")

        self.assertGreater(len(pages), 5)
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("attention", " ".join(p.text for p in pages).lower())


if __name__ == "__main__":
    unittest.main()
