"""Tests for paper ingestion, chunking, lexical retrieval and refusal.

Run: source ../lang/bin/activate && python -m unittest discover -s tests -v
"""

import unittest

from relay.retrieval import (
    Chunk,
    Page,
    chunk_pages,
    is_answerable,
    normalize_text,
    retrieve,
)


class ChunkingTest(unittest.TestCase):
    def test_chunks_split_long_text_with_overlap_and_keep_citation_metadata(self):
        page = Page(paper_id="1706.03762v7", page_number=2, text="x" * 1000)

        chunks = chunk_pages([page], size=400, overlap=50)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].paper_id, "1706.03762v7")
        self.assertEqual(chunks[0].page_number, 2)
        self.assertTrue(chunks[0].cite().startswith("1706.03762v7 p.2"))
        # overlap means the tail of one chunk reappears at the head of the next
        self.assertEqual(chunks[0].text[-50:], chunks[1].text[:50])
        self.assertEqual(chunks[1].text[-50:], chunks[2].text[:50])

    def test_short_text_produces_one_chunk(self):
        page = Page(paper_id="p", page_number=1, text="short text")

        chunks = chunk_pages([page], size=400, overlap=50)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short text")

    def test_empty_pages_are_skipped(self):
        pages = [
            Page(paper_id="p", page_number=1, text="   \n  "),
            Page(paper_id="p", page_number=2, text="real content here"),
        ]

        chunks = chunk_pages(pages, size=400, overlap=50)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 2)


class NormalizeTest(unittest.TestCase):
    def test_normalize_lowercases_and_collapses_whitespace(self):
        self.assertEqual(normalize_text("  Hello\t\nWorld  "), "hello world")

    def test_normalize_removes_arxiv_side_watermark(self):
        text = "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\nreal content"
        self.assertEqual(normalize_text(text), "real content")


class TightCorpus:
    """Tiny in-memory corpus so tests do not depend on the PDFs."""

    def __init__(self):
        self.chunks = [
            Chunk("a", 1, 0, "the transformer uses self attention and multi head attention"),
            Chunk("a", 2, 0, "positional encodings inject order information"),
            Chunk("b", 1, 0, "moshi is a speech text model for real time dialogue"),
            Chunk("c", 1, 0, "dynamic chunking learns hierarchical sequence boundaries"),
        ]


class RetrievalTest(unittest.TestCase):
    def test_retrieve_returns_most_relevant_chunk_first(self):
        corpus = TightCorpus()

        hits = retrieve("what is multi head attention", corpus.chunks, k=2)

        self.assertEqual(len(hits), 2)
        self.assertIn("multi head attention", hits[0].text)

    def test_retrieve_respects_k(self):
        hits = retrieve("attention", TightCorpus().chunks, k=1)
        self.assertEqual(len(hits), 1)

    def test_retrieve_with_rerank_true_puts_chunk_with_more_query_terms_first(self):
        corpus = TightCorpus()

        hits = retrieve("speech text dialogue model", corpus.chunks, k=2, rerank=True)

        self.assertEqual(hits[0].paper_id, "b")

    def test_retrieve_returns_empty_for_empty_corpus(self):
        self.assertEqual(retrieve("attention", [], k=3), [])


class AnswerableTest(unittest.TestCase):
    def test_low_scores_mean_not_answerable(self):
        corpus = TightCorpus()
        hits = retrieve("quantum chromodynamics lattice gauge", corpus.chunks, k=3)

        self.assertFalse(is_answerable(hits, min_score=0.2))

    def test_stopword_only_query_is_not_answerable(self):
        corpus = TightCorpus()
        hits = retrieve("what is the of and", corpus.chunks, k=3)

        self.assertFalse(is_answerable(hits, min_score=0.2))

    def test_relevant_hits_are_answerable(self):
        corpus = TightCorpus()
        hits = retrieve("multi head attention", corpus.chunks, k=3)

        self.assertTrue(is_answerable(hits, min_score=0.2))


if __name__ == "__main__":
    unittest.main()
