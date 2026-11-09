"""Tests for grading and the question set."""

import unittest

from relay.grading import QuestionSpec, grade, load_questions, quality_summary


def spec(**kw):
    base = dict(id="x", question="q", answerable=True, paper="1706.03762v7",
                must_include=["8"], any_of=None)
    base.update(kw)
    return QuestionSpec(**base)


class QuestionSetTest(unittest.TestCase):
    def test_at_least_15_questions_covering_all_three_papers(self):
        qs = load_questions()

        self.assertGreaterEqual(len(qs), 15)
        papers = {q.paper for q in qs if q.paper}
        self.assertEqual(papers, {"1706.03762v7", "2410.00037v1", "2507.07955v2"})

    def test_has_unanswerable_questions_including_france(self):
        qs = load_questions()
        unanswerable = [q for q in qs if not q.answerable]

        self.assertGreaterEqual(len(unanswerable), 3)
        self.assertIn("u-france", {q.id for q in unanswerable})


class GradeTest(unittest.TestCase):
    def test_correct_answer_needs_fact_and_right_paper(self):
        self.assertEqual(
            grade(spec(), "ok", "The model uses 8 heads.", ["1706.03762v7 p.9"]),
            "correct",
        )

    def test_missing_fact_is_wrong(self):
        self.assertEqual(
            grade(spec(), "ok", "The model uses 12 heads.", ["1706.03762v7 p.9"]),
            "wrong",
        )

    def test_citation_to_wrong_paper_is_wrong(self):
        self.assertEqual(
            grade(spec(), "ok", "The model uses 8 heads.", ["2410.00037v1 p.9"]),
            "wrong",
        )

    def test_no_citation_is_wrong(self):
        self.assertEqual(grade(spec(), "ok", "It uses 8 heads.", []), "wrong")

    def test_any_of_needs_one_alternative(self):
        s = spec(must_include=[], any_of=["sine", "cosine"])
        self.assertEqual(grade(s, "ok", "Sine and cosine.", ["1706.03762v7 p.5"]), "correct")
        self.assertEqual(grade(s, "ok", "Fourier features.", ["1706.03762v7 p.5"]), "wrong")

    def test_unanswerable_refusal_is_correct(self):
        s = spec(answerable=False, paper=None, must_include=[])
        self.assertEqual(grade(s, "refused", "The papers do not say.", []), "refused_correctly")

    def test_unanswerable_answer_is_wrong(self):
        s = spec(answerable=False, paper=None, must_include=[])
        self.assertEqual(grade(s, "ok", "Paris.", ["2410.00037v1 p.63"]), "wrong")

    def test_answerable_refusal_is_refused_wrongly(self):
        self.assertEqual(grade(spec(), "refused", "The papers do not say.", []), "refused_wrongly")


class NormalizationTest(unittest.TestCase):
    def test_numbers_with_thousands_separator_match(self):
        self.assertEqual(
            grade(spec(must_include=["4000"]), "ok", "It uses 4,000 warmup steps.",
                  ["1706.03762v7 p.7"]),
            "correct",
        )

    def test_unicode_spaces_and_nfkc_are_normalised(self):
        self.assertEqual(
            grade(spec(must_include=["0.98"]), "ok", "beta2\u00a0=\u00a00.98",
                  ["1706.03762v7 p.7"]),
            "correct",
        )


class FailureGradingTest(unittest.TestCase):
    def test_failed_request_is_not_graded(self):
        self.assertEqual(grade(spec(), "rate_limited", "", []), "")
        self.assertEqual(grade(spec(), "timeout", "", []), "")

    def test_summary_excludes_unscored_rows(self):
        out = quality_summary(["correct", "", "wrong"])

        self.assertEqual(out["graded"], 2)
        self.assertEqual(out["correct_pct"], 50.0)
        self.assertEqual(out["wrong_pct"], 50.0)


class SummaryTest(unittest.TestCase):
    def test_percentages(self):
        out = quality_summary(["correct", "correct", "wrong", "refused_correctly"])

        self.assertEqual(out["correct_pct"], 50.0)
        self.assertEqual(out["wrong_pct"], 25.0)
        self.assertEqual(out["refused_correctly_pct"], 25.0)
        self.assertEqual(out["refused_wrongly_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
