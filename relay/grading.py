"""Deterministic answer grading against the hand-written question set.

Grades are honest proxies, not a judge model: an answerable question is
`correct` only when every required fact appears AND the cited paper is right.
Cheap, reproducible, and impossible for a model to sweet-talk.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuestionSpec:
    id: str
    question: str
    answerable: bool
    paper: str | None = None
    must_include: list[str] | None = None
    any_of: list[str] | None = None


def load_questions(path: str | Path = "questions.json") -> list[QuestionSpec]:
    raw = json.loads(Path(path).read_text())
    return [QuestionSpec(**q) for q in raw]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 4,000 -> 4000
    return re.sub(r"\s+", " ", text)


def grade(spec: QuestionSpec, status: str, text: str, citations: list[str]) -> str:
    """Return correct | wrong | refused_correctly | refused_wrongly | ''.

    An empty grade means the request failed (rate limit, timeout, error) and
    was never answered, so it is excluded from quality percentages.
    """
    if status not in ("ok", "refused"):
        return ""
    refused = status == "refused"
    if not spec.answerable:
        return "refused_correctly" if refused else "wrong"
    if refused:
        return "refused_wrongly"

    lowered = _normalize(text)
    facts_ok = all(_normalize(term) in lowered for term in (spec.must_include or []))
    if spec.any_of:
        facts_ok = facts_ok and any(_normalize(term) in lowered for term in spec.any_of)
    citation_ok = bool(spec.paper) and any(c.startswith(f"{spec.paper} p.") for c in citations)
    return "correct" if facts_ok and citation_ok else "wrong"


def quality_summary(grades: list[str]) -> dict:
    scored = [g for g in grades if g]
    n = len(scored) or 1
    return {
        "questions": len(grades),
        "graded": len(scored),
        "correct_pct": round(100 * scored.count("correct") / n, 1),
        "wrong_pct": round(100 * scored.count("wrong") / n, 1),
        "refused_correctly_pct": round(100 * scored.count("refused_correctly") / n, 1),
        "refused_wrongly_pct": round(100 * scored.count("refused_wrongly") / n, 1),
    }
