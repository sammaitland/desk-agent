"""Cost prediction from past traces.

Predicts what a query will cost before it runs, by finding the most similar
questions the system has already answered and reading their cost. No model is
trained; this is kNN over the trace store, embedded with the same retriever the
documentation search uses.

Why kNN rather than a learned router: a 2025 result showed simple kNN matching
complex learned routers on routing benchmarks, and kNN needs nothing this
system does not already have. Every trace records the question, the tool
sequence, and the token cost. That is the training set, and it grows with use.

Why this works at all: token cost is dominated by tool results resent on every
turn, so the tool path largely determines the cost, and the tool path is
largely determined by the question type. "What went wrong last week" calls
`detect_anomalies` and costs about the same every time. The stochastic part is
small.

**The honest limit:** this predicts the cost of the investigation the agent
will *probably* take, from questions it has *already* seen. A genuinely novel
question has no near neighbours, and the predictor says so — low confidence
routes to the default configuration, not to a guess.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.agent.trace import TRACE_DIR


@dataclass
class PastRun:
    """One historical trace, reduced to what prediction needs."""

    question: str
    tool_sequence: list[str]
    input_tokens: int
    output_tokens: int
    turns: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Prediction:
    """What a query will probably cost, and how sure we are."""

    expected_tokens: int
    expected_turns: float
    expected_tools: list[str]
    confidence: float           # similarity of the nearest neighbour, 0..1
    neighbours: int             # how many past runs informed this
    nearest_question: str | None = None
    basis: str = "knn"          # or "cold_start"

    def as_dict(self) -> dict:
        return {
            "expected_tokens": self.expected_tokens,
            "expected_turns": round(self.expected_turns, 1),
            "expected_tools": self.expected_tools,
            "confidence": round(self.confidence, 3),
            "neighbours": self.neighbours,
            "nearest_question": self.nearest_question,
            "basis": self.basis,
        }


def load_runs(directory: Path | None = None) -> list[PastRun]:
    """Read every saved trace into a PastRun. Errored runs are skipped: a
    trace that failed on transport tells you nothing about cost."""
    directory = directory or TRACE_DIR
    runs: list[PastRun] = []
    if not directory.exists():
        return runs
    for path in sorted(directory.glob("trace_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("error") or not data.get("question"):
            continue
        runs.append(PastRun(
            question=data["question"],
            tool_sequence=list(data.get("tool_sequence", [])),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            turns=int(data.get("turns", 0)),
        ))
    return runs


# Below this similarity, the nearest past question is not really "like" the
# new one, and the prediction should not be trusted to route. Tuned on the
# lexical backend; embeddings score higher and would want a higher floor.
MIN_CONFIDENCE = 0.25

# Default when there is nothing to predict from: a mid-range estimate that
# routes to the standard tier. Deliberately not optimistic.
COLD_START = Prediction(expected_tokens=12_000, expected_turns=3.0,
                        expected_tools=[], confidence=0.0, neighbours=0,
                        basis="cold_start")


class CostPredictor:
    """kNN over past traces, embedded with the documentation retriever's vectoriser."""

    def __init__(self, runs: list[PastRun], k: int = 5):
        self.runs = runs
        self.k = k
        self._vectoriser = None
        self._matrix = None
        if runs:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                               stop_words="english")
            self._matrix = self._vectoriser.fit_transform(r.question for r in runs)

    @classmethod
    def from_disk(cls, directory: Path | None = None, k: int = 5) -> "CostPredictor":
        return cls(load_runs(directory), k=k)

    def predict(self, question: str) -> Prediction:
        if not self.runs or self._matrix is None:
            return COLD_START

        vector = self._vectoriser.transform([question])
        scores = (self._matrix @ vector.T).toarray().ravel()
        order = np.argsort(-scores)[: self.k]
        top = [(self.runs[i], float(scores[i])) for i in order if scores[i] > 0]
        if not top:
            return COLD_START

        best_run, best_score = top[0]
        if best_score < MIN_CONFIDENCE:
            # Neighbours exist but none is close. Report them, but the router
            # treats this as cold start — a weak match is worse than no match
            # because it looks like evidence.
            return Prediction(
                expected_tokens=COLD_START.expected_tokens,
                expected_turns=COLD_START.expected_turns,
                expected_tools=[], confidence=best_score, neighbours=len(top),
                nearest_question=best_run.question, basis="cold_start",
            )

        # Similarity-weighted mean, so a close match counts more than a loose one.
        weights = np.array([s for _, s in top])
        weights = weights / weights.sum()
        tokens = sum(w * r.total_tokens for (r, _), w in zip(top, weights))
        turns = sum(w * r.turns for (r, _), w in zip(top, weights))

        # Tool sequence: the most common among neighbours, not a blend — a
        # blend of two sequences is not a sequence.
        sequences = Counter(tuple(r.tool_sequence) for r, _ in top)
        tools = list(sequences.most_common(1)[0][0])

        return Prediction(
            expected_tokens=int(round(tokens)),
            expected_turns=float(turns),
            expected_tools=tools,
            confidence=best_score,
            neighbours=len(top),
            nearest_question=best_run.question,
        )
