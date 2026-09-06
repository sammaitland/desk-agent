"""Cost prediction from past traces.

Predicts what a query will cost before it runs, by finding the most similar
questions the system has already answered and reading their cost. No model is
trained; this is kNN over a trace corpus, embedded with the same vectoriser the
documentation search uses.

## What the corpus is, and what it is not

The predictor consumes **only the baseline corpus**: traces generated under a
fixed, recorded inference configuration (Sonnet, 8 turns, caching on,
compression off, routing off) from **development eval cases only**. Three
rules, each enforced in `load_runs` and tested:

1. **Schema version.** Traces from before per-turn usage, model identity and
   cache accounting existed are schema 1. They record a different system and
   cannot train a router for this one. Rejected.
2. **Held-out exclusion.** A trace whose question matches a held-out eval case
   is rejected regardless of where it was saved. If the router had seen a
   held-out question, the held-out comparison would be contaminated — the
   router would route from memory, not from prediction.
3. **Corpus tag.** Only traces tagged `corpus="baseline"` are loaded. Ad-hoc
   CLI runs, routed runs and compressed runs are not baseline measurements
   and must not stand in for them.

## Leave-one-question-out

When the predictor is evaluated on a question that is *in* its corpus, it
must exclude every run of that question — not just the one file — or it finds
itself as its own nearest neighbour and reports perfect confidence. The
comparison harness passes `exclude_question`; production use does not.

Why kNN: a 2025 result showed simple kNN matching complex learned routers, and
kNN needs nothing this system does not already have. Token cost is dominated
by tool results resent on every turn, so the tool path largely determines
cost, and the tool path is largely determined by the question type.

**The honest limit:** this predicts the cost of the investigation the agent
will *probably* take, from questions it has *already* seen. A novel question
has no near neighbours, and the predictor says so.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.agent.trace import BASELINE_DIR, TRACE_SCHEMA_VERSION

# The one configuration under which baseline traces are valid. A trace that
# records anything else — a different model, a different turn budget,
# compression on, routing on — measured a different system and is rejected
# even if tagged "baseline". The tag says what was intended; the config says
# what happened, and the config wins.
BASELINE_CONFIG = {
    "model": "claude-sonnet-4-6",
    "max_turns": 8,
    "caching": True,
    "compress": False,
    "routed": False,
}


def _norm(question: str) -> str:
    """Normalise a question for identity comparison.

    Case-folded; every run of non-alphanumeric characters — punctuation,
    tabs, repeated spaces, hyphens — collapsed to a single space; leading and
    trailing space removed. "What  went-wrong?!" and "what went wrong" are the
    same question. Identity matching is what keeps held-out questions out of
    the corpus, so it must not be defeatable by formatting.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", question.lower()).split())


def _config_matches(data: dict) -> bool:
    """Does this trace's recorded config equal the baseline config exactly?
    Also requires trace.model to agree, because a trace could carry a config
    dict it did not actually run under."""
    config = data.get("config") or {}
    if any(config.get(k) != v for k, v in BASELINE_CONFIG.items()):
        return False
    return data.get("model") == BASELINE_CONFIG["model"]


@dataclass
class PastRun:
    """One baseline trace, reduced to what prediction needs."""

    question: str
    tool_sequence: list[str]
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    turns: int
    model: str

    @property
    def processed_tokens(self) -> int:
        """Every token the model handled, whatever it was billed at."""
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write

    @property
    def key(self) -> str:
        return _norm(self.question)


@dataclass
class Prediction:
    expected_tokens: int
    expected_turns: float
    expected_tools: list[str]
    confidence: float
    neighbours: int
    nearest_question: str | None = None
    basis: str = "knn"

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


@dataclass
class LoadReport:
    """What load_runs accepted and why it rejected the rest."""

    accepted: int = 0
    rejected_schema: int = 0
    rejected_held_out: int = 0
    rejected_corpus: int = 0
    rejected_config: int = 0
    rejected_error: int = 0

    @property
    def rejected(self) -> int:
        return (self.rejected_schema + self.rejected_held_out + self.rejected_corpus
                + self.rejected_config + self.rejected_error)


def _held_out_keys() -> set[str]:
    from src.evals.cases import HELD_OUT

    return {_norm(c.question) for c in HELD_OUT}


def load_runs(directory: Path | None = None,
              require_corpus: str | None = "baseline") -> tuple[list[PastRun], LoadReport]:
    """Read baseline traces. Returns the runs and a report of what was rejected.

    The report exists so a thin corpus is visible rather than silent: a router
    trained on three traces because the other twenty were rejected should say
    so.
    """
    directory = directory or BASELINE_DIR
    runs: list[PastRun] = []
    report = LoadReport()
    held_out = _held_out_keys()

    if not directory.exists():
        return runs, report

    for path in sorted(directory.glob("trace_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            report.rejected_error += 1
            continue
        if data.get("error") or not data.get("question"):
            report.rejected_error += 1
            continue
        if int(data.get("schema_version", 1)) < TRACE_SCHEMA_VERSION:
            report.rejected_schema += 1
            continue
        if require_corpus and data.get("corpus") != require_corpus:
            report.rejected_corpus += 1
            continue
        if require_corpus == "baseline" and not _config_matches(data):
            report.rejected_config += 1
            continue
        if _norm(data["question"]) in held_out:
            report.rejected_held_out += 1
            continue

        runs.append(PastRun(
            question=data["question"],
            tool_sequence=list(data.get("tool_sequence", [])),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cache_read=int(data.get("cache_read_tokens", 0)),
            cache_write=int(data.get("cache_write_tokens", 0)),
            turns=int(data.get("turns", 0)),
            model=data.get("model") or "unknown",
        ))
        report.accepted += 1
    return runs, report


MIN_CONFIDENCE = 0.25

COLD_START = Prediction(expected_tokens=12_000, expected_turns=3.0,
                        expected_tools=[], confidence=0.0, neighbours=0,
                        basis="cold_start")


class CostPredictor:
    """kNN over the baseline corpus."""

    def __init__(self, runs: list[PastRun], k: int = 5, report: LoadReport | None = None):
        self.runs = runs
        self.k = k
        self.report = report or LoadReport(accepted=len(runs))
        self._vectoriser = None
        self._matrix = None
        if runs:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                               stop_words="english")
            self._matrix = self._vectoriser.fit_transform(r.question for r in runs)

    @classmethod
    def from_disk(cls, directory: Path | None = None, k: int = 5) -> "CostPredictor":
        runs, report = load_runs(directory)
        return cls(runs, k=k, report=report)

    def predict(self, question: str, exclude_question: str | None = None) -> Prediction:
        """Predict cost. `exclude_question` removes every run of that question
        from consideration — leave-one-question-out, for evaluation."""
        if not self.runs or self._matrix is None:
            return COLD_START

        vector = self._vectoriser.transform([question])
        scores = (self._matrix @ vector.T).toarray().ravel()

        if exclude_question is not None:
            key = _norm(exclude_question)
            for i, run in enumerate(self.runs):
                if run.key == key:
                    scores[i] = -1.0   # below any real score; never selected

        order = np.argsort(-scores)[: self.k]
        top = [(self.runs[i], float(scores[i])) for i in order if scores[i] > 0]
        if not top:
            return COLD_START

        best_run, best_score = top[0]
        if best_score < MIN_CONFIDENCE:
            return Prediction(
                expected_tokens=COLD_START.expected_tokens,
                expected_turns=COLD_START.expected_turns,
                expected_tools=[], confidence=best_score, neighbours=len(top),
                nearest_question=best_run.question, basis="cold_start",
            )

        weights = np.array([s for _, s in top])
        weights = weights / weights.sum()
        tokens = sum(w * r.processed_tokens for (r, _), w in zip(top, weights))
        turns = sum(w * r.turns for (r, _), w in zip(top, weights))
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
