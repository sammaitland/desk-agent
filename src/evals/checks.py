"""Eval checks.

Each check takes a Trace and returns a CheckResult. They are deliberately
deterministic — no model judges another model here. A check that is itself
unreliable cannot tell you whether the system regressed.

The centrepiece is `numeric_fidelity`. The tool layer guarantees the numbers are
correct; nothing so far guarantees the model *reports* them correctly. Rounding
23.3 to "roughly 20bps", or stating a figure no tool returned, is the failure
mode that survives every other test — the answer stays fluent and plausible
while quietly ceasing to be true.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from src import config as cfg
from src.agent.trace import Trace

# Figures the model may legitimately state without a tool returning them:
# thresholds given in the system prompt, and trivially small integers used in
# prose ("two halts", "the 45-second timeout").
PROMPT_CONSTANTS = {
    cfg.MAX_ACCOUNT_LEVERAGE, cfg.EMERGENCY_LEVERAGE_THRESHOLD,
    cfg.PREFILTER_MAX_SPREAD_BPS, cfg.MAX_LIMIT_ORDER_SPREAD_BPS,
    float(cfg.LIMIT_ORDER_TIMEOUT), cfg.STOP_LOSS_ALPHA_THRESHOLD,
    cfg.MAX_INDEX_GROSS_EXPOSURE_PCT, float(cfg.MAX_NEW_POSITIONS_PER_TICKER),
    cfg.MIN_POSITION_SIZE, cfg.MAX_POSITION_SIZE,
    0.0, 0.7, 1.0, 1.1, 1.3, 1.4,  # bucket multipliers
    0.2,                            # retrieval confidence floor, stated in the prompt
}

# Dates, times, tags and identifiers produce digits that are not claims.
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTHS = ("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
           "|January|February|March|April|June|July|August|September|October|November|December")
# Written dates produce digits that assert nothing: "Aug 25" is not a figure.
_DATE_WORDS = re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}\b|\b\d{{1,2}}\s+(?:{_MONTHS})\b",
                         re.IGNORECASE)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
_TAG = re.compile(r"\b[A-Z]{3}_[A-Z.]+_[A-Z.]+_[LU]_\d{8}_\d{3}\b")
_ID = re.compile(r"\b(?:ord|pos|sig|fil|evt|chk|alc|run|evl|ib)_[0-9a-f]{6,}\b")
# Trailing guard is (?!\d) not (?!\w): a figure is usually followed by its
# unit ("4.8bps", "12%"), and rejecting those truncates 4.8 to 4.
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d)")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


Check = Callable[[Trace], CheckResult]


# --- helpers --------------------------------------------------------------

def _strip_non_claims(text: str) -> str:
    """Remove digits that are identifiers or timestamps, not asserted figures."""
    for pattern in (_TAG, _ID, _DATE, _DATE_WORDS, _TIME):
        text = pattern.sub(" ", text)
    return text


def _numbers_in_text(text: str) -> list[float]:
    values = []
    for match in _NUMBER.finditer(_strip_non_claims(text)):
        whole = match.group(1).replace(",", "")
        frac = match.group(2)
        values.append(float(f"{whole}.{frac}" if frac else whole))
    return values


def _numbers_in_payload(payload: Any, out: set[float]) -> None:
    """Recursively collect every number a tool returned, including inside strings."""
    if isinstance(payload, bool):
        return
    if isinstance(payload, (int, float)):
        out.add(float(payload))
    elif isinstance(payload, str):
        out.update(_numbers_in_text(payload))
    elif isinstance(payload, dict):
        for value in payload.values():
            _numbers_in_payload(value, out)
    elif isinstance(payload, list):
        for item in payload:
            _numbers_in_payload(item, out)


def _supported(value: float, sources: set[float], tolerance: float = 0.01) -> bool:
    """Is a stated figure traceable to something a tool returned?

    Allows for rounding: a model may quote 4.8 for 4.81, or 61 for 60.7. It may
    not invent a figure, and it may not round so hard the number changes
    meaning — hence a tolerance rather than a free pass.
    """
    if value in sources or value in PROMPT_CONSTANTS:
        return True
    # Compare magnitudes: answers render negatives with en/em dashes that the
    # extractor cannot distinguish from ordinary hyphenation, so a stated
    # "-1.197%" arrives here as 1.197.
    value = abs(value)
    if value in {abs(v) for v in PROMPT_CONSTANTS}:
        return True
    for raw in sources:
        source = abs(raw)
        if abs(source - value) <= max(source * tolerance, 0.05):
            return True
        # Percentages stated as their decimal equivalent, or vice versa.
        if source and abs(source * 100 - value) <= 0.05:
            return True
        if value and abs(value * 100 - source) <= 0.05:
            return True
    return False


def _tool_numbers(trace: Trace) -> set[float]:
    numbers: set[float] = set()
    for call in trace.tool_calls:
        _numbers_in_payload(call.summary, numbers)
        _numbers_in_payload(call.arguments, numbers)
        raw = getattr(call, "raw_result", None)
        if raw is not None:
            _numbers_in_payload(raw, numbers)
        # Provenance carries event counts, windows and scores — all legitimate
        # figures for the model to quote. Missing this produced false
        # negatives on every incident review.
        prov = getattr(call, "provenance", None)
        if prov:
            _numbers_in_payload(prov, numbers)
    return numbers


# --- checks ---------------------------------------------------------------

def called_tool(name: str) -> Check:
    def check(trace: Trace) -> CheckResult:
        used = name in trace.tool_sequence
        return CheckResult(f"calls {name}", used,
                           "" if used else f"called {trace.tool_sequence or 'nothing'}")
    return check


def called_any_of(*names: str) -> Check:
    def check(trace: Trace) -> CheckResult:
        hit = [n for n in names if n in trace.tool_sequence]
        return CheckResult(f"calls one of {'/'.join(names)}", bool(hit),
                           "" if hit else f"called {trace.tool_sequence or 'nothing'}")
    return check


def chained(minimum: int = 2) -> Check:
    """Multi-step investigation: the behaviour that separates an agent from a lookup."""
    def check(trace: Trace) -> CheckResult:
        count = len(trace.tool_calls)
        return CheckResult(f"chains >= {minimum} tools", count >= minimum,
                           f"called {count}: {trace.tool_sequence}")
    return check


def tool_budget(maximum: int) -> Check:
    def check(trace: Trace) -> CheckResult:
        count = len(trace.tool_calls)
        return CheckResult(f"uses <= {maximum} tools", count <= maximum,
                           f"called {count}: {trace.tool_sequence}")
    return check


def no_tool_errors() -> Check:
    def check(trace: Trace) -> CheckResult:
        failed = [c.name for c in trace.tool_calls if c.error]
        return CheckResult("no tool errors", not failed,
                           f"errors in {failed}" if failed else "")
    return check


def forbids(*phrases: str) -> Check:
    """Terms that indicate a domain misunderstanding, e.g. calling alpha 'profit'."""
    def check(trace: Trace) -> CheckResult:
        answer = trace.answer.lower()
        hits = [p for p in phrases if p.lower() in answer]
        return CheckResult(f"avoids {'/'.join(phrases)}", not hits,
                           f"used {hits}" if hits else "")
    return check


def mentions(*phrases: str) -> Check:
    def check(trace: Trace) -> CheckResult:
        answer = trace.answer.lower()
        missing = [p for p in phrases if p.lower() not in answer]
        return CheckResult(f"mentions {'/'.join(phrases)}", not missing,
                           f"missing {missing}" if missing else "")
    return check


def numeric_fidelity(max_unsupported: int = 0) -> Check:
    """Every figure stated must trace to something a tool returned.

    The check the whole architecture rests on: the tool layer makes the numbers
    right, and this confirms the model reported them rather than approximating.
    """
    def check(trace: Trace) -> CheckResult:
        sources = _tool_numbers(trace)
        if not sources:
            return CheckResult("numeric fidelity", True, "no tool numbers to check against")
        stated = _numbers_in_text(trace.answer)
        unsupported = [v for v in stated if not _supported(v, sources)]
        passed = len(unsupported) <= max_unsupported
        return CheckResult(
            "numeric fidelity", passed,
            "" if passed else f"{len(unsupported)} unsupported: {sorted(set(unsupported))[:8]}",
        )
    return check


def bounded_retries(tool: str, maximum: int = 2) -> Check:
    """The same tool called more than `maximum` times is a hunt, not an
    investigation. Observed live: four documentation searches at scores
    between 0.06 and 0.15, all resent every turn, to reach a conclusion the
    second search already supported."""
    def check(trace: Trace) -> CheckResult:
        count = trace.tool_sequence.count(tool)
        return CheckResult(f"<= {maximum} calls to {tool}", count <= maximum,
                           f"called {count} times")
    return check


def brevity(max_words: int) -> Check:
    def check(trace: Trace) -> CheckResult:
        words = len(trace.answer.split())
        return CheckResult(f"under {max_words} words", words <= max_words, f"{words} words")
    return check


def no_deferred_investigation() -> Check:
    """The agent should run the next step, not recommend the user run it."""
    patterns = [
        r"\bI'?d (?:recommend|suggest)\b",
        r"\bwant me to\b",
        r"\bshall I\b",
        r"\bworth (?:pulling|reviewing|checking)\b",
        r"\byou (?:may|might|could) want to (?:pull|check|review|run)\b",
    ]
    def check(trace: Trace) -> CheckResult:
        hits = [p for p in patterns if re.search(p, trace.answer, re.IGNORECASE)]
        return CheckResult("does not defer investigation", not hits,
                           f"deferred: {hits}" if hits else "")
    return check


def no_absence_overclaim() -> Check:
    """Do not read 'not in these results' as 'did not happen'."""
    patterns = [
        r"\bonly (?:one|two|\d+) evaluations?\b",
        r"\bwas not evaluated\b",
        r"\bnever (?:evaluated|considered)\b",
        r"\bdoes not appear .{0,30}\bwhich means\b",
    ]
    def check(trace: Trace) -> CheckResult:
        hits = [p for p in patterns if re.search(p, trace.answer, re.IGNORECASE)]
        return CheckResult("no absence over-claim", not hits,
                           f"over-claimed: {hits}" if hits else "")
    return check


def challenges_premise() -> Check:
    """When the data contradicts the question, say so rather than confabulate."""
    patterns = [
        r"\bdoes ?n'?o?t support\b", r"\bpremise\b", r"\bnothing .{0,30}\bindicates?\b",
        r"\bactually (?:fine|clean|normal)\b", r"\bwas ?n'?o?t (?:bad|poor)\b",
        r"\bno evidence\b", r"\bnormal performance\b",
    ]
    def check(trace: Trace) -> CheckResult:
        hits = [p for p in patterns if re.search(p, trace.answer, re.IGNORECASE)]
        return CheckResult("challenges false premise", bool(hits),
                           "" if hits else "accepted the premise without question")
    return check


def reports_scope() -> Check:
    """State how much was examined, using provenance."""
    patterns = [r"\bacross\b", r"\b\d+ (?:trades|orders|events|positions|evaluations)\b",
                r"\bbetween \d{4}-\d{2}-\d{2}\b", r"\bin the window\b"]
    def check(trace: Trace) -> CheckResult:
        hits = [p for p in patterns if re.search(p, trace.answer, re.IGNORECASE)]
        return CheckResult("reports scope", bool(hits),
                           "" if hits else "no scope statement found")
    return check
