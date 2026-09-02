"""Eval runner.

Runs each case against the live API and applies its checks to the resulting
trace. Results are written to disk so runs can be compared: an eval suite that
only prints to a terminal tells you the state today but not whether it moved.

Live by design. These test model behaviour, and model behaviour is what the
scripted client deliberately removes — `tests/test_agent.py` covers the loop
mechanics for free on every commit; this costs API calls and runs on demand.
Conflating the two would give you either expensive unit tests or evals that
never see the model.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import Connection

from src.agent.loop import MODEL, run_agent
from src.agent.trace import Trace
from src.evals.cases import UNIVERSAL, EvalCase
from src.evals.checks import CheckResult

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"


@dataclass
class CaseResult:
    name: str
    question: str
    passed: bool
    checks: list[dict] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    answer: str = ""
    words: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"]]


@dataclass
class SuiteResult:
    started_at: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (sum(c.input_tokens for c in self.cases),
                sum(c.output_tokens for c in self.cases))

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or RESULTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "")[:15]
        path = directory / f"evals_{stamp}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def run_case(case: EvalCase, conn: Connection, client=None, model: str = MODEL) -> CaseResult:
    """Run one case and apply its checks plus the universal ones."""
    trace = run_agent(case.question, conn, client=client, model=model)
    if trace.error:
        return CaseResult(
            name=case.name, question=case.question, passed=False,
            tool_sequence=trace.tool_sequence, answer=trace.answer,
            turns=trace.turns, duration_ms=trace.duration_ms, error=trace.error,
        )

    results: list[CheckResult] = [check(trace) for check in case.checks + UNIVERSAL]
    return CaseResult(
        name=case.name,
        question=case.question,
        passed=all(r.passed for r in results),
        checks=[asdict(r) for r in results],
        tool_sequence=trace.tool_sequence,
        answer=trace.answer,
        words=len(trace.answer.split()),
        turns=trace.turns,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        duration_ms=trace.duration_ms,
    )


def run_suite(
    cases: list[EvalCase],
    conn: Connection,
    client=None,
    model: str = MODEL,
    verbose: bool = True,
) -> SuiteResult:
    suite = SuiteResult(
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=model,
    )
    for case in cases:
        if verbose:
            print(f"  {case.name} ... ", end="", flush=True)
        started = time.perf_counter()
        result = run_case(case, conn, client=client, model=model)
        suite.cases.append(result)
        if verbose:
            elapsed = time.perf_counter() - started
            mark = "PASS" if result.passed else "FAIL"
            print(f"{mark}  ({elapsed:.1f}s, {len(result.tool_sequence)} tools)")
            for failure in result.failures:
                print(f"      - {failure['name']}: {failure['detail']}")
            if result.error:
                print(f"      ! {result.error}")
    return suite


def report(suite: SuiteResult) -> str:
    """Human-readable summary, including a per-check pass rate.

    The per-check view is the more useful one over time: a case failing tells
    you something broke, but a check failing across several cases tells you
    what kind of thing broke.
    """
    lines = [
        "",
        "=" * 66,
        f"EVAL SUITE  {suite.started_at}  model={suite.model}",
        "=" * 66,
        f"{suite.passed}/{suite.total} cases passed",
    ]
    tokens_in, tokens_out = suite.total_tokens
    lines.append(f"tokens: {tokens_in:,} in / {tokens_out:,} out")

    tally: dict[str, list[int]] = {}
    for case in suite.cases:
        for check in case.checks:
            passed, total = tally.setdefault(check["name"], [0, 0])
            tally[check["name"]] = [passed + int(check["passed"]), total + 1]

    lines += ["", "per-check:"]
    for name, (passed, total) in sorted(tally.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        flag = "" if passed == total else "   <-- "
        lines.append(f"  {passed}/{total}  {name}{flag}")

    failing = [c for c in suite.cases if not c.passed]
    if failing:
        lines += ["", "failing cases:"]
        for case in failing:
            lines.append(f"  {case.name}: {[f['name'] for f in case.failures]}")

    lines.append("")
    return "\n".join(lines)
