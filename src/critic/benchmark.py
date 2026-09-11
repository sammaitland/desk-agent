"""Benchmark for the critic: claims with known expected outcomes.

## What the first live run taught

The 100% recall and 67% precision it reported were not measures of the
critic. External review of the 25 saved traces found the benchmark had
mislabelled a control (zero timeouts, derived from a mistyped event key,
when the orders table held ten), built a fixture that asked a nonsense
question (a "momentum" factor check substituted into a trade-rejection
sentence), credited UNSUPPORTED cautions as catches, counted a falsely
flagged target twice in coverage, and let span overlap conflate "VIS had
34.57%" (true) with "VIS was the weakest" (false) because both sat in one
sentence. Underneath, the generator had logged a share-price-vs-cap
comparison as a position-size failure, so the "$18 claim" the critic
existed to catch was the agent explaining inconsistent data.

The critic itself also overreached: it used entry-to-exit returns over
different holding periods to "disprove" an intraday cause, and contradicted
a true figure because it did not answer the original question. Those are
verifier problems, handled in the prompt. The rest are measurement problems,
handled here.

## Expected outcomes, not binary truth

Each target now carries what the critic *should* say:

    contradicted    the claim is false and the blotter shows it
    supported       the claim is true and the blotter shows it
    undetermined    the blotter cannot settle it; a confident verdict either
                    way is an error

The third is what makes the stop-out case a test of restraint. A critic that
"disproves" an intraday cause from holding-period returns is overreaching,
and the benchmark now counts that against it rather than rewarding it.

## Alignment is a candidate, not an identity

A claim's source span overlapping a target's span makes them candidates. A
claim overlapping exactly one target is aligned to it. A claim overlapping
several is *ambiguous* and reported for adjudication — no credit either way.
Targets in mixed sentences are cut tight: "VIS was the weakest" and "34.57%"
are separate targets with separate expected outcomes.

## Two measurements, not one

`--atomic` feeds hand-fixed claims straight to the verifier, bypassing
extraction, so verification accuracy is measured on its own. The end-to-end
run measures extraction and verification together. A miss is attributed to
whichever stage lost it.

## Controls come from the canonical table

A count of timed-out orders comes from `orders`, not from an event key the
benchmark hoped existed. Every control asserts its own prerequisite and
setup fails loudly if one is absent.

## Provenance

Every saved run records what produced it: source hashes, prompt hashes,
model, the case set's hash, and a fingerprint of the database. Stale data
and mixed revisions become testable instead of suspected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from src import config as cfg
from src.critic.critique import Claim, Critique, Verdict, critique, verify_claim
from src.tools import dispatch

EXPECTED = ("contradicted", "supported", "undetermined")


class BenchmarkSetupError(RuntimeError):
    """A required case could not be constructed from the data."""


@dataclass(frozen=True)
class Target:
    id: str
    span: tuple[int, int]
    expected: str                  # one of EXPECTED
    claim: str                     # the atomic proposition, for --atomic mode
    note: str = ""

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.span[1] and end > self.span[0]


@dataclass
class BenchCase:
    name: str
    kind: str
    question: str
    answer: str
    targets: list[Target]

    def by_expected(self, e: str) -> list[Target]:
        return [t for t in self.targets if t.expected == e]


def _span(answer: str, needle: str) -> tuple[int, int]:
    i = answer.index(needle)
    return (i, i + len(needle))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    case: BenchCase
    critique: Critique | None
    # per-target outcomes, keyed by target id
    outcome: dict[str, str] = field(default_factory=dict)
    ambiguous_claims: list[str] = field(default_factory=list)
    unmatched_flags: list[str] = field(default_factory=list)
    target_verdicts: dict[str, list[Verdict]] = field(default_factory=dict)

    # Outcome vocabulary:
    #   caught          expected contradicted, got contradicted (grounded)
    #   confirmed       expected supported, got verified (grounded)
    #   restrained      expected undetermined, got undetermined
    #   caution         expected contradicted, got undetermined  (not a catch; not wrong)
    #   unresolved      expected supported, got undetermined     (not a false alarm; not confirmed)
    #   false_alarm     expected supported, got contradicted
    #   false_verify    expected contradicted, got verified
    #   overreach       expected undetermined, got verified or contradicted
    #   not_extracted   no claim overlapped the target
    #   ambiguous       overlapping claim also overlaps another target

    def count(self, *kinds: str) -> int:
        return sum(1 for o in self.outcome.values() if o in kinds)

    @property
    def errors(self) -> int:
        return self.count("false_alarm", "false_verify", "overreach") + len(self.unmatched_flags)

    @property
    def passed(self) -> bool:
        if self.critique is not None and not self.critique.complete:
            return False
        return self.errors == 0 and self.count("not_extracted", "ambiguous", "assessment_error") == 0 \
            and self.count("caution", "unresolved") == 0


@dataclass
class BenchSummary:
    results: list[BenchResult]
    mode: str = "end_to_end"

    def total(self, *kinds: str) -> int:
        return sum(r.count(*kinds) for r in self.results)

    @property
    def targets(self) -> int:
        return sum(len(r.case.targets) for r in self.results)

    @property
    def extraction_coverage(self) -> float | None:
        """Unique targets that some claim overlapped, over all targets."""
        if self.mode == "atomic":
            return None
        covered = self.targets - self.total("not_extracted")
        return covered / self.targets if self.targets else 0.0

    @property
    def recall(self) -> float:
        """Grounded contradictions over targets expected contradicted."""
        expected = sum(len(r.case.by_expected("contradicted")) for r in self.results)
        return self.total("caught") / expected if expected else 0.0

    @property
    def precision(self) -> float:
        """Grounded contradictions over all contradictions issued."""
        issued = sum(v.verdict == "contradicted" for r in self.results for v in _all_verdicts(r))
        correct = sum(v.verdict == "contradicted" for r in self.results
                      for t in r.case.by_expected("contradicted")
                      for v in r.target_verdicts.get(t.id, []))
        return correct / issued if issued else 1.0

    @property
    def gate_passed(self) -> bool:
        return (bool(self.results) and self.recall >= 0.6
                and all(r.errors == 0 and not r.ambiguous_claims
                        and r.count("not_extracted", "ambiguous", "assessment_error") == 0
                        and (r.critique is None or r.critique.complete) for r in self.results))

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "gate_passed": self.gate_passed,
            "metrics": {
                "extraction_coverage": self.extraction_coverage,
                "recall": self.recall, "precision": self.precision, "restraint": self.restraint,
                "restraint_targets": sum(len(r.case.by_expected("undetermined")) for r in self.results),
                **{name: self.total(outcome) for name, outcome in {
                    "caught": "caught", "confirmed": "confirmed", "restrained": "restrained",
                    "cautions": "caution", "unresolved": "unresolved", "false_alarms": "false_alarm",
                    "false_verifications": "false_verify", "overreach": "overreach",
                    "assessment_errors": "assessment_error", "not_extracted": "not_extracted",
                    "ambiguous_targets": "ambiguous"}.items()},
                "proposed_contradictions": sum(v.proposed == "contradicted" for r in self.results for v in _all_verdicts(r)),
                "downgrades": sum(bool(v.downgraded) for r in self.results for v in _all_verdicts(r)),
            },
            "cases": [{"name": r.case.name, "kind": r.case.kind, "passed": r.passed,
                       "question": r.case.question, "answer": r.case.answer,
                       "outcome": r.outcome, "ambiguous": r.ambiguous_claims, "unmatched": r.unmatched_flags,
                       "targets": [asdict(t) for t in r.case.targets],
                       "target_verdicts": {key: [asdict(v) for v in vs] for key, vs in r.target_verdicts.items()},
                       "critique": r.critique.as_dict() if r.critique is not None else None}
                      for r in self.results],
        }

    @property
    def restraint(self) -> float:
        expected = sum(len(r.case.by_expected("undetermined")) for r in self.results)
        return self.total("restrained") / expected if expected else 1.0

    def render(self) -> str:
        coverage = "N/A (bypassed)" if self.extraction_coverage is None else f"{self.extraction_coverage:.0%}"
        restraint_n = sum(len(r.case.by_expected("undetermined")) for r in self.results)
        lines = ["", "=" * 74, f"CRITIC BENCHMARK ({self.mode})", "=" * 74,
                 f"cases {len(self.results)}   targets {self.targets}   "
                 f"expected: {sum(len(r.case.by_expected('contradicted')) for r in self.results)} contradicted, "
                 f"{sum(len(r.case.by_expected('supported')) for r in self.results)} supported, "
                 f"{sum(len(r.case.by_expected('undetermined')) for r in self.results)} undetermined",
                 f"extraction coverage {coverage}   recall {self.recall:.0%}   "
                 f"precision {self.precision:.0%}   restraint {self.total('restrained')}/{restraint_n}",
                 f"caught {self.total('caught')}  confirmed {self.total('confirmed')}  restrained {self.total('restrained')}  "
                 f"cautions {self.total('caution')}  unresolved {self.total('unresolved')}  "
                 f"assessment errors {self.total('assessment_error')}",
                 f"ERRORS — false alarms {self.total('false_alarm')}  false verifications {self.total('false_verify')}  "
                 f"overreach {self.total('overreach')}  unmatched flags {sum(len(r.unmatched_flags) for r in self.results)}  "
                 f"ambiguous {sum(len(r.ambiguous_claims) for r in self.results)}",
                 ""]
        for r in self.results:
            comp = "" if r.critique is None else ("" if r.critique.complete else "  [INCOMPLETE]")
            lines.append(f"{'PASS' if r.passed else 'FAIL'}  {r.case.name:<34} [{r.case.kind}]{comp}")
            for t in r.case.targets:
                o = r.outcome.get(t.id, "?")
                mark = "  " if o in ("caught", "confirmed", "restrained") else "!!" if o in ("false_alarm", "false_verify", "overreach") else "  "
                lines.append(f"     {mark} {t.id:<22} expected {t.expected:<13} -> {o}")
            for a in r.ambiguous_claims:
                lines.append(f"        ambiguous claim (adjudicate): {a}")
            for u in r.unmatched_flags:
                lines.append(f"        unmatched flag (adjudicate): {u}")
        lines.append("")
        return "\n".join(lines)


def _all_verdicts(r: BenchResult) -> list[Verdict]:
    if r.critique is not None:
        return r.critique.verdicts
    return [v for vs in r.target_verdicts.values() for v in vs]


# ---------------------------------------------------------------------------
# Case construction — anchored, asserted, from canonical tables
# ---------------------------------------------------------------------------

def _require(row, what: str):
    if row is None:
        raise BenchmarkSetupError(f"cannot build case: {what}")
    return row


def build_cases(conn: Connection) -> list[BenchCase]:
    cases: list[BenchCase] = []
    cap = cfg.MAX_LIMIT_ORDER_SPREAD_BPS

    # --- invented_causal: reversed reason, paired ----------------------------
    rej = _require(conn.execute(text("""
        SELECT eval_id, pair, rejection_reason, SUBSTR(evaluated_at, 1, 10) AS d
        FROM pair_evaluations
        WHERE evaluation_result = 'Rejected' AND rejection_reason IS NOT NULL
          AND rejection_reason != 'position_size' LIMIT 1""")).mappings().first(),
        "a rejected evaluation with a non-sizing reason")
    a_false = f"{rej['pair']} was rejected on {rej['d']} because its notional exceeded the position-size cap."
    a_true = f"{rej['pair']} was rejected on {rej['d']} because of {rej['rejection_reason']}."
    answer = f"{a_false} {a_true}"
    cases.append(BenchCase("invented_causal_reversed_reason", "invented_causal",
                           f"Why was {rej['pair']} rejected on {rej['d']}?", answer, [
        Target("false_reason", _span(answer, "because its notional exceeded the position-size cap"),
               "contradicted", f"{rej['pair']} was rejected on {rej['d']} because its notional exceeded the position-size cap",
               f"recorded reason is {rej['rejection_reason']}"),
        Target("true_reason", _span(answer, f"because of {rej['rejection_reason']}"),
               "supported", f"{rej['pair']} was rejected on {rej['d']} because of {rej['rejection_reason']}"),
    ]))

    # --- invented_causal: a PASSED position-size check described as a rejection
    passed = _require(conn.execute(text("""
        SELECT check_id, subject, threshold, current_value, SUBSTR(checked_at, 1, 10) AS d
        FROM risk_checks WHERE check_name = 'position_size' AND result = 'Pass'
          AND current_value < threshold ORDER BY threshold - current_value LIMIT 1""")).mappings().first(),
        "a passed position_size check (generator must record passing checks)")
    gap = round(passed["threshold"] - passed["current_value"], 2)
    answer = (f"The position-size check on {passed['subject']} on {passed['d']} failed: its notional of "
              f"${passed['current_value']:,.2f} came within ${gap:,.2f} of the ${passed['threshold']:,.0f} cap, "
              f"which triggered a rejection.")
    cases.append(BenchCase("invented_causal_near_cap", "invented_causal",
                           f"What did the position-size check on {passed['subject']} on {passed['d']} show?", answer, [
        Target("false_failed", _span(answer, f"on {passed['d']} failed"), "contradicted",
               f"The position-size check on {passed['subject']} on {passed['d']} failed",
               f"check {passed['check_id']} passed"),
        Target("false_triggered", _span(answer, "which triggered a rejection"), "contradicted",
               f"The position-size check on {passed['subject']} on {passed['d']} triggered a rejection",
               "under a cap is not a breach; the check passed"),
        Target("true_values", _span(answer, f"${passed['current_value']:,.2f}"), "supported",
               f"The position-size check on {passed['subject']} on {passed['d']} recorded a value of {passed['current_value']:.2f}"),
    ]))

    # --- sign_flip, paired ---------------------------------------------------
    pos = _require(conn.execute(text("""
        SELECT tag, final_alpha_return_pct FROM positions
        WHERE status='closed' AND final_alpha_return_pct < -1 LIMIT 1""")).mappings().first(),
        "a closed position with alpha below -1%")
    a = pos["final_alpha_return_pct"]
    answer = f"{pos['tag']} closed with alpha of +{abs(a):.2f}%. Its recorded final alpha was {a:.2f}%."
    cases.append(BenchCase("sign_flip_alpha", "sign_flip", f"How did {pos['tag']} perform?", answer, [
        Target("false_sign", _span(answer, f"alpha of +{abs(a):.2f}%"), "contradicted",
               f"{pos['tag']} closed with alpha of +{abs(a):.2f}%"),
        Target("true_sign", _span(answer, f"final alpha was {a:.2f}%"), "supported",
               f"{pos['tag']}'s recorded final alpha was {a:.2f}%"),
    ]))

    # --- wrong_count, paired: BOTH controls from the canonical tables --------
    win = ("2026-08-17", "2026-08-26")
    partials = conn.execute(text("""
        SELECT COUNT(*) FROM system_events WHERE event_type = 'partial_fill'
          AND SUBSTR(occurred_at, 1, 10) BETWEEN :a AND :b"""), {"a": win[0], "b": win[1]}).scalar()
    timeouts = conn.execute(text("""
        SELECT COUNT(*) FROM orders WHERE fallback_reason = 'timeout'
          AND SUBSTR(placed_at, 1, 10) BETWEEN :a AND :b"""), {"a": win[0], "b": win[1]}).scalar()
    if not partials or not timeouts:
        raise BenchmarkSetupError("cannot build case: window has no partial fills or no timeouts")
    answer = (f"Between {win[0]} and {win[1]} there were {partials + 3} partial-fill events "
              f"and {timeouts} orders that timed out and fell back to market.")
    cases.append(BenchCase("wrong_count_partials", "wrong_count", "What went wrong that week?", answer, [
        Target("false_count", _span(answer, f"{partials + 3} partial-fill events"), "contradicted",
               f"Between {win[0]} and {win[1]} there were {partials + 3} partial-fill events", f"actual {partials}"),
        Target("true_count", _span(answer, f"{timeouts} orders that timed out"), "supported",
               f"Between {win[0]} and {win[1]}, {timeouts} orders timed out and fell back to market",
               "from orders.fallback_reason, the canonical source"),
    ]))

    # --- reversed_ranking: figure and ranking are SEPARATE targets -----------
    attr = dispatch("alpha_attribution", {"group_by": "idx"}, conn)
    rows = sorted(attr.data.get("breakdown", []), key=lambda r: r["total_alpha_pct"], reverse=True)
    if len(rows) < 2:
        raise BenchmarkSetupError("cannot build case: fewer than two sectors")
    best, worst = rows[0], rows[-1]
    answer = (f"{best['grouping']} was the weakest sector. Its summed per-trade alpha was "
              f"{best['total_alpha_pct']}%. {worst['grouping']} was the strongest.")
    cases.append(BenchCase("reversed_ranking_sector", "reversed_ranking", "Which sector was weakest?", answer, [
        Target("false_weakest", _span(answer, f"{best['grouping']} was the weakest sector"), "contradicted",
               f"{best['grouping']} was the weakest sector by summed per-trade alpha"),
        Target("true_figure", _span(answer, f"was {best['total_alpha_pct']}%"), "supported",
               f"{best['grouping']}'s summed per-trade alpha was {best['total_alpha_pct']}%",
               "a true figure adjacent to a false ranking; must be verified, not contradicted"),
        Target("false_strongest", _span(answer, f"{worst['grouping']} was the strongest"), "contradicted",
               f"{worst['grouping']} was the strongest sector by summed per-trade alpha"),
    ]))

    # --- inference: count supported, cause UNDETERMINED ----------------------
    day = _require(conn.execute(text("""
        SELECT SUBSTR(triggered_at, 1, 10) AS d, COUNT(*) AS n FROM stop_orders
        WHERE status = 'triggered' AND triggered_at IS NOT NULL
        GROUP BY d ORDER BY n DESC LIMIT 1""")).mappings().first(), "a day with triggered stops")
    answer = (f"{day['n']} stop-loss orders triggered on {day['d']}. "
              f"The cluster was caused by a sharp intraday sector move that hit the short legs.")
    cases.append(BenchCase("inference_as_fact_cluster", "inference_as_fact",
                           f"Why did stops trigger on {day['d']}?", answer, [
        Target("true_count", _span(answer, f"{day['n']} stop-loss orders triggered"), "supported",
               f"{day['n']} stop-loss orders triggered on {day['d']}"),
        Target("undetermined_cause", _span(answer, "was caused by a sharp intraday sector move"), "undetermined",
               f"The stop-loss cluster on {day['d']} was caused by a sharp intraday sector move",
               "the blotter records exits, not intraday causes; holding-period returns do not settle this"),
    ]))

    # --- conflicting evidence: the recorded reason disagrees with the value --
    # Kept as its own test now the generator is consistent: a check whose
    # value does NOT breach its threshold but which is nonetheless recorded
    # as failed would be a data inconsistency. The critic should report the
    # conflict, not resolve it. With consistent data, this case asserts the
    # absence of such records.
    inconsistent = conn.execute(text("""
        SELECT COUNT(*) FROM risk_checks WHERE check_name = 'position_size'
          AND result = 'Fail' AND current_value <= threshold""")).scalar()
    if inconsistent:
        raise BenchmarkSetupError(
            f"data inconsistency: {inconsistent} position_size failures with value <= threshold")

    # --- all_true control ----------------------------------------------------
    order = _require(conn.execute(text("""
        SELECT order_id, ticker, side FROM orders
        WHERE fallback_reason = 'spread_validation_failure' LIMIT 1""")).mappings().first(),
        "an order routed to market on spread validation")
    eq = dispatch("execution_quality", {"order_id": order["order_id"]}, conn).data
    answer = (f"Order {order['order_id']} ({order['ticker']} {order['side']}) was routed to market "
              f"because its spread of {eq['spread_bps']}bps exceeded the {cap}bps limit-order cap. "
              f"It filled with {eq['slippage_bps']}bps of slippage against arrival mid.")
    cases.append(BenchCase("all_true_control", "control",
                           f"Why was order {order['order_id']} routed to market?", answer, [
        Target("true_cause", _span(answer, f"because its spread of {eq['spread_bps']}bps exceeded"), "supported",
               f"Order {order['order_id']} was routed to market because its spread of {eq['spread_bps']}bps exceeded the {cap}bps cap"),
        Target("true_slippage", _span(answer, f"{eq['slippage_bps']}bps of slippage"), "supported",
               f"Order {order['order_id']} filled with {eq['slippage_bps']}bps of slippage against arrival mid"),
    ]))

    return cases


def case_set_hash(cases: list[BenchCase]) -> str:
    payload = json.dumps([(c.name, c.question, c.answer,
                           [(t.id, t.span, t.expected, t.claim) for t in c.targets]) for c in cases],
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scoring — alignment is a candidate; every flag accounted for
# ---------------------------------------------------------------------------

def _outcome(expected: str, verdict: str) -> str:
    if expected == "contradicted":
        return {"contradicted": "caught", "verified": "false_verify",
                "undetermined": "caution"}.get(verdict, "caution")
    if expected == "supported":
        return {"verified": "confirmed", "contradicted": "false_alarm",
                "undetermined": "unresolved"}.get(verdict, "unresolved")
    return {"undetermined": "restrained"}.get(verdict, "overreach")


def score(case: BenchCase, crit: Critique) -> BenchResult:
    r = BenchResult(case=case, critique=crit)
    located = []
    for v in crit.verdicts:
        i = case.answer.find(v.claim.source)
        located.append((v, (i, i + len(v.claim.source)) if i >= 0 else None))

    # Align each claim to targets it overlaps
    claim_targets: dict[int, list[Target]] = {}
    for idx, (v, span) in enumerate(located):
        if span is None:
            continue
        claim_targets[idx] = [t for t in case.targets if t.overlaps(*span)]

    for t in case.targets:
        aligned = [located[idx][0] for idx, ts in claim_targets.items() if ts == [t]]
        touching = [located[idx][0] for idx, ts in claim_targets.items() if t in ts]
        if not touching:
            r.outcome[t.id] = "not_extracted"
        elif not aligned:
            r.outcome[t.id] = "ambiguous"
        else:
            # If several claims align uniquely, the worst verdict governs
            r.target_verdicts[t.id] = aligned
            verdicts = [v.verdict for v in aligned if v.verdict != "skipped"]
            if any(v.assessment_error for v in aligned):
                r.outcome[t.id] = "assessment_error"
            elif not verdicts:
                r.outcome[t.id] = "not_extracted"
            else:
                outs = [_outcome(t.expected, vd) for vd in verdicts]
                bad = [o for o in outs if o in ("false_alarm", "false_verify", "overreach")]
                r.outcome[t.id] = bad[0] if bad else ("caution" if "caution" in outs else "unresolved" if "unresolved" in outs else outs[0])

    for idx, ts in claim_targets.items():
        v = located[idx][0]
        if len(ts) > 1:
            r.ambiguous_claims.append(v.claim.source[:70])
        elif not ts and v.verdict == "contradicted":
            r.unmatched_flags.append(v.claim.source[:70])
    return r


def score_atomic(case: BenchCase, verdicts: dict[str, Verdict]) -> BenchResult:
    """Score verifier-only results: one verdict per target, no extraction."""
    r = BenchResult(case=case, critique=None)
    for t in case.targets:
        v = verdicts.get(t.id)
        r.target_verdicts[t.id] = [v] if v else []
        r.outcome[t.id] = ("assessment_error" if v.assessment_error else _outcome(t.expected, v.verdict)) if v else "not_extracted"
    return r


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_benchmark(conn: Connection, client=None, cases: list[BenchCase] | None = None,
                  verbose: bool = True, trace_dir=None) -> BenchSummary:
    cases = cases if cases is not None else build_cases(conn)
    results = []
    for case in cases:
        if verbose:
            print(f"  {case.name} ... ", end="", flush=True)
        crit = critique(case.question, case.answer, conn, client=client, trace_dir=trace_dir)
        r = score(case, crit)
        results.append(r)
        if verbose:
            print(f"{'PASS' if r.passed else 'FAIL'}  errors {r.errors}  "
                  f"{'complete' if crit.complete else 'INCOMPLETE'}  ({crit.tokens:,} tok)")
    return BenchSummary(results, mode="end_to_end")


def run_atomic(conn: Connection, client=None, cases: list[BenchCase] | None = None,
               verbose: bool = True, trace_dir=None) -> BenchSummary:
    """Verifier only: each target's atomic claim goes straight to verify_claim."""
    cases = cases if cases is not None else build_cases(conn)
    results = []
    for case in cases:
        if verbose:
            print(f"  {case.name} ... ", end="", flush=True)
        verdicts = {}
        tokens = 0
        for t in case.targets:
            claim = Claim(source=t.claim, text=t.claim, type="figure", checkable=True, stated_as_fact=True)
            v = verify_claim(claim, case.question, conn, client=client, trace_dir=trace_dir)
            verdicts[t.id] = v
            tokens += v.tokens
        r = score_atomic(case, verdicts)
        results.append(r)
        if verbose:
            print(f"{'PASS' if r.passed else 'FAIL'}  errors {r.errors}  ({tokens:,} tok)")
    return BenchSummary(results, mode="atomic")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def database_digest(conn: Connection) -> tuple[str, dict[str, int]]:
    """Logical SHA-256 of every table's schema/rows; row order is immaterial.

    Intended for a frozen local evaluation database. Row digests keep memory
    bounded to 32 bytes per row while retaining duplicates. This is not a
    claim of a cross-backend byte-identical representation.
    """
    digest = hashlib.sha256()
    counts = {}
    inspector = inspect(conn)
    quote = conn.dialect.identifier_preparer.quote
    for table in sorted(inspector.get_table_names()):
        columns = sorted(c["name"] for c in inspector.get_columns(table))
        digest.update(json.dumps([table, columns], ensure_ascii=False).encode())
        rows = []
        query = f"SELECT {', '.join(quote(c) for c in columns)} FROM {quote(table)}"
        for row in conn.execute(text(query)):
            encoded = json.dumps(list(row), ensure_ascii=False, default=str, separators=(",", ":")).encode()
            rows.append(hashlib.sha256(encoded).digest())
        counts[table] = len(rows)
        digest.update(str(len(rows)).encode())
        for row_digest in sorted(rows):
            digest.update(row_digest)
    return digest.hexdigest(), counts


def provenance(conn: Connection, cases: list[BenchCase], model: str) -> dict:
    """Hash source files, actual imports, tool schemas and database contents."""
    root = Path(__file__).resolve().parent.parent.parent
    files = sorted((root / "src").rglob("*.py")) + [root / "schema.sql", root / "run_critic_benchmark.py"]
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in files if f.is_file()}
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                imports[name] = {"path": str(Path(path).resolve()),
                                 "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    from src.critic.prompts import EXTRACTOR_PROMPT, VERIFIER_PROMPT
    from src.tools import TOOL_SCHEMAS
    fingerprint, counts = database_digest(conn)
    first, last = conn.execute(text("SELECT MIN(run_date), MAX(run_date) FROM workflow_runs")).one()
    return {
        "model": model, "source_hashes": hashes, "imported_modules": imports,
        "tool_schema_hash": hashlib.sha256(json.dumps(TOOL_SCHEMAS, sort_keys=True).encode()).hexdigest(),
        "extractor_prompt_hash": hashlib.sha256(EXTRACTOR_PROMPT.encode()).hexdigest(),
        "verifier_prompt_hash": hashlib.sha256(VERIFIER_PROMPT.encode()).hexdigest(),
        "case_set_hash": case_set_hash(cases),
        "database": {"fingerprint": fingerprint, "fingerprint_kind": "logical_rows_sha256_v1",
                     "row_counts": counts, "window": [first, last]},
    }
