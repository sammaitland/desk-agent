"""Three-arm comparison: baseline, compression-only, routed.

Runs every eval case three times under controlled configurations and reports
cost and pass rate per arm, per case. This is the measurement that turns
"cost-aware routing" from a feature into a claim — and the design of the
measurement is most of the work.

## The three arms

    baseline       Sonnet, 8 turns, caching on, no compression, no routing
    compression    as baseline, compression on
    routed         router picks tier; compression per tier

Two arms would confound the mechanisms: a saving under "routed" could be
compression, routing, or both. Three arms isolate each. If compression saves
30% and routed saves 32%, routing contributed 2% and the interesting number is
the compression one.

## Cost accounting

Four token classes, billed at different rates:

    input          full rate
    output         full rate (higher than input)
    cache_read     ~10% of input rate
    cache_write    ~125% of input rate

`processed_tokens` counts all four at face value — how much the model handled.
`cost_usd` weights them by published rates — what it was billed. The two
diverge under caching and under Haiku routing, and reporting only one of them
misstates the saving. Both are reported.

## Casewise reporting

Aggregate pass rate hides a case that flipped pass→fail alongside one that
flipped fail→pass. Every case is reported against baseline: regressed,
improved, or unchanged. "No worse" means no regressions, not equal totals.

## Leave-one-question-out

The router's corpus is built from development cases. When routing a
development case in this comparison, the predictor excludes every run of that
question — otherwise it finds itself and routes with false confidence. Held-out
cases are never in the corpus, so they need no exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Connection

from src.agent.loop import MODEL, run_agent
from src.agent.trace import Trace
from src.evals.cases import UNIVERSAL, EvalCase
from src.evals.checks import CheckResult
from src.routing.router import Router, route

# USD per million tokens: (input, output). Cache read is 10% of input; cache
# write is 125% of input. Keep in sync with published pricing.
PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25

ARMS = ("baseline", "compression", "routed")

WARMUP_QUESTION = "__cache_warmup__"


def warm_caches(conn: Connection, models: list[str], client=None) -> list[str]:
    """Prime the prompt cache for each model before measurement.

    The system prompt and tool schemas are cached per model. Whichever arm
    runs first on a cold cache pays cache *creation* (~125% of input rate)
    while later arms get cache *reads* (~10%). Since baseline always runs
    first, an uncontrolled experiment would systematically overcharge
    baseline and flatter the other arms.

    So the comparison is defined as **warm-cache, steady-state**: one
    discarded call per distinct model, before any measured run, so every arm
    sees the same cache state. Warm-up cost is excluded from all figures and
    reported separately. Returns the models warmed, for the report.
    """
    warmed = []
    for model in dict.fromkeys(models):   # distinct, order preserved
        run_agent(WARMUP_QUESTION, conn, client=client, model=model, max_turns=1)
        warmed.append(model)
    return warmed


@dataclass
class ArmResult:
    """One eval case under one arm."""

    name: str
    arm: str
    passed: bool
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    compressed_chars: int
    held_out: bool
    tool_sequence: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    # For the routed arm: the router's decision, so a test can prove the
    # exclusion was applied rather than inferring it from a separate call.
    routing: dict | None = None

    @property
    def processed_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICE_PER_MTOK.get(self.model, PRICE_PER_MTOK[MODEL])
        return (self.input_tokens * p_in
                + self.cache_read * p_in * CACHE_READ_MULT
                + self.cache_write * p_in * CACHE_WRITE_MULT
                + self.output_tokens * p_out) / 1_000_000


@dataclass
class CaseComparison:
    """One case across all arms, judged against baseline."""

    name: str
    held_out: bool
    arms: dict[str, ArmResult]

    def status(self, arm: str) -> str:
        base, other = self.arms["baseline"].passed, self.arms[arm].passed
        if base and not other:
            return "REGRESSED"
        if not base and other:
            return "improved"
        return "unchanged"


@dataclass
class Comparison:
    results: list[ArmResult] = field(default_factory=list)
    warmed_models: list[str] = field(default_factory=list)
    cache_protocol: str = "warm"   # "warm" (steady-state) or "cold" (uncontrolled)

    def arm(self, name: str) -> list[ArmResult]:
        return [r for r in self.results if r.arm == name]

    def casewise(self) -> list[CaseComparison]:
        by_case: dict[str, dict[str, ArmResult]] = {}
        for r in self.results:
            by_case.setdefault(r.name, {})[r.arm] = r
        return [CaseComparison(name, arms["baseline"].held_out, arms)
                for name, arms in by_case.items() if "baseline" in arms]

    def summary(self) -> dict:
        def agg(rows: list[ArmResult]) -> dict:
            n = len(rows) or 1
            return {
                "cases": len(rows),
                "passed": sum(r.passed for r in rows),
                "pass_rate": round(100 * sum(r.passed for r in rows) / n, 1),
                "processed_tokens": sum(r.processed_tokens for r in rows),
                "cost_usd": round(sum(r.cost_usd for r in rows), 4),
            }

        arms = {a: agg(self.arm(a)) for a in ARMS if self.arm(a)}
        base = arms.get("baseline", agg([]))
        cases = self.casewise()

        def delta(a: str) -> dict:
            if a not in arms or not base["cost_usd"]:
                return {}
            regressed = [c.name for c in cases if a in c.arms and c.status(a) == "REGRESSED"]
            improved = [c.name for c in cases if a in c.arms and c.status(a) == "improved"]
            return {
                "token_saving_pct": round(100 * (1 - arms[a]["processed_tokens"] / base["processed_tokens"]), 1)
                if base["processed_tokens"] else 0.0,
                "cost_saving_pct": round(100 * (1 - arms[a]["cost_usd"] / base["cost_usd"]), 1),
                "regressed": regressed,
                "improved": improved,
                "no_worse": not regressed,
            }

        tiers: dict[str, int] = {}
        for r in self.arm("routed"):
            tiers[r.tier] = tiers.get(r.tier, 0) + 1

        held = {a: agg([r for r in self.arm(a) if r.held_out]) for a in ARMS if self.arm(a)}

        compression, routed = delta("compression"), delta("routed")
        # Overall acceptance: every non-baseline arm that ran must have zero
        # casewise regressions. An arm that did not run cannot fail the gate,
        # but the report says which arms were assessed so a two-arm run is
        # not mistaken for a three-arm one.
        assessed = [a for a, d in (("compression", compression), ("routed", routed)) if d]
        no_worse = bool(assessed) and all(
            d["no_worse"] for d in (compression, routed) if d)

        return {
            "arms": arms,
            "compression": compression,
            "routed": routed,
            "tier_mix": tiers,
            "held_out": held if any(h["cases"] for h in held.values()) else None,
            "assessed_arms": assessed,
            "no_worse": no_worse,
        }


def _judge(case: EvalCase, trace: Trace) -> tuple[bool, list[str]]:
    results: list[CheckResult] = [c(trace) for c in case.checks + UNIVERSAL]
    return all(r.passed for r in results), [r.name for r in results if not r.passed]


def _arm(case: EvalCase, arm: str, trace: Trace) -> ArmResult:
    passed, failed = _judge(case, trace)
    tier = trace.routing["tier"] if trace.routing else arm
    return ArmResult(
        name=case.name, arm=arm, passed=passed, model=trace.model or MODEL, tier=tier,
        input_tokens=trace.input_tokens, output_tokens=trace.output_tokens,
        cache_read=trace.cache_read_tokens, cache_write=trace.cache_write_tokens,
        compressed_chars=trace.compressed_chars, held_out=case.held_out,
        tool_sequence=trace.tool_sequence, failed_checks=failed,
        routing=trace.routing,
    )


def run_comparison(cases: list[EvalCase], conn: Connection, client=None,
                   router: Router | None = None, verbose: bool = True,
                   arms: tuple[str, ...] = ARMS, warm_cache: bool = True) -> Comparison:
    """Every case under every arm. Same client, same data, same checks.

    With `warm_cache=True` (the default and the defined protocol), every
    model any arm could use is primed before the first measured run, so cache
    state does not depend on arm order. `warm_cache=False` is a cold-start
    measurement and is reported as such.
    """
    router = router or Router()
    comparison = Comparison(cache_protocol="warm" if warm_cache else "cold")

    if warm_cache:
        from src.routing.router import TIERS

        candidates = [MODEL] + [t.model for t in TIERS.values()]
        comparison.warmed_models = warm_caches(conn, candidates, client=client)
        if verbose:
            print(f"  warmed caches: {comparison.warmed_models} (excluded from figures)\n")

    for case in cases:
        if verbose:
            print(f"  {case.name:<26}", end="", flush=True)

        if "baseline" in arms:
            t = run_agent(case.question, conn, client=client, compress=False)
            comparison.results.append(_arm(case, "baseline", t))

        if "compression" in arms:
            t = run_agent(case.question, conn, client=client, compress=True)
            comparison.results.append(_arm(case, "compression", t))

        if "routed" in arms:
            # Leave-one-question-out for development cases: the predictor must
            # not find this question in its own corpus.
            exclude = None if case.held_out else case.question
            decision = route(router.predictor.predict(case.question, exclude_question=exclude))
            t = run_agent(case.question, conn, client=client,
                          model=decision.tier.model, max_turns=decision.tier.max_turns,
                          compress=decision.tier.compress, routing=decision.as_dict())
            comparison.results.append(_arm(case, "routed", t))

        if verbose:
            latest = [r for r in comparison.results if r.name == case.name]
            cells = [f"{r.arm[:4]} {'P' if r.passed else 'F'} {r.processed_tokens:>6}" for r in latest]
            print("  ".join(cells) + ("  [held-out]" if case.held_out else ""))

    return comparison


def report(comparison: Comparison) -> str:
    s = comparison.summary()
    lines = ["", "=" * 74, "THREE-ARM COMPARISON", "=" * 74,
             f"cache protocol: {comparison.cache_protocol}"
             + (f" — warmed {comparison.warmed_models}, warm-up excluded"
                if comparison.warmed_models else " — UNCONTROLLED, first arm pays cache creation"),
             f"acceptance: zero casewise regressions on {', '.join(s['assessed_arms']) or 'no arms'}"
             f" -> {'PASS' if s['no_worse'] else 'FAIL'}", "",
             f"{'arm':<13}{'cases':>6}{'pass':>6}{'pass%':>7}{'processed':>11}{'cost $':>9}"]
    for a in ARMS:
        if a in s["arms"]:
            r = s["arms"][a]
            lines.append(f"{a:<13}{r['cases']:>6}{r['passed']:>6}{r['pass_rate']:>7}"
                         f"{r['processed_tokens']:>11,}{r['cost_usd']:>9.4f}")

    for a in ("compression", "routed"):
        d = s.get(a)
        if not d:
            continue
        lines += ["", f"{a} vs baseline:",
                  f"  tokens {d['token_saving_pct']:+.1f}%   cost {d['cost_saving_pct']:+.1f}%",
                  f"  regressed: {d['regressed'] or 'none'}",
                  f"  improved:  {d['improved'] or 'none'}",
                  f"  NO WORSE: {'YES' if d['no_worse'] else 'NO'}"]

    if s["tier_mix"]:
        lines += ["", f"routed tier mix: {s['tier_mix']}"]

    if s["held_out"]:
        lines += ["", "held-out (out-of-sample):"]
        for a, h in s["held_out"].items():
            lines.append(f"  {a:<13}{h['passed']}/{h['cases']}  {h['processed_tokens']:,} tok  ${h['cost_usd']:.4f}")

    lines += ["", "casewise:"]
    for c in comparison.casewise():
        marks = "  ".join(f"{a[:4]}:{c.status(a)[:3]}" for a in ARMS if a in c.arms and a != "baseline")
        flag = "  [held-out]" if c.held_out else ""
        lines.append(f"  {c.name:<26}{marks}{flag}")
    lines.append("")
    return "\n".join(lines)
