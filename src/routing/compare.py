"""Routed-versus-baseline comparison.

Runs the eval suite twice — once at the fixed default configuration, once
through the router — and reports the difference in cost and pass rate. This is
the measurement that turns "cost-aware routing" from a feature into a claim.

The bar is explicit: **routed must be no worse on pass rate**. A router that
saves 40% and fails two more cases has not saved anything; it has traded
quality for a number. The report states both, side by side, per tier.

Held-out cases are the honest measure. The router's thresholds were set
looking at development-case traces, so a saving on development cases is
partly in-sample. Run with `--include-held-out` for the figure to quote.

**Cost** here is tokens, not money. Model prices differ (Haiku is roughly a
tenth of Sonnet per token), so a token saving understates the cost saving
when a query routes to `light`. The report gives both: tokens, and a
price-weighted estimate using published per-model rates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.engine import Connection

from src.agent.loop import MODEL, run_agent, run_agent_routed
from src.evals.cases import EvalCase
from src.evals.runner import CaseResult, run_case
from src.routing.router import Router

# USD per million tokens, input/output. Approximate; the ratio matters more
# than the absolute. Keep in sync with published pricing.
PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}


@dataclass
class ArmResult:
    """One eval case run under one configuration."""

    name: str
    passed: bool
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    compressed_chars: int
    held_out: bool

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICE_PER_MTOK.get(self.model, PRICE_PER_MTOK[MODEL])
        # Cached input is billed at ~10% of the input rate.
        return (self.input_tokens * p_in + self.cache_read * p_in * 0.1
                + self.output_tokens * p_out) / 1_000_000


@dataclass
class Comparison:
    baseline: list[ArmResult] = field(default_factory=list)
    routed: list[ArmResult] = field(default_factory=list)

    def summary(self) -> dict:
        def agg(arm: list[ArmResult]) -> dict:
            n = len(arm) or 1
            return {
                "cases": len(arm),
                "passed": sum(a.passed for a in arm),
                "pass_rate": round(100 * sum(a.passed for a in arm) / n, 1),
                "tokens": sum(a.input_tokens + a.output_tokens for a in arm),
                "cost_usd": round(sum(a.cost_usd for a in arm), 4),
            }

        b, r = agg(self.baseline), agg(self.routed)
        tiers: dict[str, int] = {}
        for a in self.routed:
            tiers[a.tier] = tiers.get(a.tier, 0) + 1

        held_b = [a for a in self.baseline if a.held_out]
        held_r = [a for a in self.routed if a.held_out]

        return {
            "baseline": b,
            "routed": r,
            "token_saving_pct": round(100 * (1 - r["tokens"] / b["tokens"]), 1) if b["tokens"] else 0.0,
            "cost_saving_pct": round(100 * (1 - r["cost_usd"] / b["cost_usd"]), 1) if b["cost_usd"] else 0.0,
            "pass_rate_delta": round(r["pass_rate"] - b["pass_rate"], 1),
            "tier_mix": tiers,
            "held_out": {"baseline": agg(held_b), "routed": agg(held_r)} if held_b else None,
            "no_worse": r["passed"] >= b["passed"],
        }


def _arm(result: CaseResult, trace, tier: str, held_out: bool) -> ArmResult:
    return ArmResult(
        name=result.name, passed=result.passed, model=trace.model or MODEL,
        tier=tier, input_tokens=trace.input_tokens, output_tokens=trace.output_tokens,
        cache_read=trace.cache_read_tokens, compressed_chars=trace.compressed_chars,
        held_out=held_out,
    )


def run_comparison(cases: list[EvalCase], conn: Connection, client=None,
                   router: Router | None = None, verbose: bool = True) -> Comparison:
    """Every case twice: baseline, then routed. Same client, same data."""
    from src.evals.cases import UNIVERSAL
    from src.evals.checks import CheckResult

    router = router or Router()
    comparison = Comparison()

    for case in cases:
        if verbose:
            print(f"  {case.name} ... ", end="", flush=True)

        # --- baseline: fixed model, fixed budget, no compression ---------
        trace_b = run_agent(case.question, conn, client=client)
        checks_b = [c(trace_b) for c in case.checks + UNIVERSAL]
        result_b = CaseResult(name=case.name, question=case.question,
                              passed=all(c.passed for c in checks_b),
                              checks=[c.__dict__ for c in checks_b],
                              tool_sequence=trace_b.tool_sequence, held_out=case.held_out)
        comparison.baseline.append(_arm(result_b, trace_b, "baseline", case.held_out))

        # --- routed ------------------------------------------------------
        trace_r = run_agent_routed(case.question, conn, client=client, router=router)
        checks_r = [c(trace_r) for c in case.checks + UNIVERSAL]
        result_r = CaseResult(name=case.name, question=case.question,
                              passed=all(c.passed for c in checks_r),
                              checks=[c.__dict__ for c in checks_r],
                              tool_sequence=trace_r.tool_sequence, held_out=case.held_out)
        tier = trace_r.routing["tier"] if trace_r.routing else "standard"
        comparison.routed.append(_arm(result_r, trace_r, tier, case.held_out))

        if verbose:
            b, r = comparison.baseline[-1], comparison.routed[-1]
            flag = "" if r.passed >= b.passed else "  <-- REGRESSION"
            print(f"baseline {'PASS' if b.passed else 'FAIL'} {b.input_tokens + b.output_tokens:>6}tok"
                  f" | routed [{tier}] {'PASS' if r.passed else 'FAIL'} "
                  f"{r.input_tokens + r.output_tokens:>6}tok{flag}")

    return comparison


def report(comparison: Comparison) -> str:
    s = comparison.summary()
    lines = [
        "", "=" * 66, "ROUTING COMPARISON", "=" * 66,
        f"{'':<12} {'cases':>6} {'passed':>7} {'pass%':>7} {'tokens':>9} {'cost $':>9}",
        f"{'baseline':<12} {s['baseline']['cases']:>6} {s['baseline']['passed']:>7} "
        f"{s['baseline']['pass_rate']:>7} {s['baseline']['tokens']:>9,} {s['baseline']['cost_usd']:>9.4f}",
        f"{'routed':<12} {s['routed']['cases']:>6} {s['routed']['passed']:>7} "
        f"{s['routed']['pass_rate']:>7} {s['routed']['tokens']:>9,} {s['routed']['cost_usd']:>9.4f}",
        "",
        f"token saving   {s['token_saving_pct']:+.1f}%",
        f"cost saving    {s['cost_saving_pct']:+.1f}%  (price-weighted; Haiku ~1/3 Sonnet rate)",
        f"pass-rate Δ    {s['pass_rate_delta']:+.1f} pts",
        f"tier mix       {s['tier_mix']}",
        "",
        f"NO WORSE: {'YES' if s['no_worse'] else 'NO — routed failed cases the baseline passed'}",
    ]
    if s["held_out"]:
        hb, hr = s["held_out"]["baseline"], s["held_out"]["routed"]
        lines += ["", "held-out (the honest measure):",
                  f"  baseline {hb['passed']}/{hb['cases']}  {hb['tokens']:,} tok",
                  f"  routed   {hr['passed']}/{hr['cases']}  {hr['tokens']:,} tok"]
    lines.append("")
    return "\n".join(lines)
