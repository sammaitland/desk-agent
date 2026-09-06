#!/usr/bin/env python3
"""Run the eval suite against the live API.

    python run_evals.py                        # all cases
    python run_evals.py --tags chaining        # a subset
    python run_evals.py --case false_premise   # one case
    python run_evals.py --show-answers         # print each answer in full
    python run_evals.py --include-held-out     # the honest measure; run sparingly
    python run_evals.py --build-baseline                       # router corpus, dev cases only
    python run_evals.py --compare-routing --include-held-out   # three arms, casewise

Costs real API calls — roughly one per case plus a turn per tool used. Run on
demand or nightly, not on every commit; `pytest` covers the loop mechanics for
free.
"""

from __future__ import annotations

import argparse
import os
import sys

from src import env
from src.agent.loop import MODEL
from src.db import get_engine
from src.evals.cases import CASES, select
from src.evals.runner import report, run_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent evals against the live API.")
    parser.add_argument("--case", action="append", help="run only named case(s)")
    parser.add_argument("--tags", nargs="*", help="run only cases with these tags")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--include-held-out", action="store_true",
                        help="also run held-out cases (never tune the prompt against these)")
    parser.add_argument("--compare-routing", action="store_true",
                        help="three arms per case (baseline, compression, routed); report deltas")
    parser.add_argument("--build-baseline", action="store_true",
                        help="generate the router's training corpus: development cases only, "
                             "fixed config (Sonnet, 8 turns, caching, no compression, no routing)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    env.require("anthropic")

    cases = select(names=args.case, tags=args.tags, include_held_out=args.include_held_out)
    if args.include_held_out:
        held = [c.name for c in cases if c.held_out]
        print(f"\nHELD-OUT cases included: {', '.join(held)}")
        print("Do not edit the prompt in response to a held-out failure. Promote the case first.")
    if not cases:
        print(f"No cases matched. Available: {', '.join(c.name for c in CASES)}", file=sys.stderr)
        return 2

    engine = get_engine(args.db_url)

    if args.build_baseline:
        from src.agent.trace import BASELINE_DIR
        from src.evals.cases import DEVELOPMENT

        if args.include_held_out:
            print("Refusing: held-out cases must never enter the baseline corpus.", file=sys.stderr)
            return 2
        cases = select(names=args.case, tags=args.tags)   # development only, by construction
        print(f"\nBuilding baseline corpus: {len(cases)} development case(s) -> {BASELINE_DIR}")
        print("Config: Sonnet 4.6, 8 turns, caching on, compression off, routing off\n")
        with engine.connect() as conn:
            for case in cases:
                print(f"  {case.name} ... ", end="", flush=True)
                from src.agent.loop import run_agent
                t = run_agent(case.question, conn, save_trace=True,
                              corpus="baseline", trace_dir=BASELINE_DIR)
                print(f"{t.processed_tokens:,} tok")
        print(f"\nDone. Predictor will load from {BASELINE_DIR}.")
        return 0

    if args.compare_routing:
        from src.routing.compare import report as routing_report, run_comparison
        from src.routing.router import Router

        router = Router()
        rep = router.corpus_report
        print(f"\nThree-arm comparison over {len(cases)} case(s)")
        print(f"predictor corpus: {rep.accepted} accepted, {rep.rejected} rejected "
              f"(schema {rep.rejected_schema}, held-out {rep.rejected_held_out}, "
              f"corpus-tag {rep.rejected_corpus}, error {rep.rejected_error})")
        if rep.accepted < 5:
            print("WARNING: thin corpus — most queries will route to default. "
                  "Run --build-baseline first.")
        print()
        with engine.connect() as conn:
            comparison = run_comparison(cases, conn, router=router)
        print(routing_report(comparison))
        return 0 if comparison.summary()["no_worse"] else 1

    print(f"\nRunning {len(cases)} case(s) against {args.model}\n")
    with engine.connect() as conn:
        suite = run_suite(cases, conn, model=args.model)

    print(report(suite))

    if args.show_answers:
        for case in suite.cases:
            print("-" * 66)
            print(f"{case.name}  ({case.words} words, tools: {case.tool_sequence})")
            print("-" * 66)
            print(case.answer, "\n")

    if not args.no_save:
        print(f"Results written to {suite.save()}\n")

    return 0 if suite.passed == suite.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
