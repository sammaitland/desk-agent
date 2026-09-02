#!/usr/bin/env python3
"""Run the eval suite against the live API.

    python run_evals.py                        # all cases
    python run_evals.py --tags chaining        # a subset
    python run_evals.py --case false_premise   # one case
    python run_evals.py --show-answers         # print each answer in full

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
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    env.require("anthropic")

    cases = select(names=args.case, tags=args.tags)
    if not cases:
        print(f"No cases matched. Available: {', '.join(c.name for c in CASES)}", file=sys.stderr)
        return 2

    print(f"\nRunning {len(cases)} case(s) against {args.model}\n")
    engine = get_engine(args.db_url)
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
