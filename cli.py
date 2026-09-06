#!/usr/bin/env python3
"""Ask the desk agent a question.

    python cli.py "why did the AVGO order fill badly on the 19th?"
    python cli.py "how has alpha broken down by bucket?" --trace
    python cli.py "what went wrong last week?" --save-trace

Credentials load from .env at the project root (copy .env.example).

Requires a populated blotter:

    python -m src.generate_blotter --days 120 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

from src import env
from src.agent.loop import MAX_TURNS, MODEL, run_agent
from src.db import get_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the trading blotter in natural language.")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--trace", action="store_true", help="print the tool sequence")
    parser.add_argument("--save-trace", action="store_true", help="write the trace to traces/")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--routed", action="store_true",
                        help="let the router pick model and budget from past traces")
    parser.add_argument("--compress", action="store_true",
                        help="compress stale tool results in history")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    env.require("anthropic")

    engine = get_engine(args.db_url)
    with engine.connect() as conn:
        if args.routed:
            from src.agent.loop import run_agent_routed

            trace = run_agent_routed(args.question, conn, save_trace=args.save_trace)
        else:
            trace = run_agent(
                args.question, conn,
                model=args.model, max_turns=args.max_turns,
                save_trace=args.save_trace, compress=args.compress,
            )

    if args.trace:
        print(trace.render())
        print()
    print(trace.answer)

    charts = [c for c in trace.tool_calls if c.name == "make_chart"]
    if charts:
        print("\nCharts written to charts/")

    return 1 if trace.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
