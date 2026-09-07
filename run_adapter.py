#!/usr/bin/env python3
"""Load archived V9.2C output into the desk agent's blotter.

    python run_adapter.py ~/Desktop/V9/archive
    python run_adapter.py ~/Desktop/V9/archive --since 2026-09-08
    python run_adapter.py ~/Desktop/V9/archive --dry-run     # report only

Idempotent: re-running over days already loaded changes nothing. The report
lists every source column and enum value it could not map — on the first
real run, that list is the set of corrections to make in src/adapter/mapping.py.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src import env  # noqa: F401
from src.adapter.load import archived_days, load_archive
from src.db import create_schema, get_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Load V9.2C archives into the blotter.")
    parser.add_argument("archive", type=Path, help="directory of YYYY-MM-DD subdirectories")
    parser.add_argument("--since", type=date.fromisoformat, default=None)
    parser.add_argument("--dry-run", action="store_true", help="list days; load nothing")
    parser.add_argument("--init", action="store_true", help="create the schema first (drops existing tables)")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    days = archived_days(args.archive)
    if not days:
        print(f"No dated directories under {args.archive}")
        return 1
    print(f"{len(days)} archived day(s): {days[0]} .. {days[-1]}")
    if args.dry_run:
        return 0

    engine = get_engine(args.db_url)
    if args.init:
        create_schema(engine)
        print("schema created")
    with engine.connect() as conn:
        report = load_archive(args.archive, conn, since=args.since)
    print()
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
