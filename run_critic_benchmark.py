#!/usr/bin/env python3
"""Run the critic over claims with known expected outcomes.

    python run_critic_benchmark.py                 # end-to-end: extraction + verification
    python run_critic_benchmark.py --atomic        # verifier only, on hand-fixed atomic claims
    python run_critic_benchmark.py --save          # write results + provenance to eval_results/

Two measurements. End-to-end tests the whole pipeline; atomic tests the
verifier alone, so an error can be placed at the stage that made it. Every
saved run records source hashes, prompt hashes, the case-set hash and a
database fingerprint, so stale data and mixed revisions are testable rather
than suspected.

Exit code 0 requires: zero errors (false alarms, false verifications,
overreach, unmatched flags), every case assessed completely, and recall of
grounded contradictions at or above 60%.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src import env  # noqa: F401
from src.agent.loop import MODEL
from src.critic.benchmark import build_cases, database_digest, provenance, run_atomic, run_benchmark
from src.db import get_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic", action="store_true", help="verifier only, bypassing extraction")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--preflight", action="store_true", help="print provenance and exit without model calls")
    parser.add_argument("--expect-db", default=None, help="require this content fingerprint before running")
    args = parser.parse_args()

    engine = get_engine(args.db_url)
    with engine.connect() as conn:
        cases = build_cases(conn)
        prov = provenance(conn, cases, MODEL)
        if args.expect_db and args.expect_db != prov["database"]["fingerprint"]:
            parser.error("database content fingerprint differs from --expect-db; no model calls made")
        if args.preflight:
            print(json.dumps(prov, indent=2, default=str))
            return 0
        print(f"\nCritic benchmark — {'atomic' if args.atomic else 'end-to-end'}")
        print(f"case set {prov['case_set_hash']}   db {prov['database']['fingerprint']}   "
              f"verifier prompt {prov['verifier_prompt_hash']}\n")
        summary = (run_atomic if args.atomic else run_benchmark)(conn, cases=cases)
        final_digest, _ = database_digest(conn)
        database_unchanged = final_digest == prov["database"]["fingerprint"]
    print(summary.render())

    if args.save:
        out = Path("eval_results")
        out.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out / f"critic_{summary.mode}_{stamp}.json"
        payload = {**summary.as_dict(), "provenance": prov, "database_unchanged": database_unchanged}
        payload["gate_passed"] = summary.gate_passed and database_unchanged
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"saved {path}  (verifier traces in traces/critic/)")

    if not database_unchanged:
        print("FAIL: database contents changed during the run; comparison is invalid.")
    return 0 if summary.gate_passed and database_unchanged else 1



if __name__ == "__main__":
    raise SystemExit(main())
