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
from src.critic.benchmark import build_cases, provenance, run_atomic, run_benchmark
from src.db import get_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic", action="store_true", help="verifier only, bypassing extraction")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    engine = get_engine(args.db_url)
    with engine.connect() as conn:
        cases = build_cases(conn)
        prov = provenance(conn, cases, MODEL)
        print(f"\nCritic benchmark — {'atomic' if args.atomic else 'end-to-end'}")
        print(f"case set {prov['case_set_hash']}   db {prov['database']['fingerprint']}   "
              f"verifier prompt {prov['verifier_prompt_hash']}\n")
        summary = (run_atomic if args.atomic else run_benchmark)(conn, cases=cases)
    print(summary.render())

    if args.save:
        out = Path("eval_results")
        out.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out / f"critic_{summary.mode}_{stamp}.json"
        path.write_text(json.dumps({
            "mode": summary.mode,
            "provenance": prov,
            "metrics": {
                "extraction_coverage": summary.extraction_coverage,
                "recall": summary.recall, "precision": summary.precision,
                "restraint": summary.restraint,
                "caught": summary.total("caught"), "confirmed": summary.total("confirmed"),
                "restrained": summary.total("restrained"), "cautions": summary.total("caution"),
                "unresolved": summary.total("unresolved"),
                "false_alarms": summary.total("false_alarm"),
                "false_verifications": summary.total("false_verify"),
                "overreach": summary.total("overreach"),
            },
            "cases": [{
                "name": r.case.name, "kind": r.case.kind, "passed": r.passed,
                "outcome": r.outcome, "ambiguous": r.ambiguous_claims,
                "unmatched": r.unmatched_flags,
                "targets": [{"id": t.id, "span": t.span, "expected": t.expected,
                             "claim": t.claim, "note": t.note} for t in r.case.targets],
                "critique": r.critique.as_dict() if r.critique else None,
            } for r in summary.results],
        }, indent=2, default=str))
        print(f"saved {path}  (verifier traces in traces/critic/)")

    errors = sum(r.errors for r in summary.results)
    incomplete = sum(1 for r in summary.results if r.critique and not r.critique.complete)
    return 0 if errors == 0 and incomplete == 0 and summary.recall >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
