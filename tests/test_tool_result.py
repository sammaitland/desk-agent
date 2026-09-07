"""Backend-independent numerical evidence at the tool-result boundary."""

import json
from decimal import Decimal

import pytest

from src.agent.trace import Trace
from src.evals.checks import numeric_fidelity
from src.tools.base import ToolResult


def test_decimal_results_match_sqlite_numeric_types():
    result = ToolResult(
        data={"totals": {"alpha": Decimal("7.89"), "trades": 3},
              "breakdown": [{"alpha": Decimal("-2.34"), "missing": None,
                             "approved": True}]},
        provenance={"score": Decimal("0.25")},
    )
    assert type(result.data["totals"]["alpha"]) is float
    assert abs(7.89 - result.data["totals"]["alpha"]) < 0.05
    assert type(result.data["totals"]["trades"]) is int
    assert result.data["breakdown"][0]["approved"] is True
    assert result.data["breakdown"][0]["missing"] is None
    # No default=str: decimals must become JSON numbers, not quoted strings.
    restored = json.loads(json.dumps(result.as_dict()))
    assert restored["data"]["breakdown"][0]["alpha"] == -2.34
    assert restored["provenance"]["score"] == 0.25


def test_normalisation_does_not_mutate_database_payload():
    source = {"rows": [{"alpha": Decimal("7.89")} ]}
    ToolResult(data=source)
    assert isinstance(source["rows"][0]["alpha"], Decimal)


@pytest.mark.parametrize("answer,passes", [
    ("Alpha was -7.89%.", True),
    ("Alpha was +7.89%.", False),
])
def test_postgres_decimal_evidence_reaches_numeric_fidelity(answer, passes):
    trace = Trace(question="What was alpha?", answer=answer)
    # No numbers in the summary: the check must use the structured evidence.
    result = ToolResult(data={"alpha": Decimal("-7.89")}, summary="Measured alpha.")
    trace.record_tool("alpha_attribution", {}, result, duration_ms=0, turn=1)
    assert numeric_fidelity()(trace).passed is passes
