"""Shared plumbing for the tool layer.

Every tool returns the same envelope: structured `data`, a machine-readable
`provenance` block, and a one-line `summary`. The agent composes narrative from
`data`; `provenance` is what lets it say "across 47 orders in that window"
rather than implying a scope it never checked.

Design rule for this whole package: the agent orchestrates, these functions
compute. No tool here returns a number the model could have guessed, and no
tool writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

MAX_ROWS = 500  # hard cap: keeps tool results inside a sane context budget


@dataclass
class ToolResult:
    """Uniform return type for every tool."""

    data: Any
    provenance: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "provenance": self.provenance, "data": self.data}


def empty(reason: str, **provenance) -> ToolResult:
    """A well-formed 'nothing found' result.

    Returned rather than raising so the agent can say so plainly instead of
    receiving an error it may narrate as a system fault.
    """
    return ToolResult(data=[], provenance={"rows": 0, **provenance}, summary=reason)


def rows_to_dicts(result) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in result]


def blotter_date_range(conn: Connection) -> tuple[str | None, str | None]:
    """Earliest and latest run dates held in the blotter."""
    row = conn.execute(text("SELECT MIN(run_date), MAX(run_date) FROM workflow_runs")).one()
    return row[0], row[1]


def resolve_window(conn: Connection, start_date: str | None, end_date: str | None) -> tuple[str, str]:
    """Fill either side of a date window from what the blotter actually holds."""
    first, last = blotter_date_range(conn)
    return (start_date or first or "1900-01-01", end_date or last or "2999-12-31")


def date_clause(column: str, alias: str = "") -> str:
    """A date-window predicate that works on SQLite and PostgreSQL alike.

    Uses SUBSTR rather than DATE(): SQLite coerces text through DATE(), but
    PostgreSQL has no date(text) function and errors. Timestamps are stored as
    ISO strings, so the first ten characters are the date and lexical
    comparison is chronological.
    """
    prefix = f"{alias}." if alias else ""
    return f"SUBSTR({prefix}{column}, 1, 10) BETWEEN :start_date AND :end_date"


def as_date(column: str) -> str:
    """Portable date extraction from an ISO timestamp column."""
    return f"SUBSTR({column}, 1, 10)"


def rounded(expression: str, places: int = 2) -> str:
    """Portable ROUND.

    PostgreSQL defines round(numeric, int) but not round(double precision,
    int), so a float expression must be cast first. SQLite accepts the cast
    and behaves identically.
    """
    return f"ROUND(CAST({expression} AS NUMERIC), {places})"


def clamp_limit(limit) -> int:
    """Coerce a requested row limit into [1, MAX_ROWS].

    `min(limit, MAX_ROWS)` alone accepted negative numbers, and SQLite treats
    LIMIT -1 as no limit at all — external review returned 1,298 rows with
    truncated=False. Non-integers and zero are also invalid; all collapse to
    the safe range rather than raising, so a bad argument degrades to a
    bounded result instead of an error the model must recover from.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return MAX_ROWS
    return max(1, min(value, MAX_ROWS))


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0
