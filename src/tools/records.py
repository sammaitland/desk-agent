"""Direct, read-only access to checks and stops, including successful checks.

Counts describe the complete filtered population; records are a bounded page.
Neither an absent rejection nor an entry-date cohort substitutes for these rows.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.tools.base import ToolResult, clamp_limit, date_clause, empty, invalid_date_window, rows_to_dicts


def query_records(conn: Connection, entity: str, start_date: str | None = None,
                  end_date: str | None = None, record_id: str | None = None,
                  subject: str | None = None, check_name: str | None = None,
                  result: str | None = None, tag: str | None = None,
                  status: str | None = None, limit: int = 50) -> ToolResult:
    """Retrieve risk_checks by checked_at or stop_orders by triggered_at.

    Omitted dates cover all non-null timestamps in this entity, independently
    of the runs/positions window. Stop rows without a trigger have no trigger
    date and are excluded. Query by record_id to disambiguate repeated checks.
    """
    date_error = invalid_date_window(start_date, end_date)
    if date_error:
        return empty(f"Invalid arguments: {date_error}.", error=True)
    if entity not in ("risk_checks", "stop_orders"):
        return empty("Invalid arguments: entity must be risk_checks or stop_orders.", error=True)
    filters = {"record_id": record_id, "subject": subject, "check_name": check_name,
               "result": result, "tag": tag, "status": status}
    allowed = ({"record_id": "check_id", "subject": "subject", "check_name": "check_name",
                "result": "result"} if entity == "risk_checks" else
               {"record_id": "stop_order_tag", "tag": "tag", "status": "status"})
    if any(v is not None and not isinstance(v, str) for v in filters.values()):
        return empty("Invalid arguments: record filters must be strings.", error=True)
    unused = [k for k, v in filters.items() if v is not None and k not in allowed]
    if unused:
        return empty(f"Invalid arguments for {entity}: {', '.join(unused)}.", error=True)
    if result is not None and result not in ("Pass", "Fail"):
        return empty("Invalid arguments: result must be Pass or Fail.", error=True)
    if status is not None and status not in ("active", "triggered", "cancelled"):
        return empty("Invalid arguments: status must be active, triggered or cancelled.", error=True)
    basis = "checked_at" if entity == "risk_checks" else "triggered_at"
    first, last = conn.execute(text(
        f"SELECT MIN(SUBSTR({basis},1,10)), MAX(SUBSTR({basis},1,10)) FROM {entity}"
    )).one()
    start, end = start_date or first or "1900-01-01", end_date or last or "2999-12-31"
    if start > end:
        return empty("Invalid arguments: start_date must not follow end_date.", error=True)
    params = {"start_date": start, "end_date": end, "limit": clamp_limit(limit)}
    where = [date_clause(basis)]
    for key, column in allowed.items():
        if filters[key] is not None:
            where.append(f"{column} = :{key}")
            params[key] = filters[key]
    predicate = " AND ".join(where)
    count = conn.execute(text(f"SELECT COUNT(*) FROM {entity} WHERE {predicate}"), params).scalar_one()
    rows = rows_to_dicts(conn.execute(text(
        f"SELECT * FROM {entity} WHERE {predicate} ORDER BY {basis}, {allowed['record_id']} LIMIT :limit"
    ), params))
    scope = {"population": entity, "date_basis": basis, "window": [start, end],
             "filters": {k: v for k, v in filters.items() if v is not None}}
    return ToolResult(
        data={"records": rows, "count": count},
        provenance={**scope, "rows": len(rows), "count_complete": True,
                    "filters_validated": True, "truncated": count > len(rows)},
        summary=f"{count} {entity} by {basis} between {start} and {end}; returned {len(rows)} records.",
    )
