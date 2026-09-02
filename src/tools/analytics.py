"""Analytical tools: execution quality, alpha attribution, anomaly detection.

All arithmetic lives here rather than in the agent. Slippage, fill rates and
attribution are computed in SQL or Python and returned as facts; the model's
job is to decide which of these to run and how to narrate the result.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src import config as cfg
from src.tools.base import (
    MAX_ROWS,
    ToolResult,
    as_date,
    date_clause,
    empty,
    pct,
    resolve_window,
    rounded,
    rows_to_dicts,
)

# Slippage is signed by direction: positive means the fill was worse than the
# arrival mid (paid up on a buy, sold down on a sell).
SLIPPAGE_BPS = ("(CASE WHEN o.side = 'BUY' THEN 1 ELSE -1 END) "
                "* (f.price - o.arrival_mid) / o.arrival_mid * 10000")


def execution_quality(
    conn: Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    order_id: str | None = None,
    group_by: str | None = None,
) -> ToolResult:
    """Measure execution cost and fill quality — transaction cost analysis.

    Computes slippage against the arrival mid (the benchmark price when the
    order was raised), effective spread paid, fill rates, limit-order timeout
    fallback rates, and the split between limit and market routing.

    Use for "why did X fill badly", "how are our fills generally", or "is limit
    or market routing costing us more". `group_by` accepts 'ticker',
    'order_type' or 'date' to break results down; omit it for a single
    aggregate. `order_id` returns the detail for one specific order.

    Positive slippage means the fill was worse than the arrival mid.
    """
    start, end = resolve_window(conn, start_date, end_date)
    params = {"start_date": start, "end_date": end}
    where = [date_clause("placed_at", "o")]

    if ticker:
        where.append("o.ticker = :ticker")
        params["ticker"] = ticker
    if order_id:
        where.append("o.order_id = :order_id")
        params["order_id"] = order_id

    # Single-order detail: return the full causal chain, not an average.
    if order_id:
        row = conn.execute(text(f"""
            SELECT o.order_id, o.ticker, o.side, o.order_type, o.total_shares,
                   o.filled_shares, o.status, o.spread_bps, o.arrival_mid, o.bid, o.ask,
                   o.limit_price, o.fell_back_to_mkt, o.fallback_reason,
                   o.elapsed_seconds, o.placed_at, f.price AS fill_price,
                   f.commission, ROUND(CAST({SLIPPAGE_BPS} AS NUMERIC), 2) AS slippage_bps
            FROM orders o LEFT JOIN fills f ON f.order_id = o.order_id
            WHERE o.order_id = :order_id"""), params).mappings().first()
        if row is None:
            return empty(f"No order found with id '{order_id}'.", order_id=order_id)
        row = dict(row)
        row["exceeded_limit_spread_cap"] = (
            row["spread_bps"] is not None and row["spread_bps"] > cfg.MAX_LIMIT_ORDER_SPREAD_BPS
        )
        row["limit_spread_cap_bps"] = cfg.MAX_LIMIT_ORDER_SPREAD_BPS
        row["limit_timeout_seconds"] = cfg.LIMIT_ORDER_TIMEOUT
        return ToolResult(
            data=row,
            provenance={"order_id": order_id, "rows": 1},
            summary=(f"{row['ticker']} {row['side']} {row['order_type']}: "
                     f"{row['slippage_bps']}bps slippage on a {row['spread_bps']}bps spread."),
        )

    group_sql = {"ticker": "o.ticker", "order_type": "o.order_type",
                 "date": as_date("o.placed_at")}
    if group_by and group_by not in group_sql:
        return empty(f"Unknown group_by '{group_by}'. Use ticker, order_type or date.")

    select_group = f"{group_sql[group_by]} AS grouping," if group_by else ""
    group_clause = f"GROUP BY {group_sql[group_by]}" if group_by else ""
    order_clause = "ORDER BY orders_filled DESC" if group_by else ""

    rows = rows_to_dicts(conn.execute(text(f"""
        SELECT {select_group}
               COUNT(*) AS orders_filled,
               ROUND(CAST(AVG({SLIPPAGE_BPS}) AS NUMERIC), 2) AS avg_slippage_bps,
               ROUND(CAST(MAX({SLIPPAGE_BPS}) AS NUMERIC), 2) AS worst_slippage_bps,
               ROUND(CAST(AVG(o.spread_bps) AS NUMERIC), 2) AS avg_spread_bps,
               SUM(o.fell_back_to_mkt) AS timeout_fallbacks,
               SUM(CASE WHEN o.order_type = 'LMT' THEN 1 ELSE 0 END) AS limit_orders,
               SUM(CASE WHEN o.order_type = 'MKT' THEN 1 ELSE 0 END) AS market_orders,
               ROUND(CAST(SUM(f.commission) AS NUMERIC), 2) AS total_commission
        FROM orders o JOIN fills f ON f.order_id = o.order_id
        WHERE {' AND '.join(where)}
        {group_clause} {order_clause} LIMIT {MAX_ROWS}"""), params))

    if not rows:
        return empty("No filled orders matched those filters.",
                     window=[start, end], filters={"ticker": ticker})

    totals = conn.execute(text(f"""
        SELECT COUNT(*) AS total_orders,
               SUM(CASE WHEN o.status = 'Filled' THEN 1 ELSE 0 END) AS filled,
               SUM(CASE WHEN o.status = 'Partial' THEN 1 ELSE 0 END) AS partial,
               SUM(CASE WHEN o.status = 'Failed' THEN 1 ELSE 0 END) AS failed
        FROM orders o WHERE {' AND '.join(where)}"""), params).mappings().one()
    totals = dict(totals)
    totals["fill_rate_pct"] = pct(totals["filled"], totals["total_orders"])

    headline = rows[0] if not group_by else None
    summary = (
        f"{totals['total_orders']} orders {start} to {end}: "
        f"{totals['fill_rate_pct']}% filled"
        + (f", average slippage {headline['avg_slippage_bps']}bps, "
           f"worst {headline['worst_slippage_bps']}bps." if headline else ".")
    )

    return ToolResult(
        data={"breakdown": rows, "totals": totals},
        provenance={
            "window": [start, end], "group_by": group_by,
            "filters": {"ticker": ticker}, "rows": len(rows),
            "benchmark": "arrival_mid",
            "limit_spread_cap_bps": cfg.MAX_LIMIT_ORDER_SPREAD_BPS,
        },
        summary=summary,
    )


def alpha_attribution(
    conn: Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    group_by: str = "idx",
    status: str = "closed",
) -> ToolResult:
    """Break down realised alpha across the book.

    Alpha here is index-relative and market-neutral by construction:
    W1*co1_return - W2*co2_return - beta*index_return. It is not raw P&L, and
    should not be described as such.

    `group_by` accepts 'idx' (sector index), 'sum_dev_bucket' (CDF bucket),
    'tail' (L/U), 'exit_reason', or 'month'. Returns trade counts, mean and
    total alpha, win rate and average holding period per group.
    """
    start, end = resolve_window(conn, start_date, end_date)
    columns = {
        "idx": "idx", "sum_dev_bucket": "sum_dev_bucket", "tail": "tail",
        "exit_reason": "exit_reason", "month": "SUBSTR(termination_date, 1, 7)",
    }
    if group_by not in columns:
        return empty(f"Unknown group_by '{group_by}'. Use "
                     f"{', '.join(columns)}.")

    params = {"start_date": start, "end_date": end}
    where = [date_clause("trade_initiation_date"), "final_alpha_return_pct IS NOT NULL"]
    if status:
        where.append("status = :status")
        params["status"] = status

    rows = rows_to_dicts(conn.execute(text(f"""
        SELECT {columns[group_by]} AS grouping,
               COUNT(*) AS trades,
               ROUND(CAST(AVG(final_alpha_return_pct) AS NUMERIC), 3) AS avg_alpha_pct,
               ROUND(CAST(SUM(final_alpha_return_pct) AS NUMERIC), 2) AS total_alpha_pct,
               ROUND(CAST(MIN(final_alpha_return_pct) AS NUMERIC), 2) AS worst_alpha_pct,
               ROUND(CAST(MAX(final_alpha_return_pct) AS NUMERIC), 2) AS best_alpha_pct,
               SUM(CASE WHEN final_alpha_return_pct > 0 THEN 1 ELSE 0 END) AS winners,
               ROUND(CAST(AVG(holding_days) AS NUMERIC), 1) AS avg_holding_days,
               ROUND(CAST(AVG(total_notional) AS NUMERIC), 0) AS avg_notional
        FROM positions WHERE {' AND '.join(where)}
        GROUP BY {columns[group_by]}
        ORDER BY total_alpha_pct DESC LIMIT {MAX_ROWS}"""), params))

    if not rows:
        return empty("No closed positions matched those filters.",
                     window=[start, end], group_by=group_by)

    for row in rows:
        row["win_rate_pct"] = pct(row["winners"], row["trades"])

    trades = sum(r["trades"] for r in rows)
    total = round(sum(r["total_alpha_pct"] for r in rows), 2)
    best, worst = rows[0], rows[-1]

    return ToolResult(
        data={"breakdown": rows,
              "totals": {"trades": trades, "total_alpha_pct": total,
                         "avg_alpha_pct": round(total / trades, 3) if trades else 0.0}},
        provenance={
            "window": [start, end], "group_by": group_by, "status": status,
            "rows": len(rows),
            "measure": "index-relative alpha (W1*co1 - W2*co2 - beta*index)",
        },
        summary=(f"{trades} closed trades {start} to {end}, total alpha {total}%. "
                 f"Best {group_by}: {best['grouping']} ({best['total_alpha_pct']}%); "
                 f"worst: {worst['grouping']} ({worst['total_alpha_pct']}%)."),
    )


def detect_anomalies(
    conn: Connection,
    start_date: str | None = None,
    end_date: str | None = None,
    severity: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
) -> ToolResult:
    """Find operational problems: what went wrong and when.

    Covers system events (stale data, reconciliation mismatches, order
    timeouts, partial fills, spread validation failures, connection errors,
    delistings, orphaned legs after a stop trigger) and failed risk checks
    (leverage, index concentration, factor exposure, portfolio beta).

    Use for "what went wrong yesterday", "have we had any halts", or as the
    first call when a question implies something broke. `severity` accepts
    info, warning or halt.
    """
    start, end = resolve_window(conn, start_date, end_date)
    params = {"start_date": start, "end_date": end, "limit": min(limit, MAX_ROWS)}
    where = [date_clause("occurred_at")]

    if severity:
        where.append("severity = :severity")
        params["severity"] = severity
    if event_type:
        where.append("event_type = :event_type")
        params["event_type"] = event_type

    events = rows_to_dicts(conn.execute(text(f"""
        SELECT occurred_at, event_type, severity, ticker, tag, order_id,
               detail, remedy_action, resolved, resolution
        FROM system_events WHERE {' AND '.join(where)}
        ORDER BY CASE severity WHEN 'halt' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                 occurred_at DESC
        LIMIT :limit"""), params))

    failed_checks = rows_to_dicts(conn.execute(text("""
        SELECT checked_at, check_name, subject, current_value, threshold, action
        FROM risk_checks
        WHERE SUBSTR(checked_at, 1, 10) BETWEEN :start_date AND :end_date AND result = 'Fail'
        ORDER BY checked_at DESC LIMIT 50"""),
        {"start_date": start, "end_date": end}))

    halted_runs = rows_to_dicts(conn.execute(text("""
        SELECT run_id, run_date, outcome FROM workflow_runs
        WHERE run_date BETWEEN :start_date AND :end_date AND outcome <> 'completed'
        ORDER BY run_date DESC"""),
        {"start_date": start, "end_date": end}))

    if not events and not failed_checks and not halted_runs:
        return empty(f"No anomalies recorded between {start} and {end}.",
                     window=[start, end])

    tally: dict[str, int] = {}
    for event in events:
        tally[event["event_type"]] = tally.get(event["event_type"], 0) + 1
    halts = sum(1 for e in events if e["severity"] == "halt")

    return ToolResult(
        data={"events": events, "failed_risk_checks": failed_checks,
              "halted_runs": halted_runs},
        provenance={
            "window": [start, end], "rows": len(events),
            "event_type_counts": tally,
            "failed_risk_checks": len(failed_checks),
            "halted_runs": len(halted_runs),
            "filters": {"severity": severity, "event_type": event_type},
            "truncated": len(events) == params["limit"],
        },
        summary=(f"{len(events)} events between {start} and {end} "
                 f"({halts} halts), {len(failed_checks)} failed risk checks, "
                 f"{len(halted_runs)} incomplete runs."),
    )
