"""Blotter lookup tools.

Docstrings in this module are written for the model as much as for a human:
they become the tool descriptions passed to the API, and a vague description is
the most common cause of an agent picking the wrong tool.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.tools.base import (
    MAX_ROWS,
    ToolResult,
    date_clause,
    empty,
    resolve_window,
    rows_to_dicts,
)


def query_blotter(
    conn: Connection,
    entity: str = "positions",
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    pair: str | None = None,
    index: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> ToolResult:
    """Look up raw records from the trade blotter.

    The general-purpose lookup. Use it to find positions, orders or trading runs
    matching simple filters, or to establish what exists before drilling in with
    a more specific tool.

    entity:
      'positions' — pair trades, open and closed
      'orders'    — ticker-level orders sent to the broker
      'runs'      — daily execution runs and their outcome

    Dates are ISO (YYYY-MM-DD) and inclusive; omitting them covers the whole
    blotter. `ticker` matches either leg of a position. `status` accepts
    open/closed for positions and Filled/Partial/Failed for orders.
    """
    start, end = resolve_window(conn, start_date, end_date)
    limit = min(limit, MAX_ROWS)
    params = {"start_date": start, "end_date": end, "limit": limit}
    where = []

    if entity == "positions":
        sql = """SELECT tag, pair, co1, co2, idx, tail, status, sum_dev_bucket,
                        position_multiplier, total_notional, trade_initiation_date,
                        termination_date, holding_days, exit_reason,
                        final_alpha_return_pct, entry_spread_bps
                 FROM positions WHERE """ + date_clause("trade_initiation_date")
        if ticker:
            where.append("(co1 = :ticker OR co2 = :ticker)")
            params["ticker"] = ticker
        if pair:
            where.append("pair = :pair")
            params["pair"] = pair
        if index:
            where.append("idx = :index")
            params["index"] = index
        if status:
            where.append("status = :status")
            params["status"] = status
        order = "ORDER BY trade_initiation_date DESC"

    elif entity == "orders":
        sql = """SELECT o.order_id, o.ticker, o.side, o.order_type, o.total_shares,
                        o.filled_shares, o.status, o.spread_bps, o.arrival_mid,
                        o.limit_price, o.fell_back_to_mkt, o.fallback_reason,
                        o.elapsed_seconds, o.placed_at,
                        (SELECT price FROM fills f WHERE f.order_id = o.order_id LIMIT 1) AS fill_price
                 FROM orders o WHERE """ + date_clause("placed_at", "o")
        if ticker:
            where.append("o.ticker = :ticker")
            params["ticker"] = ticker
        if status:
            where.append("o.status = :status")
            params["status"] = status
        order = "ORDER BY o.placed_at DESC"

    elif entity == "runs":
        sql = """SELECT r.run_id, r.run_date, r.outcome, s.position_count,
                        s.account_value, s.total_gross_exposure, s.leverage,
                        s.dollar_weighted_beta
                 FROM workflow_runs r
                 LEFT JOIN portfolio_snapshots s ON s.run_id = r.run_id
                 WHERE SUBSTR(r.run_date, 1, 10) BETWEEN :start_date AND :end_date"""
        order = "ORDER BY r.run_date DESC"

    else:
        return empty(f"Unknown entity '{entity}'. Use positions, orders or runs.",
                     entity=entity)

    if where:
        sql += " AND " + " AND ".join(where)
    sql += f" {order} LIMIT :limit"

    rows = rows_to_dicts(conn.execute(text(sql), params))
    if not rows:
        return empty(f"No {entity} matched those filters.",
                     entity=entity, window=[start, end], filters=params)

    return ToolResult(
        data=rows,
        provenance={
            "entity": entity, "rows": len(rows), "window": [start, end],
            "filters": {k: v for k, v in params.items()
                        if k not in ("start_date", "end_date", "limit")},
            "truncated": len(rows) == limit,
        },
        summary=f"{len(rows)} {entity} between {start} and {end}.",
    )


def explain_position(conn: Connection, tag: str) -> ToolResult:
    """Reconstruct the full decision context for one position.

    Answers "why did the system take this trade, and what happened to it".
    Returns the entry rationale (CDF bucket, position multiplier, leg weights,
    composite score, entry spread), the execution record for both legs, the
    stop-loss state, the alpha path since entry, and the exit reason.

    `tag` is the position identifier, e.g. VGT_AAPL_MSFT_L_20260815_001.
    Use query_blotter first if you need to find the tag.
    """
    position = conn.execute(text("SELECT * FROM positions WHERE tag = :tag"), {"tag": tag}).mappings().first()
    if position is None:
        return empty(f"No position found with tag '{tag}'.", tag=tag)
    position = dict(position)

    orders = rows_to_dicts(conn.execute(text("""
        SELECT o.order_id, o.ticker, o.side, o.order_type, o.spread_bps, o.arrival_mid,
               o.limit_price, o.status, o.fell_back_to_mkt, o.fallback_reason,
               a.requested_shares, a.allocated_shares, a.allocation_status,
               (SELECT price FROM fills f WHERE f.order_id = o.order_id LIMIT 1) AS fill_price
        FROM order_allocations a
        JOIN orders o ON o.order_id = a.order_id
        WHERE a.tag = :tag ORDER BY o.placed_at"""), {"tag": tag}))

    stop = conn.execute(text("SELECT * FROM stop_orders WHERE tag = :tag"), {"tag": tag}).mappings().first()

    updates = rows_to_dicts(conn.execute(text("""
        SELECT update_date, live_alpha_return_pct, co1_return_pct, co2_return_pct,
               index_return_pct, days_held
        FROM position_updates WHERE tag = :tag ORDER BY update_date"""), {"tag": tag}))

    events = rows_to_dicts(conn.execute(text("""
        SELECT occurred_at, event_type, severity, detail
        FROM system_events WHERE tag = :tag ORDER BY occurred_at"""), {"tag": tag}))

    w1, w2 = position["w1"], position["w2"]
    long_leg = position["co1"] if position["tail"] == "L" else position["co2"]
    short_leg = position["co2"] if position["tail"] == "L" else position["co1"]

    rationale = {
        "sum_dev_bucket": position["sum_dev_bucket"],
        "sum_dev_percentile": position["sum_dev_percentile"],
        "position_multiplier": position["position_multiplier"],
        "leg_weights": {"w1": w1, "w2": w2},
        "tail": position["tail"],
        "long_leg": long_leg,
        "short_leg": short_leg,
        "composite_score": position["composite_score"],
        "weighted_score": position["weighted_score"],
        "entry_spread_bps": position["entry_spread_bps"],
        "beta": position["beta"],
    }

    return ToolResult(
        data={
            "position": position,
            "entry_rationale": rationale,
            "execution": orders,
            "stop_loss": dict(stop) if stop else None,
            "alpha_path": updates,
            "related_events": events,
        },
        provenance={
            "tag": tag, "orders": len(orders), "daily_marks": len(updates),
            "events": len(events),
        },
        summary=(
            f"{tag}: {position['status']}"
            + (f", exited via {position['exit_reason']} after {position['holding_days']}d "
               f"at {position['final_alpha_return_pct']}% alpha"
               if position["status"] == "closed" else ", still open")
            + f". Bucket {position['sum_dev_bucket']} at {position['position_multiplier']}x."
        ),
    )


def explain_rejection(
    conn: Connection,
    pair: str | None = None,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> ToolResult:
    """Explain why candidate pairs were screened out rather than traded.

    Answers "why wasn't X traded". Every pair is evaluated on every run; this
    returns the gate that stopped it and the value it failed on, whether that
    was a primary filter (spread hurdle, earnings proximity, trend, direction,
    t-stat) or trade evaluation (untradeable CDF bucket, leverage, index
    concentration, factor shock, duplicate, position size, per-ticker cap).

    Filter by `pair` ('AAPL_MSFT'), by `ticker` (either leg), or neither to see
    the whole rejection picture for a date window.
    """
    start, end = resolve_window(conn, start_date, end_date)
    params = {"start_date": start, "end_date": end, "limit": min(limit, MAX_ROWS)}
    where = [date_clause("evaluated_at")]

    if pair:
        where.append("pair = :pair")
        params["pair"] = pair
    if ticker:
        where.append("(co1 = :ticker OR co2 = :ticker)")
        params["ticker"] = ticker

    sql = f"""
        SELECT evaluated_at, pair, idx, tail, primary_result, primary_fail_reason,
               weighted_spread_bps, earnings_days_out, co1_trending, co2_trending,
               tstat, sum_dev_bucket, sum_dev_percentile, position_multiplier,
               is_tradeable_bucket, composite_score, composite_priority_score,
               shocked_factors, factor_action, evaluation_result, rejection_reason
        FROM pair_evaluations
        WHERE {' AND '.join(where)}
          AND (primary_result = 'Fail' OR evaluation_result = 'Rejected')
        ORDER BY evaluated_at DESC LIMIT :limit"""

    rows = rows_to_dicts(conn.execute(text(sql), params))
    if not rows:
        return empty("No rejections matched those filters — the pair may have been "
                     "approved, or not evaluated in this window.",
                     window=[start, end], filters={"pair": pair, "ticker": ticker})

    tally: dict[str, int] = {}
    for row in rows:
        reason = row["primary_fail_reason"] or row["rejection_reason"] or "unknown"
        tally[reason] = tally.get(reason, 0) + 1
    top = max(tally, key=tally.get)

    return ToolResult(
        data=rows,
        provenance={
            "rows": len(rows), "window": [start, end],
            "filters": {"pair": pair, "ticker": ticker},
            "reason_counts": tally, "truncated": len(rows) == params["limit"],
        },
        summary=(f"{len(rows)} rejections between {start} and {end}; "
                 f"most common reason: {top} ({tally[top]})."),
    )
