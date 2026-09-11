"""MCP server exposing the blotter tools.

The same functions the agent loop calls, published over the Model Context
Protocol so any MCP host — Claude Desktop, an IDE, another agent — can query
the blotter directly.

This file is deliberately thin. The tool layer already had typed signatures,
explicit schemas and docstrings written for a model to read, so publishing it
over a second transport is a wrapper rather than a rewrite. That was the point
of separating orchestration from computation: the transport is not the
architecture.

Run it:

    python -m src.mcp_server.server        # stdio, for Claude Desktop
    mcp dev src/mcp_server/server.py       # MCP Inspector

Built against mcp 2.x, where FastMCP was renamed MCPServer. Pin `mcp>=2` —
v1 code will not import.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from src import env  # noqa: F401  (loads .env on import)
from src.db import get_engine
from src.tools import dispatch

mcp = MCPServer("desk-agent-blotter")

_engine = None


def _connection():
    """Lazy engine, so importing this module does not touch the database."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine.connect()


def _run(tool: str, **arguments: Any) -> dict:
    """Execute a tool and return its envelope.

    Every result carries `summary`, `provenance` and `data` — the same shape
    the agent loop receives. Provenance travels with the result so an MCP host
    can report scope as honestly as the in-house agent does.
    """
    clean = {k: v for k, v in arguments.items() if v is not None}
    with _connection() as conn:
        return dispatch(tool, clean, conn).as_dict()


# --- lookup ---------------------------------------------------------------

@mcp.tool()
def query_blotter(
    entity: str = "positions",
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    pair: str | None = None,
    index: str | None = None,
    status: str | None = None,
    limit: int = 50,
    date_basis: str | None = None,
) -> dict:
    """Look up raw records from the trade blotter.

    The general-purpose lookup: find positions, orders or daily runs by simple
    filters, typically to obtain a position tag or order id before drilling in
    with a more specific tool. Not for execution cost, rejection reasons or
    alpha breakdowns — dedicated tools cover those.

    Args:
        entity: 'positions' (pair trades), 'orders' (ticker-level broker
            orders) or 'runs' (daily execution runs).
        start_date: Inclusive ISO date. Omit for the earliest data held.
        end_date: Inclusive ISO date. Omit for the latest data held.
        ticker: Single ticker; for positions, matches either leg.
        pair: Pair identifier, e.g. AAPL_MSFT.
        index: Sector index, e.g. VGT, VFH, VIS, VHT, VCR, VOX.
        status: positions: open|closed. orders: Filled|Partial|Failed.
        limit: Maximum rows returned.
        date_basis: positions only: trade_initiation_date (default) or termination_date.
    """
    return _run("query_blotter", entity=entity, start_date=start_date,
                end_date=end_date, ticker=ticker, pair=pair, index=index,
                status=status, limit=limit, date_basis=date_basis)


@mcp.tool()
def query_records(entity: str, start_date: str | None = None, end_date: str | None = None,
                  record_id: str | None = None, subject: str | None = None,
                  check_name: str | None = None, result: str | None = None,
                  tag: str | None = None, status: str | None = None, limit: int = 50) -> dict:
    """Read authoritative checks (Pass and Fail) or stops with trigger timestamps.

    entity=risk_checks filters checked_at; subject/check_name/result select checks.
    entity=stop_orders filters triggered_at; tag/status select stops. record_id
    is a check_id or stop_order_tag. Count covers all matches even if records
    are truncated. Omitted dates cover this table's timestamps. Rejections and
    entry-date position cohorts cannot substitute for these records.
    """
    return _run("query_records", entity=entity, start_date=start_date, end_date=end_date,
                record_id=record_id, subject=subject, check_name=check_name,
                result=result, tag=tag, status=status, limit=limit)


@mcp.tool()
def explain_position(tag: str) -> dict:
    """Reconstruct the full decision context for one position.

    Returns why the trade was taken (CDF bucket, position multiplier, leg
    weights, composite score, entry spread), how both legs executed, the
    stop-loss state, the alpha path since entry, and the exit reason. Resolves
    the L/U tail into explicit long and short legs.

    Args:
        tag: Position identifier, e.g. VGT_AAPL_MSFT_L_20260815_001. Use
            query_blotter first if you do not have one.
    """
    return _run("explain_position", tag=tag)


@mcp.tool()
def explain_rejection(
    pair: str | None = None,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    """Explain why candidate pairs were screened out instead of traded.

    Returns the gate that stopped each pair and the value it failed on —
    primary filters (spread hurdle, earnings proximity, trending, direction
    checks, t-stat) or trade evaluation (untradeable CDF bucket, leverage,
    index concentration, factor shock, duplicate, position size, per-ticker
    cap).

    Returns rejections ONLY. A pair absent from the results may have been
    approved, or may not have been evaluated — absence is not evidence that
    nothing happened.

    Args:
        pair: Pair identifier, e.g. AAPL_MSFT.
        ticker: Ticker matching either leg.
        start_date: Inclusive ISO date.
        end_date: Inclusive ISO date.
        limit: Maximum rows returned.
    """
    return _run("explain_rejection", pair=pair, ticker=ticker,
                start_date=start_date, end_date=end_date, limit=limit)


# --- analytics ------------------------------------------------------------

@mcp.tool()
def execution_quality(
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str | None = None,
    order_id: str | None = None,
    group_by: str | None = None,
) -> dict:
    """Transaction cost analysis: slippage, spreads, fill rates and routing.

    Measures slippage against the arrival mid (the benchmark price when the
    order was raised), effective spread paid, fill rates, limit-order timeout
    fallbacks, and the limit-versus-market routing split. Positive slippage
    means the fill was worse than the arrival mid.

    Args:
        start_date: Inclusive ISO date.
        end_date: Inclusive ISO date.
        ticker: Restrict to one ticker.
        order_id: Return the full causal detail for a single order — the right
            call when investigating one bad fill.
        group_by: 'ticker', 'order_type' or 'date'. Omit for a single total.
    """
    return _run("execution_quality", start_date=start_date, end_date=end_date,
                ticker=ticker, order_id=order_id, group_by=group_by)


@mcp.tool()
def alpha_attribution(
    start_date: str | None = None,
    end_date: str | None = None,
    group_by: str = "idx",
    status: str = "closed",
    date_basis: str = "trade_initiation_date",
) -> dict:
    """Break down realised alpha across the book.

    Alpha is index-relative and market-neutral by construction
    (W1*co1_return - W2*co2_return - beta*index_return). It is NOT profit and
    loss and should not be described as such. Returns trade counts, mean and
    total alpha, win rate and average holding period per group.

    Args:
        start_date: Inclusive ISO date.
        end_date: Inclusive ISO date.
        group_by: 'idx', 'sum_dev_bucket', 'tail', 'exit_reason' or 'month'.
        status: Position status; only closed trades carry a final alpha.
        date_basis: trade_initiation_date (default) selects entry cohorts; termination_date selects exits.
    """
    return _run("alpha_attribution", start_date=start_date, end_date=end_date,
                group_by=group_by, status=status, date_basis=date_basis)


@mcp.tool()
def detect_anomalies(
    start_date: str | None = None,
    end_date: str | None = None,
    severity: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
) -> dict:
    """Find operational problems: what went wrong and when.

    Covers system events (stale data, reconciliation mismatches, order
    timeouts, partial fills, spread validation failures, connection errors,
    delistings, orphaned legs after a stop trigger) and failed risk checks
    (leverage, index concentration, factor exposure, portfolio beta).

    Args:
        start_date: Inclusive ISO date.
        end_date: Inclusive ISO date.
        severity: 'info', 'warning' or 'halt'. A halt means trading stopped.
        event_type: Canonical label, e.g. partial_fill. Unknown labels return errors.
            Stop triggers are queried through query_records(entity=stop_orders).
        limit: Maximum events returned.
    """
    return _run("detect_anomalies", start_date=start_date, end_date=end_date,
                severity=severity, event_type=event_type, limit=limit)


# --- documentation --------------------------------------------------------

@mcp.tool()
def search_documentation(query: str, k: int = 4) -> dict:
    """Search the trading system's design documentation.

    Answers WHY the system is built the way it is — the rationale behind a
    filter or threshold, what a gate protects against, how calibration and
    implementation relate, or what a term means. Returns ranked passages, each
    with a citation (file and section) and a relevance score.

    Counterpart to the blotter tools, which answer what HAPPENED.

    Args:
        query: The question or topic, in natural language.
        k: Number of passages to return.
    """
    from src.rag.tool import search_documentation as _search

    return _search(query=query, k=k).as_dict()


# --- resources ------------------------------------------------------------

@mcp.resource("blotter://coverage")
def coverage() -> str:
    """The date range and size of the blotter currently loaded."""
    from sqlalchemy import text

    from src.tools.base import blotter_date_range

    with _connection() as conn:
        first, last = blotter_date_range(conn)
        positions = conn.execute(text("SELECT COUNT(*) FROM positions")).scalar_one()
        orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar_one()
    if not first:
        return "The blotter is empty."
    return (f"Blotter covers {first} to {last}: "
            f"{positions} positions, {orders} orders. Read-only.")


def main() -> None:
    # stdio transport: the host launches this as a subprocess and speaks
    # JSON-RPC over the pipes. No port, no network exposure.
    mcp.run()


if __name__ == "__main__":
    main()
