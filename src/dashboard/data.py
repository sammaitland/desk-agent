"""Data access for the dashboard.

Reads the same blotter through the same tool layer the agent uses. No new
analytics: if a figure appears here and in an agent answer, it came from the
same tested function, so the two cannot disagree. A dashboard that recomputed
its own version of alpha would eventually contradict the agent, and there would
be no way to tell which was right.

Streamlit re-runs the whole script on every interaction, so anything touching
the database is cached. `ttl` keeps it honest against a blotter that changes
underneath a long-running session.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.db import get_engine
from src.tools import dispatch

CACHE_TTL = 300  # seconds


@st.cache_resource
def engine():
    """One engine per session — cache_resource, not cache_data: connections
    are not serialisable and must not be copied per caller."""
    return get_engine()


@st.cache_data(ttl=CACHE_TTL)
def run_tool(name: str, arguments: dict) -> dict:
    """Call a tool and return its envelope. Cached on (name, arguments)."""
    with engine().connect() as conn:
        return dispatch(name, arguments, conn).as_dict()


@st.cache_data(ttl=CACHE_TTL)
def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Direct read for presentation-only queries the tool layer does not cover.

    Used for shaping display tables, never for computing a metric the agent
    could also report — those go through run_tool.
    """
    with engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.mappings().all())


@st.cache_data(ttl=CACHE_TTL)
def coverage() -> tuple[str | None, str | None]:
    df = query("SELECT MIN(run_date) AS first, MAX(run_date) AS last FROM workflow_runs")
    if df.empty or df.iloc[0]["first"] is None:
        return None, None
    return df.iloc[0]["first"], df.iloc[0]["last"]


@st.cache_data(ttl=CACHE_TTL)
def headline() -> dict:
    """Top-line numbers for the overview strip."""
    snapshot = query("""
        SELECT position_count, account_value, total_gross_exposure, leverage,
               dollar_weighted_beta, snapshot_date
        FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 1""")
    closed = query("""
        SELECT COUNT(*) AS trades,
               ROUND(CAST(SUM(final_alpha_return_pct) AS NUMERIC), 2) AS total_alpha,
               ROUND(CAST(AVG(final_alpha_return_pct) AS NUMERIC), 3) AS avg_alpha,
               SUM(CASE WHEN final_alpha_return_pct > 0 THEN 1 ELSE 0 END) AS winners
        FROM positions WHERE status = 'closed'""")
    halts = query("SELECT COUNT(*) AS n FROM system_events WHERE severity = 'halt'")

    row = closed.iloc[0]
    return {
        "snapshot": snapshot.iloc[0].to_dict() if not snapshot.empty else {},
        "trades": int(row["trades"] or 0),
        "total_alpha": float(row["total_alpha"] or 0),
        "avg_alpha": float(row["avg_alpha"] or 0),
        "win_rate": round(100 * (row["winners"] or 0) / row["trades"], 1) if row["trades"] else 0.0,
        "halts": int(halts.iloc[0]["n"] or 0),
    }


@st.cache_data(ttl=CACHE_TTL)
def alpha_over_time() -> pd.DataFrame:
    """Cumulative realised alpha by termination date."""
    df = query("""
        SELECT SUBSTR(termination_date, 1, 10) AS date,
               ROUND(CAST(SUM(final_alpha_return_pct) AS NUMERIC), 4) AS daily_alpha,
               COUNT(*) AS trades
        FROM positions
        WHERE status = 'closed' AND termination_date IS NOT NULL
        GROUP BY SUBSTR(termination_date, 1, 10) ORDER BY date""")
    if df.empty:
        return df
    df["cumulative_alpha"] = df["daily_alpha"].astype(float).cumsum()
    return df


@st.cache_data(ttl=CACHE_TTL)
def leverage_over_time() -> pd.DataFrame:
    return query("""
        SELECT snapshot_date AS date, leverage, position_count,
               dollar_weighted_beta
        FROM portfolio_snapshots ORDER BY snapshot_date""")


@st.cache_data(ttl=CACHE_TTL)
def screening_funnel() -> pd.DataFrame:
    """Where candidates die: the shape of the daily screen."""
    return query("""
        SELECT COALESCE(primary_fail_reason, rejection_reason, 'approved') AS outcome,
               COUNT(*) AS n
        FROM pair_evaluations
        GROUP BY COALESCE(primary_fail_reason, rejection_reason, 'approved')
        ORDER BY n DESC""")


def format_money(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"
