"""Desk agent dashboard.

The standing view over the blotter, complementing the ask-a-question agent.
Same database, same tool layer, different mode of access: a dashboard answers
questions you knew to ask in advance, the agent answers the ones you did not.

    streamlit run src/dashboard/app.py

Streamlit re-runs this entire file on every interaction, which is why the data
layer is cached and why the code reads as a linear script rather than as event
handlers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import env  # noqa: F401  (loads .env)
from src.dashboard import data

st.set_page_config(page_title="Desk Agent", page_icon="▲", layout="wide")


def main() -> None:
    """Render the dashboard.

    The body lives in a function rather than at module level so that importing
    this module does not execute it. A Streamlit script runs top to bottom, so
    module-level code hits the database on import — which breaks CI, static
    analysis and any tool that walks the package. CI's import check caught
    exactly that: "no such table: workflow_runs" on a fresh checkout.
    """
    # --- header -----------------------------------------------------------
    first, last = data.coverage()
    if first is None:
        st.error("The blotter is empty. Run: `python -m src.generate_blotter --days 120 --seed 42`")
        st.stop()

    st.title("Desk Agent")
    st.caption(f"Systematic pairs-trading blotter · {first} to {last} · read-only")

    metrics = data.headline()
    snapshot = metrics["snapshot"]

    columns = st.columns(6)
    columns[0].metric("Open positions", int(snapshot.get("position_count", 0)))
    columns[1].metric("Leverage", f"{float(snapshot.get('leverage', 0)):.2f}x",
                      help="Cap 1.9x, emergency halt at 1.8x")
    columns[2].metric("Account value", data.format_money(snapshot.get("account_value")))
    columns[3].metric("Closed trades", metrics["trades"])
    columns[4].metric("Total alpha", f"{metrics['total_alpha']:.1f}%",
                      help="Index-relative: W1·co1 − W2·co2 − β·index. Not P&L.")
    columns[5].metric("Win rate", f"{metrics['win_rate']:.1f}%")

    overview, performance, execution, screening, ask = st.tabs(
        ["Overview", "Performance", "Execution", "Screening", "Ask"]
    )


    # --- overview -------------------------------------------------------------

    with overview:
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Cumulative alpha")
            alpha = data.alpha_over_time()
            if alpha.empty:
                st.info("No closed positions yet.")
            else:
                st.line_chart(alpha.set_index("date")["cumulative_alpha"], height=280)
                st.caption("Realised alpha on closed trades, index-relative.")

            st.subheader("Leverage and positions")
            leverage = data.leverage_over_time()
            if not leverage.empty:
                st.line_chart(leverage.set_index("date")[["leverage", "position_count"]],
                              height=220)
                st.caption("Leverage cap 1.9x; emergency halt 1.8x.")

        with right:
            st.subheader("Recent operational events")
            events = data.run_tool("detect_anomalies", {"limit": 40})
            rows = events["data"].get("events", []) if isinstance(events["data"], dict) else []
            if not rows:
                st.success("No events recorded.")
            else:
                counts = events["provenance"].get("event_type_counts", {})
                st.bar_chart(pd.Series(counts).sort_values(ascending=False), height=200)
                halts = [e for e in rows if e["severity"] == "halt"]
                if halts:
                    st.error(f"{len(halts)} halt(s) in the window")
                    for event in halts[:3]:
                        st.caption(f"**{event['occurred_at'][:10]}** — {event['detail']}")


    # --- performance ----------------------------------------------------------

    with performance:
        dimension = st.selectbox(
            "Attribute alpha by",
            ["idx", "sum_dev_bucket", "tail", "exit_reason", "month"],
            format_func=lambda d: {
                "idx": "Sector index", "sum_dev_bucket": "CDF bucket",
                "tail": "Tail (L/U)", "exit_reason": "Exit reason", "month": "Month",
            }[d],
        )

        result = data.run_tool("alpha_attribution", {"group_by": dimension})
        breakdown = result["data"].get("breakdown", []) if isinstance(result["data"], dict) else []

        if not breakdown:
            st.info("No closed positions to attribute.")
        else:
            df = pd.DataFrame(breakdown)
            left, right = st.columns([2, 3])
            with left:
                st.bar_chart(df.set_index("grouping")["total_alpha_pct"], height=320)
            with right:
                st.dataframe(
                    df[["grouping", "trades", "avg_alpha_pct", "total_alpha_pct",
                        "win_rate_pct", "avg_holding_days"]],
                    width='stretch', hide_index=True,
                )
            st.caption(result["summary"])

            if dimension == "sum_dev_bucket":
                st.info(
                    "The 40–70% buckets are disabled at a 0.0x multiplier, so they "
                    "never appear. Edge should concentrate in the extremes — that is "
                    "what the position multipliers are sized against."
                )


    # --- execution ------------------------------------------------------------

    with execution:
        grouping = st.radio("Group by", ["order_type", "ticker", "date"],
                            horizontal=True, format_func=str.title)
        quality = data.run_tool("execution_quality", {"group_by": grouping})
        payload = quality["data"]

        if not isinstance(payload, dict) or not payload.get("breakdown"):
            st.info("No filled orders in the window.")
        else:
            totals = payload["totals"]
            columns = st.columns(4)
            columns[0].metric("Orders", totals["total_orders"])
            columns[1].metric("Fill rate", f"{totals['fill_rate_pct']}%")
            columns[2].metric("Partial", totals["partial"])
            columns[3].metric("Failed", totals["failed"])

            df = pd.DataFrame(payload["breakdown"])
            st.dataframe(
                df[["grouping", "orders_filled", "avg_slippage_bps", "worst_slippage_bps",
                    "avg_spread_bps", "timeout_fallbacks", "total_commission"]],
                width='stretch', hide_index=True,
            )
            st.caption("Positive slippage means the fill was worse than the arrival mid. "
                       "Limit orders above a 24bps spread are routed to market.")

            st.subheader("Worst fills")
            worst = data.query("""
                SELECT o.ticker, o.side, o.order_type, o.spread_bps,
                       SUBSTR(o.placed_at, 1, 10) AS date, o.fallback_reason,
                       ROUND(CAST(ABS(f.price - o.arrival_mid) / o.arrival_mid * 10000
                                  AS NUMERIC), 1) AS slippage_bps
                FROM orders o JOIN fills f ON f.order_id = o.order_id
                ORDER BY slippage_bps DESC LIMIT 10""")
            st.dataframe(worst, width='stretch', hide_index=True)


    # --- screening ------------------------------------------------------------

    with screening:
        st.subheader("Where candidates are rejected")
        funnel = data.screening_funnel()
        if funnel.empty:
            st.info("No evaluations recorded.")
        else:
            left, right = st.columns([3, 2])
            with left:
                st.bar_chart(funnel.set_index("outcome")["n"], height=320)
            with right:
                st.dataframe(funnel, width='stretch', hide_index=True)
            st.caption(
                "Primary filters (spread hurdle, earnings, trend, direction, t-stat) "
                "run before trade evaluation (bucket, leverage, concentration, factor "
                "shock, duplicate, size, per-ticker cap)."
            )

        st.subheader("Look up a specific pair")
        pair = st.text_input("Pair", placeholder="AAPL_MSFT")
        if pair:
            rejection = data.run_tool("explain_rejection", {"pair": pair.strip().upper(),
                                                            "limit": 20})
            if not rejection["data"]:
                st.warning(
                    f"No rejections found for {pair}. It may have been approved, or "
                    f"not evaluated — this view shows rejections only."
                )
            else:
                st.dataframe(pd.DataFrame(rejection["data"]), width='stretch',
                             hide_index=True)
                st.caption(rejection["summary"])


    # --- ask ------------------------------------------------------------------

    with ask:
        st.subheader("Ask the agent")
        st.caption(
            "The same agent the CLI and Slack bot use. It investigates by calling "
            "the tools behind the other tabs — every figure it states comes from one "
            "of them, not from the model."
        )

        question = st.text_input("Question",
                                 placeholder="why did the C order fill badly on the 24th?")
        if question:
            import os

            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY is not set. Add it to .env.")
            else:
                with st.spinner("Investigating…"):
                    from src.agent.loop import run_agent

                    with data.engine().connect() as conn:
                        trace = run_agent(question, conn)

                if trace.error:
                    st.error(trace.error)
                else:
                    st.markdown(trace.answer)
                    st.divider()
                    st.caption(
                        f"`{', '.join(trace.tool_sequence) or 'no tools'}` · "
                        f"{trace.turns} turns · {trace.duration_ms / 1000:.1f}s · "
                        f"{trace.input_tokens:,} in / {trace.output_tokens:,} out"
                    )
                    with st.expander("Trace"):
                        for i, call in enumerate(trace.tool_calls, start=1):
                            st.code(f"{i}. {call.name}({call.arguments})\n   -> {call.summary}")


# Streamlit executes this file as a script, so main() runs here. Importing the
# module — as CI's import check does — runs nothing.
if __name__ == "__main__":
    main()
else:  # pragma: no cover - Streamlit runs the file without __main__
    import sys
    if "streamlit" in sys.modules and hasattr(st, "runtime") and st.runtime.exists():
        main()
