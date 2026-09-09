"""Tool registry and API-facing schemas.

TOOL_SCHEMAS is the list passed to the Anthropic API in Phase 2. The
description text is load-bearing: it is the only thing the model sees when
deciding which tool to call, so it is written to disambiguate against the
others, not merely to describe in isolation.

Descriptions are kept in sync with the function docstrings by a test.
"""

from __future__ import annotations

from typing import Callable

from src.tools.analytics import alpha_attribution, detect_anomalies, execution_quality
from src.tools.base import ToolResult
from src.tools.blotter import explain_position, explain_rejection, query_blotter
from src.tools.charts import make_chart
from src.rag.tool import search_documentation

# Tools taking a database connection as their first argument. make_chart does
# not touch the blotter, so the loop passes it through differently.
DB_TOOLS: dict[str, Callable[..., ToolResult]] = {
    "query_blotter": query_blotter,
    "explain_position": explain_position,
    "explain_rejection": explain_rejection,
    "execution_quality": execution_quality,
    "alpha_attribution": alpha_attribution,
}

PURE_TOOLS: dict[str, Callable[..., ToolResult]] = {
    "make_chart": make_chart,
    "search_documentation": search_documentation,
}

TOOLS: dict[str, Callable[..., ToolResult]] = {**DB_TOOLS, **PURE_TOOLS}
DB_TOOLS["detect_anomalies"] = detect_anomalies
TOOLS["detect_anomalies"] = detect_anomalies


_DATE_WINDOW = {
    "start_date": {"type": "string",
                   "description": "Inclusive start date, ISO YYYY-MM-DD. Omit for the earliest data held."},
    "end_date": {"type": "string",
                 "description": "Inclusive end date, ISO YYYY-MM-DD. Omit for the latest data held."},
}


TOOL_SCHEMAS = [
    {
        "name": "query_blotter",
        "description": (
            "Look up raw records from the trade blotter: positions, orders or daily runs. "
            "This is the general-purpose lookup — use it to find what exists (for example to "
            "obtain a position tag or order id) before drilling in with a more specific tool. "
            "Do not use it for execution cost analysis, rejection reasons or alpha breakdowns; "
            "dedicated tools exist for those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string", "enum": ["positions", "orders", "runs", "candidates"],
                    "description": (
                        "positions = pair trades; orders = ticker-level broker orders; "
                        "runs = daily execution runs; candidates = pairs evaluated on a run, "
                        "carrying the stage each reached (prefilter -> longlist -> shortlist) "
                        "and whether a position was opened on it that day."
                    ),
                },
                **_DATE_WINDOW,
                "ticker": {"type": "string", "description": "Single ticker. For positions, matches either leg."},
                "pair": {"type": "string", "description": "Pair identifier, e.g. AAPL_MSFT."},
                "index": {"type": "string", "description": "Sector index, e.g. VGT, VFH, VIS, VHT, VCR."},
                "status": {"type": "string",
                           "description": "positions: open|closed. orders: Filled|Partial|Failed."},
                "stage": {"type": "string", "enum": ["prefilter", "longlist", "shortlist", "rejected"],
                          "description": "candidates only: how far the pair got through the pipeline."},
                "traded": {"type": "boolean",
                           "description": "candidates only: true = a position was opened on this pair "
                                          "that day; false = it was not. Shortlisted-but-not-traded is "
                                          "stage='shortlist', traded=false."},
                "limit": {"type": "integer", "description": "Max rows, default 50."},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "explain_position",
        "description": (
            "Reconstruct the full decision context for ONE position, identified by its tag. "
            "Returns why the trade was taken (CDF bucket, position multiplier, leg weights, "
            "composite score, entry spread), how both legs executed, stop-loss state, the alpha "
            "path since entry, and the exit reason. Use when asked why a specific trade was "
            "taken or what happened to it. Requires a tag — call query_blotter first if you do "
            "not have one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string",
                        "description": "Position identifier, e.g. VGT_AAPL_MSFT_L_20260815_001."},
            },
            "required": ["tag"],
        },
    },
    {
        "name": "explain_rejection",
        "description": (
            "Explain why candidate pairs were screened out INSTEAD of being traded. Every pair is "
            "evaluated on every run; this returns the gate that stopped each one and the value it "
            "failed on — primary filters (spread hurdle, earnings proximity, trending, direction "
            "checks, t-stat) or trade evaluation (untradeable CDF bucket, leverage, index "
            "concentration, factor shock, duplicate, position size, per-ticker cap). Use for 'why "
            "wasn't X traded'. This is the counterpart to explain_position, which covers trades "
            "that WERE taken. IMPORTANT: this returns rejections ONLY. A pair absent from the "
            "results may have been approved, or may not have been evaluated at all — absence "
            "here is not evidence that nothing happened."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Pair identifier, e.g. AAPL_MSFT."},
                "ticker": {"type": "string", "description": "Ticker matching either leg."},
                **_DATE_WINDOW,
                "limit": {"type": "integer", "description": "Max rows, default 20."},
            },
            "required": [],
        },
    },
    {
        "name": "execution_quality",
        "description": (
            "Transaction cost analysis: slippage against the arrival mid, effective spread paid, "
            "fill rates, limit-order timeout fallbacks, and limit-versus-market routing. Use for "
            "'why did this order fill badly', 'how good are our fills', or 'is our routing costing "
            "us'. Pass order_id for the full causal detail of one order — that is the right call "
            "when investigating a single bad fill. Positive slippage means worse than arrival mid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_DATE_WINDOW,
                "ticker": {"type": "string", "description": "Restrict to one ticker."},
                "order_id": {"type": "string",
                             "description": "Return full detail for a single order rather than an aggregate."},
                "group_by": {"type": "string", "enum": ["ticker", "order_type", "date"],
                             "description": "Break the aggregate down. Omit for a single total."},
            },
            "required": [],
        },
    },
    {
        "name": "alpha_attribution",
        "description": (
            "Break down realised alpha across the book by sector index, CDF bucket, tail, exit "
            "reason or month. Returns trade counts, mean and total alpha, win rate and average "
            "holding period per group. Alpha is index-relative and market-neutral "
            "(W1*co1_return - W2*co2_return - beta*index_return) — it is NOT raw profit and loss, "
            "and should not be described as such."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_DATE_WINDOW,
                "group_by": {"type": "string",
                             "enum": ["idx", "sum_dev_bucket", "tail", "exit_reason", "month"],
                             "description": "Attribution dimension. Default idx (sector index)."},
                "status": {"type": "string",
                           "description": "Position status, default 'closed'. Only closed trades have final alpha."},
            },
            "required": [],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Find operational problems: what went wrong and when. Covers system events (stale "
            "data, reconciliation mismatches, order timeouts, partial fills, spread validation "
            "failures, connection errors, delistings, orphaned legs after a stop trigger) and "
            "failed risk checks (leverage, index concentration, factor exposure, portfolio beta). "
            "Call this first when a question implies something broke, or for 'what went wrong "
            "yesterday'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_DATE_WINDOW,
                "severity": {"type": "string", "enum": ["info", "warning", "halt"],
                             "description": "Restrict by severity. 'halt' means trading stopped."},
                "event_type": {"type": "string", "description": "Restrict to one event type."},
                "limit": {"type": "integer", "description": "Max events, default 50."},
            },
            "required": [],
        },
    },
    {
        "name": "search_documentation",
        "description": (
            "Search the trading system's design documentation. Answers WHY the system is "
            "built the way it is: the rationale behind a filter or threshold, what a gate "
            "protects against, how calibration and implementation relate, or what a term "
            "means in this system's vocabulary. Returns the most relevant passages, each "
            "with a citation (file and section) and a relevance score. This is the "
            "counterpart to the blotter tools, which answer what HAPPENED — use this for "
            "design and definitions, the blotter tools for data, and both when a question "
            "asks what happened AND why the system responded that way."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The question or topic to search for, in natural language."},
                "k": {"type": "integer", "description": "Number of passages to return, default 4."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Render a chart from values you ALREADY have and return its file path. This tool "
            "performs no analysis and does not read the blotter — it plots exactly what it is "
            "given. Call an analytical tool first, then pass its output here. Use when a visual "
            "comparison would help the reader; do not chart a single number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"},
                           "description": "Category or x-axis labels."},
                "values": {"type": "array", "items": {"type": "number"},
                           "description": "Numeric values, same length as labels."},
                "title": {"type": "string", "description": "Chart title."},
                "chart_type": {"type": "string", "enum": ["bar", "barh", "line", "scatter"],
                               "description": "bar = categories; barh = long labels; line = over time; scatter = relationship."},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "series_label": {"type": "string", "description": "Legend label for line charts."},
            },
            "required": ["labels", "values", "title"],
        },
    },
]


def dispatch(name: str, arguments: dict, conn=None) -> ToolResult:
    """Execute a tool by name. The agent loop's single entry point.

    Unknown names and bad arguments return a ToolResult rather than raising, so
    the loop can hand the problem back to the model to correct.
    """
    tool = TOOLS.get(name)
    if tool is None:
        from src.tools.base import empty
        return empty(f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOLS))}.")

    try:
        if name in PURE_TOOLS:
            return tool(**arguments)
        return tool(conn, **arguments)
    except TypeError as exc:
        from src.tools.base import empty
        return empty(f"Invalid arguments for '{name}': {exc}")


__all__ = ["TOOLS", "TOOL_SCHEMAS", "dispatch"]
