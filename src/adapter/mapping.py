"""Mapping from V9.2C output files to the blotter schema.

This module is the adapter's specification, expressed as data rather than
code, so that every column name and enum value lives in one place and can be
corrected against real files without touching the loader.

**Status: built from the code review's output inventory, not from real files.**
Every entry marked VERIFY is Claude Code's reading of what V9.2C writes; the
first archived day will confirm or correct it. Expect some to be wrong. The
loader reports every source column it could not map, so the gaps are visible
rather than silently dropped.

Three kinds of mapping:

  * `COLUMNS` — source column -> blotter column, per file
  * `ENUMS`   — source value -> blotter value, per field
  * `DERIVED` — blotter columns that must be computed rather than copied

Anything in the blotter schema not covered here is left NULL and reported.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source files, relative to an archived day's directory
# ---------------------------------------------------------------------------

FILES = {
    "portfolio": "Portfolio.xlsx",                # open positions; sheet "Portfolio"
    "completed": "Completed_Trades.xlsx",         # closed positions
    "terminated": "daily_terminated_trades.xlsx", # today's exits (subset of completed)
    "shortlist": "V9_Shortlist.xlsx",             # primary-filter output + secondary signals
    "longlist": "V9_Longlist.xlsx",               # pre-filter candidates (manual inspection)
    "filter_details": "V9_Filter_Details.xlsx",   # filter-by-filter pass/fail
    "execution": "Execution_Summary.xlsx",        # per-run execution results — VERIFY shape
    "log": "v9c_trading.log",                     # orders, fills, stages, events — needs parser
}

# ---------------------------------------------------------------------------
# Column mappings. Source names are case-sensitive as written by V9.2C; the
# loader also tries a case-insensitive match before reporting a miss.
# ---------------------------------------------------------------------------

# Portfolio.xlsx -> positions. VERIFY every name against the real header row.
POSITION_COLUMNS = {
    "Tag": "tag",
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Index": "idx",
    "Tail": "tail",
    "Model_Version": "version",
    "W1": "w1",
    "W2": "w2",
    "Quantity_1": "quantity1",
    "Quantity_2": "quantity2",
    "Trade_Value_Co1": "trade_value_co1",
    "Trade_Value_Co2": "trade_value_co2",
    "Sum_Deviation": "sum_deviation",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Weighted_Score": "weighted_score",
    "Composite_Score": "composite_score",
    "Beta": "beta",
    "Trade_Initiation_Date": "trade_initiation_date",
    "Position_Multiplier": "position_multiplier",
    "Co1_At_Initiation": "co1_at_initiation",
    "Co2_At_Initiation": "co2_at_initiation",
    "Index_At_Initiation": "index_at_initiation",
    "Live_Alpha_Return": "_live_alpha",           # not a positions column; used for updates
}

# Completed_Trades.xlsx -> positions (closed). Superset of POSITION_COLUMNS.
COMPLETED_COLUMNS = {
    **POSITION_COLUMNS,
    "Termination_Date": "termination_date",
    "Exit_Reason": "exit_reason",
    "Holding_Days": "holding_days",
    "Co1_At_Exit": "co1_at_exit",
    "Co2_At_Exit": "co2_at_exit",
    "Index_At_Exit": "index_at_exit",
    "Final_Alpha_Return": "final_alpha_return_pct",
}

# V9_Shortlist.xlsx -> pair_evaluations. VERIFY. The review lists these names.
EVALUATION_COLUMNS = {
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Index": "idx",
    "Tail": "tail",
    "Tstat": "tstat",
    "Weighted_Spread_BPS": "weighted_spread_bps",
    "Earnings_Days_Out": "earnings_days_out",
    "Co1_Trending": "co1_trending",
    "Co2_Trending": "co2_trending",
    "Same_Direction_Result": "same_direction_result",
    "Nominal_Direction_Result": "nominal_direction_result",
    "Primary_Result": "primary_result",
    "Primary_Fail_Reason": "primary_fail_reason",
    "Volume_Ratio": "volume_ratio",
    "Rolling_Intraday_Vol": "rolling_intraday_vol",
    "IV_Percentile": "iv_percentile",
    "Volume_Dominance": "volume_dominance",
    "True_Last_Hour_Volatility": "true_last_hour_volatility",
    "Weighted_Score": "weighted_score",
    "Composite_Score": "composite_score",
    "Sum_Deviation": "sum_deviation_15d",
    "Sum_Dev_Percentile": "sum_dev_percentile",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Position_Multiplier": "position_multiplier",
    "Is_Tradeable_Bucket": "is_tradeable_bucket",
}

# Blotter columns in pair_evaluations that V9.2C's shortlist may not carry.
# Left NULL if absent; the review flagged these as possibly missing.
EVALUATION_OPTIONAL = {"shocked_factors", "factor_action", "spread_quality_score",
                       "sum_dev_extremity_score", "composite_priority_score",
                       "evaluation_result", "rejection_reason"}

# ---------------------------------------------------------------------------
# Enum normalisation. Keys are lower-cased source values; the loader
# lower-cases before lookup so "Stop_Loss", "stop_loss", "STOP_LOSS" all map.
# ---------------------------------------------------------------------------

EXIT_REASONS = {
    "date": "Date Reached",
    "date reached": "Date Reached",
    "earnings": "Earnings Alert",
    "earnings alert": "Earnings Alert",
    "early_exit": "Alpha Reached",
    "early exit": "Alpha Reached",
    "alpha": "Alpha Reached",
    "alpha reached": "Alpha Reached",
    "stop_loss": "Stop Loss Triggered",
    "stop loss": "Stop Loss Triggered",
    "stop loss triggered": "Stop Loss Triggered",
    "delisting": "Delisting",
    "delisted": "Delisting",
    "manual": "Manual",
}

TAILS = {"l": "L", "lower": "L", "u": "U", "upper": "U"}

ORDER_TYPES = {"lmt": "LMT", "limit": "LMT", "mkt": "MKT", "market": "MKT"}

STATUS = {"open": "open", "active": "open", "closed": "closed", "completed": "closed"}

PASS_FAIL = {"pass": "Pass", "passed": "Pass", "true": "Pass", "1": "Pass",
             "fail": "Fail", "failed": "Fail", "false": "Fail", "0": "Fail"}

# Sum_Dev_Bucket may arrive as "0-10%", "0-10", or a decile integer 0..9.
def normalise_bucket(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith("%") and "-" in text:
        return text
    if "-" in text:
        return text + "%"
    try:
        decile = int(float(text))
        return f"{decile * 10}-{decile * 10 + 10}%"
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# Derived: columns the loader must compute because V9.2C does not write them.
# ---------------------------------------------------------------------------

DERIVED_NOTES = {
    "positions.total_notional": "trade_value_co1 + trade_value_co2",
    "positions.status": "'open' if in Portfolio.xlsx, 'closed' if in Completed_Trades.xlsx",
    "positions.co1_return_pct etc.": "from Completed_Trades exit prices vs initiation prices",
    "position_updates": "diff of consecutive archived Portfolio.xlsx snapshots",
    "workflow_runs / workflow_stages": "parsed from v9c_trading.log — parser pending log sample",
    "orders / fills": "parsed from v9c_trading.log and Execution_Summary.xlsx — parser pending",
    "order_allocations": "1:1 with orders; aggregation is disabled in V9.2C",
    "system_events / risk_checks": "parsed from v9c_trading.log — parser pending",
    "instruments": "seeded once from the universe file; delistings from log",
}
