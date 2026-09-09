"""Mapping from V9.2C/V9.3 output files to the blotter schema.

**Verified against real files from the run of 2026-09-08.** The previous
version was built from a code review's inventory and got six things wrong;
they are recorded here because each is a trap for anyone reading the schema
and assuming the files match it.

## What the real files taught

**`Tag` is not the position tag.** It is an integer row index into the
parameters file — `1`, `1686`, `2650` — stable for a pair across a run but
not a position identifier and not unique across days. The blotter's `tag`
(`VGT_AAPL_MSFT_L_20260815_001`) does not exist anywhere in these files. It
is constructed at execution time, so pre-trade rows are keyed by
`(run_date, Tag)` instead.

**`Pair` uses a hyphen, not an underscore.** `IOT-GLW`, not `IOT_GLW`. The
blotter and the event schema both use underscores.

**Ticker column names differ per file.** The prefilter and shortlist say
`Co1`/`Co2`; the longlist says `Ticker1`/`Ticker2`. Same data.

**`Active` is 0/1, not Pass/Fail**, and `Reason` carries free text that
embeds the whole trending-filter message — ticker, percentage, threshold —
so it is a sentence, not an enum. It has to be parsed to a category.

**Filter results are `Pass`/`Fail` strings in the longlist** under
human-readable headers with spaces: `15-Day Alpha`, `2-Day Dev`,
`Same Direction`, `Nominal Direction`, `Co1 Trend`.

**The rejected archive is not a rejection log.** Its `Status` is `Pending`
and it carries `Would_Be_Initiation_Date` / `Would_Be_Termination_Date` —
candidates that failed the score threshold, archived for later analysis, with
`Score_Threshold` and `Score_Shortfall`. It also carries treasury columns
(`Treasury_at_Entry`, `Co1_Treasury_Beta`) that the V9.3 single-factor model
no longer uses; they are present and empty.

## Still unknown

No `Portfolio.xlsx` or `Completed_Trades.xlsx` in this batch — the portfolio
was empty and no trades executed. Those mappings remain unverified, and are
marked. Orders, fills, workflow stages and events live in the log, which has
not been seen.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Source files, relative to an archived day's directory
# ---------------------------------------------------------------------------

FILES = {
    # VERIFIED against the 2026-09-08 run
    "prefilter": "trade_prefilter_active.xlsx",   # sheets: Active_Pairs, Summary, Failure_Reasons
    "longlist": "V9_Longlist.xlsx",               # sheets: Longlist, Summary
    "shortlist": "V9_Shortlist.xlsx",             # sheet: Shortlist
    "rejected": "Rejected_Trades_Archive.xlsx",   # sheet: Sheet1
    # UNVERIFIED — absent from the first archive (empty portfolio, no trades)
    "portfolio": "Portfolio.xlsx",
    "completed": "Completed_Trades.xlsx",
    "terminated": "daily_terminated_trades.xlsx",
    "log": "v9c_trading.log",
}

SHEETS = {
    "prefilter": "Active_Pairs",
    "longlist": "Longlist",
    "shortlist": "Shortlist",
    "rejected": 0,
    "portfolio": "Portfolio",
    "completed": 0,
    "terminated": 0,
}

# ---------------------------------------------------------------------------
# Column mappings, per file. Verified names on the left.
# ---------------------------------------------------------------------------

# trade_prefilter_active.xlsx -> pair_evaluations (the 4,500-row universe pass)
PREFILTER_COLUMNS = {
    "Tag": "source_tag",              # integer row index, NOT the position tag
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Tail": "tail",
    "Index": "idx",
    "Active": "_active",              # 0/1 -> primary_result
    "Reason": "_reason",              # free text -> primary_fail_reason
    "Sum_Deviation": "sum_deviation_15d",
    "Sum_Dev_Percentile": "sum_dev_percentile",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Category1": "category1",
    "Category2": "category2",
    "Model_Version": "version",
}

# V9_Longlist.xlsx -> pair_evaluations (the 1,072 that reached LAM, with filter detail)
LONGLIST_COLUMNS = {
    "Tag": "source_tag",
    "Pair": "pair",
    "Ticker1": "co1",                 # NB: Ticker1/2 here, Co1/2 elsewhere
    "Ticker2": "co2",
    "Tail": "tail",
    "Index": "idx",
    "Category1": "category1",
    "Category2": "category2",
    "Status": "_status",              # Pass/Fail overall
    "Earnings": "_earnings_result",
    "Spread": "_spread_result",
    "15-Day Alpha": "_alpha_result",
    "Alpha_CDF": "alpha_cdf",
    "2-Day Dev": "_two_day_result",
    "Same Direction": "same_direction_result",
    "Nominal Direction": "nominal_direction_result",
    "Co1 Trend": "_trend_result",
    "Co1_Tstat": "tstat",
    "Weighted_Score": "weighted_score",
    "Composite_Score": "composite_score",
    "Sum_Dev_Value": "sum_deviation_15d",
    "Sum_Dev_CDF": "sum_dev_percentile",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Volume_Ratio": "volume_ratio",
    "Rolling_Intraday_Vol": "rolling_intraday_vol",
    "Volume_Dominance": "volume_dominance",
    "Last_Hour_Vol": "true_last_hour_volatility",
    "IV_Percentile": "iv_percentile",
    "Volume_Ratio_Pct": "volume_ratio_pct",
    "Intraday_Vol_Pct": "intraday_vol_pct",
    "Volume_Dom_Pct": "volume_dominance_pct",
    "Last_Hour_Pct": "last_hour_pct",
    "IV_Pct_Pct": "iv_percentile_pct",
    "Index_Bias": "index_bias",
}

# V9_Shortlist.xlsx -> pair_evaluations (the 110 selected). Subset of the longlist.
SHORTLIST_COLUMNS = {
    k: v for k, v in LONGLIST_COLUMNS.items()
    if k not in {"Ticker1", "Ticker2", "Status", "Earnings", "Spread", "15-Day Alpha",
                 "Alpha_CDF", "2-Day Dev", "Same Direction", "Nominal Direction",
                 "Co1 Trend", "Co1_Tstat"}
} | {"Co1": "co1", "Co2": "co2"}

# Rejected_Trades_Archive.xlsx -> pair_evaluations (score-threshold failures)
REJECTED_COLUMNS = {
    "Tag": "source_tag",
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Index": "idx",
    "Tail": "tail",
    "Co1_at_Entry": "co1_price",
    "Co2_at_Entry": "co2_price",
    "Index_at_Entry": "index_price",
    "Co1_SubSector_Beta": "co1_beta",
    "Co2_SubSector_Beta": "co2_beta",
    "W1": "w1",
    "W2": "w2",
    "sum_dev_bucket": "sum_dev_bucket",     # lower case here, Title case elsewhere
    "Sum_Dev_CDF": "sum_dev_percentile",
    "Sum_Dev_Value": "sum_deviation_15d",
    "Volume_Ratio": "volume_ratio",
    "Rolling_Intraday_Vol": "rolling_intraday_vol",
    "Volume_Dominance": "volume_dominance",
    "Last_Hour_Vol": "true_last_hour_volatility",
    "IV_Percentile": "iv_percentile",
    "Composite_Score": "composite_score",
    "Score_Threshold": "score_threshold",
    "Score_Shortfall": "score_shortfall",
    "Archived_At": "evaluated_at",
    "Model": "version",
}

# Columns present in the source that the blotter has no home for. Listed so
# the loader can report a genuine gap rather than flagging known extras.
KNOWN_UNMAPPED: dict[str, set[str]] = {
    "rejected": {"Treasury_at_Entry", "Co1_Treasury_Beta", "Co2_Treasury_Beta",
                 "Volume_Ratio_Pct", "Intraday_Vol_Pct", "Volume_Dom_Pct",
                 "Last_Hour_Pct", "IV_Pct_Pct", "Would_Be_Initiation_Date",
                 "Would_Be_Termination_Date", "Status"},
}

# Portfolio.xlsx -> positions (open). VERIFIED against the real file: 66
# columns across 6 sheets, of which only "Portfolio" holds positions. Note the
# spaces and the "($)" suffixes — these are display headers, not identifiers.
POSITION_COLUMNS = {
    "Tag": "source_tag",                    # integer, as everywhere else
    "Version": "version",
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Tail": "tail",
    "Index": "idx",
    "Quantity1": "quantity1",
    "Quantity2": "quantity2",
    "W1": "w1",
    "W2": "w2",
    "Position_Multiplier": "position_multiplier",
    "Co1 at Initiation": "co1_at_initiation",
    "Co2 at Initiation": "co2_at_initiation",
    "Index at Initiation": "index_at_initiation",
    "Trade Value Co1 ($)": "trade_value_co1",
    "Trade Value Co2 ($)": "trade_value_co2",
    "Total_Notional": "total_notional",     # supplied, not derived
    "Sum_Dev_Value": "sum_deviation",
    "Sum_Dev_CDF": "sum_dev_percentile",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Beta": "beta",
    "Weighted_Score": "weighted_score",
    "Composite_Score": "composite_score",
    "Index_Bias": "index_bias",
    "Entry_Spread_BPS": "entry_spread_bps",
    "Trade Initiation Date": "trade_initiation_date",
    "Trade Termination Date": "scheduled_termination_date",
    "Stop_Order_ID": "stop_order_id",
    "Stop_Price": "stop_price",
    "Live Alpha Return (%)": "_live_alpha",
    "Co1 Return (%)": "_co1_return",
    "Co2 Return (%)": "_co2_return",
    "Index Return (%)": "_index_return",
}

# Completed_Trades.xlsx -> positions (closed). VERIFIED: 45 columns, single
# sheet. Carries the full return decomposition the open file does not.
COMPLETED_COLUMNS = {
    "Tag": "source_tag",
    "Version": "version",
    "Pair": "pair",
    "Co1": "co1",
    "Co2": "co2",
    "Index": "idx",
    "Tail": "tail",
    "Trade Initiation Date": "trade_initiation_date",
    "Trade Termination Date": "termination_date",
    "Holding_Days": "holding_days",
    "Exit_Reason": "_exit_reason",          # free text; see classify_exit_reason
    "Co1 at Initiation": "co1_at_initiation",
    "Co2 at Initiation": "co2_at_initiation",
    "Index at Initiation": "index_at_initiation",
    "Co1 at Exit": "co1_at_exit",
    "Co2 at Exit": "co2_at_exit",
    "Index at Exit": "index_at_exit",
    "Quantity1": "quantity1",
    "Quantity2": "quantity2",
    "Trade Value Co1 ($)": "trade_value_co1",
    "Trade Value Co2 ($)": "trade_value_co2",
    "Total_Notional": "total_notional",
    "W1": "w1",
    "W2": "w2",
    "Beta": "beta",
    "Position_Multiplier": "position_multiplier",
    "Sum_Dev_Value": "sum_deviation",
    "Sum_Dev_CDF": "sum_dev_percentile",
    "Sum_Dev_Bucket": "sum_dev_bucket",
    "Co1_Return_Pct": "co1_return_pct",
    "Co2_Return_Pct": "co2_return_pct",
    "Index_Return_Pct": "index_return_pct",
    "Co1_Alpha_Pct": "co1_alpha_pct",
    "Co2_Alpha_Pct": "co2_alpha_pct",
    "Final_Alpha_Return_Pct": "final_alpha_return_pct",
    "Entry_Spread_BPS": "entry_spread_bps",
    "Weighted_Score": "weighted_score",
    "Index_Bias": "index_bias",
    "Composite_Score": "composite_score",
    "Volume_Ratio": "volume_ratio",
    "Rolling_Intraday_Vol": "rolling_intraday_vol",
    "Volume_Dominance": "volume_dominance",
    "IV_Percentile": "iv_percentile",
}

# Treasury columns survive from the two-factor model V9.3 removed. Present,
# entirely empty in the real file (185/185 null on exit), and correctly
# unmapped. IGV and factor-beta columns likewise: the hedge was retired and
# the four-factor betas are portfolio-level bookkeeping the blotter does not
# model.
KNOWN_UNMAPPED.update({
    "portfolio": {"Treasury at Initiation", "Concentration",
                  "USMV_Long_Beta", "USMV_Short_Beta", "USMV_Net_Contrib",
                  "VLUE_Long_Beta", "VLUE_Short_Beta", "VLUE_Net_Contrib",
                  "MTUM_Long_Beta", "MTUM_Short_Beta", "MTUM_Net_Contrib",
                  "SOXX_Long_Beta", "SOXX_Short_Beta", "SOXX_Net_Contrib",
                  "Volume_Ratio", "Rolling_Intraday_Vol", "Volume_Dominance",
                  "Last_Hour_Vol", "IV_Percentile", "Volume_Ratio_Pct",
                  "Intraday_Vol_Pct", "Volume_Dom_Pct", "Last_Hour_Pct", "IV_Pct_Pct",
                  "Spread_Quality_Score", "Sum_Dev_Extremity_Score",
                  "Composite_Priority_Score", "IGV_Long", "IGV_Short", "IGV_Pair_Type",
                  "Version.1", "Index Return"},
    "completed": {"Treasury at Initiation", "Treasury at Exit"},
})


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

TAILS = {"l": "L", "lower": "L", "u": "U", "upper": "U"}

PASS_FAIL = {"pass": "Pass", "passed": "Pass", "true": "Pass", "1": "Pass", "1.0": "Pass",
             "fail": "Fail", "failed": "Fail", "false": "Fail", "0": "Fail", "0.0": "Fail"}

# Exit_Reason is free text carrying the detail, exactly like the prefilter's
# Reason. The real file holds 48 distinct values across 185 trades:
#   "Date Reached"
#   "Early Exit - Day15_TakeProfit_8pct"
#   "Earnings - NSSC reports 2026-02-02"
#   "Pre-Holiday Exit (term date 2025-12-25 is non-trading day)"
#   "Past Due (was 2026-02-17)"
# The schema's five canonical reasons do not cover Pre-Holiday or Past Due,
# both of which are scheduled exits displaced by the calendar. They map to
# Date Reached with the original preserved.
EXIT_REASON_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^early exit", re.I), "Alpha Reached"),
    (re.compile(r"takeprofit", re.I), "Alpha Reached"),
    (re.compile(r"^earnings", re.I), "Earnings Alert"),
    (re.compile(r"stop.?loss", re.I), "Stop Loss Triggered"),
    (re.compile(r"delist", re.I), "Delisting"),
    (re.compile(r"^manual", re.I), "Manual"),
    (re.compile(r"date reached|pre-holiday|past due", re.I), "Date Reached"),
]


def classify_exit_reason(text) -> tuple[str | None, str | None]:
    """(canonical reason, original). Same treatment as rejection reasons: the
    category is queryable, the sentence keeps the day number, the reporting
    ticker or the displaced date."""
    if text is None:
        return None, None
    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return None, None
    for pattern, canonical in EXIT_REASON_PATTERNS:
        if pattern.search(raw):
            return canonical, raw
    return "Manual", raw

ORDER_TYPES = {"lmt": "LMT", "limit": "LMT", "mkt": "MKT", "market": "MKT"}


def normalise_pair(value) -> str | None:
    """`IOT-GLW` -> `IOT_GLW`. The files hyphenate; the schema underscores."""
    if value is None:
        return None
    return str(value).strip().replace("-", "_")


def normalise_bucket(value) -> str | None:
    """Already `0-10%` .. `90-100%` in every verified file. Integers and
    unsuffixed ranges are handled for the files not yet seen."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if re.fullmatch(r"\d+-\d+%", text):
        return text
    if re.fullmatch(r"\d+-\d+", text):
        return text + "%"
    try:
        d = int(float(text))
        return f"{d * 10}-{d * 10 + 10}%"
    except ValueError:
        return text


# The prefilter's Reason column is free text that embeds the trending
# message, so it cannot be an enum lookup. Ordered patterns -> the event
# schema's rejection categories.
REASON_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^trending filter", re.I), "trend_filter"),
    (re.compile(r"alpha_variance", re.I), "alpha_variance"),
    (re.compile(r"\bspread\b", re.I), "spread_hurdle"),
    (re.compile(r"missing historical data", re.I), "missing_data"),
    (re.compile(r"two_day_deviation", re.I), "two_day_deviation"),
    (re.compile(r"same_direction", re.I), "same_direction"),
    (re.compile(r"nominal_direction", re.I), "nominal_direction"),
    (re.compile(r"sum deviation|sum_dev", re.I), "sum_dev_exclusion"),
    (re.compile(r"earnings", re.I), "earnings_filter"),
    (re.compile(r"tstat|t-stat", re.I), "tstat"),
]


def classify_reason(text) -> tuple[str | None, str | None]:
    """(category, original). Returns the raw text alongside so nothing is lost:
    'Trending filter: ASAN - Negative trending: -73.49% excess return over 12M'
    carries the ticker and the magnitude, and both are worth keeping."""
    if text is None:
        return None, None
    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return None, None
    for pattern, category in REASON_PATTERNS:
        if pattern.search(raw):
            return category, raw
    return "other", raw


def position_tag(idx, pair: str, tail: str | None, init_date: str, source_tag=None) -> str:
    """Reconstruct a blotter position tag from what the files carry.

    The event schema's tag — `VGT_AAPL_MSFT_L_20260815_001` — is built at
    execution time and appears in none of the pre- or post-trade files. What
    identifies a position across them is index, legs, tail and initiation
    date; the trailing sequence disambiguates the same pair opened twice on
    one day, using the parameters-file row index when present. The real
    Completed_Trades held 185 rows over 177 tags, so a tag-only key would
    have silently merged eight positions.
    """
    date_part = str(init_date)[:10].replace("-", "")
    try:
        seq = f"{int(source_tag) % 1000:03d}"
    except (TypeError, ValueError):
        seq = "000"
    return f"{idx or 'UNK'}_{pair}_{tail or 'X'}_{date_part}_{seq}"


def disambiguate(tag: str, exit_date, n: int) -> str:
    """Suffix a tag that collides with one already seen.

    The real Completed_Trades held five positions recorded TWICE — same
    index, legs, tail and initiation date, but different termination dates,
    exit reasons and alpha. AXSM_DAWN appears as both "Date Reached" and
    "Past Due (was 2026-02-17)" with alphas of +0.49 and -0.48; all ten rows
    fall on 17-18 February. Whether that is a duplicate-exit bug or two
    genuine records, the adapter is not the place to decide — it keeps both,
    distinguished by exit date, and the report says how many it found. A
    silent overwrite would have lost half of a contradiction.
    """
    suffix = str(exit_date)[:10].replace("-", "") if exit_date else f"dup{n}"
    return f"{tag}#{suffix}"


def evaluation_key(run_date: str, source_tag, stage: str) -> str:
    """Pre-trade rows have no position tag, so they are keyed by run, stage
    and the parameters-file row index.

    Stage is part of the key because Tag is unique *within* a file but the
    same Tag appears in several: pair 130 is row 130 of the prefilter, of the
    longlist if it stayed active, and of the shortlist if it was selected.
    Keying on (run, tag) alone collapsed 5,682 rows into 2,533 — each later
    stage overwriting the earlier one, losing the prefilter's rejection reason
    for every pair that survived to the longlist.
    """
    return f"v92c_{run_date}_{stage}_{int(source_tag):05d}"


# ---------------------------------------------------------------------------
# Derived and pending
# ---------------------------------------------------------------------------

DERIVED_NOTES = {
    "pair_evaluations.evaluation_result": "prefilter Active 1/0 -> Approved/Rejected",
    "pair_evaluations.rejection_reason": "classify_reason(Reason)",
    "pair_evaluations.primary_result": "longlist Status Pass/Fail",
    "pair_evaluations.is_tradeable_bucket": "derive from Sum_Dev_Bucket vs disabled buckets",
    "pair_evaluations.stage": "prefilter | longlist | shortlist | rejected — which file it came from",
    "positions.*": "UNVERIFIED — no Portfolio.xlsx yet",
    "orders / fills / workflow_stages / risk_checks / system_events":
        "from v9c_trading.log — parser pending a sample",
}
