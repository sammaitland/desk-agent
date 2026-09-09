"""Tests for the V9.2C -> blotter adapter.

Real V9.2C output does not exist yet, so these build synthetic Excel files
shaped to the review's column inventory and check that the loader maps,
normalises, diffs and upserts correctly — and that it reports every gap
rather than hiding it.

The mapping itself is unverified until a real archived day arrives. What is
verified here is the loader's behaviour given a mapping: idempotence, enum
normalisation, the day-to-day diff, and honest reporting. When the real
columns turn out to differ, mapping.py changes and these tests still hold.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from src.adapter import mapping as M
from src.adapter.load import LoadReport, LogParser, archived_days, load_archive
from src.db import create_schema, get_engine


# ---------------------------------------------------------------------------
# Synthetic archive builder
# ---------------------------------------------------------------------------

# (index, co1, co2, tail, source Tag) -> initiation date
P1 = ("VGT", "AAPL", "MSFT", "L", 101)
P2 = ("VFH", "JPM", "BAC", "U", 202)
P3 = ("VHT", "LLY", "JNJ", "L", 303)
OPENED = {P1: date(2026, 9, 5), P2: date(2026, 9, 5), P3: date(2026, 9, 9)}


def _tag(name):
    from src.adapter.mapping import position_tag

    idx, co1, co2, tail, src = name
    return position_tag(idx, f"{co1}_{co2}", tail, OPENED[name].isoformat(), src)


def _portfolio_row(name, day, alpha):
    """Real Portfolio.xlsx column names: spaces, "($)" suffixes, "Quantity1"."""
    idx, co1, co2, tail, src = name
    return {
        "Version": "9.3", "Tag": src, "Pair": f"{co1}-{co2}", "Co1": co1, "Co2": co2,
        "Tail": tail, "Index": idx, "Quantity1": 10, "Quantity2": 6,
        "W1": 0.6, "W2": 0.4, "Position_Multiplier": 1.3,
        "Co1 at Initiation": 200.0, "Co2 at Initiation": 216.0,
        "Index at Initiation": 620.0, "Trade Value Co1 ($)": 2000.0,
        "Trade Value Co2 ($)": 1300.0, "Total_Notional": 3300.0,
        "Sum_Dev_Value": -1.2, "Sum_Dev_CDF": 0.04, "Sum_Dev_Bucket": "0-10%",
        "Beta": 0.12, "Weighted_Score": 0.7, "Composite_Score": 0.65,
        "Entry_Spread_BPS": 12.5,
        "Trade Initiation Date": OPENED[name].isoformat(),
        "Trade Termination Date": (OPENED[name] + timedelta(days=21)).isoformat(),
        "Live Alpha Return (%)": alpha,
    }


def _completed_row(name, day):
    r = _portfolio_row(name, day, 1.8)
    r.pop("Live Alpha Return (%)")
    r.update({"Trade Termination Date": day.isoformat(),
              "Exit_Reason": "Early Exit - Day15_TakeProfit_8pct",
              "Holding_Days": 3.0, "Co1 at Exit": 205.0, "Co2 at Exit": 210.0,
              "Index at Exit": 625.0, "Co1_Return_Pct": 2.5, "Co2_Return_Pct": -2.8,
              "Index_Return_Pct": 0.8, "Final_Alpha_Return_Pct": -1.9})
    return r


def _shortlist_row(pair, idx, tail, result, reason=None):
    co1, co2 = pair.split("-")
    return {
        "Tag": abs(hash(pair)) % 4000, "Pair": pair, "Co1": co1, "Co2": co2,
        "Index": idx, "Tail": tail,
        "Tstat": 2.9, "Weighted_Spread_BPS": 11.0, "Earnings_Days_Out": 12,
        "Co1_Trending": 0, "Co2_Trending": 0, "Same_Direction_Result": "Pass",
        "Nominal_Direction_Result": "Pass", "Primary_Result": result,
        "Primary_Fail_Reason": reason, "Volume_Ratio": 1.2, "Rolling_Intraday_Vol": 0.02,
        "IV_Percentile": 40.0, "Volume_Dominance": 0.6, "True_Last_Hour_Volatility": 0.01,
        "Weighted_Score": 0.5, "Composite_Score": 0.55, "Sum_Deviation": -1.1,
        "Sum_Dev_Percentile": 8.0, "Sum_Dev_Bucket": 0, "Position_Multiplier": 1.3,
        "Is_Tradeable_Bucket": 1,
    }


def build_archive(root: Path, days: list[date]) -> Path:
    """Two positions open on day 1; one closes on day 2; a third opens on day 3."""
    for i, day in enumerate(days):
        d = root / day.isoformat()
        d.mkdir(parents=True)
        open_names = [P1, P2]
        closed = []
        if i >= 1:
            open_names.remove(P2)
            closed.append(P2)
        if i >= 2:
            open_names.append(P3)

        with pd.ExcelWriter(d / "Portfolio.xlsx") as w:
            pd.DataFrame([_portfolio_row(n, day, 0.5 * (i + 1)) for n in open_names]) \
                .to_excel(w, sheet_name="Portfolio", index=False)
            pd.DataFrame([{"Symbol": "IGV"}]).to_excel(w, sheet_name="Options", index=False)
        completed = pd.DataFrame([_completed_row(n, day) for n in closed])
        if completed.empty:
            completed = pd.DataFrame(columns=list(_completed_row(P1, day)))
        completed.to_excel(d / "Completed_Trades.xlsx", index=False)
        with pd.ExcelWriter(d / "V9_Shortlist.xlsx") as w:
            pd.DataFrame([
                _shortlist_row("AAPL-MSFT", "VGT", "L", "Pass"),
                _shortlist_row("ORCL-ADBE", "VGT", "U", "Fail", "spread_hurdle"),
                _shortlist_row("GS-MS", "VFH", "Lower", "PASS"),        # enum variants
            ]).to_excel(w, sheet_name="Shortlist", index=False)
        pd.DataFrame([{"Tag": 1, "Pair": "AAPL-MSFT", "Co1": "AAPL", "Co2": "MSFT",
                       "Index": "VGT", "Tail": "L", "Active": 1, "Reason": None,
                       "Sum_Deviation": -1.2, "Sum_Dev_Percentile": 0.04,
                       "Sum_Dev_Bucket": "0-10%", "Category1": "Tech_Core",
                       "Category2": "Tech_Core", "Model_Version": "V9.3"}]) \
            .to_excel(d / "trade_prefilter_active.xlsx", sheet_name="Active_Pairs", index=False)
        (d / "v9c_trading.log").write_text("2026-09-08 14:30:00 INFO stage 1 portfolio_load\n")
    return root


@pytest.fixture
def archive(tmp_path):
    days = [date(2026, 9, 8) + timedelta(days=i) for i in range(3)]
    return build_archive(tmp_path / "archive", days), days


@pytest.fixture
def conn(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'blotter.db'}")
    create_schema(engine)
    with engine.connect() as c:
        yield c


# ---------------------------------------------------------------------------
# archive discovery
# ---------------------------------------------------------------------------

def test_archived_days_are_sorted_and_filtered(tmp_path):
    for name in ["2026-09-10", "2026-09-08", "notes", "2026-09-09", "2026-9-1"]:
        (tmp_path / name).mkdir()
    assert archived_days(tmp_path) == [date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10)]


def test_missing_archive_is_empty(tmp_path):
    assert archived_days(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_positions_load_with_status_from_source_file(archive, conn):
    root, days = archive
    load_archive(root, conn)
    rows = conn.execute(text("SELECT tag, status, exit_reason, total_notional FROM positions ORDER BY tag")).all()
    by_tag = {r[0]: r for r in rows}
    assert by_tag[_tag(P1)][1] == "open"
    assert by_tag[_tag(P2)][1] == "closed"
    assert by_tag[_tag(P2)][2] == "Alpha Reached"      # free text normalised
    assert by_tag[_tag(P1)][3] == 3300.0


def test_instruments_seeded_from_positions(archive, conn):
    """positions.co1/co2 are foreign keys. The first run failed on this
    constraint; the adapter now seeds instruments from every ticker it sees."""
    root, _ = archive
    report = load_archive(root, conn)
    tickers = {r[0] for r in conn.execute(text("SELECT ticker FROM instruments")).all()}
    assert {"AAPL", "MSFT", "JPM", "BAC", "LLY", "JNJ", "ORCL", "ADBE", "GS", "MS"} <= tickers
    assert report.rows["instruments"] >= 10
    idx = conn.execute(text("SELECT idx FROM instruments WHERE ticker='JPM'")).scalar()
    assert idx == "VFH"


def test_evaluations_load_with_enum_normalisation(archive, conn):
    root, _ = archive
    load_archive(root, conn)
    rows = conn.execute(text(
        "SELECT pair, tail, sum_dev_bucket FROM pair_evaluations "
        "WHERE run_id = 'v92c_2026-09-08' AND stage='shortlist' ORDER BY pair")).all()
    by_pair = {r[0]: r for r in rows}
    assert by_pair["GS_MS"][1] == "L"                 # "Lower" -> "L"
    assert by_pair["AAPL_MSFT"][2] == "0-10%"         # decile 0 -> "0-10%"


def test_position_updates_come_from_daily_snapshots(archive, conn):
    root, days = archive
    load_archive(root, conn)
    updates = conn.execute(text(
        "SELECT update_date, live_alpha_return_pct, days_held FROM position_updates "
        "WHERE tag = :t ORDER BY update_date"), {"t": _tag(P1)}).all()
    assert [u[0] for u in updates] == [d.isoformat() for d in days]
    assert [u[1] for u in updates] == [0.5, 1.0, 1.5]
    assert [u[2] for u in updates] == [3, 4, 5]


def test_snapshot_per_day(archive, conn):
    root, days = archive
    load_archive(root, conn)
    snaps = conn.execute(text(
        "SELECT snapshot_date, position_count, total_gross_exposure FROM portfolio_snapshots ORDER BY 1")).all()
    assert [s[1] for s in snaps] == [2, 1, 2]
    assert all(s[2] > 0 for s in snaps)


# ---------------------------------------------------------------------------
# idempotence
# ---------------------------------------------------------------------------

def test_reload_is_a_noop(archive, conn):
    root, _ = archive
    load_archive(root, conn)
    counts_before = {t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                     for t in ("positions", "pair_evaluations", "position_updates", "portfolio_snapshots")}
    load_archive(root, conn)
    counts_after = {t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                    for t in counts_before}
    assert counts_after == counts_before


def test_since_filters_days(archive, conn):
    root, days = archive
    report = load_archive(root, conn, since=days[2])
    assert report.days == [days[2].isoformat()]


def test_position_status_transitions_open_to_closed(archive, conn):
    """A tag in Portfolio on day 1 and Completed on day 2 ends up closed once."""
    root, _ = archive
    load_archive(root, conn)
    rows = conn.execute(text(
        "SELECT COUNT(*), MAX(status) FROM positions WHERE tag = :t"), {"t": _tag(P2)}).one()
    assert rows == (1, "closed")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_report_lists_unmapped_columns(archive, conn):
    root, days = archive
    df = pd.read_excel(root / days[0].isoformat() / "V9_Shortlist.xlsx")
    df["Mystery_Column"] = 1
    df.to_excel(root / days[0].isoformat() / "V9_Shortlist.xlsx", index=False)
    report = load_archive(root, conn)
    assert "Mystery_Column" in report.unmapped_columns.get("shortlist", set())


def test_report_lists_unmapped_enums(archive, conn):
    root, days = archive
    d = root / days[1].isoformat()
    df = pd.read_excel(d / "Completed_Trades.xlsx")
    df["Tail"] = "Sideways"
    df.to_excel(d / "Completed_Trades.xlsx", index=False)
    report = load_archive(root, conn)
    assert any(k[0] == "tail" and k[1] == "Sideways" for k in report.unmapped_enums)


def test_report_flags_pending_log_tables(archive, conn):
    root, _ = archive
    report = load_archive(root, conn)
    assert set(report.pending) == set(LogParser.TABLES)
    assert conn.execute(text("SELECT COUNT(*) FROM orders")).scalar() == 0


def test_report_renders(archive, conn):
    root, _ = archive
    text_out = load_archive(root, conn).render()
    assert "days loaded: 3" in text_out and "pending a log parser" in text_out


def test_missing_file_is_reported_not_fatal(archive, conn):
    """A missing file is a reported gap, not an exception — the other files
    still load."""
    root, days = archive
    (root / days[0].isoformat() / "Completed_Trades.xlsx").unlink()
    report = load_archive(root, conn)
    assert "Completed_Trades.xlsx" in report.missing_files
    assert report.rows["pair_evaluations"] > 0


# ---------------------------------------------------------------------------
# real V9.3 files — fixtures taken from the run of 2026-09-08
# ---------------------------------------------------------------------------

REAL = Path(__file__).resolve().parent / "fixtures" / "v9"


@pytest.fixture
def real_archive(tmp_path):
    import shutil

    d = tmp_path / "archive" / "2026-09-08"
    d.mkdir(parents=True)
    for f in REAL.glob("*.xlsx"):
        shutil.copy(f, d / f.name)
    return tmp_path / "archive"


def test_real_files_load(real_archive, conn):
    """The four pre-trade files the live system actually writes."""
    report = load_archive(real_archive, conn)
    assert report.rows["pair_evaluations"] > 0
    stages = dict(conn.execute(text(
        "SELECT stage, COUNT(*) FROM pair_evaluations GROUP BY stage")).all())
    assert set(stages) == {"prefilter", "longlist", "shortlist", "rejected"}


def test_tag_is_not_unique_across_files(real_archive, conn):
    """Tag is a parameters-file row index, unique within a file but repeated
    across them. Keying on (run, tag) alone collapsed 5,682 real rows into
    2,533 — each stage silently overwriting the last."""
    load_archive(real_archive, conn)
    rows = conn.execute(text("""
        SELECT source_tag, COUNT(DISTINCT stage) FROM pair_evaluations
        GROUP BY source_tag HAVING COUNT(DISTINCT stage) > 1 LIMIT 1""")).all()
    assert rows, "fixture should contain a pair present at more than one stage"
    tag = rows[0][0]
    ids = conn.execute(text("SELECT eval_id FROM pair_evaluations WHERE source_tag = :t"),
                       {"t": tag}).all()
    assert len(ids) == len(set(i[0] for i in ids))


def test_pair_separator_is_normalised(real_archive, conn):
    """Files write IOT-GLW; the schema uses IOT_GLW."""
    load_archive(real_archive, conn)
    pairs = [r[0] for r in conn.execute(text("SELECT pair FROM pair_evaluations")).all()]
    assert pairs and not any("-" in p for p in pairs)
    assert all("_" in p for p in pairs)


def test_active_flag_becomes_evaluation_result(real_archive, conn):
    """Active is 0/1, not Pass/Fail."""
    load_archive(real_archive, conn)
    results = dict(conn.execute(text("""
        SELECT evaluation_result, COUNT(*) FROM pair_evaluations
        WHERE stage='prefilter' GROUP BY 1""")).all())
    assert set(results) == {"Approved", "Rejected"}


def test_free_text_reason_is_classified_and_preserved(real_archive, conn):
    """'Trending filter: CAKE - Positive trending: 60.14% excess return over
    3M (threshold: 60.00%)' is a sentence, not an enum. Category and original
    are both kept."""
    load_archive(real_archive, conn)
    row = conn.execute(text("""
        SELECT rejection_reason, rejection_detail FROM pair_evaluations
        WHERE rejection_reason = 'trend_filter' LIMIT 1""")).first()
    if row:
        assert row[0] == "trend_filter"
        assert "Trending filter" in row[1]
    categories = {r[0] for r in conn.execute(text(
        "SELECT DISTINCT rejection_reason FROM pair_evaluations WHERE rejection_reason IS NOT NULL")).all()}
    assert categories <= {"trend_filter", "alpha_variance", "spread_hurdle", "missing_data",
                          "two_day_deviation", "same_direction", "nominal_direction",
                          "sum_dev_exclusion", "earnings_filter", "tstat", "other"}


def test_approved_rows_carry_no_rejection_reason(real_archive, conn):
    load_archive(real_archive, conn)
    n = conn.execute(text("""
        SELECT COUNT(*) FROM pair_evaluations
        WHERE evaluation_result = 'Approved' AND rejection_reason IS NOT NULL""")).scalar()
    assert n == 0


def test_cumulative_archive_is_filtered_to_the_day(real_archive, conn):
    """Rejected_Trades_Archive spans every run since it was created — the
    first real file ran from 2025-11-25 to today. Loading it wholesale would
    attribute ten months of rejections to one day."""
    report = load_archive(real_archive, conn)
    assert report.filtered_rows.get("Rejected_Trades_Archive.xlsx", 0) > 0
    dates = {r[0][:10] for r in conn.execute(text(
        "SELECT evaluated_at FROM pair_evaluations WHERE stage='rejected'")).all()}
    assert dates == {"2026-09-08"}


def test_ticker_column_aliases(real_archive, conn):
    """Prefilter says Co1/Co2; longlist says Ticker1/Ticker2."""
    load_archive(real_archive, conn)
    for stage in ("prefilter", "longlist"):
        n = conn.execute(text(
            "SELECT COUNT(*) FROM pair_evaluations WHERE stage=:s AND co1 IS NOT NULL"),
            {"s": stage}).scalar()
        assert n > 0, f"{stage} lost its tickers"


def test_real_completed_trades_load(real_archive, conn):
    """Positions from the real file: 45 columns, free-text exit reasons,
    nulls in ten columns the schema had marked NOT NULL."""
    load_archive(real_archive, conn)
    n = conn.execute(text("SELECT COUNT(*) FROM positions WHERE status='closed'")).scalar()
    assert n > 0
    reasons = {r[0] for r in conn.execute(text(
        "SELECT DISTINCT exit_reason FROM positions WHERE exit_reason IS NOT NULL")).all()}
    assert reasons <= {"Date Reached", "Earnings Alert", "Alpha Reached",
                       "Stop Loss Triggered", "Delisting", "Manual"}


def test_exit_detail_preserves_the_original_text(real_archive, conn):
    """'Early Exit - Day15_TakeProfit_8pct' -> Alpha Reached, with the day
    number kept. 'Pre-Holiday Exit (term date ... is non-trading day)' and
    'Past Due (was ...)' are scheduled exits displaced by the calendar; the
    schema has no reason for them, so they map to Date Reached."""
    load_archive(real_archive, conn)
    rows = conn.execute(text(
        "SELECT exit_reason, exit_detail FROM positions WHERE exit_detail IS NOT NULL")).all()
    assert rows
    for reason, detail in rows:
        if detail.startswith("Early Exit"):
            assert reason == "Alpha Reached" and "Day" in detail
        if detail.startswith("Earnings -"):
            assert reason == "Earnings Alert" and "reports" in detail
        if detail.startswith(("Past Due", "Pre-Holiday")):
            assert reason == "Date Reached"


def test_duplicate_positions_are_kept_not_merged(real_archive, conn):
    """The real file recorded five positions twice — same pair, same
    initiation date, DIFFERENT termination dates, exit reasons and alpha.
    AXSM_DAWN is both 'Date Reached' (+0.49%) and 'Past Due' (-0.48%). A
    silent overwrite would have lost half of a contradiction."""
    report = load_archive(real_archive, conn)
    assert report.duplicate_positions.get("completed")
    dup = report.duplicate_positions["completed"][0]
    base = dup.split("#")[0]
    rows = conn.execute(text(
        "SELECT tag, exit_detail, final_alpha_return_pct FROM positions WHERE tag LIKE :p"),
        {"p": base + "%"}).all()
    assert len(rows) == 2
    assert rows[0][2] != rows[1][2], "the two records disagree; both must survive"


def test_rows_without_a_key_are_skipped_and_counted(real_archive, conn):
    """One row in the real file is entirely blank below Tag."""
    report = load_archive(real_archive, conn)
    assert report.skipped_rows.get("completed", 0) >= 1


def test_position_tag_is_reconstructed(real_archive, conn):
    """Files carry no position tag; it is rebuilt from index, legs, tail and
    initiation date."""
    load_archive(real_archive, conn)
    import re

    for (tag,) in conn.execute(text("SELECT tag FROM positions")).all():
        base = tag.split("#")[0]
        # index _ legs _ tail _ YYYYMMDD _ NNN — legs may contain underscores
        # where a ticker was hyphenated (LILA-K -> LILA_K).
        assert re.fullmatch(r"(VGT|VFH|VHT|VCR|VIS|UNK)_.+_[LUX]_\d{8}_\d{3}", base), tag


def test_empty_portfolio_sheet_loads_without_error(real_archive, conn):
    """The real Portfolio.xlsx has six sheets and zero position rows."""
    report = load_archive(real_archive, conn)
    assert "Portfolio.xlsx" not in report.missing_files


def test_retired_columns_are_not_reported_as_gaps(real_archive, conn):
    """Treasury and IGV columns survive from features V9.3 removed."""
    report = load_archive(real_archive, conn)
    for key in ("portfolio", "completed"):
        gaps = report.unmapped_columns.get(key, set())
        assert not {c for c in gaps if "Treasury" in c or c.startswith("IGV")}


def test_known_extra_columns_are_not_reported_as_gaps(real_archive, conn):
    """The rejected archive carries treasury columns V9.3 no longer uses.
    Known extras must not drown the report's real findings."""
    report = load_archive(real_archive, conn)
    assert "Treasury_at_Entry" not in report.unmapped_columns.get("rejected", set())


# ---------------------------------------------------------------------------
# mapping sanity
# ---------------------------------------------------------------------------

def test_bucket_normalisation():
    assert M.normalise_bucket("0-10%") == "0-10%"
    assert M.normalise_bucket("90-100") == "90-100%"
    assert M.normalise_bucket(0) == "0-10%"
    assert M.normalise_bucket(9) == "90-100%"
    assert M.normalise_bucket(None) is None


def test_reason_classification():
    from src.adapter.mapping import classify_reason

    cat, raw = classify_reason("Trending filter: ASAN - Negative trending: -73.49% excess return over 12M (threshold: -70.00%)")
    assert cat == "trend_filter" and "ASAN" in raw
    assert classify_reason("Failed alpha_variance with leniency")[0] == "alpha_variance"
    assert classify_reason("Failed spread with leniency")[0] == "spread_hurdle"
    assert classify_reason("Missing historical data")[0] == "missing_data"
    assert classify_reason(None) == (None, None)
    assert classify_reason("something new")[0] == "other"


def test_exit_reason_classification():
    from src.adapter.mapping import classify_exit_reason

    assert classify_exit_reason("Date Reached")[0] == "Date Reached"
    assert classify_exit_reason("Early Exit - Day15_TakeProfit_8pct")[0] == "Alpha Reached"
    assert classify_exit_reason("Earnings - NSSC reports 2026-02-02")[0] == "Earnings Alert"
    assert classify_exit_reason("Pre-Holiday Exit (term date 2025-12-25 is non-trading day)")[0] == "Date Reached"
    assert classify_exit_reason("Past Due (was 2026-02-17)")[0] == "Date Reached"
    assert classify_exit_reason(None) == (None, None)
    canonical, raw = classify_exit_reason("Earnings - DXC reports 2026-01-30")
    assert "DXC" in raw and "2026-01-30" in raw


def test_position_tag_disambiguation():
    from src.adapter.mapping import disambiguate, position_tag

    t = position_tag("VHT", "AXSM_DAWN", "U", "2026-01-27", 3173)
    assert t == "VHT_AXSM_DAWN_U_20260127_173"
    assert disambiguate(t, "2026-02-17", 1).endswith("#20260217")
    assert disambiguate(t, None, 1).endswith("#dup1")


def test_pair_normalisation():
    from src.adapter.mapping import normalise_pair

    assert normalise_pair("IOT-GLW") == "IOT_GLW"
    assert normalise_pair("BRK-B-AAPL") == "BRK_B_AAPL"
    assert normalise_pair(None) is None


def test_every_mapped_target_exists_in_schema(conn):
    """A mapping to a column the schema lacks would silently drop data."""
    def cols(table):
        return {c[1] for c in conn.execute(text(f"PRAGMA table_info({table})")).all()}
    positions = cols("positions")
    for src, dst in M.COMPLETED_COLUMNS.items():
        if not dst.startswith("_") and dst != "source_tag":
            assert dst in positions, f"{src} -> {dst} not in positions"
    evaluations = cols("pair_evaluations")
    for name, columns in (("prefilter", M.PREFILTER_COLUMNS), ("longlist", M.LONGLIST_COLUMNS),
                          ("shortlist", M.SHORTLIST_COLUMNS), ("rejected", M.REJECTED_COLUMNS)):
        for src, dst in columns.items():
            if not dst.startswith("_") and dst != "source_tag":
                assert dst in evaluations, f"{name}: {src} -> {dst} not in pair_evaluations"


def test_adapter_does_not_import_the_trading_system():
    """The adapter reads V9.2C's files as external data. It must not reach
    into the live-proven code."""
    import pathlib

    for path in (pathlib.Path(__file__).resolve().parent.parent / "src" / "adapter").glob("*.py"):
        source = path.read_text()
        import re

        for forbidden in ("workflow_v9", "portfolio_management", "tool_box", "config_helper"):
            assert not re.search(rf"^\s*(from|import)\s+\S*{forbidden}", source, re.M), \
                f"{path.name} imports {forbidden}"
