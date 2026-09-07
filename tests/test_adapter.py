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

OPENED = {"VGT_AAPL_MSFT_L_20260901_001": date(2026, 9, 5),
          "VFH_JPM_BAC_U_20260901_002": date(2026, 9, 5),
          "VHT_LLY_JNJ_L_20260903_003": date(2026, 9, 9)}


def _portfolio_row(tag, day, alpha):
    idx, co1, co2, tail = tag.split("_")[:4]
    return {
        "Tag": tag, "Pair": f"{co1}_{co2}", "Co1": co1, "Co2": co2, "Index": idx,
        "Tail": tail, "Model_Version": "V9.2C", "W1": 0.6, "W2": 0.4,
        "Quantity_1": 10, "Quantity_2": 6, "Trade_Value_Co1": 2000.0,
        "Trade_Value_Co2": 1300.0, "Sum_Deviation": -1.2, "Sum_Dev_Bucket": "0-10%",
        "Weighted_Score": 0.7, "Composite_Score": 0.65, "Beta": 0.12,
        "Trade_Initiation_Date": OPENED.get(tag, day - timedelta(days=3)).isoformat(),
        "Position_Multiplier": 1.3, "Co1_At_Initiation": 200.0,
        "Co2_At_Initiation": 216.0, "Index_At_Initiation": 620.0,
        "Live_Alpha_Return": alpha,
    }


def _completed_row(tag, day):
    r = _portfolio_row(tag, day, 1.8)
    r.update({"Termination_Date": day.isoformat(), "Exit_Reason": "Stop_Loss",
              "Holding_Days": 3, "Co1_At_Exit": 205.0, "Co2_At_Exit": 210.0,
              "Index_At_Exit": 625.0, "Final_Alpha_Return": -1.9})
    return r


def _shortlist_row(pair, idx, tail, result, reason=None):
    co1, co2 = pair.split("_")
    return {
        "Pair": pair, "Co1": co1, "Co2": co2, "Index": idx, "Tail": tail,
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
        open_tags = ["VGT_AAPL_MSFT_L_20260901_001", "VFH_JPM_BAC_U_20260901_002"]
        closed = []
        if i >= 1:
            open_tags.remove("VFH_JPM_BAC_U_20260901_002")
            closed.append("VFH_JPM_BAC_U_20260901_002")
        if i >= 2:
            open_tags.append("VHT_LLY_JNJ_L_20260903_003")

        with pd.ExcelWriter(d / "Portfolio.xlsx") as w:
            pd.DataFrame([_portfolio_row(t, day, 0.5 * (i + 1)) for t in open_tags]) \
                .to_excel(w, sheet_name="Portfolio", index=False)
            pd.DataFrame([{"note": "options sheet"}]).to_excel(w, sheet_name="Options", index=False)
        pd.DataFrame([_completed_row(t, day) for t in closed] or
                     [dict.fromkeys(_completed_row("VGT_X_Y_L_20260901_000", day))][:0]) \
            .to_excel(d / "Completed_Trades.xlsx", index=False)
        pd.DataFrame([
            _shortlist_row("AAPL_MSFT", "VGT", "L", "Pass"),
            _shortlist_row("ORCL_ADBE", "VGT", "U", "Fail", "spread_hurdle"),
            _shortlist_row("GS_MS", "VFH", "Lower", "PASS"),           # enum variants
        ]).to_excel(d / "V9_Shortlist.xlsx", index=False)
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
    assert by_tag["VGT_AAPL_MSFT_L_20260901_001"][1] == "open"
    assert by_tag["VFH_JPM_BAC_U_20260901_002"][1] == "closed"
    assert by_tag["VFH_JPM_BAC_U_20260901_002"][2] == "Stop Loss Triggered"   # normalised
    assert by_tag["VGT_AAPL_MSFT_L_20260901_001"][3] == 3300.0               # derived


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
        "SELECT pair, tail, primary_result, primary_fail_reason, sum_dev_bucket "
        "FROM pair_evaluations WHERE run_id = 'v92c_2026-09-08' ORDER BY pair")).all()
    by_pair = {r[0]: r for r in rows}
    assert by_pair["GS_MS"][1] == "L"                 # "Lower" -> "L"
    assert by_pair["GS_MS"][2] == "Pass"              # "PASS" -> "Pass"
    assert by_pair["ORCL_ADBE"][3] == "spread_hurdle"
    assert by_pair["AAPL_MSFT"][4] == "0-10%"         # decile 0 -> "0-10%"


def test_position_updates_come_from_daily_snapshots(archive, conn):
    root, days = archive
    load_archive(root, conn)
    updates = conn.execute(text(
        "SELECT update_date, live_alpha_return_pct, days_held FROM position_updates "
        "WHERE tag = 'VGT_AAPL_MSFT_L_20260901_001' ORDER BY update_date")).all()
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
        "SELECT COUNT(*), MAX(status) FROM positions WHERE tag = 'VFH_JPM_BAC_U_20260901_002'")).one()
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
    df["Exit_Reason"] = "Weird_Reason"
    df.to_excel(d / "Completed_Trades.xlsx", index=False)
    report = load_archive(root, conn)
    assert any(k[0] == "exit_reason" and k[1] == "Weird_Reason" for k in report.unmapped_enums)


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
    root, days = archive
    (root / days[0].isoformat() / "Completed_Trades.xlsx").unlink()
    report = load_archive(root, conn)
    assert "Completed_Trades.xlsx" in report.missing_files
    assert report.rows["positions"] > 0


# ---------------------------------------------------------------------------
# mapping sanity
# ---------------------------------------------------------------------------

def test_bucket_normalisation():
    assert M.normalise_bucket("0-10%") == "0-10%"
    assert M.normalise_bucket("90-100") == "90-100%"
    assert M.normalise_bucket(0) == "0-10%"
    assert M.normalise_bucket(9) == "90-100%"
    assert M.normalise_bucket(None) is None


def test_every_mapped_target_exists_in_schema(conn):
    """A mapping to a column the schema lacks would silently drop data."""
    def cols(table):
        return {c[1] for c in conn.execute(text(f"PRAGMA table_info({table})")).all()}
    positions = cols("positions")
    for src, dst in M.COMPLETED_COLUMNS.items():
        if not dst.startswith("_"):
            assert dst in positions, f"{src} -> {dst} not in positions"
    evaluations = cols("pair_evaluations")
    for src, dst in M.EVALUATION_COLUMNS.items():
        assert dst in evaluations, f"{src} -> {dst} not in pair_evaluations"


def test_adapter_does_not_import_the_trading_system():
    """The adapter reads V9.2C's files as external data. It must not reach
    into the live-proven code."""
    import pathlib

    for path in (pathlib.Path(__file__).resolve().parent.parent / "src" / "adapter").glob("*.py"):
        source = path.read_text()
        for forbidden in ("workflow_v9", "portfolio_management", "tool_box", "LAM", "config_helper"):
            assert forbidden not in source, f"{path.name} references {forbidden}"
