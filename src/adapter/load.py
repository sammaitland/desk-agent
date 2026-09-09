"""Load V9.2C output files into the desk agent's blotter.

Reads a directory of dated archives — one subdirectory per trading day, each
holding that day's copies of the trading system's Excel outputs and log — and
populates the blotter tables. Runs any number of times over the same archive
without duplicating anything.

## What it does and does not touch

The adapter reads V9.2C's files as external data. It does not import from,
modify, or depend on V9.2C's code. That separation is deliberate: V9.2C is the
live-proven system, and a bridge that reached into it would risk changing what
it is bridging.

## Idempotence

Every insert is keyed — positions by tag, evaluations by (run date, pair),
updates by (tag, date) — and existing rows are updated rather than duplicated.
Re-running over a day already loaded is a no-op. This is what makes "archive
daily, load whenever" safe.

## What is built and what is pending

Structured files are handled here: Portfolio.xlsx, Completed_Trades.xlsx,
V9_Shortlist.xlsx, and the day-to-day diff that yields position_updates and
portfolio_snapshots. Orders, fills, workflow stages, risk checks and system
events live in v9c_trading.log, which needs a parser written against a real
sample. `LogParser` is the interface it will fill; until then those tables stay
empty and the report says so.

## Gaps are reported, not hidden

Every source column that could not be mapped, every enum value that did not
normalise, every blotter column left NULL for want of a source — all of it
lands in `LoadReport`. The first real run is expected to produce a long one.
That is the point: it is the list of corrections to make to mapping.py.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.adapter import mapping as M


# ---------------------------------------------------------------------------
# Archive layout
# ---------------------------------------------------------------------------

DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def archived_days(archive: Path) -> list[date]:
    """Dated subdirectories, oldest first."""
    if not archive.exists():
        return []
    return sorted(date.fromisoformat(p.name) for p in archive.iterdir()
                  if p.is_dir() and DATE_DIR.match(p.name))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    days: list[str] = field(default_factory=list)
    rows: Counter = field(default_factory=Counter)          # table -> rows written
    unmapped_columns: dict[str, set] = field(default_factory=dict)  # file -> source cols
    unmapped_enums: Counter = field(default_factory=Counter)       # (field, value) -> count
    missing_files: list[str] = field(default_factory=list)
    filtered_rows: dict[str, int] = field(default_factory=dict)  # cumulative files, rows outside the day
    skipped_rows: dict[str, int] = field(default_factory=dict)   # rows lacking a usable key
    duplicate_positions: dict[str, list] = field(default_factory=dict)  # same position recorded twice
    pending: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"days loaded: {len(self.days)}" + (f" ({self.days[0]} .. {self.days[-1]})" if self.days else "")]
        lines.append("rows written:")
        for table, n in sorted(self.rows.items()):
            lines.append(f"  {table:<22} {n:>6}")
        if self.unmapped_columns:
            lines.append("source columns with no blotter mapping (correct mapping.py):")
            for f, cols in sorted(self.unmapped_columns.items()):
                lines.append(f"  {f}: {sorted(cols)}")
        if self.unmapped_enums:
            lines.append("enum values that did not normalise:")
            for (fld, val), n in self.unmapped_enums.most_common(20):
                lines.append(f"  {fld} = {val!r}  x{n}")
        if self.filtered_rows:
            for f, n in sorted(self.filtered_rows.items()):
                lines.append(f"cumulative file {f}: {n} row(s) outside the loaded day, skipped")
        if self.duplicate_positions:
            for f, tags in sorted(self.duplicate_positions.items()):
                lines.append(f"{f}: {len(tags)} position(s) recorded more than once — "
                             f"kept and suffixed, NOT merged: {tags[:3]}")
        if self.skipped_rows:
            for f, n in sorted(self.skipped_rows.items()):
                lines.append(f"{f}: {n} row(s) skipped — no pair or initiation date")
        if self.missing_files:
            lines.append(f"files absent in some days: {sorted(set(self.missing_files))}")
        if self.pending:
            lines.append("tables pending a log parser: " + ", ".join(self.pending))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading and mapping
# ---------------------------------------------------------------------------

def _read(path: Path, sheet: str | int = 0) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        return pd.read_excel(path, sheet_name=0)


def _map_columns(df: pd.DataFrame, columns: dict[str, str], file_key: str,
                 report: LoadReport) -> pd.DataFrame:
    """Rename source columns to blotter columns; report what did not map."""
    lower = {c.lower(): c for c in df.columns}
    rename, matched = {}, set()
    for src, dst in columns.items():
        actual = src if src in df.columns else lower.get(src.lower())
        if actual is not None:
            rename[actual] = dst
            matched.add(actual)
    unmapped = set(df.columns) - matched - M.KNOWN_UNMAPPED.get(file_key, set())
    if unmapped:
        report.unmapped_columns.setdefault(file_key, set()).update(unmapped)
    return df.rename(columns=rename)[list(rename.values())]


def _enum(value, table: dict, field_name: str, report: LoadReport):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    if key in table:
        return table[key]
    report.unmapped_enums[(field_name, str(value))] += 1
    return str(value)


def _clean(value):
    """NaN -> None; numpy scalars -> Python; timestamps -> ISO."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ", timespec="seconds")
    if hasattr(value, "item"):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert(conn: Connection, table: str, rows: list[dict], key: tuple[str, ...]) -> int:
    """Insert or update by key, leaving child rows intact.

    An earlier version deleted then re-inserted, which fails the moment a
    child table references the row — position_updates from day one blocked
    the day-two reload of the same position. INSERT ... ON CONFLICT DO UPDATE
    is supported by SQLite (3.24+) and PostgreSQL with identical syntax and
    updates in place.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    non_key = [c for c in cols if c not in key]
    sets = ", ".join(f"{c} = excluded.{c}" for c in non_key) or f"{key[0]} = excluded.{key[0]}"
    conn.execute(text(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)}) "
        f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}"), rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Instruments: seeded from whatever tickers the day's files mention
# ---------------------------------------------------------------------------

def seed_instruments(day_dir: Path, conn: Connection, report: LoadReport) -> None:
    """positions.co1/co2 and orders.ticker are foreign keys to instruments.
    V9.2C has no instrument file the adapter can rely on daily, so every
    ticker seen in the day's positions and evaluations is upserted with its
    index. The universe file, when loaded, refines names and delisting state;
    this guarantees referential integrity before any position row lands.
    The first test run failed on exactly this constraint."""
    seen: dict[str, str] = {}
    for key, cols in (("portfolio", M.POSITION_COLUMNS),
                      ("completed", M.COMPLETED_COLUMNS),
                      ("prefilter", M.PREFILTER_COLUMNS),
                      ("shortlist", M.SHORTLIST_COLUMNS)):
        sheet = M.SHEETS[key]
        df = _read(day_dir / M.FILES[key], sheet=sheet)
        if df is None or df.empty:
            continue
        mapped = _map_columns(df, cols, key, LoadReport())   # gaps reported by the real loaders
        for _, r in mapped.iterrows():
            idx = _clean(r.get("idx")) or "UNKNOWN"
            for leg in ("co1", "co2"):
                t = _clean(r.get(leg))
                if t:
                    seen.setdefault(str(t), str(idx))
    existing = {r[0] for r in conn.execute(text("SELECT ticker FROM instruments")).all()}
    rows = [{"ticker": t, "name": t, "idx": i, "is_active": 1,
             "delisted_date": None, "delisting_type": None, "acquirer": None}
            for t, i in seen.items() if t not in existing]
    if rows:
        conn.execute(text("INSERT INTO instruments (ticker, name, idx, is_active, delisted_date, "
                          "delisting_type, acquirer) VALUES (:ticker, :name, :idx, :is_active, "
                          ":delisted_date, :delisting_type, :acquirer)"), rows)
        report.rows["instruments"] += len(rows)


# ---------------------------------------------------------------------------
# Per-file loaders
# ---------------------------------------------------------------------------

def _position_rows(df, columns: dict, file_key: str, status: str,
                   report: LoadReport) -> list[dict]:
    """Map Portfolio.xlsx or Completed_Trades.xlsx to positions.

    The blotter's primary key is the position tag
    (`VGT_AAPL_MSFT_L_20260815_001`), which these files do not carry — their
    `Tag` is the parameters-file row index and is not unique either: the real
    Completed_Trades held 185 rows over 177 tags, the same pair traded again
    later. The key is reconstructed from what does identify a position:
    index, legs, tail and initiation date.
    """
    mapped = _map_columns(df, columns, file_key, report)
    rows, seen = [], {}
    for _, r in mapped.iterrows():
        rec = {c: _clean(r.get(c)) for c in mapped.columns if not c.startswith("_")}
        rec["pair"] = M.normalise_pair(rec.get("pair"))
        rec["tail"] = _enum(rec.get("tail"), M.TAILS, "tail", report)
        rec["sum_dev_bucket"] = M.normalise_bucket(rec.get("sum_dev_bucket"))
        rec["status"] = status
        if rec.get("version") is not None:
            rec["version"] = str(rec["version"])

        source_tag = rec.pop("source_tag", None)
        init = str(rec.get("trade_initiation_date") or "")[:10]
        if not rec.get("pair") or not init or init in ("None", "NaT", "nan"):
            report.skipped_rows[file_key] = report.skipped_rows.get(file_key, 0) + 1
            continue
        tag = M.position_tag(rec.get("idx"), rec["pair"], rec.get("tail"), init, source_tag)
        if tag in seen:
            tag = M.disambiguate(tag, rec.get("termination_date"), seen[tag])
            report.duplicate_positions.setdefault(file_key, []).append(tag)
        seen[tag] = seen.get(tag, 0) + 1
        rec["tag"] = tag
        rec["trade_initiation_date"] = init
        if rec.get("termination_date"):
            rec["termination_date"] = str(rec["termination_date"])[:10]

        if "_exit_reason" in mapped.columns:
            canonical, raw = M.classify_exit_reason(_clean(r.get("_exit_reason")))
            rec["exit_reason"] = canonical
            rec["exit_detail"] = raw

        # Total_Notional is supplied by the real files; derive only if absent.
        if rec.get("total_notional") is None:
            tv1, tv2 = rec.get("trade_value_co1") or 0, rec.get("trade_value_co2") or 0
            rec["total_notional"] = round(float(tv1) + float(tv2), 2)
        rows.append(rec)
    return rows


def load_positions(day_dir: Path, conn: Connection, report: LoadReport) -> None:
    for key, status, columns in (("portfolio", "open", M.POSITION_COLUMNS),
                                 ("completed", "closed", M.COMPLETED_COLUMNS)):
        df = _read(day_dir / M.FILES[key], sheet=M.SHEETS[key])
        if df is None:
            report.missing_files.append(M.FILES[key])
            continue
        rows = _position_rows(df, columns, key, status, report)
        report.rows["positions"] += _upsert(conn, "positions",
                                            _fit(conn, "positions", rows), ("tag",))


def _evaluation_rows(df, columns: dict, file_key: str, stage: str, day: date,
                     report: LoadReport) -> list[dict]:
    """Common mapping for every pre-trade file. Each contributes rows to
    pair_evaluations, tagged with the stage it came from, so a query can ask
    'what reached the shortlist' as well as 'what was evaluated'."""
    mapped = _map_columns(df, columns, file_key, report)
    run_id = f"v92c_{day.isoformat()}"
    rows = []
    for _, r in mapped.iterrows():
        rec = {c: _clean(r.get(c)) for c in mapped.columns if not c.startswith("_")}
        source_tag = rec.pop("source_tag", None)
        if source_tag is None:
            continue
        rec["eval_id"] = M.evaluation_key(day.isoformat(), source_tag, stage)
        rec["run_id"] = run_id
        rec["stage"] = stage
        rec["pair"] = M.normalise_pair(rec.get("pair"))
        rec["tail"] = _enum(rec.get("tail"), M.TAILS, "tail", report)
        rec["sum_dev_bucket"] = M.normalise_bucket(rec.get("sum_dev_bucket"))
        rec.setdefault("evaluated_at", f"{day.isoformat()} 14:30:00")

        # Prefilter: Active 0/1 plus free-text Reason
        active = _clean(r.get("_active")) if "_active" in mapped.columns else None
        if active is not None:
            approved = int(active) == 1
            rec["evaluation_result"] = "Approved" if approved else "Rejected"
            if not approved:
                category, raw = M.classify_reason(_clean(r.get("_reason")))
                rec["rejection_reason"] = category
                rec["rejection_detail"] = raw

        # Longlist: Pass/Fail per filter
        for src, dst in (("_status", "primary_result"),
                         ("_earnings_result", "earnings_result"),
                         ("_spread_result", "spread_result"),
                         ("_alpha_result", "alpha_result"),
                         ("_two_day_result", "two_day_result"),
                         ("_trend_result", "trend_result")):
            if src in mapped.columns:
                rec[dst] = _enum(_clean(r.get(src)), M.PASS_FAIL, dst, report)
        for f in ("same_direction_result", "nominal_direction_result"):
            if rec.get(f) is not None:
                rec[f] = _enum(rec[f], M.PASS_FAIL, f, report)
        rows.append(rec)
    return rows


def load_evaluations(day_dir: Path, day: date, conn: Connection, report: LoadReport) -> None:
    """Four files, one table. Later stages overwrite earlier ones on the same
    eval_id, so a pair that reached the shortlist carries its shortlist row —
    and the stage column records how far it got."""
    run_id = f"v92c_{day.isoformat()}"
    _upsert(conn, "workflow_runs", [{"run_id": run_id, "run_date": day.isoformat(),
                                     "started_at": f"{day.isoformat()} 14:30:00",
                                     "completed_at": None, "outcome": "completed"}], ("run_id",))
    report.rows["workflow_runs"] += 1

    for key, stage, columns in (
        ("prefilter", "prefilter", M.PREFILTER_COLUMNS),
        ("longlist", "longlist", M.LONGLIST_COLUMNS),
        ("shortlist", "shortlist", M.SHORTLIST_COLUMNS),
        ("rejected", "rejected", M.REJECTED_COLUMNS),
    ):
        df = _read(day_dir / M.FILES[key], sheet=M.SHEETS[key])
        if df is None:
            report.missing_files.append(M.FILES[key])
            continue
        if key == "rejected" and "Archived_At" in df.columns:
            # The archive is cumulative — it spans every run since it was
            # created (2025-11-25 to today in the first real file). Only
            # today's rows belong to today's evaluation.
            before = len(df)
            df = df[df["Archived_At"].astype(str).str[:10] == day.isoformat()]
            report.filtered_rows[M.FILES[key]] = before - len(df)
        rows = _evaluation_rows(df, columns, key, stage, day, report)
        report.rows["pair_evaluations"] += _upsert(
            conn, "pair_evaluations", _fit(conn, "pair_evaluations", rows), ("eval_id",))


def load_snapshot_and_updates(day_dir: Path, day: date, prev_dir: Path | None,
                              conn: Connection, report: LoadReport) -> None:
    """position_updates from today's Portfolio.xlsx; the snapshot from its totals."""
    df = _read(day_dir / M.FILES["portfolio"], sheet=M.SHEETS["portfolio"])
    if df is None:
        return
    mapped = _map_columns(df, M.POSITION_COLUMNS, "portfolio", report)
    run_id = f"v92c_{day.isoformat()}"
    updates, gross = [], 0.0
    for _, r in mapped.iterrows():
        # Same reconstruction as _position_rows — the file has no tag column.
        pair = M.normalise_pair(_clean(r.get("pair")))
        init_date = str(_clean(r.get("trade_initiation_date")) or "")[:10]
        if not pair or not init_date or init_date in ("None", "NaT", "nan"):
            continue
        tag = M.position_tag(_clean(r.get("idx")), pair,
                             _enum(_clean(r.get("tail")), M.TAILS, "tail", report),
                             init_date, _clean(r.get("source_tag")))
        try:
            held = (day - date.fromisoformat(init_date)).days
        except ValueError:
            held = 0
        gross += float(_clean(r.get("trade_value_co1")) or 0) + float(_clean(r.get("trade_value_co2")) or 0)
        updates.append({
            "tag": tag, "update_date": day.isoformat(), "run_id": run_id,
            "live_co1_price": None, "live_co2_price": None, "live_index_price": None,
            # The real Portfolio.xlsx carries live returns per leg alongside
            # the alpha; the synthetic generator only ever had the alpha.
            "co1_return_pct": _clean(r.get("_co1_return")),
            "co2_return_pct": _clean(r.get("_co2_return")),
            "index_return_pct": _clean(r.get("_index_return")),
            "co1_alpha_pct": None, "co2_alpha_pct": None,
            "live_alpha_return_pct": _clean(r.get("_live_alpha")),
            "days_held": held,
        })
    if updates:
        report.rows["position_updates"] += _upsert(
            conn, "position_updates", _fit(conn, "position_updates", updates), ("tag", "update_date"))
    _upsert(conn, "portfolio_snapshots", [{
        "run_id": run_id, "snapshot_date": day.isoformat(),
        "position_count": len(updates), "account_value": None,
        "total_gross_exposure": round(gross, 2), "leverage": None,
        "dollar_weighted_beta": None, "staleness_minutes": None,
    }], ("run_id",))
    report.rows["portfolio_snapshots"] += 1


def _fit(conn: Connection, table: str, rows: list[dict]) -> list[dict]:
    """Keep only columns the table has; fill the rest with NULL so every row
    has the same key set (the executemany needs that)."""
    if not rows:
        return rows
    cols = [c[1] for c in conn.execute(text(f"PRAGMA table_info({table})")).all()] \
        if conn.engine.dialect.name == "sqlite" else \
        [r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t ORDER BY ordinal_position"),
            {"t": table}).all()]
    return [{c: r.get(c) for c in cols} for r in rows]


# ---------------------------------------------------------------------------
# Log parser interface — pending a real sample
# ---------------------------------------------------------------------------

class LogParser:
    """Turns v9c_trading.log into orders, fills, workflow stages, risk checks
    and system events. Not implemented: the log format is unknown until a real
    day is archived. This is the interface the parser will fill."""

    TABLES = ("orders", "fills", "order_allocations", "workflow_stages",
              "risk_checks", "system_events")

    def parse(self, log_path: Path, day: date) -> dict[str, list[dict]]:
        return {t: [] for t in self.TABLES}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_archive(archive: Path, conn: Connection, since: date | None = None,
                 parser: LogParser | None = None) -> LoadReport:
    """Load every archived day (optionally from `since`) into the blotter."""
    parser = parser or LogParser()
    report = LoadReport()
    days = [d for d in archived_days(archive) if since is None or d >= since]
    prev_dir = None
    for day in days:
        day_dir = archive / day.isoformat()
        # SQLAlchemy autobegins a transaction on the first read; an explicit
        # begin() then raises. Commit whatever is open so each day is its own
        # unit of work — a day that fails leaves earlier days committed.
        if conn.in_transaction():
            conn.commit()
        with conn.begin():
            seed_instruments(day_dir, conn, report)
            load_positions(day_dir, conn, report)
            load_evaluations(day_dir, day, conn, report)
            load_snapshot_and_updates(day_dir, day, prev_dir, conn, report)
            parsed = parser.parse(day_dir / M.FILES["log"], day)
            for table, rows in parsed.items():
                if rows:
                    key = {"orders": ("order_id",), "fills": ("fill_id",),
                           "order_allocations": ("allocation_id",),
                           "workflow_stages": ("run_id", "stage_number"),
                           "risk_checks": ("check_id",), "system_events": ("event_id",)}[table]
                    report.rows[table] += _upsert(conn, table, _fit(conn, table, rows), key)
        report.days.append(day.isoformat())
        prev_dir = day_dir
    if not any(report.rows.get(t) for t in LogParser.TABLES):
        report.pending = list(LogParser.TABLES)
    return report
