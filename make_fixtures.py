#!/usr/bin/env python3
"""Build the adapter's test fixtures from a real archived day.

    python make_fixtures.py ~/Desktop/V9/archive/2026-09-08

The adapter's tests run against small real-shaped Excel files rather than
synthetic ones, because the whole point of them is that the real files have
properties a generator would not invent: Tag repeated across stages, free-text
reasons, a cumulative archive, five positions recorded twice, one blank row.

This script samples those properties out of a real archive so the fixtures
stay small enough to commit while still exhibiting every case the tests name.
Run it once; the output is committed and does not need regenerating unless the
live system's file formats change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEST = Path(__file__).resolve().parent / "tests" / "fixtures" / "v9"


def build(src: Path) -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    # --- prefilter: 20 rejected + 20 active, keeping the free-text reasons ---
    pf = pd.read_excel(src / "trade_prefilter_active.xlsx", sheet_name="Active_Pairs")
    reasons = pf[pf.Active == 0]
    varied = pd.concat([reasons[reasons.Reason.astype(str).str.startswith(p)].head(3)
                        for p in ("Trending", "Failed", "Missing")])
    sample = pd.concat([varied, reasons.head(12), pf[pf.Active == 1].head(20)])
    sample = sample[~sample.index.duplicated()]
    with pd.ExcelWriter(DEST / "trade_prefilter_active.xlsx") as w:
        sample.to_excel(w, sheet_name="Active_Pairs", index=False)
        try:
            pd.read_excel(src / "trade_prefilter_active.xlsx", sheet_name="Summary") \
              .to_excel(w, sheet_name="Summary", index=False)
        except ValueError:
            pass

    # --- longlist: rows whose Tag also appears in the prefilter sample, so a
    #     pair exists at more than one stage (the key-collision test) ---------
    ll = pd.read_excel(src / "V9_Longlist.xlsx", sheet_name="Longlist")
    overlap = ll[ll.Tag.isin(sample.Tag)].head(20)
    if overlap.empty:
        overlap = ll.head(20)
    with pd.ExcelWriter(DEST / "V9_Longlist.xlsx") as w:
        overlap.to_excel(w, sheet_name="Longlist", index=False)

    # --- shortlist: rows overlapping the longlist, for a third stage --------
    sl = pd.read_excel(src / "V9_Shortlist.xlsx", sheet_name="Shortlist")
    short = sl[sl.Tag.isin(overlap.Tag)].head(10)
    if short.empty:
        short = sl.head(10)
    with pd.ExcelWriter(DEST / "V9_Shortlist.xlsx") as w:
        short.to_excel(w, sheet_name="Shortlist", index=False)

    # --- rejected archive: today's rows AND older ones, so the cumulative
    #     filter has something to filter ------------------------------------
    rj = pd.read_excel(src / "Rejected_Trades_Archive.xlsx")
    day = src.name
    stamps = rj.Archived_At.astype(str).str[:10]
    today, older = rj[stamps == day].head(10), rj[stamps < day].head(10)
    if today.empty:
        latest = stamps.max()
        today, older = rj[stamps == latest].head(10), rj[stamps < latest].head(10)
        print(f"  note: no rows dated {day}; used {latest} as the fixture's day")
    pd.concat([older, today]).to_excel(DEST / "Rejected_Trades_Archive.xlsx", index=False)

    # --- portfolio: real 66-column header, whatever rows exist --------------
    book = pd.ExcelFile(src / "Portfolio.xlsx")
    with pd.ExcelWriter(DEST / "Portfolio.xlsx") as w:
        for sheet in book.sheet_names[:2]:
            pd.read_excel(src / "Portfolio.xlsx", sheet_name=sheet).head(20) \
              .to_excel(w, sheet_name=sheet, index=False)

    # --- completed: the duplicates, the blank row, and each exit type ------
    ct = pd.read_excel(src / "Completed_Trades.xlsx")
    init = ct["Trade Initiation Date"].astype(str).str[:10]
    key = (ct.Index.astype(str) + "|" + ct.Pair.astype(str) + "|" + ct.Tail.astype(str)
           + "|" + init + "|" + (ct.Tag % 1000).astype("Int64").astype(str))
    er = ct.Exit_Reason.astype(str)
    parts = [ct.head(4),
             ct[key.duplicated(keep=False)],
             ct[ct["Trade Initiation Date"].isna()],
             ct[er.str.startswith("Early")].head(2),
             ct[er.str.startswith("Earnings")].head(2),
             ct[er.str.startswith("Pre-Holiday")].head(1)]
    done = pd.concat(parts)
    done = done[~done.index.duplicated()]
    done.to_excel(DEST / "Completed_Trades.xlsx", index=False)

    print(f"fixtures written to {DEST}")
    print(f"  prefilter {len(sample)} | longlist {len(overlap)} | shortlist {len(short)}")
    print(f"  rejected {len(older) + len(today)} ({len(older)} outside the day)")
    print(f"  completed {len(done)} | duplicates {int(key.duplicated(keep=False).sum())} "
          f"| blank {int(ct['Trade Initiation Date'].isna().sum())}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_fixtures.py <archive>/<YYYY-MM-DD>")
    source = Path(sys.argv[1]).expanduser()
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")
    build(source)
