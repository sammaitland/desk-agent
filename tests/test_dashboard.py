"""Tests for the dashboard data layer.

Streamlit's caching decorators are no-ops outside its runtime, so every data
function can be called directly. That is the whole reason the queries live in
`data.py` rather than inline in `app.py` — UI code is awkward to test, data
access is not, and separating them means the numbers on the dashboard are
covered even though the layout is not.

The central assertion is agreement: a figure shown on the dashboard and the
same figure in an agent answer must come from the same tested function. A
dashboard that recomputed its own alpha would eventually contradict the agent
with no way to tell which was right.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.db import create_schema, get_engine
from src.generate_blotter import Generator


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    from src.dashboard import data as module

    db = tmp_path_factory.mktemp("dash") / "blotter.db"
    engine = get_engine(f"sqlite:///{db}")
    create_schema(engine)
    gen = Generator(seed=42, days=90)
    gen.run()
    gen.write(engine)

    # Point the cached engine at the fixture database.
    module.engine.clear()
    module.engine = lambda: engine
    yield module


# --- coverage and headline ------------------------------------------------

def test_coverage_returns_the_window(dashboard):
    first, last = dashboard.coverage()
    assert first and last and first < last


def test_headline_numbers_are_coherent(dashboard):
    metrics = dashboard.headline()
    assert metrics["trades"] > 0
    assert 0 <= metrics["win_rate"] <= 100
    assert "leverage" in metrics["snapshot"]


def test_headline_alpha_matches_the_tool(dashboard):
    """The dashboard and the agent must not disagree about total alpha."""
    from_dashboard = dashboard.headline()["total_alpha"]
    result = dashboard.run_tool("alpha_attribution", {"group_by": "idx"})
    from_tool = result["data"]["totals"]["total_alpha_pct"]
    assert abs(from_dashboard - from_tool) < 0.05


# --- series ---------------------------------------------------------------

def test_alpha_series_is_cumulative(dashboard):
    df = dashboard.alpha_over_time()
    assert not df.empty
    assert df["date"].is_monotonic_increasing
    # The final cumulative value is the sum of the daily values.
    assert abs(df["cumulative_alpha"].iloc[-1] - df["daily_alpha"].astype(float).sum()) < 0.01


def test_leverage_series_respects_the_cap(dashboard):
    from src import config as cfg

    df = dashboard.leverage_over_time()
    assert not df.empty
    assert df["leverage"].max() <= cfg.MAX_ACCOUNT_LEVERAGE


def test_screening_funnel_covers_every_evaluation(dashboard):
    """Funnel counts must reconcile to the underlying row count, or the chart
    silently misrepresents where candidates are lost."""
    funnel = dashboard.screening_funnel()
    total = dashboard.query("SELECT COUNT(*) AS n FROM pair_evaluations").iloc[0]["n"]
    assert funnel["n"].sum() == total


# --- tool passthrough -----------------------------------------------------

def test_run_tool_returns_the_envelope(dashboard):
    result = dashboard.run_tool("detect_anomalies", {"limit": 5})
    assert {"summary", "provenance", "data"} <= set(result)


def test_run_tool_handles_an_unknown_tool(dashboard):
    result = dashboard.run_tool("no_such_tool", {})
    assert "Unknown tool" in result["summary"]


def test_query_returns_a_dataframe(dashboard):
    df = dashboard.query("SELECT COUNT(*) AS n FROM positions")
    assert isinstance(df, pd.DataFrame) and df.iloc[0]["n"] > 0


# --- presentation helpers -------------------------------------------------

def test_money_formatting_handles_missing_values(dashboard):
    # Python rounds half to even, so 1234.5 -> 1234. Asserting 1235 here was a
    # bug in the test, not the formatter.
    assert dashboard.format_money(1234.4) == "$1,234"
    assert dashboard.format_money(98765) == "$98,765"
    assert dashboard.format_money(None) == "—"
    assert dashboard.format_money("nonsense") == "—"


# --- portability ----------------------------------------------------------

def test_dashboard_sql_is_portable():
    """The dashboard writes its own SQL, so it needs the same guard as the
    tool layer: DATE(text) and ROUND(double, n) fail on PostgreSQL."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "src" / "dashboard").rglob("*.py")
    problems = []
    for path in source:
        text = path.read_text()
        problems += [f"{path.name}: DATE()" for _ in re.finditer(r"\bDATE\(\s*[a-z_.]+\s*\)", text)]
        problems += [f"{path.name}: ROUND without CAST"
                     for _ in re.finditer(r"\bROUND\(\s*(?!CAST)", text)]
    assert not problems, "; ".join(problems)


def test_dashboard_is_read_only():
    import pathlib

    for path in (pathlib.Path(__file__).resolve().parent.parent / "src" / "dashboard").rglob("*.py"):
        source = path.read_text().upper()
        for keyword in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER "):
            assert keyword not in source, f"{path.name} contains {keyword.strip()}"
