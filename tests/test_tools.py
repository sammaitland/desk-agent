"""Tests for the tool layer.

Three things are being protected here:

1. The envelope contract — every tool returns data, provenance and a summary,
   because the agent depends on provenance to describe its own scope honestly.
2. The arithmetic — slippage sign, alpha grouping, fill rates. These are the
   numbers the model will state as fact, so they are tested rather than trusted.
3. The schemas — a tool the model cannot select correctly is a broken tool,
   so names, required arguments and enums are checked against the functions.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src import config as cfg
from src.db import create_schema, get_engine
from src.generate_blotter import Generator
from src.tools import DB_TOOLS, PURE_TOOLS, TOOL_SCHEMAS, TOOLS, dispatch
from src.tools.base import ToolResult


@pytest.fixture(scope="module")
def conn(blotter_url):
    engine = get_engine(blotter_url)
    create_schema(engine)
    gen = Generator(seed=42, days=90)
    gen.run()
    gen.write(engine)
    with engine.connect() as c:
        yield c


# --- envelope contract ----------------------------------------------------

@pytest.mark.parametrize("name,args", [
    ("query_blotter", {"entity": "positions", "limit": 5}),
    ("explain_rejection", {"limit": 5}),
    ("execution_quality", {}),
    ("alpha_attribution", {}),
    ("detect_anomalies", {"limit": 5}),
])
def test_tools_return_envelope(conn, name, args):
    result = dispatch(name, args, conn)
    assert isinstance(result, ToolResult)
    assert result.summary, f"{name} returned no summary"
    assert isinstance(result.provenance, dict) and result.provenance
    assert "rows" in result.provenance or "window" in result.provenance


def test_every_tool_is_in_schemas():
    assert {s["name"] for s in TOOL_SCHEMAS} == set(TOOLS)


def test_schema_required_args_are_real_parameters():
    """A required argument the function does not accept is a runtime failure."""
    import inspect
    for schema in TOOL_SCHEMAS:
        params = inspect.signature(TOOLS[schema["name"]]).parameters
        for required in schema["input_schema"].get("required", []):
            assert required in params, f"{schema['name']} requires unknown arg '{required}'"


def test_schema_descriptions_are_substantial():
    """The description is the only thing the model sees when choosing a tool."""
    for schema in TOOL_SCHEMAS:
        assert len(schema["description"]) > 120, f"{schema['name']} description too thin"


# --- dispatch behaviour ---------------------------------------------------

def test_unknown_tool_returns_result_not_exception(conn):
    result = dispatch("no_such_tool", {}, conn)
    assert isinstance(result, ToolResult) and result.data == []
    assert "Unknown tool" in result.summary


def test_bad_arguments_return_result_not_exception(conn):
    """The loop must be able to hand the error back to the model to correct."""
    result = dispatch("explain_position", {"nonsense": 1}, conn)
    assert isinstance(result, ToolResult)
    assert "Invalid arguments" in result.summary


def test_pure_tools_need_no_connection():
    result = dispatch("make_chart", {"labels": ["a"], "values": [1.0], "title": "t"})
    assert result.data["chart_path"].endswith(".png")


# --- query_blotter --------------------------------------------------------

def test_query_blotter_respects_limit(conn):
    result = dispatch("query_blotter", {"entity": "positions", "limit": 3}, conn)
    assert len(result.data) <= 3


def test_query_blotter_ticker_matches_either_leg(conn):
    ticker = conn.execute(text("SELECT co1 FROM positions LIMIT 1")).scalar()
    result = dispatch("query_blotter", {"entity": "positions", "ticker": ticker}, conn)
    for row in result.data:
        assert ticker in (row["co1"], row["co2"])


def test_query_blotter_unknown_entity_is_handled(conn):
    result = dispatch("query_blotter", {"entity": "bananas"}, conn)
    assert result.data == [] and "Unknown entity" in result.summary


def test_query_blotter_flags_truncation(conn):
    result = dispatch("query_blotter", {"entity": "orders", "limit": 2}, conn)
    assert result.provenance["truncated"] is True


# --- explain_position -----------------------------------------------------

def test_explain_position_returns_full_context(conn):
    tag = conn.execute(text("SELECT tag FROM positions WHERE status='closed' LIMIT 1")).scalar()
    result = dispatch("explain_position", {"tag": tag}, conn)
    for key in ("position", "entry_rationale", "execution", "stop_loss", "alpha_path"):
        assert key in result.data


def test_explain_position_resolves_tail_to_legs(conn):
    """Tail L/U determines which ticker is long — the agent must not guess."""
    row = conn.execute(text("SELECT tag, co1, co2, tail FROM positions LIMIT 1")).mappings().one()
    rationale = dispatch("explain_position", {"tag": row["tag"]}, conn).data["entry_rationale"]
    if row["tail"] == "L":
        assert rationale["long_leg"] == row["co1"] and rationale["short_leg"] == row["co2"]
    else:
        assert rationale["long_leg"] == row["co2"] and rationale["short_leg"] == row["co1"]


def test_explain_position_unknown_tag(conn):
    result = dispatch("explain_position", {"tag": "NOT_A_TAG"}, conn)
    assert result.data == [] and "No position found" in result.summary


# --- explain_rejection ----------------------------------------------------

def test_explain_rejection_only_returns_failures(conn):
    result = dispatch("explain_rejection", {"limit": 30}, conn)
    for row in result.data:
        assert row["primary_result"] == "Fail" or row["evaluation_result"] == "Rejected"


def test_explain_rejection_tallies_reasons(conn):
    result = dispatch("explain_rejection", {"limit": 30}, conn)
    tally = result.provenance["reason_counts"]
    # Counts cover the window, not the page — see
    # test_rejection_counts_cover_the_window_not_the_page.
    assert tally and sum(tally.values()) == result.provenance["total_rejections"]
    assert sum(tally.values()) >= len(result.data)


# --- date window spans positions, not just runs ---------------------------

def test_window_covers_positions_outside_the_run_range(tmp_path):
    """Real data loads one run per archived day but months of position
    history. Taking the window from workflow_runs alone gave a one-day range
    and hid 184 closed positions: "how have closed positions performed?"
    returned nothing against a full book.

    Built directly rather than by deleting from a generated blotter, because
    this is exactly the shape the adapter produces: a single run row, and
    positions predating it by months.
    """
    from src.db import create_schema, get_engine
    from src.tools.base import blotter_date_range

    engine = get_engine(f"sqlite:///{tmp_path / 'narrow.db'}")
    create_schema(engine)
    with engine.connect() as c:
        c.execute(text("""INSERT INTO instruments (ticker, name, idx, is_active)
                          VALUES ('AAPL','AAPL','VGT',1), ('MSFT','MSFT','VGT',1)"""))
        c.execute(text("""INSERT INTO workflow_runs (run_id, run_date, started_at, outcome)
                          VALUES ('r1', '2026-09-08', '2026-09-08 14:30:00', 'completed')"""))
        c.execute(text("""INSERT INTO positions
            (tag, pair, co1, co2, idx, tail, status, trade_initiation_date,
             termination_date, final_alpha_return_pct, holding_days, total_notional)
            VALUES ('VGT_AAPL_MSFT_L_20251120_001','AAPL_MSFT','AAPL','MSFT','VGT','L',
                    'closed','2025-11-20','2025-12-11', 1.23, 21, 3300)"""))
        c.commit()

        first, last = blotter_date_range(c)
        assert first == "2025-11-20", f"window starts {first}, not at the earliest position"
        assert last >= "2026-09-08"
        result = dispatch("alpha_attribution", {"group_by": "idx"}, c)
        assert result.data, "a closed position months before the only run must be visible"
        assert result.data["breakdown"][0]["trades"] == 1


# --- rejection counts cover the window ------------------------------------

def test_rejection_counts_cover_the_window_not_the_page(conn):
    """An earlier version tallied reasons across the LIMITed rows, so a day
    with thousands of rejections reported the breakdown of the first 50 —
    one alphabetical block of one sector — as the day's proportions."""
    small = dispatch("explain_rejection", {"limit": 5}, conn)
    large = dispatch("explain_rejection", {"limit": 500}, conn)
    assert small.provenance["reason_counts"] == large.provenance["reason_counts"]
    assert small.provenance["total_rejections"] == large.provenance["total_rejections"]
    assert len(small.data) == 5 and small.provenance["truncated"] is True
    assert str(small.provenance["total_rejections"]) in small.summary


def test_rejection_rows_are_deterministic(conn):
    """No ORDER BY on pair meant the limited page was insertion order, which
    for a bulk-loaded file is alphabetical — and looked like a real pattern."""
    a = dispatch("explain_rejection", {"limit": 10}, conn).data
    b = dispatch("explain_rejection", {"limit": 10}, conn).data
    assert [r["pair"] for r in a] == [r["pair"] for r in b]


# --- candidates entity -----------------------------------------------------

def test_candidates_entity_exposes_stage_and_traded(conn):
    """"Which pairs reached the shortlist but weren't traded?" had no tool
    that could answer it — stage was added by the adapter and nothing
    surfaced it, so the agent answered a different question confidently."""
    result = dispatch("query_blotter", {"entity": "candidates", "limit": 5}, conn)
    assert result.data
    for row in result.data:
        assert "stage" in row and "traded" in row and row["traded"] in (0, 1)


def test_candidates_traded_flag_matches_positions(conn):
    """traded=1 must mean a position exists on that pair and day."""
    rows = dispatch("query_blotter", {"entity": "candidates", "limit": 200}, conn).data
    for row in rows:
        expected = conn.execute(text("""
            SELECT COUNT(*) FROM positions
            WHERE pair = :p AND trade_initiation_date = :d"""),
            {"p": row["pair"], "d": row["evaluated_at"][:10]}).scalar()
        assert row["traded"] == (1 if expected else 0), row["pair"]


def test_candidates_filters_by_stage_and_traded(conn):
    conn.execute(text("UPDATE pair_evaluations SET stage = 'shortlist' WHERE eval_id IN "
                      "(SELECT eval_id FROM pair_evaluations LIMIT 5)"))
    conn.commit()
    rows = dispatch("query_blotter",
                    {"entity": "candidates", "stage": "shortlist", "traded": False},
                    conn).data
    assert rows and all(r["stage"] == "shortlist" and r["traded"] == 0 for r in rows)


def test_candidates_is_registered_with_the_agent():
    from src.tools import TOOL_SCHEMAS

    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "query_blotter")
    props = schema["input_schema"]["properties"]
    assert "candidates" in props["entity"]["enum"]
    assert "stage" in props and "traded" in props


# --- fill aggregation ------------------------------------------------------

def test_multiple_fills_are_quantity_weighted(conn):
    """External review: one share at mid plus 99 shares 1% above reported 0bps
    because the first fill row was taken. The fill price must be the
    quantity-weighted average across all fills for the order."""
    order_id = conn.execute(text(
        "SELECT order_id FROM orders WHERE status='Filled' AND side='BUY' LIMIT 1")).scalar()
    mid = conn.execute(text("SELECT arrival_mid FROM orders WHERE order_id=:o"),
                       {"o": order_id}).scalar()
    conn.execute(text("DELETE FROM fills WHERE order_id=:o"), {"o": order_id})
    conn.execute(text("""INSERT INTO fills (fill_id, order_id, quantity, price, filled_at, commission)
                         VALUES ('fx1', :o, 1, :mid, '2026-08-20 15:00:00', 0.1),
                                ('fx2', :o, 99, :high, '2026-08-20 15:00:01', 0.5)"""),
                 {"o": order_id, "mid": mid, "high": round(mid * 1.01, 4)})
    conn.commit()
    detail = dispatch("execution_quality", {"order_id": order_id}, conn).data
    # 99% of the quantity filled 100bps above mid -> ~99bps weighted.
    assert 95 < detail["slippage_bps"] < 100, detail["slippage_bps"]
    assert detail["fill_count"] == 2


def test_aggregate_counts_orders_not_fill_rows(conn):
    """Joining fills directly would count an order with two fills twice."""
    orders_with_fills = conn.execute(text(
        "SELECT COUNT(DISTINCT order_id) FROM fills")).scalar()
    totals = dispatch("execution_quality", {}, conn).data["breakdown"][0]
    assert totals["orders_filled"] == orders_with_fills


# --- limit clamping --------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 0, -500, "abc", None, 10**9])
def test_limit_is_clamped(conn, bad):
    """External review: limit=-1 returned 1,298 orders with truncated=False.
    Every out-of-range or malformed limit must collapse into [1, MAX_ROWS]."""
    from src.tools.base import MAX_ROWS

    result = dispatch("query_blotter", {"entity": "orders", "limit": bad}, conn)
    assert 1 <= len(result.data) <= MAX_ROWS
    if len(result.data) == MAX_ROWS:
        assert result.provenance["truncated"] is True


# --- execution_quality ----------------------------------------------------

def test_slippage_sign_convention(conn):
    """Positive slippage must mean 'worse than arrival mid' for both sides."""
    row = conn.execute(text("""
        SELECT o.side, o.arrival_mid, f.price, o.order_id
        FROM orders o JOIN fills f ON f.order_id = o.order_id
        WHERE o.side='BUY' AND f.price > o.arrival_mid LIMIT 1""")).mappings().first()
    assert row is not None
    result = dispatch("execution_quality", {"order_id": row["order_id"]}, conn)
    assert result.data["slippage_bps"] > 0

    sell = conn.execute(text("""
        SELECT o.order_id FROM orders o JOIN fills f ON f.order_id = o.order_id
        WHERE o.side='SELL' AND f.price < o.arrival_mid LIMIT 1""")).scalar()
    if sell:
        assert dispatch("execution_quality", {"order_id": sell}, conn).data["slippage_bps"] > 0


def test_execution_quality_flags_spread_cap_breach(conn):
    """The causal link behind a bad fill: spread over cap forces market routing."""
    order_id = conn.execute(text(f"""
        SELECT o.order_id FROM orders o JOIN fills f ON f.order_id = o.order_id
        WHERE o.spread_bps > {cfg.MAX_LIMIT_ORDER_SPREAD_BPS} LIMIT 1""")).scalar()
    assert order_id, "generator should produce at least one over-cap order"
    data = dispatch("execution_quality", {"order_id": order_id}, conn).data
    assert data["exceeded_limit_spread_cap"] is True
    assert data["order_type"] == "MKT"


def test_execution_quality_group_by(conn):
    result = dispatch("execution_quality", {"group_by": "order_type"}, conn)
    groupings = {row["grouping"] for row in result.data["breakdown"]}
    assert groupings <= {"LMT", "MKT"}


def test_execution_quality_rejects_bad_group_by(conn):
    result = dispatch("execution_quality", {"group_by": "colour"}, conn)
    assert "Unknown group_by" in result.summary


def test_fill_rate_is_a_percentage(conn):
    totals = dispatch("execution_quality", {}, conn).data["totals"]
    assert 0.0 <= totals["fill_rate_pct"] <= 100.0


# --- alpha_attribution ----------------------------------------------------

def test_alpha_attribution_groups_are_exhaustive(conn):
    """Group counts must reconcile to the underlying closed-trade count."""
    result = dispatch("alpha_attribution", {"group_by": "idx"}, conn)
    grouped = sum(row["trades"] for row in result.data["breakdown"])
    actual = conn.execute(text("""
        SELECT COUNT(*) FROM positions
        WHERE status='closed' AND final_alpha_return_pct IS NOT NULL""")).scalar()
    assert grouped == actual


def test_alpha_attribution_win_rate_bounds(conn):
    for row in dispatch("alpha_attribution", {"group_by": "tail"}, conn).data["breakdown"]:
        assert 0.0 <= row["win_rate_pct"] <= 100.0
        assert row["winners"] <= row["trades"]


def test_alpha_attribution_declares_its_measure(conn):
    """Provenance must state that this is alpha, not P&L, so the agent says so."""
    provenance = dispatch("alpha_attribution", {}, conn).provenance
    assert "alpha" in provenance["measure"].lower()


def test_alpha_attribution_rejects_bad_group_by(conn):
    assert "Unknown group_by" in dispatch("alpha_attribution", {"group_by": "wat"}, conn).summary


# --- detect_anomalies -----------------------------------------------------

def test_detect_anomalies_severity_filter(conn):
    result = dispatch("detect_anomalies", {"severity": "halt"}, conn)
    for event in result.data["events"]:
        assert event["severity"] == "halt"


def test_detect_anomalies_orders_halts_first(conn):
    events = dispatch("detect_anomalies", {"limit": 50}, conn).data["events"]
    severities = [e["severity"] for e in events]
    if "halt" in severities:
        assert severities.index("halt") == 0


def test_detect_anomalies_empty_window(conn):
    result = dispatch("detect_anomalies", {"start_date": "1999-01-01", "end_date": "1999-01-02"}, conn)
    assert result.data == [] and "No anomalies" in result.summary


# --- make_chart -----------------------------------------------------------

def test_make_chart_rejects_mismatched_lengths():
    result = dispatch("make_chart", {"labels": ["a", "b"], "values": [1.0], "title": "t"})
    assert "Mismatched lengths" in result.summary


def test_make_chart_rejects_empty_input():
    assert "Nothing to chart" in dispatch("make_chart", {"labels": [], "values": [], "title": "t"}).summary


def test_make_chart_rejects_unknown_type():
    result = dispatch("make_chart", {"labels": ["a"], "values": [1.0], "title": "t", "chart_type": "pie"})
    assert "Unknown chart_type" in result.summary


# --- read-only guarantee --------------------------------------------------

def test_tool_layer_contains_no_write_statements():
    """The agent must not be able to mutate the blotter. Enforced structurally."""
    import pathlib
    forbidden = ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ")
    tools_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "tools"
    for path in tools_dir.glob("*.py"):
        source = path.read_text().upper()
        for keyword in forbidden:
            assert keyword not in source, f"{path.name} contains '{keyword.strip()}'"


def test_db_tools_and_pure_tools_are_disjoint():
    assert not set(DB_TOOLS) & set(PURE_TOOLS)


# --- backend selection ----------------------------------------------------

def test_integration_fixture_honours_db_url(blotter_url):
    """If DB_URL is set the fixtures must use it, or the Postgres CI job tests
    nothing. Without it, SQLite is the fallback. This test asserts the fixture
    reflects the environment either way."""
    import os

    configured = os.getenv("DB_URL")
    if configured:
        assert blotter_url == configured
        assert not blotter_url.startswith("sqlite")
    else:
        assert blotter_url.startswith("sqlite:///")


# --- SQL portability ------------------------------------------------------

def test_no_sqlite_only_sql_remains():
    """The README claims Postgres is a config change. This keeps it true.

    Two constructs silently work on SQLite and fail on Postgres:
    `DATE(text_column)` (Postgres has no date(text)) and `ROUND(double, int)`
    (Postgres defines only round(numeric, int)). Both appeared throughout an
    earlier version, which meant the portability claim was untested and false.
    CI runs the whole suite against a real Postgres; this catches it sooner.
    """
    import pathlib
    import re

    problems = []
    for path in (pathlib.Path(__file__).resolve().parent.parent / "src").rglob("*.py"):
        source = path.read_text()
        for match in re.finditer(r"\bDATE\(\s*[a-z_.]+\s*\)", source):
            problems.append(f"{path.name}: {match.group(0)}")
        for match in re.finditer(r"\bROUND\(\s*(?!CAST)", source):
            line = source[:match.start()].count("\n") + 1
            problems.append(f"{path.name}:{line}: ROUND without CAST")
    assert not problems, "SQLite-only SQL: " + "; ".join(problems)


def test_portable_helpers_emit_expected_sql():
    from src.tools.base import as_date, date_clause, rounded

    assert date_clause("placed_at", "o").startswith("SUBSTR(o.placed_at, 1, 10)")
    assert as_date("x") == "SUBSTR(x, 1, 10)"
    assert rounded("AVG(x)") == "ROUND(CAST(AVG(x) AS NUMERIC), 2)"
