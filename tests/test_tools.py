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
def conn(tmp_path_factory):
    db = tmp_path_factory.mktemp("tools") / "blotter.db"
    engine = get_engine(f"sqlite:///{db}")
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
    assert tally and sum(tally.values()) == len(result.data)


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
