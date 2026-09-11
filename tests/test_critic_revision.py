"""Regressions for the observed atomic failures and the revised evidence contract."""

from datetime import date
import json

import pytest
from sqlalchemy import text

from src.agent.loop import run_agent
from src.agent.scripted import ScriptedClient, final_turn, tool_turn
from src.agent.trace import Trace
from src.critic.benchmark import BenchCase, BenchSummary, Target, database_digest, score_atomic
from src.critic.critique import Claim, Verdict, verify_claim
from src.critic.evidence import SCOPE_KEYS, evidence_is_grounded, parse_assessment
from src.db import create_schema, get_engine
from src.generate_blotter import Generator
from src.tools import dispatch
from src.tools.base import ToolResult


@pytest.fixture(scope="module")
def database(blotter_url):
    engine = get_engine(blotter_url)
    create_schema(engine)
    gen = Generator(seed=42, days=120, as_of=date(2026, 9, 7))
    gen.run()
    gen.write(engine)
    with engine.connect() as conn:
        yield conn


def assessment(trace, path, value, relation="record", call=1, verdict="verified"):
    provenance = trace.tool_calls[call - 1].provenance or {}
    return {"verdict": verdict, "relation": relation, "evidence": "Specific returned observation",
            "references": [{"call": call, "path": path, "value": value,
                            "scope": {k: provenance[k] for k in SCOPE_KEYS if k in provenance}}]}


def trace_for(data, provenance=None, tool="query_blotter"):
    trace = Trace("q")
    trace.record_tool(tool, {}, ToolResult(data, provenance or {}), 0, 1)
    return trace


@pytest.mark.parametrize("observed,cited", [(-7.89, 7.89), (0.07, 7), (True, 1), ("2026-08-24", 24), (7.89, 7.9)])
def test_reference_rejects_sign_unit_type_date_and_rounding_changes(observed, cited):
    t = trace_for({"value": observed})
    assert not evidence_is_grounded(assessment(t, "/data/value", cited), t)[0]


def test_other_call_cannot_lend_a_number_to_a_reference():
    t = trace_for({"alpha": -7.89})
    t.record_tool("query_blotter", {}, ToolResult({"alpha": 7.89}), 0, 2)
    assert not evidence_is_grounded(assessment(t, "/data/alpha", 7.89), t)[0]


def test_reference_scope_cannot_relabel_entry_cohort_as_exits():
    t = trace_for({"count": 1}, {"population": "positions", "date_basis": "trade_initiation_date"})
    a = assessment(t, "/data/count", 1)
    a["references"][0]["scope"]["date_basis"] = "termination_date"
    assert not evidence_is_grounded(a, t)[0]


def test_aggregate_cannot_use_a_date_or_row_quantity_as_a_count():
    t = trace_for({"quantity": 24, "date": "2026-08-24"})
    assert not evidence_is_grounded(assessment(t, "/data/quantity", 24, "aggregate"), t)[0]


def test_chart_echo_is_not_database_evidence():
    t = trace_for({"values": [24]}, tool="make_chart")
    assert not evidence_is_grounded(assessment(t, "/data/values/0", 24), t)[0]


def test_recorded_cause_needs_a_reason_field():
    t = trace_for({"stop_loss": {"status": "triggered"}}, tool="explain_position")
    assert not evidence_is_grounded(assessment(t, "/data/stop_loss/status", "triggered", "recorded_reason"), t)[0]


@pytest.mark.parametrize("answer", ["VERDICT: verified\nEVIDENCE: 24", "{}", "[]",
    '{"verdict":"verified","evidence":"x","relation":"record","references":NaN}',
    '{"verdict":"verified"}\n{"verdict":"contradicted"}'])
def test_malformed_assessments_are_errors(answer):
    assert parse_assessment(answer)[1]


def test_entry_and_exit_windows_select_different_positions(database):
    args = {"entity": "positions", "status": "closed", "start_date": "2026-08-24", "end_date": "2026-08-24"}
    entries = dispatch("query_blotter", args, database)
    exits = dispatch("query_blotter", {**args, "date_basis": "termination_date"}, database)
    assert entries.data and exits.data
    assert any(r["termination_date"][:10] != "2026-08-24" for r in entries.data)
    assert all(r["termination_date"][:10] == "2026-08-24" for r in exits.data)
    assert entries.provenance["date_basis"] == "trade_initiation_date"
    assert exits.provenance["date_basis"] == "termination_date"
    attributed = dispatch("alpha_attribution", {k: v for k, v in args.items() if k != "entity"} |
                          {"date_basis": "termination_date"}, database)
    assert attributed.data["totals"]["trades"] == len(exits.data)


def test_four_stops_are_counted_by_trigger_time_even_with_one_row_page(database):
    r = dispatch("query_records", {"entity": "stop_orders", "status": "triggered",
        "start_date": "2026-08-24", "end_date": "2026-08-24", "limit": 1}, database)
    assert r.data["count"] == 4 and len(r.data["records"]) == 1
    assert r.data["records"][0]["triggered_at"].startswith("2026-08-24")
    assert r.provenance["date_basis"] == "triggered_at" and r.provenance["truncated"]
    t = Trace("q")
    t.record_tool("query_records", {}, r, 0, 1)
    assert evidence_is_grounded(assessment(t, "/data/count", 4, "aggregate"), t)[0]


def test_passing_check_is_retrievable_by_id_and_shares_position_notional(database):
    row = database.execute(text("""SELECT r.check_id, r.subject, r.current_value, r.checked_at
        FROM risk_checks r JOIN positions p ON p.pair=r.subject
        AND SUBSTR(p.trade_initiation_date,1,10)=SUBSTR(r.checked_at,1,10)
        AND p.total_notional=r.current_value
        WHERE r.check_name='position_size' AND r.result='Pass' LIMIT 1""")).mappings().one()
    r = dispatch("query_records", {"entity": "risk_checks", "record_id": row["check_id"]}, database)
    assert r.data["count"] == 1
    assert r.data["records"][0]["result"] == "Pass"
    assert r.data["records"][0]["current_value"] == row["current_value"]


def test_unknown_event_label_is_a_trace_error_not_zero(database):
    t = run_agent("q", database, client=ScriptedClient([
        tool_turn("detect_anomalies", {"event_type": "stop_trigger"}), final_turn("x")]))
    assert t.tool_calls[0].error
    assert "stop_orders" in t.tool_calls[0].summary


def test_partial_fill_alias_and_counts_are_complete(database):
    short = dispatch("detect_anomalies", {"event_type": "partial fill", "limit": 1}, database)
    full = dispatch("detect_anomalies", {"event_type": "partial_fill", "limit": 500}, database)
    assert short.provenance["event_type_counts"] == full.provenance["event_type_counts"]
    assert short.provenance["filters"]["event_type"] == "partial_fill"
    assert short.provenance["event_count"] > len(short.data["events"])


def test_multiple_events_for_one_order_do_not_become_order_count(database):
    with database.begin_nested():
        before = dispatch("detect_anomalies", {"event_type": "partial_fill"}, database).provenance["event_count"]
        orders_before = database.execute(text("SELECT COUNT(*) FROM orders WHERE status='Partial'")).scalar()
        database.execute(text("""INSERT INTO system_events(event_id, occurred_at, event_type, severity, order_id, detail)
            SELECT 'test_duplicate_event', occurred_at, event_type, severity, order_id, detail
            FROM system_events WHERE event_type='partial_fill' LIMIT 1"""))
        after = dispatch("detect_anomalies", {"event_type": "partial_fill"}, database).provenance["event_count"]
        assert after == before + 1
        assert database.execute(text("SELECT COUNT(*) FROM orders WHERE status='Partial'")).scalar() == orders_before
        database.execute(text("DELETE FROM system_events WHERE event_id='test_duplicate_event'"))


def test_valid_zero_count_grounds_without_any_rows(database):
    r = dispatch("query_records", {"entity": "stop_orders", "start_date": "1999-01-01", "end_date": "1999-01-01"}, database)
    t = Trace("q")
    t.record_tool("query_records", {}, r, 0, 1)
    a = assessment(t, "/data/count", 0, "aggregate")
    assert r.data == {"records": [], "count": 0}
    assert evidence_is_grounded(a, t)[0]
    t.tool_calls[0].provenance["count_complete"] = False
    assert not evidence_is_grounded(a, t)[0]


def test_stop_lookup_rejects_filters_that_would_be_ignored(database):
    r = dispatch("query_records", {"entity": "stop_orders", "subject": "GM_NKE"}, database)
    assert r.provenance["error"] is True


def test_atomic_precision_retains_overreach_and_trace_metadata():
    targets = [Target("f", (0, 1), "contradicted", "f"), Target("u", (2, 3), "undetermined", "u")]
    case = BenchCase("c", "k", "q", "f u", targets)
    vs = {t.id: Verdict(Claim(t.claim, t.claim, "figure", True, True), "contradicted",
                       proposed="contradicted", trace_run_id=f"trace_{t.id}", model="test-model") for t in targets}
    summary = BenchSummary([score_atomic(case, vs)], mode="atomic")
    assert summary.precision == 0.5 and summary.extraction_coverage is None
    saved = json.loads(json.dumps(summary.as_dict()))
    assert saved["cases"][0]["target_verdicts"]["u"][0]["trace_run_id"] == "trace_u"
    assert "N/A (bypassed)" in summary.render() and "restraint 0/1" in summary.render()


def test_assessment_error_does_not_earn_restraint():
    t = Target("u", (0, 1), "undetermined", "u")
    c = BenchCase("c", "k", "q", "u", [t])
    v = Verdict(Claim("u", "u", "causal", True, True), "undetermined", assessment_error="timeout")
    r = score_atomic(c, {"u": v})
    assert r.outcome["u"] == "assessment_error" and not r.passed
    assert BenchSummary([r], mode="atomic").total("restrained") == 0


@pytest.mark.parametrize("claim", ["A sector-wide move caused the stops.",
    "The stops were triggered by a sharp intraday move.", "The cluster was due to a liquidity shock."])
def test_market_cause_cannot_be_settled_by_a_record_relation(database, tmp_path, claim):
    # Deliberately mislabel the evidence relation: an extra heuristic catches
    # these explicit phrases. It does not claim to classify every paraphrase.
    a = {"verdict": "contradicted", "relation": "record", "evidence": "Prior losses exclude a later shock.", "references": []}
    v = verify_claim(Claim(claim, claim, "figure", True, True), "q", database,
                     client=ScriptedClient([final_turn(json.dumps(a))]), trace_dir=tmp_path)
    assert v.verdict == "undetermined" and v.proposed == "contradicted"
    assert "market cause" in v.downgraded


def test_budget_exhaustion_is_an_assessment_error(database, tmp_path):
    client = ScriptedClient([tool_turn("query_records", {"entity": "stop_orders", "limit": 1}) for _ in range(4)])
    v = verify_claim(Claim("c", "c", "figure", True, True), "q", database, client=client, trace_dir=tmp_path)
    assert v.assessment_error and "max_turns" in v.assessment_error


def test_loop_exposes_stable_evidence_call_numbers(database):
    client = ScriptedClient([tool_turn("query_records", {"entity": "stop_orders"}),
                             tool_turn("query_records", {"entity": "risk_checks", "limit": 1}), final_turn("x")])
    t = run_agent("q", database, client=client)
    assert len(t.tool_calls) == 2
    # ScriptedClient captures exact request payloads; these IDs are available
    # to the model and refer to the same ordered calls saved in the trace.
    blocks = [b for m in client.requests[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
              for b in m["content"] if b["type"] == "tool_result"]
    assert [json.loads(b["content"])["evidence_call"] for b in blocks] == [1, 2]


def test_database_digest_detects_value_change_without_count_change(database):
    before, counts = database_digest(database)
    row = database.execute(text("SELECT check_id, current_value FROM risk_checks WHERE current_value IS NOT NULL LIMIT 1")).one()
    try:
        database.execute(text("UPDATE risk_checks SET current_value=:v WHERE check_id=:id"), {"v": row[1] + 1, "id": row[0]})
        after, new_counts = database_digest(database)
        assert counts == new_counts and before != after
    finally:
        database.execute(text("UPDATE risk_checks SET current_value=:v WHERE check_id=:id"), {"v": row[1], "id": row[0]})
    assert database_digest(database)[0] == before


@pytest.mark.parametrize("args", [{"start_date": "wrong"}, {"start_date": "20260824"},
                                  {"start_date": "2026-09-01", "end_date": "2026-08-01"}])
def test_malformed_date_filters_are_errors(database, args):
    for name, extra in (("query_records", {"entity": "stop_orders"}),
                        ("detect_anomalies", {"event_type": "partial_fill"})):
        assert dispatch(name, {**extra, **args}, database).provenance["error"]


def test_duplicate_verdict_keys_are_not_silently_overwritten():
    assert parse_assessment('{"verdict":"verified","verdict":"undetermined",'
                            '"evidence":"x","relation":"record","references":[]}')[1]


def test_stop_count_verification_runs_through_real_tools_and_saves_references(database, tmp_path):
    args = {"entity": "stop_orders", "status": "triggered", "start_date": "2026-08-24", "end_date": "2026-08-24"}
    result = dispatch("query_records", args, database)
    t = Trace("q")
    t.record_tool("query_records", args, result, 0, 1)
    a = assessment(t, "/data/count", 4, "aggregate")
    claim = Claim("four stops", "Four stop orders triggered on 2026-08-24.", "figure", True, True)
    v = verify_claim(claim, "q", database, trace_dir=tmp_path,
                     client=ScriptedClient([tool_turn("query_records", args), final_turn(json.dumps(a))]))
    assert v.verdict == "verified" and not v.assessment_error
    assert v.references[0]["value"] == 4
    from pathlib import Path
    saved = json.loads(Path(v.trace_path).read_text())
    assert saved["tool_calls"][0]["raw_result"]["count"] == 4


def test_mixed_anomaly_sections_keep_their_own_scope():
    scope = {"population": "risk_checks", "date_basis": "checked_at", "filters": {"result": "Fail"}}
    t = trace_for({"failed_risk_checks": [{"result": "Fail"}]},
                  {"population": "system_events", "date_basis": "occurred_at",
                   "scopes": {"failed_risk_checks": scope}}, tool="detect_anomalies")
    a = assessment(t, "/data/failed_risk_checks/0/result", "Fail")
    assert not evidence_is_grounded(a, t)[0]
    a["references"][0]["scope"] = scope
    assert evidence_is_grounded(a, t)[0]


def test_extractor_transport_failure_is_reported(database):
    from src.agent.scripted import FailingClient
    from src.critic.critique import critique
    c = critique("q", "Four stops.", database, extractor_client=FailingClient())
    assert not c.complete and not c.extraction_ok and "request failed" in c.extraction_error
