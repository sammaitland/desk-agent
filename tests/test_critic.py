"""Tests for the claim critic.

Each section names the failure from the first live run it guards against.
The review's list, verbatim: a correct number adjacent to a false ranking; a
real threshold unrelated to a recorded rejection; aggregate citations; empty
control extraction; skipped factual claims; inconsistent risk-check records;
wrong event-key defaults; holding-period returns used to claim intraday
causation; a proposed contradiction after unsuccessful investigation. Plus the
successful sign and rejection-reason cases, preserved.

Whether the live verifier now gets these right is what the benchmark
measures. These prove the harness scores them right when it does and wrong
when it doesn't.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text as sql

from src.agent.loop import run_agent
from src.agent.scripted import ScriptedClient, final_turn, tool_turn
from src.critic.critique import (
    Claim, Critique, Verdict, critique, evidence_is_grounded, extract_claims,
    parse_verdict, verify_claim,
)
from src.db import create_schema, get_engine
from src.generate_blotter import Generator


@pytest.fixture(scope="module")
def conn(blotter_url):
    engine = get_engine(blotter_url)
    create_schema(engine)
    g = Generator(seed=42, days=120)
    g.run()
    g.write(engine)
    with engine.connect() as c:
        yield c


def _extractor(items) -> ScriptedClient:
    return ScriptedClient([final_turn(json.dumps(items))])


def _item(source, type_="figure", checkable=True, fact=True, claim=None):
    return {"source": source, "claim": claim or source, "type": type_,
            "checkable": checkable, "stated_as_fact": fact}


def _v(source, verdict, type_="figure", checkable=True, fact=True, evidence="x", proposed=None):
    return Verdict(Claim(source, source, type_, checkable, fact), verdict,
                   evidence=evidence, proposed=proposed or verdict)


def _crit(answer, verdicts, **kw):
    return Critique("q", answer, [v.claim for v in verdicts], verdicts, **kw)


def _trace(conn, script, question="q"):
    return run_agent(question, conn, client=ScriptedClient(script), max_turns=4,
                     system_prompt="critic", save_trace=False)


# ===========================================================================
# DATA: inconsistent risk-check records (the root of the $18 story)
# ===========================================================================

def test_no_failed_check_has_a_non_breaching_value(conn):
    """The generator logged share-price-vs-cap as a position_size failure.
    A failed check whose value does not breach its threshold is a data
    inconsistency; the generator must not produce one."""
    bad = conn.execute(sql("""
        SELECT COUNT(*) FROM risk_checks
        WHERE result = 'Fail' AND check_name = 'position_size' AND current_value <= threshold""")).scalar()
    assert bad == 0
    bad2 = conn.execute(sql("""
        SELECT COUNT(*) FROM risk_checks
        WHERE result = 'Fail' AND check_name = 'min_share_quantity' AND current_value >= threshold""")).scalar()
    assert bad2 == 0


def test_passing_position_size_checks_are_recorded(conn):
    """A benchmark needs a passed check a critic can retrieve."""
    n = conn.execute(sql("SELECT COUNT(*) FROM risk_checks WHERE check_name='position_size' AND result='Pass'")).scalar()
    assert n > 0


# ===========================================================================
# GROUNDING: aggregates, arguments, failed calls
# ===========================================================================

def test_aggregate_evidence_grounds_on_a_figure(conn):
    """Review: '11 partial fills' and 'VIS at 34.57%' were downgraded for
    naming no record. A figure present in a result now grounds."""
    t = _trace(conn, [tool_turn("detect_anomalies", {"start_date": "2026-08-17", "end_date": "2026-08-26"}),
                      final_turn("x")])
    counts = t.tool_calls[0].provenance["event_type_counts"]
    n = counts["partial_fill"]
    ok, why = evidence_is_grounded(f"detect_anomalies — event_type_counts shows {n} partial_fill events", t)
    assert ok, why


def test_ranking_evidence_grounds_on_breakdown_figures(conn):
    t = _trace(conn, [tool_turn("alpha_attribution", {"group_by": "idx"}), final_turn("x")])
    row = t.tool_calls[0].raw_result["breakdown"][0]
    ok, why = evidence_is_grounded(
        f"alpha_attribution — {row['grouping']} total_alpha_pct {row['total_alpha_pct']}", t)
    assert ok, why


def test_arguments_never_ground(conn):
    """Review: a nonexistent pair present only in the query arguments passed."""
    t = _trace(conn, [tool_turn("explain_rejection", {"pair": "ZZZZ_QQQQ"}), final_turn("x")])
    ok, why = evidence_is_grounded("explain_rejection — ZZZZ_QQQQ rejection_reason differs", t)
    assert not ok


def test_failed_or_empty_call_does_not_ground_even_if_another_succeeded(conn):
    """Review: a reference found only in a failed call grounded because an
    unrelated call had succeeded."""
    tag = conn.execute(sql("SELECT tag FROM positions LIMIT 1")).scalar()
    t = _trace(conn, [tool_turn("explain_position", {"tag": tag}),           # succeeds
                      tool_turn("explain_position", {"tag": "VGT_NO_NO_L_20260101_999"}),  # empty
                      final_turn("x")])
    ok, why = evidence_is_grounded("explain_position — VGT_NO_NO_L_20260101_999 alpha 12.34", t)
    assert not ok and "absent" in why


def test_figure_must_be_a_value_in_a_result_not_a_digit_in_text(conn):
    """'5' inside 'chk_5a...' or a date must not ground a claim of five."""
    t = _trace(conn, [tool_turn("query_blotter", {"entity": "runs", "limit": 1}), final_turn("x")])
    # 999999 appears nowhere as a value
    ok, why = evidence_is_grounded("query_blotter — there were 999999", t)
    assert not ok and "absent" in why
    # a count that IS a value in the result grounds
    rows = t.tool_calls[0].provenance["rows"]
    ok, why = evidence_is_grounded(f"query_blotter — returned {rows}", t)
    assert ok, why


def test_proposed_verdict_is_preserved_when_downgraded(conn, tmp_path):
    """Review: preserve raw proposed verdicts alongside final ones."""
    claim = Claim("x", "x", "causal", True, True)
    client = ScriptedClient([final_turn("VERDICT: contradicted\nEVIDENCE: intuition")])
    v = verify_claim(claim, "q", conn, client=client, trace_dir=tmp_path)
    assert v.proposed == "contradicted" and v.verdict == "undetermined"


def test_unverifiable_is_normalised_to_undetermined():
    assert parse_verdict("VERDICT: unverifiable\nEVIDENCE: none") == ("undetermined", "none")
    assert parse_verdict("VERDICT: undetermined\nEVIDENCE: none") == ("undetermined", "none")


# ===========================================================================
# STATUS: findings and completion kept apart
# ===========================================================================

def test_skipped_factual_claim_makes_assessment_incomplete():
    """Review: the incomplete check looked for skipped+checkable, a state the
    skip branch never produces. Now: skipped + stated_as_fact."""
    v = _v("11 partial fills", "skipped", checkable=False, fact=True)
    c = _crit("a", [v])
    assert not c.complete and c.status == "incomplete"


def test_skipped_hedged_claim_is_fine():
    v = _v("may reflect", "skipped", "inferential", checkable=False, fact=False)
    assert _crit("a", [v]).complete


def test_empty_extraction_is_never_complete():
    """Review: the twelve-word cutoff let empty extraction on a short factual
    statement read as clean. Brevity is not absence of claims."""
    assert not Critique("q", "There were 11 partial fills.", [], []).complete
    assert not Critique("q", "No.", [], []).complete


def test_contradiction_visible_even_when_incomplete():
    """Review: a malformed extraction entry must not conceal a confirmed
    contradiction. Findings and completion are separate."""
    c = _crit("a", [_v("x", "contradicted")], rejected_items=1)
    assert c.findings == "flagged" and not c.complete and c.status == "flagged"


def test_undetermined_factual_claim_is_unresolved():
    c = _crit("a", [_v("x", "undetermined", "causal")])
    assert c.findings == "unresolved" and c.verdicts[0].flag == "UNSUPPORTED"


def test_verified_inferential_claim_carries_no_flag():
    assert _v("x", "verified", "inferential").flag is None


# ===========================================================================
# ALIGNMENT: true figure adjacent to false ranking; ambiguity reported
# ===========================================================================

def _case(answer, targets):
    from src.critic.benchmark import BenchCase, Target
    return BenchCase("t", "k", "q", answer,
                     [Target(i, (answer.index(s), answer.index(s) + len(s)), e, s) for i, s, e in targets])


def test_true_figure_next_to_false_ranking_scores_separately():
    """Review: 'VIS had 34.57%' (true) contradicted because the question
    asked about weakest; span overlap credited it as a catch of the false
    ranking. Tight targets + unique alignment separate them."""
    from src.critic.benchmark import score

    answer = "VIS was the weakest sector. Its summed per-trade alpha was 34.57%."
    case = _case(answer, [("false_weakest", "VIS was the weakest sector", "contradicted"),
                          ("true_figure", "was 34.57%", "supported")])
    # Good critic: figure verified, ranking contradicted
    crit = _crit(answer, [_v("was 34.57%", "verified"),
                          _v("VIS was the weakest sector", "contradicted")])
    r = score(case, crit)
    assert r.outcome == {"false_weakest": "caught", "true_figure": "confirmed"}
    # Bad critic (the observed one): contradicts the true figure
    crit = _crit(answer, [_v("was 34.57%", "contradicted"),
                          _v("VIS was the weakest sector", "contradicted")])
    r = score(case, crit)
    assert r.outcome["true_figure"] == "false_alarm" and r.errors == 1


def test_claim_overlapping_two_targets_is_ambiguous_not_credited():
    """Review: overlap is a candidate alignment, not identity."""
    from src.critic.benchmark import score

    answer = "VIS was the weakest sector at 34.57%."
    case = _case(answer, [("false_weakest", "VIS was the weakest", "contradicted"),
                          ("true_figure", "34.57%", "supported")])
    crit = _crit(answer, [_v("VIS was the weakest sector at 34.57%", "contradicted")])
    r = score(case, crit)
    assert r.outcome == {"false_weakest": "ambiguous", "true_figure": "ambiguous"}
    assert r.ambiguous_claims and not r.passed


def test_unmatched_flag_is_reported():
    from src.critic.benchmark import score

    answer = "A is true. Something else."
    case = _case(answer, [("a", "A is true", "supported")])
    r = score(case, _crit(answer, [_v("Something else", "contradicted")]))
    assert r.unmatched_flags and r.errors == 1


# ===========================================================================
# EXPECTED OUTCOMES: cautions are not catches; overreach is an error
# ===========================================================================

def test_undetermined_on_a_false_claim_is_a_caution_not_a_catch():
    """Review: UNSUPPORTED warnings were credited as catches. Now only a
    grounded contradiction counts."""
    from src.critic.benchmark import score

    answer = "X was rejected because of Y."
    case = _case(answer, [("f", "because of Y", "contradicted")])
    r = score(case, _crit(answer, [_v("because of Y", "undetermined")]))
    assert r.outcome["f"] == "caution" and r.count("caught") == 0


def test_confident_verdict_on_undetermined_target_is_overreach():
    """Review: the verifier 'disproved' an intraday cause with holding-period
    returns and the benchmark rewarded it. An expected-undetermined target
    now counts a confident verdict as an error."""
    from src.critic.benchmark import score

    answer = "The cluster was caused by a sharp intraday move."
    case = _case(answer, [("cause", "was caused by a sharp intraday move", "undetermined")])
    for wrong in ("contradicted", "verified"):
        r = score(case, _crit(answer, [_v("was caused by a sharp intraday move", wrong)]))
        assert r.outcome["cause"] == "overreach" and r.errors == 1
    r = score(case, _crit(answer, [_v("was caused by a sharp intraday move", "undetermined")]))
    assert r.outcome["cause"] == "restrained" and r.errors == 0


def test_verifying_a_false_claim_is_an_error():
    from src.critic.benchmark import score

    answer = "X was rejected because of Y."
    case = _case(answer, [("f", "because of Y", "contradicted")])
    r = score(case, _crit(answer, [_v("because of Y", "verified")]))
    assert r.outcome["f"] == "false_verify" and r.errors == 1


# ===========================================================================
# METRICS: unique coverage; controls must be assessed to pass
# ===========================================================================

def test_coverage_counts_each_target_once():
    """Review: 107% coverage — a falsely flagged true target counted twice."""
    from src.critic.benchmark import BenchSummary, score

    answer = "A is true. B is true."
    case = _case(answer, [("a", "A is true", "supported"), ("b", "B is true", "supported")])
    r = score(case, _crit(answer, [_v("A is true", "contradicted"), _v("B is true", "verified")]))
    s = BenchSummary([r])
    assert s.extraction_coverage == 1.0


def test_control_with_no_claims_extracted_fails():
    """Review: an unassessed all-true control could PASS."""
    from src.critic.benchmark import score

    answer = "Order X filled with 12bps slippage."
    case = _case(answer, [("t", "12bps slippage", "supported")])
    r = score(case, Critique("q", answer, [], []))
    assert not r.passed and r.outcome["t"] == "not_extracted"


def test_precision_counts_only_contradictions():
    from src.critic.benchmark import BenchSummary, score

    answer = "A. B. C."
    case = _case(answer, [("a", "A", "contradicted"), ("b", "B", "supported"), ("c", "C", "undetermined")])
    r = score(case, _crit(answer, [_v("A", "contradicted"), _v("B", "contradicted"), _v("C", "contradicted")]))
    s = BenchSummary([r])
    # 3 contradictions issued, 1 correct
    assert s.precision == pytest.approx(1 / 3)
    assert s.total("caught") == 1 and s.total("false_alarm") == 1 and s.total("overreach") == 1


# ===========================================================================
# GROUND TRUTH: canonical tables, prerequisites, consistency
# ===========================================================================

def test_timeout_control_comes_from_orders_table(conn):
    """Review: the zero-timeout 'control' came from a mistyped event key."""
    from src.critic.benchmark import build_cases

    case = next(c for c in build_cases(conn) if c.kind == "wrong_count")
    t = next(t for t in case.targets if t.id == "true_count")
    n = conn.execute(sql("""SELECT COUNT(*) FROM orders WHERE fallback_reason='timeout'
        AND SUBSTR(placed_at,1,10) BETWEEN '2026-08-17' AND '2026-08-26'""")).scalar()
    assert n > 0 and f"{n} orders that timed out" in case.answer


def test_near_cap_fixture_is_a_position_size_check(conn):
    """Review: the fallback picked a 'momentum' factor check."""
    from src.critic.benchmark import build_cases

    case = next(c for c in build_cases(conn) if c.name == "invented_causal_near_cap")
    assert "position-size check" in case.question
    t = next(t for t in case.targets if t.id == "false_failed")
    assert "chk_" in t.note


def test_setup_fails_on_inconsistent_data(conn):
    """The benchmark asserts the data it runs on is self-consistent."""
    from src.critic.benchmark import BenchmarkSetupError, build_cases

    conn.execute(sql("""INSERT INTO risk_checks (check_id, run_id, checked_at, check_name, subject,
        current_value, threshold, result, action)
        SELECT 'chk_bad', run_id, '2026-08-25 14:32:00', 'position_size', 'SBUX_BKNG', 4981.9, 5000, 'Fail', 'reject_trade'
        FROM workflow_runs LIMIT 1"""))
    conn.commit()
    try:
        with pytest.raises(BenchmarkSetupError, match="inconsistency"):
            build_cases(conn)
    finally:
        conn.execute(sql("DELETE FROM risk_checks WHERE check_id='chk_bad'"))
        conn.commit()


def test_every_target_has_an_atomic_claim(conn):
    from src.critic.benchmark import build_cases

    for c in build_cases(conn):
        for t in c.targets:
            assert t.claim and t.expected in ("contradicted", "supported", "undetermined")


def test_provenance_records_what_matters(conn):
    from src.critic.benchmark import build_cases, provenance

    p = provenance(conn, build_cases(conn), "m")
    assert {"src/critic/critique.py", "src/critic/prompts.py", "src/generate_blotter.py"} <= set(p["source_hashes"])
    assert p["verifier_prompt_hash"] and p["case_set_hash"] and p["database"]["fingerprint"]


def test_atomic_mode_scores_without_extraction(conn):
    """The verifier can be measured on its own."""
    from src.critic.benchmark import build_cases, score_atomic

    case = next(c for c in build_cases(conn) if c.kind == "sign_flip")
    verdicts = {t.id: _v(t.claim, "contradicted" if t.expected == "contradicted" else "verified")
                for t in case.targets}
    r = score_atomic(case, verdicts)
    assert r.passed and r.critique is None


# ===========================================================================
# PROMPT: the verifier is told to judge the claim, not the question
# ===========================================================================

def test_verifier_prompt_separates_claim_from_question():
    from src.critic.prompts import VERIFIER_PROMPT

    assert "Judge the claim as stated" in VERIFIER_PROMPT
    assert "Entry-to-exit returns over different holding periods" in VERIFIER_PROMPT
    assert "absent reason in one table does not prove" in VERIFIER_PROMPT


# ===========================================================================
# PRESERVED: the successes
# ===========================================================================

def test_grounded_contradiction_of_reversed_reason_stands(conn, tmp_path):
    pair, reason = conn.execute(sql(
        "SELECT pair, rejection_reason FROM pair_evaluations WHERE evaluation_result='Rejected' AND rejection_reason IS NOT NULL LIMIT 1")).one()
    claim = Claim("x", f"{pair} was rejected because of position size", "causal", True, True)
    client = ScriptedClient([
        tool_turn("explain_rejection", {"pair": pair}),
        final_turn(f"VERDICT: contradicted\nEVIDENCE: explain_rejection — {pair} rejection_reason is {reason}"),
    ])
    v = verify_claim(claim, "q", conn, client=client, trace_dir=tmp_path)
    assert v.verdict == "contradicted" and v.downgraded is None


def test_cli_critique_flag_is_read(monkeypatch, capsys):
    import sys

    import cli
    from src.critic import critique as crit_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["cli.py", "q", "--critique"])

    class T:
        answer, tool_calls, error = "Answer.", [], None
        def render(self): return "trace"

    monkeypatch.setattr("cli.run_agent", lambda *a, **k: T())
    called = {}
    def fake(q, a, conn, **kw):
        called["q"] = q
        return Critique(q, a, [], [])
    monkeypatch.setattr(crit_module, "critique", fake)
    cli.main()
    assert called["q"] == "q" and "CRITIQUE" in capsys.readouterr().out
