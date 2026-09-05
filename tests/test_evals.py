"""Tests for the eval checks.

The checks are the measuring instrument. A check that passes a bad answer, or
fails a good one, is worse than no check at all — it produces confident wrong
readings about whether the system regressed. So the instrument is calibrated
here, against traces constructed to be unambiguously right or wrong.

`numeric_fidelity` gets the most attention because it is doing the most work
and has the most ways to be subtly wrong: identifiers that look like figures,
percentages stated two ways, and legitimate rounding.
"""

from __future__ import annotations

import pytest

from src.agent.trace import Trace
from src.evals.cases import CASES, UNIVERSAL, select
from src.evals.checks import (
    bounded_retries,
    brevity,
    called_any_of,
    called_tool,
    challenges_premise,
    chained,
    forbids,
    mentions,
    no_absence_overclaim,
    no_deferred_investigation,
    no_tool_errors,
    numeric_fidelity,
    reports_scope,
    tool_budget,
)


def make_trace(answer: str, tools: list[tuple[str, dict, str, dict]] | None = None) -> Trace:
    """Build a trace by hand: (tool name, arguments, summary, raw data)."""
    trace = Trace(question="q")
    for name, arguments, summary, data in tools or []:
        result = type("R", (), {"provenance": {"rows": 1}, "summary": summary, "data": data})()
        trace.record_tool(name, arguments, result, 1, 1)
    trace.finish(answer, "end_turn")
    return trace


# --- tool selection -------------------------------------------------------

def test_called_tool():
    trace = make_trace("x", [("execution_quality", {}, "s", {})])
    assert called_tool("execution_quality")(trace).passed
    assert not called_tool("alpha_attribution")(trace).passed


def test_called_any_of():
    trace = make_trace("x", [("query_blotter", {}, "s", {})])
    assert called_any_of("query_blotter", "detect_anomalies")(trace).passed
    assert not called_any_of("alpha_attribution", "explain_position")(trace).passed


def test_chained_and_budget():
    two = make_trace("x", [("a", {}, "s", {}), ("b", {}, "s", {})])
    assert chained(2)(two).passed
    assert not chained(3)(two).passed
    assert tool_budget(2)(two).passed
    assert not tool_budget(1)(two).passed


def test_no_tool_errors_detects_failures():
    trace = Trace(question="q")
    bad = type("R", (), {"provenance": {}, "summary": "Invalid arguments for 'x': y", "data": None})()
    trace.record_tool("x", {}, bad, 1, 1)
    trace.finish("answer", "end_turn")
    assert not no_tool_errors()(trace).passed


# --- numeric fidelity -----------------------------------------------------

def test_accepts_figures_the_tool_returned():
    trace = make_trace(
        "The order filled at 23.31bps slippage on a 32.56bps spread.",
        [("execution_quality", {}, "slippage 23.31bps on 32.56bps spread",
          {"slippage_bps": 23.31, "spread_bps": 32.56})],
    )
    assert numeric_fidelity()(trace).passed


def test_rejects_invented_figures():
    trace = make_trace(
        "The order filled at 87.4bps slippage.",
        [("execution_quality", {}, "slippage 23.31bps", {"slippage_bps": 23.31})],
    )
    result = numeric_fidelity()(trace)
    assert not result.passed and "87.4" in result.detail


def test_allows_reasonable_rounding():
    """4.81 quoted as 4.8 is reporting; 23.3 quoted as 20 is not."""
    ok = make_trace("Roughly 4.8bps.", [("t", {}, "s", {"v": 4.81})])
    assert numeric_fidelity()(ok).passed
    bad = make_trace("Roughly 20bps.", [("t", {}, "s", {"v": 23.31})])
    assert not numeric_fidelity()(bad).passed


def test_ignores_identifiers_and_dates():
    """Tags, order ids and timestamps contain digits but assert nothing."""
    trace = make_trace(
        "Position VGT_AAPL_MSFT_L_20260815_001 on order ord_36af71cf7b "
        "opened 2026-08-20 at 14:58:00 and returned 1.27%.",
        [("explain_position", {}, "s", {"final_alpha_return_pct": 1.27})],
    )
    assert numeric_fidelity()(trace).passed


def test_accepts_prompt_constants_without_a_tool():
    """Thresholds are given in the system prompt, so quoting them is sourced."""
    trace = make_trace(
        "The spread exceeded the 24bps cap and the 45-second timeout applied.",
        [("execution_quality", {}, "s", {"spread_bps": 32.56})],
    )
    assert numeric_fidelity()(trace).passed


def test_accepts_percentage_restatement():
    trace = make_trace("Alpha was 2.3%.", [("t", {}, "s", {"alpha": 0.023})])
    assert numeric_fidelity()(trace).passed


def test_searches_nested_tool_output():
    """Figures often sit deep in a breakdown, not in the summary."""
    trace = make_trace(
        "VIS contributed 34.57% across 56 trades.",
        [("alpha_attribution", {}, "summary",
          {"breakdown": [{"grouping": "VIS", "total_alpha_pct": 34.57, "trades": 56}]})],
    )
    assert numeric_fidelity()(trace).passed


def test_ignores_written_dates():
    """Caught in the first live run: "Aug 25" was read as the figure 25."""
    trace = make_trace(
        "Five stop-outs hit on Aug 25 and two more on 17 August; slippage was 12.4bps.",
        [("detect_anomalies", {}, "s", {"slippage_bps": 12.4})],
    )
    assert numeric_fidelity()(trace).passed


def test_negative_figures_reconcile():
    """Answers render minus signs as en-dashes, which the extractor cannot see."""
    trace = make_trace(
        "Alpha was -1.197% at that point.",
        [("explain_position", {}, "s", {"alpha_path": [{"live_alpha_return_pct": -1.197}]})],
    )
    assert numeric_fidelity()(trace).passed


def test_passes_when_no_tools_were_called():
    """Nothing to verify against is not a failure — other checks catch that."""
    assert numeric_fidelity()(make_trace("No data available.")).passed


# --- behavioural checks ---------------------------------------------------

def test_forbids_and_mentions():
    trace = make_trace("Total profit was strong.")
    assert not forbids("profit")(trace).passed
    assert mentions("profit")(trace).passed
    assert not mentions("alpha")(trace).passed


def test_detects_deferred_investigation():
    """The failure observed live: recommending a tool it could have called."""
    deferred = make_trace("I'd recommend pulling execution quality on those orders.")
    assert not no_deferred_investigation()(deferred).passed
    acted = make_trace("Execution quality on those orders shows 12.4bps average slippage.")
    assert no_deferred_investigation()(acted).passed


def test_detects_absence_overclaim():
    """Also observed live: one rejection row read as 'evaluated only once'."""
    over = make_trace("It was evaluated only one evaluation in the window.")
    assert not no_absence_overclaim()(over).passed
    careful = make_trace("It appears once in the rejection log; it may have been "
                         "approved on other days.")
    assert no_absence_overclaim()(careful).passed


def test_detects_premise_challenge():
    pushed = make_trace("The data doesn't support the premise that this fill was bad.")
    assert challenges_premise()(pushed).passed
    accepted = make_trace("The fill was poor because the spread was wide.")
    assert not challenges_premise()(accepted).passed


def test_reports_scope():
    scoped = make_trace("Across 47 orders that week, average slippage was 3.2bps.")
    assert reports_scope()(scoped).passed
    unscoped = make_trace("Slippage is generally fine.")
    assert not reports_scope()(unscoped).passed


def test_bounded_retries():
    """Observed live: four documentation searches to reach one conclusion."""
    hunting = make_trace("x", [("search_documentation", {}, "s", [])] * 4)
    assert not bounded_retries("search_documentation", 2)(hunting).passed
    bounded = make_trace("x", [("search_documentation", {}, "s", [])] * 2)
    assert bounded_retries("search_documentation", 2)(bounded).passed


def test_brevity():
    assert brevity(10)(make_trace("one two three")).passed
    assert not brevity(2)(make_trace("one two three")).passed


# --- case definitions -----------------------------------------------------

def test_case_names_are_unique():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))


def test_every_case_has_checks_and_a_note():
    for case in CASES:
        assert case.checks, f"{case.name} has no checks"
        assert case.note, f"{case.name} has no note explaining why it exists"


def test_universal_checks_apply_everywhere():
    assert len(UNIVERSAL) >= 3


def test_selection_by_tag_and_name():
    assert select(tags=["chaining"]) and all("chaining" in c.tags for c in select(tags=["chaining"]))
    assert [c.name for c in select(names=["false_premise"])] == ["false_premise"]
    assert select(names=["nonexistent"]) == []


def test_held_out_cases_are_excluded_by_default():
    """The out-of-sample set must not leak into the default run, or it stops
    being out of sample the first time someone tunes against it."""
    from src.evals.cases import DEVELOPMENT, HELD_OUT

    assert HELD_OUT, "the suite needs held-out cases to have an honest measure"
    default = select()
    assert not any(c.held_out for c in default)
    assert len(default) == len(DEVELOPMENT)
    assert len(select(include_held_out=True)) == len(DEVELOPMENT) + len(HELD_OUT)


def test_held_out_share_is_meaningful():
    from src.evals.cases import CASES, HELD_OUT

    assert 0.2 <= len(HELD_OUT) / len(CASES) <= 0.5


def test_every_category_has_a_held_out_case():
    """Each behavioural category needs an out-of-sample probe, or a category
    could be fully in-sample without anyone noticing."""
    from src.evals.cases import CASES

    tags_dev = {t for c in CASES if not c.held_out for t in c.tags}
    tags_held = {t for c in CASES if c.held_out for t in c.tags}
    for tag in ("chaining", "honesty", "domain", "rag", "safety", "scope"):
        assert tag in tags_dev, f"no development case tagged {tag}"
        assert tag in tags_held, f"no held-out case tagged {tag}"
