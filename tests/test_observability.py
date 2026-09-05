"""Tests for the Langfuse exporter.

The exporter is a mapping from the hand-rolled Trace to Langfuse's model, and
these check the mapping — not Langfuse itself. A fake client records what the
exporter asked it to create, so the tree shape, nesting and usage figures can
be asserted without a Langfuse instance. Same pattern as the scripted API
client in the agent loop.

Three properties are protected beyond the mapping: the exporter is a no-op
when unconfigured, it never raises into the agent, and eval scores attach to
the trace they judged.
"""

from __future__ import annotations

import logging

import pytest

from src.agent.trace import Trace
from src.observability.langfuse_export import (
    _fake_for_tests,
    configured,
    export_trace,
    score_trace,
)


def make_trace(tools=None, turns=2) -> Trace:
    trace = Trace(question="why did the C order fill badly?")
    for turn, (name, args, summary, rows) in enumerate(tools or [], start=1):
        trace.turns = turn
        result = type("R", (), {"provenance": {"rows": rows}, "summary": summary, "data": []})()
        trace.record_tool(name, args, result, duration_ms=5, turn=turn)
    # one usage entry per turn
    for t in range(1, turns + 1):
        trace.turns = t
        usage = type("U", (), {"input_tokens": 1000 * t, "output_tokens": 100,
                               "cache_read_input_tokens": 2900 if t > 1 else 0,
                               "cache_creation_input_tokens": 2900 if t == 1 else 0})()
        trace.record_usage(usage)
    trace.finish("Routed to market; spread exceeded the cap.", "end_turn")
    return trace


# --- mapping --------------------------------------------------------------

def test_root_span_carries_question_and_answer():
    fake = _fake_for_tests()
    trace = make_trace()
    export_trace(trace, client=fake)
    root = fake.root[0]
    assert root.kwargs["name"] == "desk-agent"
    assert root.kwargs["input"] == {"question": trace.question}
    assert root.updates[-1]["output"] == {"answer": trace.answer}
    assert root.kwargs["metadata"]["run_id"] == trace.run_id


def test_one_generation_per_turn_with_cache_usage():
    """Cache tokens must reach Langfuse — they are the point of prompt caching
    and the dashboard is where the saving becomes visible."""
    fake = _fake_for_tests()
    export_trace(make_trace(turns=3), client=fake)
    generations = [c for c in fake.root[0].children if c.kwargs["as_type"] == "generation"]
    assert [g.kwargs["name"] for g in generations] == ["turn-1", "turn-2", "turn-3"]
    usage_turn2 = generations[1].updates[0]["usage_details"]
    assert usage_turn2["input"] == 2000
    assert usage_turn2["cache_read_input_tokens"] == 2900


def test_tool_spans_nest_under_their_turn():
    """The tree should read as the loop ran: model asked, tools answered."""
    fake = _fake_for_tests()
    trace = make_trace(tools=[
        ("detect_anomalies", {"severity": "warning"}, "3 events", 3),
        ("execution_quality", {"order_id": "ord_1"}, "23bps slippage", 1),
    ], turns=3)
    export_trace(trace, client=fake)
    generations = [c for c in fake.root[0].children if c.kwargs["as_type"] == "generation"]
    turn1_tools = [c.kwargs["name"] for c in generations[0].children]
    turn2_tools = [c.kwargs["name"] for c in generations[1].children]
    assert turn1_tools == ["detect_anomalies"]
    assert turn2_tools == ["execution_quality"]
    span = generations[1].children[0]
    assert span.kwargs["input"] == {"order_id": "ord_1"}
    assert span.updates[0]["output"] == "23bps slippage"


def test_tool_errors_are_marked():
    fake = _fake_for_tests()
    trace = Trace(question="q")
    bad = type("R", (), {"provenance": {}, "summary": "Invalid arguments for 'x': y", "data": None})()
    trace.turns = 1
    trace.record_tool("x", {}, bad, 1, 1)
    trace.record_usage(type("U", (), {"input_tokens": 1, "output_tokens": 1,
                                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})())
    trace.finish("a", "end_turn")
    export_trace(trace, client=fake)
    span = fake.root[0].children[0].children[0]
    assert span.updates[0]["level"] == "ERROR"


def test_export_flushes_and_returns_trace_id():
    fake = _fake_for_tests()
    trace_id = export_trace(make_trace(), client=fake)
    assert trace_id == "trace-fake"
    assert fake.flushed == 1


# --- safety properties ----------------------------------------------------

def test_noop_when_unconfigured(monkeypatch):
    """A machine without Langfuse keys must behave exactly like one with them,
    minus the export. No warning, no error, no network."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not configured()
    assert export_trace(make_trace()) is None


def test_export_failure_never_raises(caplog):
    """An observability failure must not become an availability failure."""
    class Broken:
        def start_as_current_observation(self, **kw):
            raise ConnectionError("langfuse down")

        def create_score(self, **kw): ...
        def flush(self): ...

    with caplog.at_level(logging.WARNING):
        result = export_trace(make_trace(), client=Broken())
    assert result is None
    assert "export failed" in caplog.text


def test_agent_answer_unaffected_by_export(monkeypatch):
    """End to end: with keys set but Langfuse unreachable, the agent still
    answers. The export is wired into _finish, so this is the real path."""
    from src.agent.loop import run_agent
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")  # nothing listens

    engine = get_engine("sqlite://")
    create_schema(engine)
    with engine.connect() as conn:
        trace = run_agent("q", conn, client=ScriptedClient([final_turn("Answer.")]))
    assert trace.answer == "Answer."


# --- scoring --------------------------------------------------------------

def test_scores_attach_to_the_trace():
    """Eval results become boolean scores on the trace they judged."""
    fake = _fake_for_tests()
    checks = [
        {"name": "numeric fidelity", "passed": True, "detail": ""},
        {"name": "under 300 words", "passed": False, "detail": "412 words"},
    ]
    written = score_trace("trace-abc", checks, client=fake)
    assert written == 2
    assert fake.scores[0]["name"] == "numeric_fidelity"
    assert fake.scores[0]["value"] == 1.0
    assert fake.scores[1]["value"] == 0.0
    assert fake.scores[1]["comment"] == "412 words"
    assert all(s["trace_id"] == "trace-abc" for s in fake.scores)


def test_scoring_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert score_trace("t", [{"name": "x", "passed": True}]) == 0


# --- optional: real client constructs -----------------------------------

def test_real_client_constructs_if_installed():
    pytest.importorskip("langfuse")
    from langfuse import Langfuse

    # tracing_enabled=False: no network, no background threads.
    client = Langfuse(public_key="pk", secret_key="sk", base_url="http://127.0.0.1:9",
                      tracing_enabled=False)
    assert hasattr(client, "start_as_current_observation")
