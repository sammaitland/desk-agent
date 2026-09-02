"""Tests for the agent loop.

These cover the loop's contract with the API, not the model's judgement. A
scripted client supplies the responses, so a failure here is a loop bug rather
than model behaviour — that separation is the point of injecting the client.

Model judgement is what the Phase 3 eval harness will test, against real calls.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from src.agent.loop import run_agent
from src.agent.prompt import SYSTEM_PROMPT, build_system_prompt
from src.agent.scripted import (
    FailingClient,
    ScriptedClient,
    final_turn,
    parallel_tool_turn,
    tool_turn,
)
from src.db import create_schema, get_engine
from src.generate_blotter import Generator


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = tmp_path_factory.mktemp("agent") / "blotter.db"
    engine = get_engine(f"sqlite:///{db}")
    create_schema(engine)
    gen = Generator(seed=42, days=90)
    gen.run()
    gen.write(engine)
    with engine.connect() as c:
        yield c


# --- protocol correctness -------------------------------------------------

def test_first_request_carries_question_and_tools(conn):
    client = ScriptedClient([final_turn("Answer.")])
    run_agent("What happened?", conn, client=client)
    request = client.requests[0]
    assert request["messages"] == [{"role": "user", "content": "What happened?"}]
    assert len(request["tools"]) == 7
    assert request["system"].startswith("You are an analytics assistant")


def test_tool_results_go_in_a_user_message(conn):
    """The API rejects tool_result blocks in an assistant message."""
    client = ScriptedClient([
        tool_turn("query_blotter", {"entity": "positions", "limit": 1}),
        final_turn("Done."),
    ])
    run_agent("Show me a position.", conn, client=client)
    messages = client.requests[1]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"


def test_every_tool_use_gets_a_matching_result(conn):
    """Unmatched tool_use ids are an API error, so ids must round-trip."""
    client = ScriptedClient([
        parallel_tool_turn([
            ("execution_quality", {"group_by": "order_type"}),
            ("alpha_attribution", {"group_by": "idx"}),
        ]),
        final_turn("Done."),
    ])
    run_agent("Two things at once.", conn, client=client)
    messages = client.requests[1]["messages"]
    requested = {b["id"] for b in messages[1]["content"] if b["type"] == "tool_use"}
    returned = {b["tool_use_id"] for b in messages[2]["content"]}
    assert requested == returned and len(requested) == 2


def test_tool_result_content_is_a_json_string(conn):
    client = ScriptedClient([
        tool_turn("query_blotter", {"entity": "positions", "limit": 1}),
        final_turn("Done."),
    ])
    run_agent("Anything.", conn, client=client)
    block = client.requests[1]["messages"][2]["content"][0]
    assert isinstance(block["content"], str)
    payload = json.loads(block["content"])
    assert {"summary", "provenance", "data"} <= set(payload)


def test_history_grows_each_turn(conn):
    """The model is stateless: every turn resends the full conversation."""
    client = ScriptedClient([
        tool_turn("query_blotter", {"entity": "positions", "limit": 1}, call_id="t1"),
        tool_turn("query_blotter", {"entity": "orders", "limit": 1}, call_id="t2"),
        final_turn("Done."),
    ])
    run_agent("Two lookups.", conn, client=client)
    assert [len(r["messages"]) for r in client.requests] == [1, 3, 5]


def test_assistant_content_is_echoed_verbatim(conn):
    client = ScriptedClient([
        tool_turn("query_blotter", {"entity": "runs", "limit": 1}, text="Checking."),
        final_turn("Done."),
    ])
    run_agent("Runs please.", conn, client=client)
    blocks = client.requests[1]["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "Checking."}
    assert blocks[1]["name"] == "query_blotter"
    assert blocks[1]["input"] == {"entity": "runs", "limit": 1}


# --- chaining -------------------------------------------------------------

def test_chained_investigation(conn):
    """The demo path: locate the problem, then pull its detail."""
    order_id = conn.execute(text("""
        SELECT o.order_id FROM orders o JOIN fills f ON f.order_id = o.order_id
        WHERE o.fallback_reason = 'spread_validation_failure' LIMIT 1""")).scalar()
    client = ScriptedClient([
        tool_turn("detect_anomalies", {"severity": "warning"}, call_id="t1"),
        tool_turn("execution_quality", {"order_id": order_id}, call_id="t2"),
        final_turn("Routed to market because the spread exceeded the cap."),
    ])
    trace = run_agent("Why did that order fill badly?", conn, client=client)
    assert trace.tool_sequence == ["detect_anomalies", "execution_quality"]
    assert trace.turns == 3
    assert not any(c.error for c in trace.tool_calls)


# --- failure handling -----------------------------------------------------

def test_bad_arguments_are_returned_not_raised(conn):
    """The model must get the chance to correct its own mistake."""
    client = ScriptedClient([
        tool_turn("explain_position", {"nonsense": True}),
        final_turn("Corrected."),
    ])
    trace = run_agent("Explain something.", conn, client=client)
    assert trace.tool_calls[0].error is True
    assert trace.answer == "Corrected."


def test_unknown_tool_is_handled(conn):
    client = ScriptedClient([tool_turn("no_such_tool", {}), final_turn("Recovered.")])
    trace = run_agent("Do a thing.", conn, client=client)
    assert trace.tool_calls[0].error is True
    assert trace.stop_reason == "end_turn"


def test_turn_cap_stops_the_loop(conn):
    """A model that never stops calling tools must not spin indefinitely."""
    script = [tool_turn("query_blotter", {"entity": "positions", "limit": 1},
                        call_id=f"t{i}") for i in range(30)]
    trace = run_agent("Loop.", conn, client=ScriptedClient(script), max_turns=3)
    assert trace.turns == 3
    assert trace.stop_reason == "max_turns"
    assert "Stopped after 3 turns" in trace.answer


def test_transport_failure_degrades_gracefully(conn):
    trace = run_agent("Anything.", conn, client=FailingClient())
    assert trace.stop_reason == "error"
    assert "ConnectionError" in trace.error
    assert "failed" in trace.answer


# --- tracing --------------------------------------------------------------

def test_trace_records_calls_and_usage(conn):
    client = ScriptedClient([
        tool_turn("alpha_attribution", {"group_by": "idx"}),
        final_turn("Alpha is concentrated in the extreme buckets."),
    ])
    trace = run_agent("Where is alpha coming from?", conn, client=client)
    assert trace.tool_sequence == ["alpha_attribution"]
    assert trace.tool_calls[0].arguments == {"group_by": "idx"}
    assert trace.tool_calls[0].summary
    assert trace.input_tokens > 0 and trace.output_tokens > 0
    assert trace.duration_ms >= 0


def test_trace_serialises_and_saves(conn, tmp_path):
    client = ScriptedClient([
        tool_turn("detect_anomalies", {"limit": 5}),
        final_turn("Two halts last week."),
    ])
    trace = run_agent("What broke?", conn, client=client)
    path = trace.save(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["question"] == "What broke?"
    assert payload["tool_sequence"] == ["detect_anomalies"]
    assert payload["answer"] == "Two halts last week."


def test_trace_renders_readably(conn):
    client = ScriptedClient([
        tool_turn("query_blotter", {"entity": "positions", "limit": 2}),
        final_turn("Two positions."),
    ])
    rendered = run_agent("Show positions.", conn, client=client).render()
    assert "query_blotter" in rendered and "entity=" in rendered


# --- prompt ---------------------------------------------------------------

def test_prompt_states_the_domain_traps():
    """Each of these, omitted, produces a confident wrong answer."""
    for phrase in ["Alpha is not profit", "Tail 'L' means long", "24bps",
                   "Never state a number that did not come from a tool",
                   "read-only"]:
        assert phrase in SYSTEM_PROMPT


def test_prompt_appends_runtime_context():
    prompt = build_system_prompt("The blotter covers 2026-01-01 to 2026-06-30.")
    assert "## Current data" in prompt and "2026-06-30" in prompt


def test_prompt_tells_the_model_its_date_range(conn):
    client = ScriptedClient([final_turn("Answer.")])
    run_agent("When does the data start?", conn, client=client)
    assert "The blotter covers" in client.requests[0]["system"]
