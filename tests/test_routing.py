"""Tests for cost-aware routing.

Four things are protected:

1. Compression removes stale tool results and leaves fresh ones alone — and
   never sends the loop's bookkeeping keys to the API.
2. The predictor finds the right neighbours, weights by similarity, and
   refuses to guess when nothing is similar.
3. The router is conservative: it sends to `light` only on strong evidence,
   and defaults to `standard` on any doubt.
4. The comparison harness reports "no worse" honestly.

None of these needs the API. The router's *quality* — whether `light` answers
are actually acceptable — is what `run_evals.py --compare-routing` measures
live, and it is deliberately not asserted here: a scripted client cannot tell
you whether Haiku would have got the answer right.
"""

from __future__ import annotations

import json

import pytest

from src.agent.trace import Trace
from src.routing.compression import COMPRESSED_MARKER, compress_history, strip_private_keys
from src.routing.predictor import COLD_START, CostPredictor, PastRun, load_runs
from src.routing.router import DEFAULT_TIER, TIERS, Router, route


# --- compression ----------------------------------------------------------

def _tool_result(turn: int, rows: int = 50) -> dict:
    envelope = {"summary": f"{rows} rows found", "provenance": {"rows": rows},
                "data": [{"id": i, "payload": "x" * 40} for i in range(rows)]}
    return {"type": "tool_result", "tool_use_id": f"t{turn}",
            "content": json.dumps(envelope), "_turn": turn}


def test_fresh_results_are_left_alone():
    """The turn immediately after a result arrives must see the full payload."""
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "text", "text": "…"}]},
                {"role": "user", "content": [_tool_result(turn=1)]}]
    out, removed = compress_history(messages, current_turn=2)
    assert removed == 0
    assert COMPRESSED_MARKER not in out[2]["content"][0]["content"]


def test_stale_results_are_compressed_to_summary():
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "text", "text": "…"}]},
                {"role": "user", "content": [_tool_result(turn=1)]},
                {"role": "assistant", "content": [{"type": "text", "text": "…"}]},
                {"role": "user", "content": [_tool_result(turn=2)]}]
    out, removed = compress_history(messages, current_turn=3)
    stale = json.loads(out[2]["content"][0]["content"])
    fresh = json.loads(out[4]["content"][0]["content"])
    assert removed > 1000
    assert "data" not in stale and stale["summary"] == "50 rows found"
    assert stale["provenance"] == {"rows": 50}       # provenance survives
    assert "data" in fresh                            # newest untouched


def test_compression_is_idempotent():
    messages = [{"role": "user", "content": [_tool_result(turn=1)]}]
    once, r1 = compress_history(messages, current_turn=5)
    twice, r2 = compress_history(once, current_turn=6)
    assert r2 == 0 and twice[0]["content"][0]["content"] == once[0]["content"][0]["content"]


def test_private_keys_never_reach_the_api():
    """_turn and _compressed are ours; the API would reject them."""
    messages = [{"role": "user", "content": [_tool_result(turn=1)]}]
    out, _ = compress_history(messages, current_turn=3)
    clean = strip_private_keys(out)
    block = clean[0]["content"][0]
    assert not any(k.startswith("_") for k in block)
    assert "tool_use_id" in block


def test_compression_survives_malformed_content():
    """A tool result that is not JSON must not crash the loop."""
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": "not json", "_turn": 1}]}]
    out, _ = compress_history(messages, current_turn=3)
    assert COMPRESSED_MARKER in out[0]["content"][0]["content"]


# --- predictor ------------------------------------------------------------

RUNS = [
    PastRun("what went wrong last week?", ["detect_anomalies"], 13000, 800, 2),
    PastRun("what went wrong yesterday?", ["detect_anomalies"], 9000, 500, 2),
    PastRun("why did the C order fill badly on the 24th?",
            ["query_blotter", "execution_quality"], 11000, 500, 3),
    PastRun("why did the AAPL order fill badly?",
            ["query_blotter", "execution_quality"], 10000, 450, 3),
    PastRun("where is alpha coming from?", ["alpha_attribution"], 7800, 575, 2),
    PastRun("what does tail mean?", ["search_documentation"], 4000, 200, 2),
]


def test_predictor_finds_the_right_neighbours():
    p = CostPredictor(RUNS, k=3)
    pred = p.predict("why did the GS order fill badly?")
    assert pred.basis == "knn"
    assert pred.expected_tools == ["query_blotter", "execution_quality"]
    assert 9_000 < pred.expected_tokens < 12_500
    assert "fill badly" in pred.nearest_question


def test_predictor_weights_by_similarity():
    """A close neighbour should dominate a loose one."""
    p = CostPredictor(RUNS, k=5)
    pred = p.predict("what went wrong last week?")     # exact match exists
    assert pred.expected_tokens == pytest.approx(13800, rel=0.15)


def test_predictor_refuses_to_guess_on_novel_questions():
    """No near neighbour means cold start, not a low-confidence guess
    presented as a prediction."""
    p = CostPredictor(RUNS)
    pred = p.predict("quantum entanglement of the portfolio")
    assert pred.basis == "cold_start"
    assert pred.expected_tokens == COLD_START.expected_tokens


def test_predictor_cold_start_with_no_history():
    assert CostPredictor([]).predict("anything") is COLD_START


def test_load_runs_skips_errored_traces(tmp_path):
    good = Trace(question="q1"); good.finish("a", "end_turn")
    bad = Trace(question="q2"); bad.error = "boom"; bad.finish("x", "error")
    good.save(tmp_path); bad.save(tmp_path)
    runs = load_runs(tmp_path)
    assert [r.question for r in runs] == ["q1"]


# --- router ---------------------------------------------------------------

def _pred(tokens, tools, confidence, basis="knn"):
    from src.routing.predictor import Prediction
    return Prediction(expected_tokens=tokens, expected_turns=2.0,
                      expected_tools=tools, confidence=confidence,
                      neighbours=3, basis=basis)


def test_cold_start_routes_to_default():
    d = route(_pred(0, [], 0.0, basis="cold_start"))
    assert d.tier is DEFAULT_TIER and "no similar" in d.reason


def test_light_requires_all_three_conditions():
    """Confident, single-tool, cheap. Missing any one is not light."""
    assert route(_pred(4000, ["search_documentation"], 0.8)).tier is TIERS["light"]
    assert route(_pred(4000, ["search_documentation"], 0.3)).tier is not TIERS["light"]
    assert route(_pred(4000, ["a", "b"], 0.8)).tier is not TIERS["light"]
    assert route(_pred(20000, ["search_documentation"], 0.8)).tier is not TIERS["light"]


def test_deep_on_multi_hop_or_expensive():
    assert route(_pred(15000, ["a", "b", "c"], 0.5)).tier is TIERS["deep"]
    assert route(_pred(30000, ["a"], 0.5)).tier is TIERS["deep"]


def test_standard_is_the_default_on_doubt():
    assert route(_pred(11000, ["a", "b"], 0.5)).tier is DEFAULT_TIER


def test_every_decision_states_a_reason():
    for pred in [_pred(0, [], 0.0, "cold_start"), _pred(4000, ["a"], 0.9),
                 _pred(30000, ["a", "b", "c"], 0.5), _pred(11000, ["a", "b"], 0.5)]:
        assert route(pred).reason


def test_router_end_to_end_with_history(tmp_path):
    for run in RUNS:
        t = Trace(question=run.question)
        for name in run.tool_sequence:
            t.turns += 1
            t.record_tool(name, {}, type("R", (), {"provenance": {}, "summary": "s", "data": []})(), 1, t.turns)
        t.input_tokens, t.output_tokens = run.input_tokens, run.output_tokens
        t.turns = run.turns
        t.finish("a", "end_turn"); t.save(tmp_path)

    router = Router(CostPredictor.from_disk(tmp_path))
    assert router.history_size == len(RUNS)
    assert router.decide("what does tail mean?").tier is TIERS["light"]
    assert router.decide("why did the MSFT order fill badly?").tier is DEFAULT_TIER


# --- loop integration -----------------------------------------------------

def test_loop_records_routing_and_compression(tmp_path):
    from src.agent.loop import run_agent_routed
    from src.agent.scripted import ScriptedClient, final_turn, tool_turn
    from src.db import create_schema, get_engine
    from src.generate_blotter import Generator

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    g = Generator(seed=42, days=60); g.run(); g.write(engine)

    router = Router(CostPredictor(RUNS))
    client = ScriptedClient([
        tool_turn("detect_anomalies", {"limit": 40}, call_id="a"),
        tool_turn("query_blotter", {"entity": "orders", "limit": 5}, call_id="b"),
        final_turn("Done."),
    ])
    with engine.connect() as conn:
        trace = run_agent_routed("what went wrong last week?", conn, client=client, router=router)

    assert trace.routing and trace.routing["tier"] in TIERS
    assert trace.model == TIERS[trace.routing["tier"]].model
    # Third request should carry the compressed first result.
    third = client.requests[2]["messages"]
    first_result = third[2]["content"][0]["content"]
    assert COMPRESSED_MARKER in first_result
    assert trace.compressed_chars > 0
    assert "compressed" in trace.render()


# --- comparison -----------------------------------------------------------

def test_comparison_reports_no_worse_honestly():
    from src.routing.compare import ArmResult, Comparison

    c = Comparison()
    c.baseline = [ArmResult("a", True, "claude-sonnet-4-6", "baseline", 10000, 500, 0, 0, False),
                  ArmResult("b", True, "claude-sonnet-4-6", "baseline", 10000, 500, 0, 0, True)]
    c.routed = [ArmResult("a", True, "claude-haiku-4-5-20251001", "light", 4000, 200, 0, 0, False),
                ArmResult("b", False, "claude-sonnet-4-6", "standard", 8000, 400, 0, 0, True)]
    s = c.summary()
    assert s["no_worse"] is False          # routed failed a case baseline passed
    assert s["token_saving_pct"] > 30
    assert s["tier_mix"] == {"light": 1, "standard": 1}
    assert s["held_out"]["routed"]["passed"] == 0


def test_price_weighting_favours_light_tier():
    from src.routing.compare import ArmResult

    sonnet = ArmResult("x", True, "claude-sonnet-4-6", "standard", 10000, 500, 0, 0, False)
    haiku = ArmResult("x", True, "claude-haiku-4-5-20251001", "light", 10000, 500, 0, 0, False)
    assert haiku.cost_usd < sonnet.cost_usd / 2
