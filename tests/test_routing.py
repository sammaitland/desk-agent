"""Tests for cost-aware routing and its measurement architecture.

Beyond the mechanics (compression, prediction, routing), these protect the
invariants that make the comparison a valid measurement. Each was a real gap
in the first implementation, caught by external review:

  * legacy traces without model identity or cache accounting must be rejected
  * held-out cases must never enter the predictor's corpus
  * only traces tagged as baseline are training data
  * a question in the corpus must not be its own nearest neighbour
  * cost accounting must include cache writes and distinguish processed from billed
  * Langfuse must attribute generations to the model that ran
  * three arms, not two, so compression and routing are separable
  * casewise reporting, so a regression cannot hide behind an aggregate

None of these needs the API. Whether Haiku *answers well* on a light-tier
query is what the live comparison measures; it is deliberately not asserted
here.
"""

from __future__ import annotations

import json

import pytest

from src.agent.trace import TRACE_SCHEMA_VERSION, Trace
from src.routing.compression import COMPRESSED_MARKER, compress_history, strip_private_keys
from src.routing.predictor import COLD_START, CostPredictor, PastRun, load_runs
from src.routing.router import DEFAULT_TIER, TIERS, Router, route


# --- helpers --------------------------------------------------------------

def _run(q, tools, tokens, turns=2, model="claude-sonnet-4-6"):
    return PastRun(question=q, tool_sequence=tools, input_tokens=tokens,
                   output_tokens=tokens // 20, cache_read=0, cache_write=0,
                   turns=turns, model=model)


RUNS = [
    _run("what went wrong last week?", ["detect_anomalies"], 13000),
    _run("what went wrong yesterday?", ["detect_anomalies"], 9000),
    _run("why did the C order fill badly on the 24th?", ["query_blotter", "execution_quality"], 11000, 3),
    _run("why did the AAPL order fill badly?", ["query_blotter", "execution_quality"], 10000, 3),
    _run("where is alpha coming from?", ["alpha_attribution"], 7800),
    _run("what is the nominal direction check?", ["search_documentation"], 4000),
]


def _save(tmp_path, question, *, schema=TRACE_SCHEMA_VERSION, corpus="baseline",
          error=None, tools=("detect_anomalies",), model="claude-sonnet-4-6"):
    from src.routing.predictor import BASELINE_CONFIG

    t = Trace(question=question)
    t.schema_version = schema
    t.corpus = corpus
    t.model = model
    t.config = dict(BASELINE_CONFIG)
    t.error = error
    for i, name in enumerate(tools, start=1):
        t.turns = i
        t.record_tool(name, {}, type("R", (), {"provenance": {}, "summary": "s", "data": []})(), 1, i)
    t.input_tokens, t.output_tokens = 5000, 300
    t.finish("a", "end_turn")
    return t.save(tmp_path)


# --- compression (unchanged mechanics) ------------------------------------

def _tool_result(turn, rows=50):
    envelope = {"summary": f"{rows} rows", "provenance": {"rows": rows},
                "data": [{"i": i, "p": "x" * 40} for i in range(rows)]}
    return {"type": "tool_result", "tool_use_id": f"t{turn}",
            "content": json.dumps(envelope), "_turn": turn}


def test_fresh_results_are_left_alone():
    msgs = [{"role": "user", "content": [_tool_result(1)]}]
    out, removed = compress_history(msgs, current_turn=2)
    assert removed == 0 and COMPRESSED_MARKER not in out[0]["content"][0]["content"]


def test_stale_results_compress_to_summary_keeping_provenance():
    msgs = [{"role": "user", "content": [_tool_result(1)]}]
    out, removed = compress_history(msgs, current_turn=3)
    compact = json.loads(out[0]["content"][0]["content"])
    assert removed > 1000 and "data" not in compact and compact["provenance"] == {"rows": 50}


def test_private_keys_never_reach_the_api():
    out, _ = compress_history([{"role": "user", "content": [_tool_result(1)]}], current_turn=3)
    block = strip_private_keys(out)[0]["content"][0]
    assert not any(k.startswith("_") for k in block) and "tool_use_id" in block


# --- corpus loading: the three rejection rules ------------------------------

def test_legacy_schema_traces_are_rejected(tmp_path):
    """August traces predate model identity and cache accounting. They
    describe a different system and cannot train this one."""
    _save(tmp_path, "what went wrong last week?", schema=1)
    _save(tmp_path, "what went wrong yesterday?", schema=TRACE_SCHEMA_VERSION)
    runs, report = load_runs(tmp_path)
    assert [r.question for r in runs] == ["what went wrong yesterday?"]
    assert report.rejected_schema == 1 and report.accepted == 1


def test_held_out_questions_are_rejected_wherever_saved(tmp_path):
    """If the router had seen a held-out question, the held-out comparison
    would be routing from memory. Rejected by question identity, not by
    directory, so a stray file cannot leak it in."""
    from src.evals.cases import HELD_OUT

    held = HELD_OUT[0].question
    _save(tmp_path, held)
    _save(tmp_path, held.upper() + "  ")           # normalisation must catch this too
    _save(tmp_path, "what went wrong yesterday?")
    runs, report = load_runs(tmp_path)
    assert all(held.lower() not in r.question.lower() for r in runs)
    assert report.rejected_held_out == 2


def test_untagged_traces_are_rejected(tmp_path):
    """Ad-hoc CLI runs and routed runs are not baseline measurements."""
    _save(tmp_path, "q1", corpus=None)
    _save(tmp_path, "q2", corpus="adhoc")
    _save(tmp_path, "q3", corpus="baseline")
    runs, report = load_runs(tmp_path)
    assert [r.question for r in runs] == ["q3"] and report.rejected_corpus == 2


def test_errored_traces_are_rejected(tmp_path):
    _save(tmp_path, "q", error="boom")
    runs, report = load_runs(tmp_path)
    assert not runs and report.rejected_error == 1


def test_load_report_makes_a_thin_corpus_visible(tmp_path):
    _save(tmp_path, "a", schema=1)
    _save(tmp_path, "b", corpus=None)
    _save(tmp_path, "c")
    _, report = load_runs(tmp_path)
    assert report.accepted == 1 and report.rejected == 2


# --- prediction -----------------------------------------------------------

def test_predictor_finds_the_right_neighbours():
    pred = CostPredictor(RUNS, k=3).predict("why did the GS order fill badly?")
    assert pred.basis == "knn"
    assert pred.expected_tools == ["query_blotter", "execution_quality"]


def test_leave_one_question_out_excludes_every_run_of_that_question():
    """Two runs of the same question: excluding one file is not enough."""
    runs = RUNS + [_run("what went wrong last week?", ["detect_anomalies"], 13500)]
    p = CostPredictor(runs, k=5)

    with_self = p.predict("what went wrong last week?")
    assert with_self.confidence > 0.99                 # found itself

    without = p.predict("what went wrong last week?",
                        exclude_question="What went wrong last week?  ")
    assert without.confidence < 0.99
    assert without.nearest_question != "what went wrong last week?"
    assert "yesterday" in without.nearest_question      # nearest legitimate neighbour


def test_predictor_refuses_to_guess_on_novel_questions():
    assert CostPredictor(RUNS).predict("quantum entanglement").basis == "cold_start"


def test_prediction_uses_processed_tokens_including_cache():
    """A cached run processed more than it was billed for; the predictor
    must estimate what the model will handle, not what it will cost."""
    cached = PastRun("what went wrong last week", ["a"], input_tokens=1000, output_tokens=100,
                     cache_read=2900, cache_write=0, turns=2, model="m")
    assert cached.processed_tokens == 4000
    pred = CostPredictor([cached, cached]).predict("what went wrong last week")
    assert pred.expected_tokens == 4000


# --- routing ---------------------------------------------------------------

def _pred(tokens, tools, conf, basis="knn"):
    from src.routing.predictor import Prediction
    return Prediction(tokens, 2.0, tools, conf, 3, basis=basis)


def test_light_requires_all_three_conditions():
    assert route(_pred(4000, ["search_documentation"], 0.8)).tier is TIERS["light"]
    assert route(_pred(4000, ["search_documentation"], 0.3)).tier is not TIERS["light"]
    assert route(_pred(4000, ["a", "b"], 0.8)).tier is not TIERS["light"]
    assert route(_pred(20000, ["search_documentation"], 0.8)).tier is not TIERS["light"]


def test_cold_start_and_doubt_route_to_default():
    assert route(_pred(0, [], 0.0, "cold_start")).tier is DEFAULT_TIER
    assert route(_pred(11000, ["a", "b"], 0.5)).tier is DEFAULT_TIER


def test_router_exposes_corpus_report(tmp_path):
    _save(tmp_path, "what went wrong yesterday?")
    _save(tmp_path, "old", schema=1)
    router = Router(CostPredictor.from_disk(tmp_path))
    assert router.history_size == 1
    assert router.corpus_report.rejected_schema == 1


# --- cost accounting --------------------------------------------------------

def test_cost_includes_all_four_token_classes():
    from src.routing.compare import CACHE_READ_MULT, CACHE_WRITE_MULT, PRICE_PER_MTOK, ArmResult

    r = ArmResult("x", "baseline", True, "claude-sonnet-4-6", "standard",
                  input_tokens=1000, output_tokens=100, cache_read=2000, cache_write=500,
                  compressed_chars=0, held_out=False)
    p_in, p_out = PRICE_PER_MTOK["claude-sonnet-4-6"]
    expected = (1000 * p_in + 2000 * p_in * CACHE_READ_MULT
                + 500 * p_in * CACHE_WRITE_MULT + 100 * p_out) / 1e6
    assert r.cost_usd == pytest.approx(expected)
    assert r.processed_tokens == 3600


def test_processed_and_billed_diverge_under_caching():
    """Same processed tokens, very different bills — both must be reported."""
    from src.routing.compare import ArmResult

    uncached = ArmResult("x", "b", True, "claude-sonnet-4-6", "s", 3000, 100, 0, 0, 0, False)
    cached = ArmResult("x", "b", True, "claude-sonnet-4-6", "s", 100, 100, 2900, 0, 0, False)
    assert uncached.processed_tokens == cached.processed_tokens
    assert cached.cost_usd < uncached.cost_usd * 0.5


def test_haiku_is_priced_as_haiku():
    from src.routing.compare import ArmResult

    sonnet = ArmResult("x", "b", True, "claude-sonnet-4-6", "s", 10000, 500, 0, 0, 0, False)
    haiku = ArmResult("x", "r", True, "claude-haiku-4-5-20251001", "light", 10000, 500, 0, 0, 0, False)
    assert haiku.cost_usd < sonnet.cost_usd / 2


# --- langfuse model attribution ---------------------------------------------

def test_langfuse_uses_the_model_that_ran():
    """A routed Haiku call logged as Sonnet would misprice every downstream figure."""
    from src.observability.langfuse_export import _fake_for_tests, export_trace

    t = Trace(question="q")
    t.model = "claude-haiku-4-5-20251001"
    t.turns = 1
    t.record_usage(type("U", (), {"input_tokens": 1, "output_tokens": 1,
                                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})())
    t.finish("a", "end_turn")
    fake = _fake_for_tests()
    export_trace(t, client=fake)
    gen = fake.root[0].children[0]
    assert gen.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_loop_records_config_and_passes_model_to_export(tmp_path, monkeypatch):
    from src.agent.loop import run_agent_routed
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    router = Router(CostPredictor(RUNS))
    with engine.connect() as conn:
        t = run_agent_routed("what is the nominal direction check?", conn,
                             client=ScriptedClient([final_turn("a")]), router=router)
    assert t.config["routed"] is True
    assert t.config["model"] == t.model == TIERS[t.routing["tier"]].model


# --- three arms and casewise reporting --------------------------------------

def _arm(name, arm, passed, held=False, tokens=10000, model="claude-sonnet-4-6", tier=None):
    from src.routing.compare import ArmResult
    return ArmResult(name, arm, passed, model, tier or arm, tokens, tokens // 20, 0, 0, 0, held)


def test_three_arms_are_separable():
    from src.routing.compare import Comparison

    c = Comparison(results=[
        _arm("a", "baseline", True, tokens=10000),
        _arm("a", "compression", True, tokens=7000),
        _arm("a", "routed", True, tokens=6800, model="claude-haiku-4-5-20251001", tier="light"),
    ])
    s = c.summary()
    assert s["compression"]["token_saving_pct"] == 30.0
    assert s["routed"]["token_saving_pct"] == 32.0
    # Compression did the work; routing added 2 points on tokens but far more on cost.
    assert s["routed"]["cost_saving_pct"] > s["compression"]["cost_saving_pct"] + 20


def test_casewise_reports_regressions_not_just_totals():
    """One case flips pass->fail, another fail->pass: totals are equal, and
    that is exactly what an aggregate would hide."""
    from src.routing.compare import Comparison

    c = Comparison(results=[
        _arm("a", "baseline", True), _arm("a", "routed", False),
        _arm("b", "baseline", False), _arm("b", "routed", True),
    ])
    s = c.summary()
    assert s["arms"]["baseline"]["passed"] == s["arms"]["routed"]["passed"] == 1
    assert s["routed"]["regressed"] == ["a"]
    assert s["routed"]["improved"] == ["b"]
    assert s["routed"]["no_worse"] is False


def test_held_out_reported_separately():
    from src.routing.compare import Comparison

    c = Comparison(results=[
        _arm("dev", "baseline", True), _arm("dev", "routed", True),
        _arm("held", "baseline", True, held=True), _arm("held", "routed", False, held=True),
    ])
    s = c.summary()
    assert s["held_out"]["routed"]["passed"] == 0
    assert s["held_out"]["baseline"]["passed"] == 1


def test_comparison_passes_exclusion_for_dev_cases_only(tmp_path):
    """Proves run_comparison() itself applied leave-one-question-out: a spy
    predictor records what exclusion it was given. Development case -> its own
    question excluded. Held-out case -> nothing excluded (it is never in the
    corpus). The earlier version of this test checked router.decide separately,
    which proved nothing about the comparison path."""
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine
    from src.evals.cases import DEVELOPMENT, HELD_OUT
    from src.routing.compare import run_comparison

    dev, held = DEVELOPMENT[0], HELD_OUT[0]

    class Spy(CostPredictor):
        calls: list[tuple[str, str | None]] = []

        def predict(self, question, exclude_question=None):
            self.calls.append((question, exclude_question))
            return super().predict(question, exclude_question=exclude_question)

    spy = Spy(RUNS + [_run(dev.question, ["alpha_attribution"], 7800)])
    router = Router(spy)

    class Always:
        @property
        def messages(self):
            return ScriptedClient([final_turn("a")]).messages

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    with engine.connect() as conn:
        comparison = run_comparison([dev, held], conn, client=Always(), router=router,
                                    verbose=False, warm_cache=False)

    assert (dev.question, dev.question) in spy.calls
    assert (held.question, None) in spy.calls
    # And the recorded decision on the routed arm reflects the exclusion.
    routed_dev = next(r for r in comparison.arm("routed") if r.name == dev.name)
    assert routed_dev.routing["prediction"]["confidence"] < 0.99


# --- point 1: acceptance criterion and the CLI path -------------------------

def test_summary_has_overall_acceptance_requiring_both_arms():
    """The CLI reads summary()["no_worse"]; it must exist and must require
    zero regressions on every assessed arm, not just one."""
    from src.routing.compare import Comparison

    both_clean = Comparison(results=[
        _arm("a", "baseline", True), _arm("a", "compression", True), _arm("a", "routed", True)])
    assert both_clean.summary()["no_worse"] is True
    assert both_clean.summary()["assessed_arms"] == ["compression", "routed"]

    compression_regresses = Comparison(results=[
        _arm("a", "baseline", True), _arm("a", "compression", False), _arm("a", "routed", True)])
    assert compression_regresses.summary()["no_worse"] is False

    routed_regresses = Comparison(results=[
        _arm("a", "baseline", True), _arm("a", "compression", True), _arm("a", "routed", False)])
    assert routed_regresses.summary()["no_worse"] is False

    baseline_only = Comparison(results=[_arm("a", "baseline", True)])
    assert baseline_only.summary()["no_worse"] is False     # nothing assessed -> not a pass


def test_cli_compare_routing_exit_code_follows_acceptance(monkeypatch, capsys):
    """Exercises the actual CLI branch that crashed: a KeyError on
    summary()["no_worse"] would surface here, not in a unit test of summary()."""
    import sys

    import run_evals
    from src.routing import compare as compare_module
    from src.routing.compare import Comparison

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--compare-routing", "--case", "off_topic"])

    def fake_run(cases, conn, client=None, router=None, verbose=True, **kw):
        return Comparison(results=[_arm("off_topic", "baseline", True),
                                   _arm("off_topic", "compression", True),
                                   _arm("off_topic", "routed", False)])

    monkeypatch.setattr(compare_module, "run_comparison", fake_run)
    monkeypatch.setattr("src.routing.router.Router.__init__", lambda self, predictor=None: setattr(self, "predictor", CostPredictor([])))
    assert run_evals.main() == 1                     # routed regressed -> fail

    def fake_run_clean(cases, conn, client=None, router=None, verbose=True, **kw):
        return Comparison(results=[_arm("off_topic", "baseline", True),
                                   _arm("off_topic", "compression", True),
                                   _arm("off_topic", "routed", True)])

    monkeypatch.setattr(compare_module, "run_comparison", fake_run_clean)
    assert run_evals.main() == 0


# --- point 2: baseline config enforced -------------------------------------

def _save_cfg(tmp_path, question, **overrides):
    from src.routing.predictor import BASELINE_CONFIG

    t = Trace(question=question)
    t.corpus = "baseline"
    t.config = {**BASELINE_CONFIG, **{k: v for k, v in overrides.items() if k != "model_field"}}
    t.model = overrides.get("model_field", t.config["model"])
    t.turns = 1
    t.record_tool("detect_anomalies", {}, type("R", (), {"provenance": {}, "summary": "s", "data": []})(), 1, 1)
    t.input_tokens, t.output_tokens = 5000, 300
    t.finish("a", "end_turn")
    return t.save(tmp_path)


@pytest.mark.parametrize("override", [
    {"model": "claude-haiku-4-5-20251001", "model_field": "claude-haiku-4-5-20251001"},
    {"compress": True},
    {"routed": True},
    {"max_turns": 12},
    {"caching": False},
    {"model_field": "claude-haiku-4-5-20251001"},   # config says Sonnet, trace.model disagrees
])
def test_baseline_config_mismatch_is_rejected(tmp_path, override):
    """A trace tagged baseline but run under any other configuration measured
    a different system. The tag says what was intended; the config wins."""
    _save_cfg(tmp_path, "what went wrong yesterday?", **override)
    runs, report = load_runs(tmp_path)
    assert not runs and report.rejected_config == 1


def test_baseline_config_match_is_accepted(tmp_path):
    _save_cfg(tmp_path, "what went wrong yesterday?")
    runs, report = load_runs(tmp_path)
    assert len(runs) == 1 and report.rejected_config == 0


def test_loop_records_a_config_that_matches_baseline_when_run_as_baseline(tmp_path):
    """The config the loop writes for a default run must equal BASELINE_CONFIG,
    or --build-baseline would produce traces load_runs rejects."""
    from src.agent.loop import run_agent
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine
    from src.routing.predictor import BASELINE_CONFIG

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    with engine.connect() as conn:
        t = run_agent("q", conn, client=ScriptedClient([final_turn("a")]))
    assert t.config == BASELINE_CONFIG


# --- point 3: cache warm-up protocol ---------------------------------------

def test_warm_cache_primes_every_model_before_any_measured_run(tmp_path):
    """Cache state must not depend on arm order. With warm_cache=True, the
    first requests are warm-ups covering every model a tier could use, and
    only then do measured runs start."""
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine
    from src.evals.cases import DEVELOPMENT
    from src.routing.compare import WARMUP_QUESTION, run_comparison
    from src.routing.router import TIERS

    class Recording:
        requests: list[dict] = []

        @property
        def messages(self):
            outer = self

            class M:
                def create(self, **kw):
                    outer.requests.append(kw)
                    return final_turn("a")
            return M()

    client = Recording()
    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    with engine.connect() as conn:
        comparison = run_comparison([DEVELOPMENT[0]], conn, client=client,
                                    router=Router(CostPredictor(RUNS)), verbose=False)

    distinct_models = list(dict.fromkeys([TIERS[t].model for t in TIERS]))
    warmups = [r for r in client.requests if r["messages"][0]["content"] == WARMUP_QUESTION]
    assert len(warmups) == len(distinct_models)
    assert {r["model"] for r in warmups} == set(distinct_models)
    # Every warm-up precedes every measured run.
    first_measured = next(i for i, r in enumerate(client.requests)
                          if r["messages"][0]["content"] != WARMUP_QUESTION)
    assert all(r["messages"][0]["content"] == WARMUP_QUESTION
               for r in client.requests[:first_measured])
    assert comparison.cache_protocol == "warm"
    assert set(comparison.warmed_models) == set(distinct_models)


def test_cold_protocol_is_labelled(tmp_path):
    from src.agent.scripted import ScriptedClient, final_turn
    from src.db import create_schema, get_engine
    from src.evals.cases import DEVELOPMENT
    from src.routing.compare import report, run_comparison

    class Always:
        @property
        def messages(self):
            return ScriptedClient([final_turn("a")]).messages

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    with engine.connect() as conn:
        c = run_comparison([DEVELOPMENT[0]], conn, client=Always(),
                           router=Router(CostPredictor(RUNS)), verbose=False, warm_cache=False)
    assert c.cache_protocol == "cold" and not c.warmed_models
    assert "UNCONTROLLED" in report(c)


# --- point 4: normalisation ------------------------------------------------

@pytest.mark.parametrize("variant", [
    "What went wrong last week?",
    "what  went   wrong last week",
    "what\twent\nwrong last week!!",
    "WHAT WENT-WRONG LAST WEEK",
    "  what went wrong, last week  ",
    "what/went/wrong/last/week",
])
def test_norm_collapses_formatting_variants(variant):
    from src.routing.predictor import _norm

    assert _norm(variant) == "what went wrong last week"


def test_held_out_leakage_blocked_across_formatting_variants(tmp_path):
    """The identity check that keeps held-out questions out of the corpus
    must survive every variant _norm handles."""
    from src.evals.cases import HELD_OUT

    held = HELD_OUT[0].question
    for variant in [held.upper(), held.replace(" ", "  "), held.replace(" ", "-"),
                    "  " + held + "?? ", held.replace("?", "!")]:
        _save(tmp_path, variant)
    runs, report = load_runs(tmp_path)
    assert not runs and report.rejected_held_out == 5


# --- point 5: processed_tokens is canonical --------------------------------

def test_trace_processed_tokens_is_the_four_way_sum():
    t = Trace(question="q")
    t.input_tokens, t.output_tokens = 1000, 100
    t.cache_read_tokens, t.cache_write_tokens = 2900, 500
    assert t.processed_tokens == 4500


def test_arm_result_agrees_with_trace_processed_tokens(tmp_path):
    """Two definitions of the same sum would drift. ArmResult must equal Trace."""
    from src.agent.scripted import Response, TextBlock, Usage
    from src.db import create_schema, get_engine
    from src.evals.cases import BY_NAME
    from src.routing.compare import _arm as build_arm
    from src.agent.loop import run_agent

    class One:
        @property
        def messages(self):
            class M:
                def create(self, **kw):
                    return Response(content=[TextBlock(text="a")], stop_reason="end_turn",
                                    usage=Usage(1000, 100, 2900, 500))
            return M()

    engine = get_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_schema(engine)
    with engine.connect() as conn:
        t = run_agent("q", conn, client=One())
    arm = build_arm(BY_NAME["off_topic"], "baseline", t)
    assert arm.processed_tokens == t.processed_tokens == 4500


def test_build_baseline_refuses_held_out(monkeypatch, capsys):
    """The CLI guard: --build-baseline with --include-held-out must exit."""
    import sys

    import run_evals

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--build-baseline", "--include-held-out"])
    assert run_evals.main() == 2
    assert "Refusing" in capsys.readouterr().err
