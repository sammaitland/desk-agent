# Desk Agent — Project Status

A read-only analytics agent over a systematic pairs-trading blotter. Ask a
question in natural language, get an investigated answer with charts and an
audit trail.

**State:** Phases 0–9 complete. 252 tests passing. Slack and CLI verified
against the live API. CI green on SQLite and PostgreSQL.

---

## 1. What exists

### The architectural rule

**The agent orchestrates; deterministic code computes.** Every number in every
answer comes from a Python function the model *called*, never from the model's
own arithmetic. The LLM decides which analysis to run and writes the narrative;
the maths is tested code.

This is the answer to "how do you stop it inventing numbers", it is what makes
the eval harness possible (behaviour decomposes into checkable units), and it
is why the MCP server was a shim rather than a rewrite.

### Phase 0 — Blotter (`schema.sql`, `src/generate_blotter.py`, `src/config.py`)

13 tables conforming to the V9.4C trade event schema, populated by a seeded
generator standing in for the lost paper-trading logs. The data is fabricated;
the *shape* is the contract — real paper-account output populates the same
tables unchanged.

Reproduces the real mechanics rather than a generic blotter: Co1/Co2 + Tail
(L/U) with no fixed long/short roles, bucket-driven leg weights and position
multipliers (40–70% buckets disabled at 0.0x), index-relative alpha
(`W1·co1 − W2·co2 − β·index`), ticker-level order aggregation with pro-rata
allocation back to tags, LMT with 45s timeout falling back to MKT, and
stop-losses on the short leg with orphan detection.

Five scenarios are planted deterministically so the demo and evals reproduce:
a wide-spread market fill, a limit timeout, a stop trigger leaving an orphan,
a delisting force-close, and a reconciliation halt.

### Phase 1 — Tool layer (`src/tools/`)

Seven deterministic functions: `query_blotter`, `explain_position`,
`explain_rejection`, `execution_quality`, `alpha_attribution`,
`detect_anomalies`, `make_chart`.

Uniform envelope — `data`, `provenance`, `summary` — where provenance carries
the filters applied, rows examined and window covered. That is what lets the
agent say "across 47 orders in that window" rather than implying a scope it
never checked.

Errors return rather than raise, so the loop can hand a bad-argument problem
back to the model to correct. Read-only is structural: a test greps the package
for write statements.

### Phase 2 — Agent loop (`src/agent/`)

Hand-rolled against the Anthropic Messages API, no orchestration framework.
`loop.py` runs the tool-use cycle; `prompt.py` holds the domain grounding and
sourcing rules; `trace.py` captures every run; `scripted.py` is a fake client
that lets the loop be tested without API calls.

The client is injected rather than constructed, which separates loop bugs from
model behaviour and makes the eval harness cheap.

### Phase 3 — Evals (`src/evals/`, `run_evals.py`)

Eight cases, each derived from an observed run rather than an assumption. Two
tiers, deliberately separate: `pytest` tests the loop with a scripted client
(free, every commit); `run_evals.py` tests the model live (costs calls, on
demand).

The centrepiece is `numeric_fidelity` — it extracts every figure from an answer
and traces each to something a tool returned, allowing rounding but not drift,
and ignoring digits that assert nothing (tags, order ids, dates, times).

`tests/test_evals.py` calibrates the checks themselves, because a check that
passes a bad answer gives confident wrong readings about whether the system
regressed.

### Phase 4 — Slack (`src/slack/`)

Socket Mode bot: mention it in a channel, get a threaded reply with charts
attached and a provenance footer. No public URL or deployment needed.

The substantive work was `format.py`. Slack uses `mrkdwn`, not Markdown: bold is
a single asterisk, there are no headers, and there are no tables. Tables are
converted to fixed-width text in a code block, which is Slack's only way to
align columns. These failures are silent — nothing raises, the message just
renders wrongly.

### Phase 5 — MCP server (`src/mcp_server/`)

Six tools published over the Model Context Protocol for any MCP host. The
wrapper adds no logic: the SDK derives every JSON schema from the existing type
hints, and the docstrings become tool descriptions unchanged. `make_chart` is
excluded — it writes PNGs to local disk, meaningless to a remote host.

Built against **mcp 2.x**, where `FastMCP` was renamed `MCPServer` and fields
moved to snake_case. Most tutorials still show v1, which will not import.

### Phase 6 — CI and containers

`.github/workflows/tests.yml` runs three jobs: pytest on SQLite (3.11 and
3.12), the whole suite against a real PostgreSQL service container, and an
import check that walks every module.

`Dockerfile` / `docker-compose.yml` / `Makefile`. Compose brings up Postgres
alongside the agent — the local equivalent of the Postgres CI job.

### Phase 7 — Dashboard (`src/dashboard/`)

Five tabs over the same blotter: Overview (cumulative alpha, leverage, recent
events), Performance (attribution by index, bucket, tail, exit reason, month),
Execution (TCA and worst fills), Screening (rejection funnel and per-pair
lookup), and Ask — the agent embedded.

Computes nothing of its own: every metric routes through the tool layer, so a
figure here and a figure in an agent answer are the same tested function.
`test_headline_alpha_matches_the_tool` enforces it. Data access lives in
`data.py` separately from the UI in `app.py`, which is what makes the numbers
testable while the layout is not.

### Phase 8 — Retrieval over documentation (`src/rag/`)

An eighth tool, `search_documentation`, retrieves passages from the trading
system's design documents in `docs/`. Two backends behind a `Retriever`
protocol: `LexicalRetriever` (TF-IDF, deterministic, no model download) and
`EmbeddingRetriever` (sentence-transformers, optional). `RAG_BACKEND` env var
selects; defaults to lexical, falls back gracefully if embeddings unavailable.

Chunking follows heading boundaries rather than token counts, and each chunk
carries its heading path as both citation and retrieval prefix. The tool
returns the standard `ToolResult` envelope with provenance, so the agent's
sourcing discipline applies unchanged.

Three new eval cases: `design_rationale`, `definition_lookup`, and
`outside_the_corpus` (held-out). `what_and_why` tests chaining a blotter
lookup with a documentation search.

### Phase 9 — Cost-aware routing (`src/routing/`)

Predicts a query's cost before it runs by kNN over the system's own baseline
traces, selects a model and turn budget, and compresses stale tool results so
any path costs less. Measured under a three-arm comparison (baseline,
compression-only, routed) with a warm-cache protocol, leave-one-question-out,
full four-class cost accounting and casewise regression reporting.

**Result** (`docs/EXPERIMENT_routing_2026-09-06.md`): compression reduced
processed tokens by 15% and price-weighted cost by 29%, with no pass-rate
change distinguishable from run-to-run variance. Routing was not evaluated —
the fifteen-question corpus was too small for leave-one-question-out to leave
neighbours. The suite's noise floor (roughly ±3 cases of 22 between identical
runs) turned out to be larger than the effect being measured, which is the
more useful finding.

### Credentials (`src/env.py`, `.env`)

`.env` at the project root, gitignored, loaded automatically by every entry
point. Not merely convenient: Claude Desktop launches the MCP server as a
subprocess with a minimal environment that does not inherit shell exports, so a
server relying on `export` would fail to authenticate with no obvious cause.

---

## 2. Test coverage

| File | Tests | Covers |
|---|---:|---|
| `test_tools.py` | 39 | envelope contract, arithmetic, empty/malformed input, schema consistency, SQL portability |
| `test_evals.py` | 28 | the eval checks themselves |
| `test_blotter.py` | 24 | domain invariants — disabled buckets never trade, alpha stays market-neutral, allocations reconcile |
| `test_rag.py` | 21 | chunking, retrieval ranking, tool envelope, integration |
| `test_slack.py` | 18 | mrkdwn conversion, truncation, chart extraction |
| `test_agent.py` | 17 | API protocol, chaining, error recovery, turn cap, tracing |
| `test_dashboard.py` | 13 | cached queries, agreement with the tool layer, SQL portability |
| `test_mcp_server.py` | 10 | registration, schema generation, envelope preservation |
| **Total** | **170** | |

---

## 3. Outstanding items

### Known behavioural gaps (measurable, not blocking)

- [ ] **Verbosity on broad questions.** "What went wrong last week?" produces
      ~400 words across six sections despite the prompt fix and the
      `brevity(300)` eval check. Narrower questions are fine, so it is specific
      to enumerable multi-category answers. Options: raise the limit and treat
      it as a ceiling, or add a harder constraint ("at most three sections;
      report counts for the rest").
- [ ] **Names the next step instead of taking it.** Ends answers with "worth
      investigating X" where X is a tool it holds. `no_deferred_investigation`
      catches some phrasings but not all.
- [ ] **Computes figures the tools did not return.** Observed stating a
      book-average alpha and an average trade count that appear derived rather
      than looked up. `numeric_fidelity` does not catch this — it verifies
      traceability, not that the model refrained from arithmetic. Decide
      whether simple arithmetic over returned values is acceptable, or forbid
      it in the prompt. The credibility argument is "every number came from
      tested code"; "except the easy ones" weakens it.
- [ ] **Invents causal links between real figures.** Stated that a trade was
      blocked because its notional "came within $18 of the 5,000 cap" — being
      *under* a cap is not a breach. The real cause is the one-share floor
      (BKNG at ~$4,850 makes one share plus the other leg unsizeable). Both
      numbers were genuine tool output; the *relationship asserted between
      them* was invented. `numeric_fidelity` cannot catch this by design — it
      verifies where figures came from, not whether the sentence built around
      them is true. The most interesting limitation of the harness, and worth
      being able to articulate.

- [ ] **Borrowed a constant across contexts.** Claimed orphans were closed
      "within 45 seconds" — 45s is the *limit order* timeout and has nothing to
      do with orphan closure. Passed `numeric_fidelity` because 45 is a
      whitelisted prompt constant. A good concrete example of the check
      verifying traceability rather than truth.

### Open questions about the real system

- [ ] **Longlist/Shortlist ordering.** The event schema has primary filters →
      Shortlist, secondary signals → Longlist. `ARCHITECTURE.md` in the trading
      repo states the reverse, and also presents Longlist as an inter-module
      data contract when it is actually a manual-inspection Excel export. Both
      need correcting in that repo.
- [ ] **Alpha exit target and stop reference** (`src/config.py`) are invented —
      1.8% target, 4.5% × 0.40 stop. Replace with the real values if known.
- [ ] **VOX** is treated as a sixth index here; in the real system its
      constituents are folded into VGT by clustering. Harmless for synthetic
      data, but a known difference from the source system


### Next work

- [x] **Streamlit dashboard** — complete (Phase 7).

- [ ] **Real paper-account data.** Fire up the paper account, let it generate
      real fills, and swap them into the same schema. The whole Phase 0 design
      exists to make this a no-op.
- [ ] **Red-team / fault-injection harness.**:
      inject faults into the blotter (stale timestamps, reconciliation
      drift, phantom fills) and check whether the agent *detects* them or
      confidently narrates corrupted data as fact.


---

## 4. Running it

```bash
cd ~/Desktop/Python/desk_agent     # venv auto-activates via the zshrc cd hook
which python                        # confirm .../desk_agent/.venv/bin/python

make test                           # 170 tests
make blotter                        # regenerate synthetic data
make ask Q="what went wrong last week?"
make evals                          # costs API calls
make slack                          # start the Slack bot
make mcp                            # start the MCP server
make docker-test                    # suite against Postgres in Docker
```

Generated artefacts — `blotter.db`, `charts/`, `traces/`, `eval_results/` — are
gitignored and reproducible from source. The seeded generator is the reason:
anyone can rebuild the identical blotter with one command.

**Determinism boundary:** same seed and same interpreter gives the same blotter
every time. Across Python versions it differs (3.11 gives 375 positions, 3.12
gives 365) because `random.Random` guarantees reproducibility per interpreter
version, not across them. Fine for demos and evals; worth knowing.

---

## 5. Bugs found and fixed

These are the substantive ones, mostly caught by tests that existed to catch
exactly this class of error.

**Alpha was −55% and should not have been.** Three causes. The hedge term used
a raw stock beta (~1.0) where the formula needs the position's *net* exposure
(`W1·β₁ − W2·β₂`, near zero for a hedged pair) — a 5× over-hedge draining alpha
in a rising market. Price paths were independent random walks, so pair legs
never converged and the strategy had no edge by construction. And
`sum_dev_percentile` was `rng.uniform(0,100)` — a random number pretending to
be a measure of the leg spread, so no bucket could predict anything. Fixed with
net-exposure beta, an Ornstein-Uhlenbeck idiosyncratic component (~11-day
half-life), and a deviation signal computed from the actual price paths. Alpha
now +109%, concentrated in the extreme buckets as the system's own logic
predicts.

**The Postgres portability claim was false.** The README said since Phase 0 that
Postgres was "a config change, not a rewrite". It wasn't: `DATE(text_column)`
(Postgres has no `date(text)`) and `ROUND(double, 2)` (Postgres defines only
`round(numeric, int)`) appeared throughout, and the first was load-bearing —
every date filter went through it. The entire agent would have failed on its
first Postgres query. Fixed with `SUBSTR(col, 1, 10)` and
`ROUND(CAST(x AS NUMERIC), n)`, and now guarded by both a unit test and a CI
job against a real Postgres.

**Slack mangled position tags.** `VGT_AAPL_NVDA_L` rendered as `VGTAAPLNVDAL`
because the table formatter stripped every underscore to prevent literal
emphasis markers inside code blocks — where mrkdwn is not interpreted anyway.
Underscores in identifiers are data, not formatting.

**Order type semantics were contradictory.** A timed-out limit order was
recorded as `order_type='MKT'` while keeping its `limit_price`. Now
`order_type` records what was *submitted* and `fell_back_to_mkt` records how it
*resolved*, with the limit price kept as forensic evidence.

**Planted scenarios fired opportunistically.** Both the delisting and the
wide-spread fill depended on the RNG producing suitable conditions, so they
vanished at different `--days` values and broke the tests. Both are now applied
deterministically after generation.

**Three of the first five eval failures were the harness, not the agent.** The
scope check demanded "across N orders" phrasing on single-order
investigations; written dates ("Aug 25") were read as figures; and negative
numbers rendered with en-dashes failed to reconcile against their sources. Each
now has a regression test. That ratio is normal, and it is why calibrating the
instrument is real work rather than a slogan.
