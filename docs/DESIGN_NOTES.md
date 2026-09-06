# Desk Agent — Design Notes

Why the system is built the way it is, and what building it taught. `PROJECT_STATUS.md` records what exists; this records the reasoning.

---

## 1. Core concepts

### Tool use / function calling

You send the model a list of tool definitions — name, description, JSON Schema
for the arguments. The model does not execute anything. It returns a
`tool_use` block naming a tool and supplying arguments; **you** run the function
and send the result back. The loop repeats until the model stops asking.

**Why it matters:** this is the entire mechanism by which a language model
touches the outside world. Everything called "an agent" is a variation on it.

The model never executes anything. It emits a structured
request, my code runs the function, and the result goes back in the next
message. So the boundary of what it can do is exactly the set of functions I
wrote.

**Q:** *How does the model know which tool to pick?* From the
description text and the schema — that is all it sees. Most tool-selection
failures are description failures, not model failures. Mine are written to
disambiguate against each other, not in isolation: `explain_rejection`
explicitly names itself the counterpart to `explain_position`.

### The agent loop

    send messages + tools -> response
    stop_reason == 'tool_use'  -> execute, append results, repeat
    stop_reason == 'end_turn'  -> return the text

Three protocol details that bite: `tool_result` blocks must go in a **user**
message (assistant is an API error); every `tool_use` id needs a matching
result in the **same** message; and result content must be a JSON string, not
an object.

**Q:** *Why did you not use LangChain?* The abstractions are thin, and
knowing exactly what is on the wire is worth more than the lines saved —
particularly when debugging why a model chose the wrong tool. I can also tell
you precisely what a framework would abstract, which I could not if I had
started with one.

### Statelessness

The model holds nothing between calls. Every turn resends the entire
conversation, so a three-turn investigation sends 1, then 3, then 5 messages,
and every tool result is resent on every subsequent turn.

**Why it matters:** it makes context a compounding cost, not a one-off. That is
why the tool layer caps results at 500 rows — an uncapped result is paid for
again on every turn that follows.

**Q:** *How would you handle a long conversation?* Summarise older
turns, or drop tool results once their conclusions are in the narrative. I have
not needed to: the turn cap is 8 and most investigations finish in 2–3.

### Why deterministic tools

The model decides *which* analysis to run and writes the narrative. Every
figure comes from a tested Python function.

It is the answer to hallucinated numbers, and it is
architectural rather than a prompt instruction. The model cannot state a wrong
number because it never computes one. It also makes the system testable —
behaviour decomposes into checkable units, so an eval can assert on the
tool-call sequence rather than on prose.

**Q:** *So it can never be wrong?* No — and this is the important part.
It can still assert a false *relationship* between two true numbers. It once
said a trade was blocked because its notional "came within $18 of the $5,000
cap", which is incoherent: being under a cap is not a breach. Both figures were
genuine tool output; the causal link was invented. My numeric-fidelity check
verifies where figures came from, not whether the sentence built around them is
true. That is the honest limit of the technique.

---

## 2. The system, layer by layer

### The blotter (Phase 0)

13 tables conforming to the real system's event schema, populated by a seeded
generator. The original paper-trading logs were lost, so the data is
fabricated — but the *shape* is the contract, and real output will populate the
same tables unchanged.

**Why it looks like this:** it reproduces the actual mechanics rather than a
generic blotter. Positions are Co1/Co2 plus a Tail (L/U), so neither leg has a
fixed long/short role. Legs are weighted by CDF bucket, with position
multipliers from 0.7x to 1.4x and three buckets disabled at 0.0x. Performance
is index-relative alpha, not P&L. Orders aggregate by ticker across pairs and
allocate fills back pro rata.

**Q:** *Is synthetic data not a weakness?* It is a limitation, stated
plainly in the README. But the design work is in the schema and the mechanics,
and those came from the real system. The generator also plants specific
failures deterministically — a wide-spread market fill, a timeout, an orphaned
leg, a delisting, a reconciliation halt — because an agent with nothing to
investigate demonstrates nothing.

### The tool layer (Phase 1)

Seven functions. Uniform envelope: `data`, `provenance`, `summary`.

**The provenance block is the interesting part.** It carries the filters
applied, rows examined and window covered, which is what lets the agent say
"across 47 orders that week" instead of "generally". Without it, an agent
sounds authoritative about a scope it never checked.

**Two design rules worth stating:** errors *return* rather than raise, so the
loop can hand a bad-argument problem back to the model to correct rather than
surfacing a stack trace it might narrate as a system fault. And read-only is
structural — there is no write path in the package, and a test greps for one.

**Q:** *Why seven tools rather than one flexible one?* Granularity is
the real design question. Too coarse and the model cannot compose an answer;
too fine and it makes ten calls to answer one question. The heuristic I used is
one tool per question a desk actually asks, not one per table.

**Q:** *Would you let it place orders?* No, and the more interesting
answer is that I have drawn this line twice. In the trading system, delisting
detection is automated but execution is human-gated, because qualification
failure is a noisy proxy that also fires on connectivity faults. Here, analysis
is automated and action is absent entirely. The question is not what the system
*can* do, it is which decisions justify autonomy.

### The agent loop (Phase 2)

Hand-rolled. The client is **injected** rather than constructed, which is the
decision that pays off repeatedly: it separates loop bugs from model behaviour
(a failing scripted test is a fault in the loop, not the model) and makes the
eval harness runnable on every commit without API calls.

**The system prompt earns its length.** Each domain fact in it corrects a
specific confident wrong answer. Without "alpha is not profit" the model calls
it P&L. Without the Tail explanation it assumes Co1 is the long leg. Without
the thresholds it invents them.

### Evals (Phase 3)

**Two tiers, deliberately separate.** `pytest` tests the loop with a scripted
client — free, deterministic, every commit. `run_evals.py` tests the model
live — costs API calls, runs on demand. Merging them gives you either expensive
unit tests or evals that never see a model.

**Cases come from observed behaviour, not assumption.** Every one was written
after watching real runs. Evals written before observing a system test what you
assumed it would do wrong.

**Q:** *How do you know the evals are any good?* I test them. A check
that passes a bad answer gives confident wrong readings about whether the
system regressed, so `tests/test_evals.py` calibrates each check against traces
built to be unambiguously right or wrong. Three of the first five eval failures
turned out to be the harness, not the agent — the scope check demanded "across
N orders" phrasing on single-order investigations, written dates were read as
figures, and en-dash negatives failed to reconcile. Each now has a regression
test.

### Slack, MCP, dashboard (Phases 4, 5, 7)

Three transports over one tool layer. **The transport is not the
architecture.** The MCP server proves it: the SDK
derives every JSON schema from the existing type hints, so the file adds no
logic at all.

The dashboard computes nothing of its own either. A figure on screen and a
figure in an agent answer are the same tested function, enforced by
`test_headline_alpha_matches_the_tool`. A dashboard with its own alpha
calculation would eventually contradict the agent with no way to tell which was
right.

**The framing for the two modes:** a dashboard answers questions you knew to
ask in advance; the agent answers the ones you did not.

### CI and containers (Phase 6)

Three jobs: SQLite on two interpreters, the whole suite against a real
PostgreSQL, and an import check that walks every module.

**Why the Postgres job exists:** see section 5. It was added because a claim in
the README turned out to be false.

### Retrieval over documentation (Phase 8)

An eighth tool, `search_documentation`, retrieves passages from the trading
system's design documents. The blotter tools answer *what happened*; this one
answers *why the system is built this way*. A question like "why was this order
routed to market?" needs both, and the agent chains them.

**RAG alongside the tool layer, not instead of it.** Two retrieval mechanisms
suited to two data shapes — SQL over structured rows, retrieval over prose —
behind one agent with one provenance discipline. Every retrieved passage
carries a citation to file and section.

**The retriever is an interface with two backends.** Lexical (TF-IDF) is
deterministic and needs no model download; embedding (sentence-transformers)
finds passages that mean the same thing in different words. Which is right is a
property of the corpus, so it is configuration.

**Q:** *Why did you build tool-calling first and RAG second?* Because
the data was structured and the questions were about data. Typed functions over
a semantic layer are more governable than vector search — every metric
definition is code, not an embedding. RAG was added for the one thing the tools
could not answer: design rationale, which lives in prose.

**Q:** *Where does lexical retrieval fail?* The query "why are limit
orders rejected above 24bps?" ranks weakly because the documentation writes it
as `MAX_LIMIT_ORDER_SPREAD_BPS = 24` — same meaning, no shared tokens. That is
precisely the gap embeddings close, and it is a concrete example rather than a
textbook one.

**Three things it taught:** chunk by heading, not token count, because
technical documentation is already organised into units of meaning; a low
retrieval score is information, not noise — the first miss was a corpus gap,
not a retriever fault; and retrieval needs a floor, because the agent re-queried
four times through noise before the prompt told it to stop.

### Cost-aware routing (Phase 9)

Predicts the cost of a query before running it and selects model and turn
budget accordingly, with tool-result compression to reduce what any path costs.
Validated against held-out evals.

**The mechanism.** Token cost is dominated by tool results resent on every
turn, so predicting the tool path predicts most of the cost. The predictor is
kNN over the system's own execution traces — "questions like this one called
these tools and cost about this much." No training; a lookup over history,
embedded with the same retriever the RAG layer uses.

**The reference points.** RouteLLM (Berkeley, ICLR 2025) is the canonical
classifier-based router. FrugalGPT (Stanford) is the cascade alternative —
cheap model first, escalate on low confidence. A 2025 paper showed simple kNN
matches complex learned routers, which is why the implementation is kNN.

**Tool-result compression** is layer one of the standard three-layer cascade:
compress tool outputs, then sliding window, then summarise. Only layer one is
implemented — after a result has been used, it is replaced in history by its
summary line.

**Q:** *Routing or cascading?* Routing: one decision before the query
runs. Cascading costs a wasted call on every hard query.

**Q:** *What was the saving?* Quote the held-out number, with the caveat
that RouteLLM's 85% was benchmark-specific and so is mine.

**Q:** *Why not LLMLingua for compression?* Token-level compressors
destroy the action grammar agents depend on — 17 of 17 test cells collapsed in
a 2026 study. Summary-replacement at the tool-result level is the right
granularity.

**The connection to the trading system.** This is the spread-based
transaction-cost model applied to inference. Both replace a fixed cost
assumption with one conditioned on observable pre-trade quantities; both are
predict-the-cost-of-the-path, not predict-the-outcome.

### Prompt caching and the Agent SDK

**Prompt caching** is on: the system prompt and tool schemas are cached across
calls, cutting the resent-every-turn cost of ~3,000 tokens by roughly 90%. It
was added before the router so the router's savings are measured against a
baseline that already had the cheap win.

**The Claude Agent SDK** ships the loop I hand-rolled — same runtime as Claude
Code, with caching, compaction and subagents built in. I built the loop by hand
to know what is on the wire; for a production deployment I would evaluate the
SDK because it maintains what I would otherwise have to.

### Observability (Langfuse)

Traces export to Langfuse alongside the local JSON. Hand-rolled tracing came
first, so I know what Langfuse abstracts — the same argument as no-framework.
Langfuse was acquired by ClickHouse in January 2026 and remains MIT-licensed.

---

## 3. Design decisions

| Decision | Rationale |
|---|---|
| No orchestration framework | The abstractions are thin; knowing what is on the wire is worth more, and I can say precisely what a framework would have hidden. |
| Model orchestrates, code computes | Architectural answer to hallucinated numbers, and what makes the system testable. |
| Injected client | Separates loop bugs from model behaviour; makes evals cheap. |
| Errors return, not raise | Lets the model correct its own mistake instead of surfacing a stack trace. |
| Read-only, structurally | A guarantee enforced by a prompt is not a guarantee. |
| Seven tools, not one | One tool per question a desk asks, not one per table. |
| Provenance on every result | The difference between sounding authoritative and being scoped. |
| Two-tier testing | Loop mechanics free on every commit; model behaviour on demand. |
| Evals from observation | Evals written first test your assumptions, not the system. |
| Seeded, deterministic data | The demo and the eval set must reproduce; a scenario that fires opportunistically is not a test. |
| Tool-calling before RAG | Structured data, structured questions; typed functions are more governable than embeddings. RAG for prose only. |
| Two retrieval backends | Which is right is a corpus property, so it is configuration, not a commitment. |
| Held-out eval cases | Once the prompt is tuned against an eval, that eval is in-sample. Same discipline as walk-forward. |
| kNN routing, not a learned router | Simple kNN matches complex routers in the literature, and it needs no training — just the traces I already have. |
| Compression at the tool-result level | Token-level compressors destroy action grammar; the result-summary is the unit of meaning. |
| Hand-rolled loop, SDK for production | Built by hand to learn; would adopt the SDK where maintenance matters more than visibility. |

---

## 4. Failures and what they taught

Each of these was found by the project's own tests or diagnostics rather than
reported from outside, and each changed the design.

### Alpha was −55% and should not have been

Three causes, found by diagnostic rather than by the system complaining.

The hedge term used a raw stock beta of about 1.0 where the formula needs the
position's **net** exposure — `W1·β₁ − W2·β₂`, near zero for a hedged pair. A
five-fold over-hedge, draining alpha in a rising market.

Price paths were independent random walks, so pair legs never converged and the
strategy had no edge by construction.

And `sum_dev_percentile` was `rng.uniform(0, 100)` — a random number pretending
to be a measure of the leg spread. **No bucket can predict anything if the
signal is noise.**

Fixed with net-exposure beta, an Ornstein-Uhlenbeck idiosyncratic component
(~11-day half-life) so deviations actually revert, and a deviation signal
computed from the real price paths. Alpha is now +109%, concentrated in the
extreme buckets exactly as the system's own sizing logic predicts.

**What it demonstrates:** the data looked entirely plausible while being wrong.
`test_alpha_is_market_neutral_scale` exists to catch that class of regression,
because a hedged book producing equity-scale returns is a silent failure.

### A claim in my own README was false

Since Phase 0 it said Postgres was "a config change, not a rewrite". It was not.
`DATE(text_column)` works on SQLite but PostgreSQL has no `date(text)`, and
`ROUND(double, 2)` fails because PostgreSQL defines only `round(numeric, int)`.
The first was load-bearing — every date filter routed through it, so the entire
agent would have failed on its first Postgres query.

Fixed with `SUBSTR(col, 1, 10)` and `ROUND(CAST(x AS NUMERIC), n)`, and now
guarded twice: a unit test that greps for the patterns, and a CI job running the
whole suite against a real Postgres container.

**What it demonstrates:** an untested claim is a guess. This is the same lesson
as the trading system's walk-forward work, arrived at independently.

### CI found a design flaw on its first run

The import check reported `src.dashboard.app: no such table: workflow_runs`.
Streamlit scripts run top to bottom, so module-level code hit the database on
import. The dashboard worked fine when run normally, which is why it would have
gone unnoticed — but a module that cannot be imported without a populated
database breaks static analysis, IDE tooling and anything that walks the
package. Body moved into `main()`.

### The retrieval layer found a hole in the documentation

The first design-rationale question returned low-confidence hits across four
rephrased searches. The retriever was fine; the trading architecture document
named the trend filter but never said why it existed. A retrieval system
surfaced a documentation gap by failing to answer, and the gap was fixed in the
trading repo. "A retrieval system is only as good as what it has to retrieve
from" is the general lesson.

### A circular import that only bit in one direction

`src.tools` imported `src.rag.tool` to register it; `src.rag.tool` imported
`src.tools.base` for the envelope. Import `src.tools` first and it worked;
import `src.rag.tool` first and it failed. The tests happened to hit the working
order. Fixed with a deferred import, and locked in with a test that spawns a
fresh interpreter per import order — because import state persists within one
process and would hide the bug.

### Slack silently mangled position tags

`VGT_AAPL_NVDA_L` rendered as `VGTAAPLNVDAL`. The table formatter stripped
every underscore to prevent literal emphasis markers inside code blocks — where
mrkdwn is not interpreted anyway. Underscores in identifiers are data, not
formatting. Nothing raised; the message just came out wrong.

---

## 5. Known limits

**It is verbose on broad questions.** "What went wrong last week?" produces
about 400 words across six sections despite a prompt fix and an eval check.
Narrow questions are fine, so it is specific to enumerable multi-category
answers.

**It names the next step instead of taking it.** Ends answers with "worth
investigating X" where X is a tool it holds.

**It computes figures the tools did not return.** Observed stating a book-wide
average that was derived rather than looked up. Correct, but the mechanism is
wrong: "every number came from tested code, except the easy ones" is a much
weaker claim.

**It asserts causal links between real figures.** The $18-from-the-cap example.
The numbers were real; the relationship was invented. This is the limit
`numeric_fidelity` cannot reach by design.

**The data is synthetic**, and the alpha target and stop threshold in
`config.py` are invented rather than the real system's values.

**Lexical retrieval misses paraphrase.** The 24bps example. Embeddings fix it;
they are optional because of the model download.

**The router's saving is benchmark-specific.** Measured on my held-out cases,
which are a small set. It is evidence the mechanism works, not a general claim.

**The eval suite is small.** Twenty-two cases, seven held out. Enough to catch
regressions in the behaviours I have seen; not enough to claim coverage of the
ones I have not.

---

## 6. Vocabulary

**Alpha** — index-relative return, `W1·co1 − W2·co2 − β·index`. Market-neutral
by construction. Not P&L. Never call it profit.

**Tail (L/U)** — L is long Co1 / short Co2; U reverses. Legs have no fixed
role.

**CDF bucket** — decile of the 15-day standardised deviation of the log price
ratio. Sets leg weights and the position multiplier. The 40–70% buckets are
disabled at 0.0x.

**Arrival mid** — the mid price when the order was raised. The TCA benchmark.

**Slippage** — signed against arrival mid; positive means worse (paid up on a
buy, sold down on a sell).

**Provenance** — the filters, row count and window a tool result was computed
over.

**Tool use / function calling** — the model requests, your code executes.

**MCP** — Model Context Protocol; Anthropic's open standard for exposing tools
to any host. Note v2 renamed `FastMCP` to `MCPServer`.

**Socket Mode** — Slack's outbound WebSocket, so no public URL is needed.

**Eval** — a test of model *behaviour*, distinct from a unit test of code.

**Held-out** — eval cases never used to tune the prompt. The out-of-sample
measure.

**RAG** — retrieval-augmented generation: retrieve passages, put them in
context, generate. Here it is one tool among eight, not the architecture.

**Chunk** — one retrievable unit of documentation, split at headings, carrying
its heading path as citation.

**Routing** — choosing model and budget per query before it runs. Distinct from
cascading, which tries cheap first and escalates.

**kNN router** — predict a query's cost from its nearest past traces.

**Prompt caching** — the API caches the system prompt and tool definitions
across calls; the resent-every-turn cost drops by ~90% on the cached portion.

**Context compression** — reducing what is resent each turn. Three layers:
compress tool outputs, sliding window, summarise. Only the first is used.

---

