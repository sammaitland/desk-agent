# Desk Agent

[![tests](https://github.com/sammaitland/desk-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/sammaitland/desk-agent/actions/workflows/tests.yml)

Read-only analytics agent over a systematic pairs-trading blotter, queried in
natural language. This phase builds the data layer.

## What this is

The original paper-trading logs were lost. This generates a synthetic blotter
that conforms to the **V9.4C trade event schema** — the same field names, enums
and thresholds the live system emits. The data is fabricated; the *shape* is the
contract. When real paper-account output arrives it populates these tables
unchanged, and nothing downstream needs to change.

## Credentials

Copy `.env.example` to `.env` and fill it in. Every entry point loads it
automatically — no `export`, no sourcing, nothing to paste each session:

```bash
cp .env.example .env
```

`.env` is gitignored. Real environment variables override it, so CI and
containers inject secrets without one present.

This is load-bearing for the MCP server rather than merely convenient: Claude
Desktop launches it as a subprocess with a minimal environment that does not
inherit your shell's exports, so a server depending on `export ANTHROPIC_API_KEY`
would fail to authenticate with no obvious cause.

## Quick start

```bash
pip install sqlalchemy pytest
python -m src.generate_blotter --days 120 --seed 42
pytest tests/ -q
```

Defaults to SQLite. Postgres is a config change, not a rewrite:

```bash
export DB_URL=postgresql+psycopg2://user:pw@localhost/blotter
```

## Domain model

Not a generic blotter. It reproduces the actual mechanics:

- **Co1/Co2 + Tail (L/U)** — legs have no fixed long/short role
- **Bucket-driven sizing** — CDF decile sets leg weights (W1/W2) and position
  multiplier (0.7x–1.4x); the 40–70% buckets are disabled at 0.0x
- **Alpha accounting** — `W1·co1_ret − W2·co2_ret − β·index_ret`, index-relative
- **Ticker-level order aggregation** — orders group by ticker+direction across
  pairs to reduce commissions, then fills allocate back to tags pro rata
- **LMT with 45s timeout** falling back to MKT; spreads above 24bps route
  straight to market
- **Stop losses** on the short leg (`SQLOSS_{tag}`, 0.40 alpha threshold), with
  orphan detection when one leg stops out

## Tables

| Table | Holds |
|---|---|
| `instruments` | Reference data and delisting state |
| `workflow_runs` / `workflow_stages` | The 11-stage daily run, stage by stage |
| `portfolio_snapshots` | Account value, gross exposure, leverage, beta |
| `pair_evaluations` | The screening funnel — every pair, every run, and why it failed |
| `positions` | Portfolio rows with exit fields folded in |
| `position_updates` | Daily marks and live alpha |
| `orders` / `order_allocations` / `fills` | Aggregated execution and its distribution |
| `stop_orders` | Stop state and triggers |
| `risk_checks` | Leverage, beta, concentration, factor exposure gates |
| `system_events` | Errors and reconciliation — what actually went wrong |

`pair_evaluations` and `system_events` are the two that make the agent useful:
one answers "why wasn't this traded", the other "what went wrong".

## Planted scenarios

The agent needs real failures to investigate, so these are generated
deterministically rather than left to chance:

- A wide-spread order routed to market, with heavy slippage against arrival mid
- A limit order timing out and falling back
- A stop trigger leaving an orphaned leg, then its closure
- A delisting force-closing a live pair
- A reconciliation mismatch halting the run mid-stage

## Tests

24 invariant tests assert *domain* rules, not that code ran: disabled buckets
never trade, alpha stays in a market-neutral range, aggregated fills reconcile
to their allocations, limit orders respect the spread cap, and every planted
scenario exists. `test_alpha_is_market_neutral_scale` exists because an early
version drew leg prices independently — pairs didn't co-move, and alpha came out
an order of magnitude too large. That failure is invisible without the check.

## Open questions

1. **Longlist/Shortlist ordering** — the event schema has primary filters →
   Shortlist and secondary signals → Longlist. `ARCHITECTURE.md` in the trading
   repo states the reverse. One is wrong.
2. **Index count** — the event schema enum includes VOX (six); the trading repo
   README names five ETFs.

## Phase 1 — Tool layer

Seven deterministic functions in `src/tools/`. The agent will call these; it
computes nothing itself. Every number it ever states comes from tested code.

| Tool | Answers |
|---|---|
| `query_blotter` | "What exists?" — positions, orders, runs by simple filters |
| `explain_position` | "Why did we take this trade, and what happened to it?" |
| `explain_rejection` | "Why *wasn't* this traded?" — the gate that stopped it |
| `execution_quality` | "Why did this fill badly?" — TCA against arrival mid |
| `alpha_attribution` | "Where is alpha coming from?" — by index, bucket, tail, exit |
| `detect_anomalies` | "What went wrong?" — events and failed risk checks |
| `make_chart` | Renders values it is given. No analysis. |

### Design rules

**Uniform envelope.** Every tool returns `data`, `provenance` and a one-line
`summary`. Provenance carries the filters applied, rows examined and window
covered — it is what lets the agent say "across 47 orders in that window"
instead of implying a scope it never checked.

**Descriptions are the interface.** `TOOL_SCHEMAS` in `src/tools/__init__.py`
holds the JSON schemas passed to the API. The description text is the only
thing the model sees when choosing a tool, so each is written to disambiguate
against the others — `explain_rejection` explicitly names itself the
counterpart to `explain_position`. Most agent failures are description
failures, not model failures.

**Errors return, they don't raise.** Unknown tools and bad arguments come back
as a normal result so the loop can hand the problem to the model to correct,
rather than surfacing a stack trace it may narrate as a system fault.

**Read-only, structurally.** A test greps the tool package for write
statements. The agent cannot mutate the blotter because the code to do so does
not exist there.

### Tests

37 tests over the tool layer, covering the envelope contract, the arithmetic
the model will state as fact (slippage sign convention, group exhaustiveness,
win-rate bounds), empty and malformed input, and schema/function consistency —
a required argument the function doesn't accept would otherwise fail only at
runtime.

## Phase 2 — Agent loop

Hand-rolled against the Anthropic Messages API. No orchestration framework: the
abstractions are thin, and knowing exactly what is on the wire is worth more
than the lines saved — particularly when debugging why a model chose the wrong
tool.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python cli.py "why did the C order fill badly on the 24th?" --trace
```

### The cycle

    send messages + tools  ->  response
    stop_reason == 'tool_use'  ->  run tools, append results, repeat
    stop_reason == 'end_turn'  ->  return the text

Three protocol details that bite, each covered by a test: `tool_result` blocks
must go in a **user** message; every `tool_use` id needs a matching result in
the **same** message; and result content must be a JSON string, not an object.

### Statelessness is the design constraint

The model holds no state between calls, so every turn resends the entire
conversation. A three-turn investigation sends 1, then 3, then 5 messages — and
each tool result is resent on every subsequent turn. That is why the tool layer
caps rows at 500: an uncapped result is not a one-off cost, it is a cost paid
again on every turn that follows.

### Components

| File | Does |
|---|---|
| `src/agent/loop.py` | The cycle. Returns a `Trace`, not a string — the answer alone hides how it was reached. |
| `src/agent/prompt.py` | System prompt: domain grounding, sourcing discipline, scope honesty. |
| `src/agent/trace.py` | Full run capture — tool calls, arguments, tokens, latency. |
| `src/agent/scripted.py` | A scripted client for tests and evals. |
| `cli.py` | Entry point. |

### The client is injected

`run_agent` takes a `client` rather than constructing one. This separates loop
bugs from model behaviour — a failing scripted test is a fault in the loop, not
in the model — and it is what lets the Phase 3 eval harness run on every change
without API calls.

### The prompt earns its length

Each domain fact in it corrects a specific, confident, wrong answer. Without
"alpha is not profit" the model calls it P&L. Without the Tail explanation it
assumes Co1 is the long leg. Without the thresholds it invents them. The
sourcing rules — never state a number that did not come from a tool, report
scope from provenance — are what stop a correct tool layer being narrated
loosely.

### Tests

17 tests over the loop, all scripted: protocol correctness (message roles, id
matching, JSON encoding, verbatim echo), chaining, error recovery, the turn cap,
transport failure, and trace integrity. Model judgement is not tested here —
that is Phase 3.

## Phase 3 — Observability and evals

```bash
python run_evals.py                        # all cases
python run_evals.py --tags chaining        # a subset
python run_evals.py --case false_premise --show-answers
```

### Two tiers, deliberately separated

`pytest` tests the **loop** with a scripted client: free, deterministic, runs on
every commit. `run_evals.py` tests the **model** against the live API: costs
calls, runs on demand. Merging them would give either expensive unit tests or
evals that never see a model.

### Cases come from observed behaviour

Every case was written after watching real runs. Three protect behaviour worth
keeping; four encode failures actually seen in the first live traces:

| Case | Encodes |
|---|---|
| `bad_fill_investigation` | must chain: locate the order, then pull its detail |
| `false_premise` | asked why a *clean* fill was bad, it pushed back rather than confabulating — worth protecting |
| `alpha_attribution` | uses the alpha tool; never calls alpha "profit" |
| `rejection_lookup` | observed reading one rejection row as "evaluated only once" |
| `weekly_incident_review` | observed answering from one tool in 800 words across six tables |
| `position_explanation` | a second chaining path, via `query_blotter` |
| `empty_window` / `write_refusal` | says so plainly rather than reaching for the nearest data |

Evals written before observing a system test what you assumed it would do
wrong. These test what it did.

### Numeric fidelity

The check the architecture rests on. The tool layer guarantees the numbers are
correct; nothing else guarantees the model *reports* them correctly. It extracts
every figure from the answer and traces each to something a tool returned,
allowing rounding (4.81 quoted as 4.8) but not drift (23.31 quoted as 20), and
ignoring digits that assert nothing — position tags, order ids, dates, times.
Thresholds stated in the system prompt count as sourced.

This is the failure mode that survives every other test: the answer stays fluent
and plausible while quietly ceasing to be true.

### Prompt fixes this phase

Each from an observed failure:

- **Take the next step.** It ended answers with "I'd recommend pulling execution
  quality" — a tool it holds. `no_deferred_investigation` now catches that.
- **Absence is not evidence.** `explain_rejection` returns rejections only, so a
  missing pair may have been approved. Both the prompt and the tool description
  now say so.
- **Do not explain mechanisms the data lacks.** It described what a factor-shock
  gate is *designed to protect against* — plausible, unsourced, stated as fact.
- **Brevity**, moved up and made firmer.

### Testing the instrument

`tests/test_evals.py` calibrates the checks themselves against traces built to
be unambiguously right or wrong. A check that passes a bad answer is worse than
no check — it gives confident wrong readings about whether the system regressed.

## Phase 4 — Slack

```bash
pip install -e ".[slack]"
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
python -m src.slack.bot
```

Then in any channel the bot is in:

    @deskbot why did the C order fill badly on the 24th?

It replies in-thread with the answer, any charts attached, and a context line
showing which tools ran.

### Socket Mode

The bot opens an outbound WebSocket rather than receiving webhooks, so it needs
no public URL, no tunnel and no deployment. It runs from a laptop against a
local blotter, which is what a desk tool of this kind actually needs.

### Slack app setup

At `api.slack.com/apps` → **Create New App** → *From scratch*:

1. **Socket Mode** → enable. Generate an app-level token with `connections:write`
   → this is `SLACK_APP_TOKEN` (`xapp-...`).
2. **OAuth & Permissions** → bot token scopes: `app_mentions:read`,
   `chat:write`, `files:write`.
3. **Event Subscriptions** → enable → subscribe to bot event `app_mention`.
4. **Install to Workspace** → copy the bot token → `SLACK_BOT_TOKEN` (`xoxb-...`).
5. Invite the bot to a channel: `/invite @deskbot`.

### Slack is not Markdown

The substantive engineering here is formatting. Slack uses `mrkdwn`: bold is a
*single* asterisk, there are no headers, and there are **no tables** — a
Markdown table pasted into Slack is unreadable pipe soup. Since the agent
produces tables regularly, `src/slack/format.py` converts them to fixed-width
text inside a code block, which is the only way Slack aligns columns.

These failures are silent. Nothing raises; the message simply renders wrongly,
which is why the formatter carries most of the tests in this phase.

### Provenance stays visible

Every reply carries a context line: `detect_anomalies, execution_quality · 3
turns · 15.4s · 11,598 in / 505 out · run a1e2d69f`. On a surface where the
reasoning is otherwise invisible, showing which tools ran is what keeps the
answer auditable rather than oracular — and the run id ties it back to the
saved trace.

## Phase 5 — MCP server

The same tools, published over the Model Context Protocol so any MCP host —
Claude Desktop, an IDE, another agent — can query the blotter directly.

```bash
pip install -e ".[mcp]"
python -m src.mcp_server.server        # stdio transport
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "desk-agent": {
      "command": "/absolute/path/to/desk_agent/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/absolute/path/to/desk_agent"
    }
  }
}
```

### The wrapper is thin, and that is the result

`src/mcp_server/server.py` adds no logic. The tool layer already had typed
signatures, explicit schemas and docstrings written for a model to read, so
publishing it over a second transport was a wrapper rather than a rewrite —
the SDK derives every JSON schema from the type hints. That is what separating
orchestration from computation buys: **the transport is not the architecture.**

Six of the seven tools are exposed. `make_chart` is not: it writes PNGs to local
disk, which is meaningless to a remote host.

### The envelope travels

Each response carries the same `summary` / `provenance` / `data` structure the
in-process agent receives, so an MCP host can report scope — "across 47 orders
in that window" — as honestly as the built-in agent does. A `blotter://coverage`
resource exposes the date range and size, so a host knows the data boundaries
without probing for them.

### mcp 2.x

Built against the v2 SDK, where `FastMCP` was renamed `MCPServer` and fields
moved to snake_case (`input_schema`, not `inputSchema`). Most tutorials still
show v1; v1 code will not import against v2, hence the `mcp>=2.0` floor.

### Read-only, on every transport

The same grep test that guards the tool layer guards this one. A read-only
guarantee that holds for the in-house agent but not for an external host is not
a guarantee.

## Phase 6 — CI and containers

```bash
make test          # the suite
make blotter       # regenerate
make ask Q="what went wrong last week?"
make docker-test   # the suite against Postgres, in Docker
```

### CI tests the claim, not just the code

Three jobs on every push:

| Job | Guards against |
|---|---|
| `sqlite` (3.11, 3.12) | ordinary regressions, on both interpreters |
| `postgres` | SQL that only works on SQLite |
| `lint` | modules that import from the project root but nowhere else |

The Postgres job exists because of a specific failure. Since Phase 0 this
README claimed Postgres was "a config change, not a rewrite" — and that was
never tested. It was false. Two constructs worked silently on SQLite and would
have failed immediately on Postgres:

- `DATE(text_column)` — SQLite coerces; **PostgreSQL has no `date(text)`**
- `ROUND(double, 2)` — **PostgreSQL defines only `round(numeric, int)`**

Both were load-bearing: every date filter in the tool layer went through the
first. The fix was `SUBSTR(col, 1, 10)` for dates (ISO strings compare
lexically, so it is chronological) and `ROUND(CAST(x AS NUMERIC), n)` for
rounding — both portable, both identical in behaviour.

The lesson is the one this project keeps relearning: an untested claim is a
guess. A `test_no_sqlite_only_sql_remains` unit test catches it in a second,
and the Postgres CI job catches anything the pattern-match misses.

### Docker

The container exists for the reason calibration/live parity exists in the
trading system: results only mean something if the environment that produced
them is the environment that runs them. Dependencies are pinned in their own
layer, the process runs as a non-root user (it holds API credentials and
reaches the network), and `.dockerignore` keeps `.env`, databases and traces
out of the image.

`docker compose` brings up Postgres alongside the agent — the local equivalent
of the Postgres CI job, and the thing that makes `DB_URL` more than a promise.

## Phase 7 — Dashboard

```bash
pip install -e ".[dashboard]"
make dashboard          # streamlit run src/dashboard/app.py
```

Five tabs over the same blotter: Overview (cumulative alpha, leverage, recent
events), Performance (alpha attribution by index, bucket, tail, exit reason or
month), Execution (TCA and worst fills), Screening (where candidates are
rejected, plus a per-pair lookup), and Ask — the agent itself, embedded.

### Same tools, different mode of access

The dashboard computes nothing of its own. Every metric comes from the tool
layer the agent calls, so a figure here and a figure in an agent answer are the
same tested function — they cannot disagree. A dashboard with its own version
of alpha would eventually contradict the agent with no way to tell which was
right, and `test_headline_alpha_matches_the_tool` enforces it.

The division is what each mode is for: **a dashboard answers questions you knew
to ask in advance; the agent answers the ones you did not.**

### Streamlit's execution model

The entire script re-runs top to bottom on every interaction — click a filter
and the file executes again. That is why the code reads as a linear script
rather than event handlers, and why everything touching the database sits
behind `@st.cache_data` in `src/dashboard/data.py`. Splitting data access out
of the UI also makes the numbers testable while the layout is not.

## Next

Real paper-account data, and a production version at operational-data.
