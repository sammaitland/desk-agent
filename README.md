# Desk Agent

[![tests](https://github.com/sammaitland/desk-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/sammaitland/desk-agent/actions/workflows/tests.yml)

A read-only analytics agent over a systematic pairs-trading blotter. Ask a
question in natural language — *why wasn't this pair traded?*, *where is alpha
coming from?*, *what went wrong last week?* — and it investigates by calling a
small set of deterministic, tested tools, then explains what it found with
every figure traced to its source.

**Status: a working prototype, not a product.** It runs, it is tested, and it
answers questions about real output from a live trading system. It is also
unfinished in ways the [open items](#open-items) below name explicitly.
Nothing here is a claim about the trading strategy's performance.

```bash
cp .env.example .env                      # add ANTHROPIC_API_KEY
pip install -e ".[dev]"
python -m src.generate_blotter --days 120 --seed 42
python cli.py "why did the C order on the 24th fill badly?" --trace
```

## The architectural rule

**The model orchestrates; the code computes.** Every number in every answer
comes from a Python function the model called, never from the model itself. It
cannot state a computed figure it invented, because it never computes one — and
it cannot write to the blotter, because the code to do so does not exist in the
tool package.

The rest of the design follows from that:

| Layer | Does |
|---|---|
| `src/tools/` | Eight typed functions over the blotter. Uniform envelope: `data`, `provenance`, `summary`. Read-only, enforced by a test that greps for write statements. |
| `src/agent/` | The loop, hand-rolled against the Anthropic Messages API. Returns a `Trace` — tool calls, arguments, tokens, latency — not a bare string. |
| `src/rag/` | Retrieval over the trading system's design documents: *why the system works this way*, as against *what happened*. |
| `src/evals/` | Behavioural evals with a held-out set, because a prompt tuned against an eval makes that eval in-sample. |
| `src/critic/` | An adversarial second pass that attacks an answer's claims one at a time. |
| `src/routing/` | Predicts a query's cost before it runs and picks a model and turn budget. |
| `src/adapter/` | Loads the live trading system's Excel and log output into the blotter. |

The same tool layer serves a CLI, a Slack bot, an MCP server and a Streamlit
dashboard. The transport is not the architecture.

## Real data, and what it broke

The blotter starts synthetic — a seeded simulator reproducing the trading
system's mechanics, with operational failures planted deterministically so the
agent has real things to investigate. `src/adapter/` then loads the live
system's actual output over the top:

```bash
python run_adapter.py ~/Desktop/V9/archive --init
```

The first real load reproduced the trading system's own run log exactly: 4,500
pairs evaluated, 1,072 active, 110 shortlisted, 480 rejected on score, plus
185 closed positions going back months.

The first three questions asked against it found four faults that months of
synthetic testing had not, because each depended on a shape the generator
never produced:

- **The date window came from runs alone.** The generator wrote a run for every
  day it opened a position, so runs and positions always spanned the same
  period. Real data loads one run per archived day and months of position
  history — so *"how have closed positions performed?"* returned nothing
  against 184 real closed trades, and the agent accurately reported what it was
  shown.
- **Rejection counts covered the page, not the window.** Reasons were tallied
  across the rows returned, so a day with 3,839 rejections reported the
  breakdown of the first fifty — which, with no `ORDER BY`, was one
  alphabetical block of one sector.
- **No tool exposed the pipeline stage.** *"Which pairs reached the shortlist
  but weren't traded?"* was unanswerable, so the agent answered a different
  question confidently.
- **PostgreSQL returns `Decimal` where SQLite returns `float`.** The
  numeric-fidelity check skipped Decimals, so on Postgres every sourced figure
  would have looked unsupported.

The adapter found more in the files themselves: `Tag` is a row index rather
than a position tag and repeats across files; rejection and exit reasons are
free text carrying tickers and magnitudes; the rejected-trades archive is
cumulative across ten months; ten `NOT NULL` columns had nulls; and five
positions were recorded twice with contradictory exits, which the loader keeps
rather than silently merging.

The pattern in all of it: the environment the system was built against was
internally consistent in ways the real one is not, and every component quietly
relying on that consistency failed on first contact.

## What is measured, and what is not

Three things this project takes seriously, because each is a way for a
confident answer to be wrong:

**Numeric fidelity.** Every figure in an answer is traced back to something a
tool returned. Rounding is allowed; drift is not. This is the check the
architecture rests on — the tool layer guarantees the numbers are correct, and
nothing else guarantees the model reports them correctly.

**Held-out evals.** Once a prompt is edited in response to an eval failure,
that eval is in-sample in exactly the sense a backtest is. Seven of the
twenty-two cases are never used for tuning.

**A claim critic.** Numeric fidelity cannot verify the *sentence* built around
the figures. *"Rejected because its notional came within $18 of the cap"* had
two real numbers, an invented causal link, and passed every check. The critic
attacks each claim separately, and may only contradict one with a tool result
behind it.

What is *not* measured: whether the routing saves anything. The first three-arm
comparison found the eval suite's run-to-run noise floor — roughly ±3 cases in
22 — larger than the effect being measured, so compression's 29% cost saving is
reported and routing's is not. Written up in
[`docs/EXPERIMENT_routing_2026-09-06.md`](docs/EXPERIMENT_routing_2026-09-06.md).

## Open items

Named rather than hidden, because a prototype that presents as finished is
worse than one that says where it stops:

- **Orders, fills, workflow stages and events are empty.** They come from the
  trading system's log and the parser is unwritten — `LogParser` is the
  interface. It needs a day on which trades actually execute.
- **The routing claim is untested.** See above.
- **Position tags are reconstructed**, not read, because the live files do not
  carry them. Reconstruction is deterministic but is not the system's own
  identifier.
- **The critic's benchmark is small** — seven cases, sixteen targets. Enough to
  catch the failure types observed; not enough to claim coverage.
- **One archived day loaded so far.** Multi-day behaviour, particularly
  position updates diffed across consecutive snapshots, is implemented but
  barely exercised.
- **The synthetic generator and the real schema have diverged** in places the
  adapter had to widen. Regenerating synthetic data still works, but it no
  longer exercises every column the real files fill.

## Documentation

| Document | Covers |
|---|---|
| [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) | Every phase in the order it was built, and the reasoning behind each decision |
| [`docs/desk_agent_design_notes.md`](docs/desk_agent_design_notes.md) | The design reasoning, condensed |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | What exists, test coverage, outstanding items |
| [`docs/EXPERIMENT_routing_2026-09-06.md`](docs/EXPERIMENT_routing_2026-09-06.md) | The routing experiment and why its result was inconclusive |

## Running it

```bash
make test                                # the suite
make blotter                             # regenerate synthetic data
make ask Q="what went wrong last week?"
make dashboard                           # Streamlit
python -m src.slack.bot                  # Slack, Socket Mode
python -m src.mcp_server.server          # MCP, stdio
python run_evals.py --include-held-out   # live API, costs calls
python run_critic_benchmark.py           # live API, costs calls
```

SQLite by default; PostgreSQL is a `DB_URL` change, tested in CI on every push
— because the claim that it was portable turned out to be false the first time
anyone checked.

## Licence

MIT.
