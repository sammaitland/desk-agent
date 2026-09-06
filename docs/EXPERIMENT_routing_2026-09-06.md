# Experiment record — three-arm routing comparison

**Date:** 2026-09-06
**Suite:** 22 eval cases (15 development, 7 held-out)
**Arms:** baseline · compression-only · routed
**Protocol:** warm-cache, steady-state; Sonnet 4.6 baseline; leave-one-question-out
for development cases; predictor corpus of 30 baseline traces (15 questions × 2)
**Commit:** 6f353de

---

## Result

| Arm | Pass | Processed tokens | Cost (USD) |
|---|---:|---:|---:|
| baseline | 12/22 | 368,469 | 0.698 |
| compression | 13/22 | 314,368 | 0.499 |
| routed | 14/22 | 334,985 | 0.584 |

Compression vs baseline: **−14.7% tokens, −28.6% cost.** Two casewise
regressions, three improvements.
Routed vs baseline: −9.1% tokens, −16.4% cost. Two regressions, four improvements.
Acceptance criterion (zero regressions on every arm): **FAIL**, on both arms.

---

## Finding 1 — the noise floor is larger than the effect

The same baseline configuration, run in the morning as a standalone eval and
again in the afternoon as the baseline arm, produced 15/22 and 12/22. Five
cases flipped pass→fail; two flipped the other way. Nothing changed but the
sampling.

**Run-to-run variance on this suite is roughly ±3 cases.** Every regression
in the comparison sits inside that band:

- `rejection_lookup` regressed under compression at 8,999 tokens vs 9,090
  baseline. It is a two-turn run; compression cannot engage before turn
  three. The flip is variance.
- `write_refusal` regressed under routed at 4,192 vs 4,205. Identical
  behaviour, different wording, a brevity check on the edge.
- `disabled_buckets` (held-out) passed at baseline after an 81,000-token
  wander and failed on both other arms at 18,000 and 55,000. The pass
  required an expensive investigation that happened to say "disabled" and
  "0.0"; the cheaper runs did not.

The acceptance criterion applied its rule correctly. The rule assumed
pass/fail was a stable measurement, and it is not. **A single-run eval suite
cannot detect a two-case effect when identical runs differ by five.**

Case-level token comparisons are also unreliable for the same reason: two
arms can take different investigation paths (`what_and_why` 16,829 vs 21,579;
`leading_question` 9,063 vs 19,922), so only aggregates are meaningful, and
those with caveats.

## Finding 2 — compression works, and the mechanism is visible

`position_explanation`, the most expensive case at 34,000 tokens, dropped to
25,000 under compression. That is the `explain_position` payload being
released after the model has read it, rather than resent on every subsequent
turn. The saving is where it was predicted to be.

Aggregate: 15% fewer processed tokens, 29% less price-weighted cost. The cost
saving exceeds the token saving because the released tokens were full-rate
input, not cached input.

## Finding 3 — the router did not route

Tier mix: 21 standard, 1 deep, 0 light. Of 15 development cases, 11 fell to
cold start under leave-one-question-out. With only 15 distinct questions in
the corpus, excluding a case's own runs usually leaves no neighbour above the
0.25 confidence floor.

The router is not wrong. **The corpus is too small for leave-one-question-out
to leave it neighbours.** The routed arm is therefore approximately
compression-on with a different random seed, which is why its figures sit
between the other two.

One escalation was mis-specified: `bad_fill_investigation` routed to deep on a
0.307-confidence match to `what_and_why`, whose four-tool sequence tripped
`DEEP_MIN_TOOLS`. The deep rule checks tool count but not confidence. A loose
match to a complex question can escalate a simple one. Recorded, not changed.

## Finding 4 — the cold-cache asymmetry was real

Before the warm-up protocol existed, the first baseline trace paid 3,794
tokens of cache creation; the other 29 paid none. The asymmetry review point 3
described is visible in the corpus. The comparison warmed both models before
measurement; warm-up cost is excluded.

---

## What is defensible today

> Tool-result compression reduced processed tokens by 15% and price-weighted
> cost by 29% across a 22-case evaluation suite, with no change in pass rate
> distinguishable from run-to-run variance.

That claim survives questioning because the variance can be shown.

## What is not defensible yet

Any claim about routing. The router has not been evaluated under conditions
where it could act.

---

## What it would take

**To evaluate routing:** a baseline corpus with enough distinct questions that
leave-one-question-out leaves neighbours — fifty or more, not fifteen. That is
a question-writing task, not a threshold change.

**To make pass/fail a measurement:** three or more runs per arm, reporting
pass rate with a range. Roughly 200 calls, a few pounds. It turns "no worse"
from a coin flip into a statement with error bars — the same discipline as a
backtest reported with confidence intervals rather than a point Sharpe.

**To fix the deep escalation:** require minimum confidence on the deep rule
as well as the light one. Deferred until there is a corpus to test it against.

---

## Decisions

- No prompt changes. No threshold changes. No routing changes.
- Compression is the reported result. Routing is reported as untested.
- The baseline corpus (30 traces, manifest SHA
  `8c133a92a306a6da018bd357c3e7d0282e0d9200ff28770b99b0c9afc61e3fbe`) is
  retained as-is.
- Phase 9 is closed pending real data and a larger question set.

The experiment was built to answer "does routing help?" It answered a prior
question instead: "can this suite detect whether anything helps?" That is the
more useful result, and it only surfaced because the measurement was
constructed carefully enough to fail honestly.
