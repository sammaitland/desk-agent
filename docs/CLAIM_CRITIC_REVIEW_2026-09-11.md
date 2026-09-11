# Claim Critic: implementation for the next review

This change addresses the failures in the 7 September atomic run. It is built
on repository commit `1396e35d3cfbc8c048815dd84e7e8901975c15d1`. At that
commit, `critique.py`, `benchmark.py`, `prompts.py`, the benchmark runner,
`test_critic.py` and `generate_blotter.py` matched the latest uploaded versions
byte for byte. Supporting tools come from that repository revision, which
also includes the subsequent adapter work.

## What changed

- **Direct evidence access.** `query_records` retrieves passing and failed
  risk checks by `checked_at`, and stop records by `triggered_at`. It supports
  exact record IDs. Counts cover the complete filtered population; the record
  page remains bounded. Unsupported filters produce errors. The lookup is
  registered in both the agent and MCP interfaces.
- **Explicit date scope.** Position lookup and alpha attribution retain their
  entry-date defaults and offer `date_basis="termination_date"` for exits.
  Results and summaries identify the date basis. This prevents the interface
  from describing an entry cohort as if it selected exits.
- **Event semantics.** Unknown event labels return an error with alternatives.
  Two explicit aliases are accepted: `partial fill` and `order timeout`.
  Event tallies use the full filtered population, even when the returned page
  is truncated. Mixed anomaly results distinguish event, risk-check and run
  scopes. A stop count comes from stop records, not orphan-event counts.
- **Specific citations.** The verifier returns a JSON assessment. A settled
  verdict requires references containing a call number, JSON pointer, copied
  scalar value and returned scope. The loop exposes the call number in tool
  responses. The guard rejects sign changes, implicit percentage conversions,
  numbers extracted from dates, values borrowed from other calls, argument
  echoes and chart outputs. It accepts authoritative zero counts. There is
  no automatic numeric rounding: the model cites the exact returned value,
  including values already rounded by an analytical tool.
- **Uncertainty and failures.** Invalid JSON, invalid settled-verdict citations,
  transport failures and exhausted turn budgets are assessment errors. They
  cannot earn restraint credit. Proposed verdicts, final verdicts, downgrades,
  references, model, usage and trace paths are retained. Extractor transport
  failures also produce an incomplete assessment instead of aborting silently.
- **Causal restraint.** The user and system prompts now ask for neutral
  assessment of the exact claim. A declared `market_cause` cannot be settled
  by the current tools. A small, explicitly heuristic check also catches some
  market-cause phrases when the model labels them as record claims.
- **Benchmark accounting.** Atomic results retain every target's verdict.
  Precision includes contradictions issued against undetermined targets;
  the regression with one correct contradiction and one overreach reports
  50%. Atomic extraction coverage is null/N/A. Restraint includes its sample
  count. Missing targets, ambiguous alignment and assessment errors fail the
  run gate. The existing 60% recall threshold remains.
- **Reproducibility.** Provenance includes hashes of source files, actual
  imported paths, prompts and tool schemas. The database fingerprint hashes
  logical row contents and column names, not just counts and date bounds.
  `--preflight` checks setup and prints provenance without model calls;
  `--expect-db` checks a content fingerprint before any calls. The runner
  checks the database digest again after the run. The generator now accepts
  `--as-of`; IDs remain UUIDs, so an existing frozen database is still the
  appropriate basis for comparison.

## Validation

Local Python 3.12 / SQLite suite: **372 passed, 1 skipped**. The skipped test
requires optional `sentence_transformers`. The local proxy environment was
cleared for the suite because the optional Langfuse construction test
otherwise requires an additional SOCKS package; application code was not
changed for that environment issue. No live model calls were made.

The new regressions include the four stops on 24 August, a position entered
on 24 August and exited later, a passing check sharing a position's notional,
an invalid stop-event label, multiple events for one order, truncated count
pages, valid zero counts, malformed dates, field/sign/unit/scope mismatches,
causal overreach, assessment failures and atomic verdict serialization.

A separate synthetic validation database was generated with seed 42 and
`--as-of 2026-09-07`. Its 13 table counts match the reported atomic run.
The preflight completes against it. This is a newly generated fixture with
new UUIDs; it is not the original evaluation database. The user's database
was not supplied or modified. PostgreSQL and a live model rerun remain to be
validated; local tests establish the code contract, not improved model recall
or precision.

## What Claude should challenge

1. **Citation fidelity is not entailment.** The guard checks that references
   faithfully reproduce the fields and declared scopes they cite. It does not
   prove that those fields justify the prose verdict. A model can faithfully
   cite a wrong population or an irrelevant fact. Scope references make that
   reviewable and reject relabelling; they do not implement natural-language
   scope matching. Review successful traces as well as failed ones.
2. **The market-cause backstop is a heuristic.** It is deliberately visible,
   not presented as a general causal classifier. Paraphrases can evade it and
   some phrasing may be overly restricted. Recorded-reason claims remain
   distinct from empirical market attribution. Judge this on new phrasing,
   not only the existing cluster sentence.
3. **Strict citations may reduce recall.** JSON pointers and complete scope
   copies add model work. We have not measured the compliance rate. Invalid
   settled citations now become visible assessment failures instead of
   apparent restraint. Check whether that cost is justified before widening
   the format or adding another model stage.
4. **Record identity remains the model's responsibility.** Check-ID access
   removes a retrieval gap, but a pair/day can still contain multiple checks.
   The verifier must select the relevant record and cite its identity. This
   patch does not pretend that matching a scalar alone resolves ambiguity.
5. **End-to-end alignment remains provisional.** Span overlap is still the
   candidate matching rule. Ambiguous claims fail the gate, but unique overlap
   does not prove semantic identity. No benchmark labels or atomic target
   propositions were changed in this patch.

## Next run

Use the review revision and the same existing evaluation database. Do not
regenerate that database to test this patch. Confirm the imported files first:

```bash
python run_critic_benchmark.py --preflight > critic_preflight.json
python run_critic_benchmark.py --atomic --save
python run_critic_benchmark.py --save
```

If the evaluation database is outside the checkout, add its explicit
`--db-url` to each command. To enforce a frozen input, copy the full
`database.fingerprint` from preflight into `--expect-db` on both live runs.
Source and prompt hashes are now full SHA-256 strings; the database hash uses
a new content-based definition and cannot be compared to the old short hash.

Return the result JSONs, preflight JSON and referenced `traces/critic/` files.
Compare final outcomes, proposed outcomes, citation failures and tool use
separately. In particular, verify that the near-cap investigations retrieve
the actual check, stop counts use `triggered_at`, partial-fill claims count
events, and market-cause claims remain undetermined for the right reason.
