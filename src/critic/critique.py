"""The Claim Critic.

Takes an agent's answer and attacks it: extract every discrete claim, verify
each against the blotter with a narrow adversarial agent, and report which
held, which broke, which could not be settled, and which were never assessed.

## Why this exists

`numeric_fidelity` verifies that every figure in an answer came from a tool.
It cannot verify the sentence built around the figures. "SBUX_BKNG was
rejected because its notional came within $18 of the $5,000 cap" passed every
check and is false — both numbers were real, the causal link was invented,
and being under a cap is not a breach. That class of error is what this
attacks.

## Four outcomes, not two

An earlier version had "clean" mean "no contradictions", which let a critique
with no verified claims at all — empty extraction, every claim skipped, every
verdict unverifiable — read as a pass. Silence is not approval. The status is
now one of:

    flagged      at least one contradiction with tool evidence
    unresolved   no contradictions, but factual claims the tools could not settle
    incomplete   extraction failed, returned nothing for a substantive answer,
                 or skipped checkable claims
    clean        every claim assessed, every factual claim verified, no flags

and the report carries coverage: how many claims, how many assessed, how many
resolved.

## Contradicted is not the same as unsupported

A causal claim the tools show to be false is *contradicted*. A causal claim
the tools cannot bear on either way, stated as fact, is *unsupported*. The
first is a finding; the second is a caution. Both are reported; only the first
is a flag with evidence behind it. Claim type is descriptive metadata and does
not gate verification: a causal claim can be unsupported inference, and an
"inferential" claim about recorded events can be checked.

## The evidence rule, enforced structurally

A contradiction or verification stands only if the verifier (a) made at least
one tool call that did not error, (b) gave a non-empty EVIDENCE line, and
(c) the evidence names a record — tag, order id, pair, check id — that appears
in one of its tool results. A verdict failing any of these is downgraded to
unverifiable and marked as such. Reference validation cannot guarantee the
model's judgement was right; it guarantees the model was looking at something
real when it made it.

Verifier traces are saved, so every verdict in a benchmark result links to
the full tool output that produced it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Connection

from src.agent.loop import MODEL, _blotter_context, run_agent
from src.agent.trace import TRACE_DIR, Trace
from src.critic.prompts import EXTRACTOR_PROMPT, verifier_question, verifier_system

CLAIM_TYPES = ("figure", "causal", "comparative", "inferential")
VERDICTS = ("verified", "contradicted", "undetermined", "skipped")
STATUSES = ("flagged", "unresolved", "incomplete", "clean")

VERIFIER_MAX_TURNS = 4
CRITIC_TRACE_DIR = TRACE_DIR / "critic"


@dataclass
class Claim:
    source: str                # verbatim span of the answer
    text: str                  # standalone restatement (may equal source)
    type: str
    checkable: bool
    stated_as_fact: bool

    @classmethod
    def from_dict(cls, d: Any, answer: str) -> "Claim | None":
        """Strict: every field present and well-typed, source a real substring
        of the answer. Anything else is rejected and counted, not silently
        dropped."""
        if not isinstance(d, dict):
            return None
        source = d.get("source")
        text = d.get("claim") or source
        ctype = d.get("type")
        checkable = d.get("checkable")
        fact = d.get("stated_as_fact")
        if not (isinstance(source, str) and source.strip()):
            return None
        if not (isinstance(text, str) and text.strip()):
            return None
        if ctype not in CLAIM_TYPES:
            return None
        if not isinstance(checkable, bool) or not isinstance(fact, bool):
            return None
        if source.strip() not in answer:
            return None
        return cls(source=source.strip(), text=text.strip(), type=ctype,
                   checkable=checkable, stated_as_fact=fact)


@dataclass
class Verdict:
    claim: Claim
    verdict: str                       # final, after grounding
    evidence: str = ""
    proposed: str | None = None        # what the verifier said before grounding
    tool_sequence: list[str] = field(default_factory=list)
    downgraded: str | None = None      # why a verdict was reduced to unverifiable
    trace_run_id: str | None = None
    trace_path: str | None = None
    tokens: int = 0

    @property
    def flag(self) -> str | None:
        if self.verdict == "contradicted":
            return "CONTRADICTED"
        if self.verdict == "undetermined" and self.claim.stated_as_fact:
            return "UNSUPPORTED"
        return None


@dataclass
class Critique:
    question: str
    answer: str
    claims: list[Claim]
    verdicts: list[Verdict]
    extraction_ok: bool = True
    extraction_error: str | None = None
    rejected_items: int = 0             # extractor entries that failed validation
    tokens: int = 0

    # -- outcomes ---------------------------------------------------------

    @property
    def contradicted(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.verdict == "contradicted"]

    @property
    def unsupported(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.flag == "UNSUPPORTED"]

    @property
    def skipped_factual(self) -> list[Verdict]:
        """Factual assertions the extractor marked non-checkable and so were
        never assessed. An earlier check looked for skipped+checkable, a
        state the skip branch never produces."""
        return [v for v in self.verdicts if v.verdict == "skipped" and v.claim.stated_as_fact]

    @property
    def coverage(self) -> dict[str, int]:
        return {
            "claims": len(self.claims),
            "assessed": sum(1 for v in self.verdicts if v.verdict != "skipped"),
            "resolved": sum(1 for v in self.verdicts if v.verdict in ("verified", "contradicted")),
            "undetermined": sum(1 for v in self.verdicts if v.verdict == "undetermined"),
            "verified": sum(1 for v in self.verdicts if v.verdict == "verified"),
            "contradicted": len(self.contradicted),
            "unsupported": len(self.unsupported),
            "skipped": sum(1 for v in self.verdicts if v.verdict == "skipped"),
            "rejected_items": self.rejected_items,
        }

    @property
    def complete(self) -> bool:
        """Did the critic assess everything it should have? Independent of
        what it found: a malformed extraction entry must not hide a confirmed
        contradiction, and a contradiction must not hide that other claims
        went unassessed."""
        if not self.extraction_ok:
            return False
        if not self.claims:
            return False          # empty extraction is never complete; brevity is not absence of claims
        if self.rejected_items or self.skipped_factual:
            return False
        return True

    @property
    def findings(self) -> str:
        if self.contradicted:
            return "flagged"
        if self.unsupported:
            return "unresolved"
        return "none"

    @property
    def status(self) -> str:
        """One word for the headline; `complete` and `findings` carry the two
        dimensions separately."""
        if self.contradicted:
            return "flagged"
        if not self.complete:
            return "incomplete"
        if self.unsupported:
            return "unresolved"
        return "clean"

    @property
    def clean(self) -> bool:
        return self.complete and self.findings == "none"

    # -- output -----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "complete": self.complete,
            "findings": self.findings,
            "coverage": self.coverage,
            "extraction_ok": self.extraction_ok,
            "extraction_error": self.extraction_error,
            "claims": [asdict(c) for c in self.claims],
            "verdicts": [{**asdict(v), "flag": v.flag} for v in self.verdicts],
            "tokens": self.tokens,
        }

    def render(self) -> str:
        c = self.coverage
        lines = [f"CRITIQUE — findings: {self.findings.upper()}   "
                 f"assessment: {'COMPLETE' if self.complete else 'INCOMPLETE'}",
                 f"  claims {c['claims']}  assessed {c['assessed']}  resolved {c['resolved']}  "
                 f"contradicted {c['contradicted']}  unsupported {c['unsupported']}  skipped {c['skipped']}"]
        if not self.extraction_ok:
            lines.append(f"  extraction failed: {self.extraction_error}")
        if self.rejected_items:
            lines.append(f"  {self.rejected_items} extractor item(s) rejected as malformed")
        marks = {"verified": "ok ", "contradicted": "XX ", "undetermined": "?  ", "skipped": "-- "}
        for v in self.verdicts:
            flag = f"  [{v.flag}]" if v.flag else ""
            down = f"  (downgraded: {v.downgraded})" if v.downgraded else ""
            lines.append(f"{marks[v.verdict]}[{v.claim.type:<11}] {v.claim.source}{flag}{down}")
            if v.evidence:
                lines.append(f"      {v.evidence}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------

_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def extract_claims(answer: str, client=None, model: str = MODEL) -> tuple[list[Claim], int, str | None]:
    """Answer in; (claims, rejected_count, error) out.

    Malformed output is an error. Malformed *entries* are counted and
    rejected, never silently dropped: a critique that lost claims in
    extraction must say so.
    """
    client = client or _client()
    response = client.messages.create(
        model=model, max_tokens=2500,
        system=EXTRACTOR_PROMPT,
        messages=[{"role": "user", "content": f"Answer to analyse:\n\n{answer}"}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content)
    match = _JSON_ARRAY.search(text)
    if not match:
        return [], 0, f"no JSON array in extractor output: {text[:120]!r}"
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], 0, f"extractor JSON invalid: {exc}"
    if not isinstance(items, list):
        return [], 0, "extractor output is not a list"
    claims, rejected = [], 0
    for item in items:
        claim = Claim.from_dict(item, answer)
        if claim is None:
            rejected += 1
        else:
            claims.append(claim)
    return claims, rejected, None


# ---------------------------------------------------------------------------
# Stage 2: verify
# ---------------------------------------------------------------------------

_VERDICT = re.compile(r"VERDICT:\s*(verified|contradicted|undetermined|unverifiable)", re.I)
_EVIDENCE = re.compile(r"EVIDENCE:\s*(.+)", re.I)

# Anchors: things in an evidence line that can be checked against a result.
# Identifiers must all be present. Figures (two or more digits, or a decimal)
# must all be present. Single digits alone do not anchor — "5" appears in
# almost any result.
_ID = re.compile(
    r"\b(?:"
    r"[A-Z]{2,5}_[A-Z0-9.]{1,6}_[A-Z0-9.]{1,6}_[LU]_\d{8}_\d{3}"   # position tag
    r"|ord_[0-9a-f]{6,}|run_[0-9a-f]{6,}|chk_[0-9a-f]{6,}|rc_[0-9a-f]{6,}"
    r"|SQLOSS_\S+"
    r"|[A-Z0-9.]{2,6}_[A-Z0-9.]{2,6}"                                # pair
    r")\b"
)
_FIGURE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])")
_TICKER = re.compile(r"\b[A-Z]{2,5}\b")


def parse_verdict(text: str) -> tuple[str, str]:
    """Pull the two-line verdict off the end of the verifier's answer.
    Missing verdict is undetermined — the critic failed to decide, which is
    itself information. 'unverifiable' is accepted as a synonym."""
    v = _VERDICT.search(text)
    e = _EVIDENCE.search(text)
    verdict = v.group(1).lower() if v else "undetermined"
    if verdict == "unverifiable":
        verdict = "undetermined"
    return verdict, (e.group(1).strip() if e else "")


def _successful_output(trace: Trace) -> str:
    """Text of every SUCCESSFUL tool call's result — summary, raw data,
    provenance — and never its arguments. An earlier version included
    arguments and failed calls, so a nonexistent pair the critic had merely
    queried for could 'ground' a verdict, and a failed call's echo could
    ground one if any other call had succeeded."""
    parts = []
    for call in trace.tool_calls:
        if call.error:
            continue
        empty = call.raw_result in (None, [], {}) and not (call.provenance or {}).get("rows")
        if empty:
            continue
        parts.append(call.summary or "")
        if call.raw_result is not None:
            parts.append(json.dumps(call.raw_result, default=str))
        if call.provenance:
            parts.append(json.dumps(call.provenance, default=str))
    return "\n".join(parts)


def _figures_in(text: str) -> list[float]:
    out = []
    for f in _FIGURE.findall(text):
        try:
            out.append(float(f.replace(",", "")))
        except ValueError:
            pass
    return out


def _numeric_values(obj, acc: set[float] | None = None) -> set[float]:
    """Every numeric leaf in a result structure. Figures in evidence are
    checked against these VALUES, not against the result's text — so a
    single-digit count grounds when it is genuinely a value in the result,
    and never because the digit happened to appear inside an identifier."""
    acc = set() if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(round(float(obj), 4))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numeric_values(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numeric_values(v, acc)
    elif isinstance(obj, str):
        # Summaries carry figures as text ("50 events", "23.13bps slippage")
        for f in _figures_in(obj):
            acc.add(round(f, 4))
    return acc


def _successful_values(trace: Trace) -> set[float]:
    vals: set[float] = set()
    for call in trace.tool_calls:
        if call.error:
            continue
        if call.raw_result in (None, [], {}) and not (call.provenance or {}).get("rows"):
            continue
        _numeric_values(call.raw_result, vals)
        _numeric_values(call.provenance or {}, vals)
        _numeric_values(call.summary or "", vals)
    return vals


def _value_present(f: float, vals: set[float]) -> bool:
    """Exact, rounding (2dp), sign-agnostic for display dashes, or
    percent/decimal equivalents."""
    cands = {f, -f, f / 100, f * 100}
    for v in vals:
        for c in cands:
            if abs(v - c) <= max(abs(c) * 0.005, 0.005):
                return True
    return False


def evidence_is_grounded(evidence: str, trace: Trace) -> tuple[bool, str]:
    """(a) at least one successful, non-empty tool result; (b) non-empty
    evidence; (c) every identifier and every figure the evidence names
    appears in a successful result's OUTPUT.

    Handles aggregates: "11 partial fills" is grounded if 11 appears in a
    count; "VIS at 34.57%" if 34.57 appears in a breakdown. Handles records
    as before. This checks the critic was looking at something real when it
    judged; it cannot check that what it saw supports the judgement.
    """
    output = _successful_output(trace)
    if not output:
        return False, "no successful non-empty tool result"
    if not evidence.strip():
        return False, "empty evidence"

    ids = set(_ID.findall(evidence))
    figs = _figures_in(evidence)
    tickers = set(_TICKER.findall(evidence)) - {"VERDICT", "EVIDENCE", "LMT", "MKT", "BUY", "SELL"}
    if not ids and not figs:
        return False, "evidence names no record or figure"

    missing_ids = [i for i in ids if i not in output]
    if missing_ids:
        return False, f"evidence cites {missing_ids} absent from every successful result"
    vals = _successful_values(trace)
    missing_figs = [f for f in figs if not _value_present(f, vals)]
    if missing_figs:
        return False, f"evidence cites figures {missing_figs} absent from every successful result"
    missing_tk = [t for t in tickers if t not in output]
    if tickers and len(missing_tk) == len(tickers):
        return False, f"evidence names tickers {missing_tk} absent from every successful result"
    return True, ""


def verify_claim(claim: Claim, question: str, conn: Connection, client=None,
                 model: str = MODEL, trace_dir: Path | None = None) -> Verdict:
    trace: Trace = run_agent(
        verifier_question(claim.text, question), conn, client=client, model=model,
        max_turns=VERIFIER_MAX_TURNS,
        system_prompt=verifier_system(_blotter_context(conn)),
        save_trace=True, trace_dir=trace_dir or CRITIC_TRACE_DIR, corpus="critic",
    )
    proposed, evidence = parse_verdict(trace.answer)
    verdict, downgraded = proposed, None

    if proposed in ("verified", "contradicted"):
        ok, why = evidence_is_grounded(evidence, trace)
        if not ok:
            downgraded = f"{proposed} without grounded evidence — {why}"
            verdict = "undetermined"

    path = (trace_dir or CRITIC_TRACE_DIR) / f"trace_{trace.run_id}.json"
    return Verdict(
        claim=claim, verdict=verdict, evidence=evidence, proposed=proposed,
        tool_sequence=trace.tool_sequence, downgraded=downgraded,
        trace_run_id=trace.run_id, trace_path=str(path), tokens=trace.processed_tokens,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def critique(question: str, answer: str, conn: Connection, client=None,
             model: str = MODEL, extractor_client=None,
             trace_dir: Path | None = None) -> Critique:
    claims, rejected, error = extract_claims(answer, client=extractor_client or client, model=model)
    if error:
        return Critique(question, answer, [], [], extraction_ok=False, extraction_error=error)

    verdicts: list[Verdict] = []
    total = 0
    for claim in claims:
        if not claim.checkable:
            verdicts.append(Verdict(claim=claim, verdict="skipped",
                                    evidence="not checkable against the blotter"))
            continue
        v = verify_claim(claim, question, conn, client=client, model=model, trace_dir=trace_dir)
        total += v.tokens
        verdicts.append(v)

    return Critique(question, answer, claims, verdicts, rejected_items=rejected, tokens=total)


def _client():
    from src.agent.loop import build_client
    return build_client()
