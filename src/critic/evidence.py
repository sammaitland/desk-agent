"""Validate references to exact tool fields, not numbers pooled across a trace.

This establishes citation fidelity and declared scope. It cannot establish
that a faithful citation entails an arbitrary natural-language proposition.
No sign changes, unit conversions, or numbers parsed from dates are allowed:
the verifier must copy the observed value as returned, and explain any
comparison in prose. Tool-computed rounded values can be copied directly.
"""

from __future__ import annotations

import json
import math
import re

from src.agent.trace import Trace

RELATIONS = {"record", "recorded_reason", "aggregate", "market_cause"}
EVIDENCE_TOOLS = {"query_records", "query_blotter", "explain_rejection",
                  "explain_position", "alpha_attribution", "execution_quality", "detect_anomalies"}
SCOPE_KEYS = ("population", "date_basis", "window", "filters", "group_by", "status")
_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n?([\s\S]*?)```", re.I)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_assessment(answer: str) -> tuple[dict, str | None]:
    """Parse exactly one JSON object, tolerating prose around one code fence."""
    value = answer.strip()
    fences = _JSON_FENCE.findall(value)
    if len(fences) > 1:
        return {}, "verifier returned multiple fenced assessments"
    if fences:
        value = fences[0].strip()
    try:
        data = json.loads(value, object_pairs_hook=_unique_object, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        return {}, "verifier did not return one valid JSON assessment"
    if not isinstance(data, dict) or data.get("verdict") not in ("verified", "contradicted", "undetermined"):
        return {}, "missing or invalid verdict"
    if not isinstance(data.get("evidence"), str) or not data["evidence"].strip():
        return data, "missing evidence explanation"
    if data.get("relation") not in RELATIONS:
        return data, "missing or invalid evidence relation"
    if not isinstance(data.get("references"), list):
        return data, "references must be a list"
    return data, None


def _field(payload, pointer: str):
    if not isinstance(pointer, str) or not pointer.startswith(("/data/", "/provenance/")):
        raise ValueError("reference must point inside data or provenance")
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(payload, list):
            if not part.isdigit():
                raise ValueError("list index must be nonnegative")
            payload = payload[int(part)]
        elif isinstance(payload, dict):
            payload = payload[part]
        else:
            raise ValueError("path descends through a scalar")
    return payload


def _same_value(observed, cited) -> bool:
    if isinstance(observed, bool) or isinstance(cited, bool):
        return type(observed) is type(cited) and observed == cited
    if isinstance(observed, (int, float)) and isinstance(cited, (int, float)):
        return math.isfinite(observed) and math.isfinite(cited) and observed == cited
    return type(observed) is type(cited) and observed == cited


def reference_scope(provenance: dict, path: str) -> dict:
    """Mixed anomaly results carry separate scopes for checks and runs."""
    for section, scope in provenance.get("scopes", {}).items():
        if isinstance(path, str) and path.startswith(f"/data/{section}/"):
            return {k: scope[k] for k in SCOPE_KEYS if k in scope}
    return {k: provenance[k] for k in SCOPE_KEYS if k in provenance}


def validate_evidence(assessment: dict, trace: Trace) -> tuple[list[dict], list[str], str | None]:
    """Resolve citations and attach scope from the cited result itself.

    A model supplies only call/path/value. Invalid *extra* citations become
    warnings when another citation grounds the verdict; they do not erase
    good evidence. A settled verdict still fails when it has no valid primary
    citation or does not meet the declared relation's requirements.
    """
    if not isinstance(assessment, dict):
        return [], [], "structured field references required"
    refs = assessment.get("references")
    if not isinstance(refs, list) or not refs:
        return [], [], "no field references"
    relation = assessment.get("relation")
    if relation not in RELATIONS:
        return [], [], "invalid evidence relation"
    if relation == "market_cause":
        return [], [], "available tools do not establish or exclude market causation"
    valid: list[dict] = []
    warnings: list[str] = []
    primary = False
    reason = False
    aggregate = False
    for number, ref in enumerate(refs, 1):
        problem = None
        if not isinstance(ref, dict) or type(ref.get("call")) is not int:
            problem = "invalid call reference"
        else:
            index = ref["call"] - 1
            if index < 0 or index >= len(trace.tool_calls):
                problem = "referenced call does not exist"
        if problem:
            warnings.append(f"reference {number}: {problem}")
            continue
        call = trace.tool_calls[index]
        if call.error or (call.provenance or {}).get("error") or call.name not in EVIDENCE_TOOLS:
            warnings.append(f"reference {number}: referenced call is not successful blotter evidence")
            continue
        p = call.provenance or {}
        path = ref.get("path")
        try:
            value = _field({"data": call.raw_result, "provenance": p}, path)
        except (ValueError, KeyError, IndexError, TypeError):
            warnings.append(f"reference {number}: referenced field does not exist")
            continue
        if isinstance(value, (dict, list)) or "value" not in ref:
            warnings.append(f"reference {number}: reference must copy a scalar field")
            continue
        if not _same_value(value, ref["value"]):
            warnings.append(f"reference {number}: referenced value differs (including sign, type or units)")
            continue
        # Scope echoes and presentation metadata are not measurements.
        is_count = (call.name == "query_records" and path == "/data/count") or (
            call.name == "detect_anomalies" and
            (path == "/provenance/event_count" or path.startswith("/provenance/event_type_counts/"))) or (
            call.name == "query_blotter" and path == "/provenance/total_rows")
        if is_count:
            if p.get("count_complete") is not True or p.get("filters_validated") is not True:
                warnings.append(f"reference {number}: count requires complete, validated query scope")
                continue
            primary = aggregate = True
        elif path.startswith("/data/"):
            primary = True
            if call.name in ("alpha_attribution", "execution_quality") and path.startswith(("/data/breakdown/", "/data/totals/")):
                aggregate = True
        else:
            warnings.append(f"reference {number}: provenance field is scope or presentation metadata, not an authoritative count")
            continue
        valid.append({"call": ref["call"], "tool": call.name, "path": path,
                      "value": value, "scope": reference_scope(p, path)})
        if value is not None and path.rsplit("/", 1)[-1] in (
                "rejection_reason", "primary_fail_reason", "fallback_reason", "exit_reason", "action"):
            reason = True
    if not valid or not primary:
        return valid, warnings, "no valid primary evidence field"
    if relation == "recorded_reason" and not reason:
        return valid, warnings, "recorded_reason requires an actual recorded reason/action field"
    if relation == "aggregate" and not aggregate:
        return valid, warnings, "aggregate requires a tool-computed aggregate, not an incidental row value"
    return valid, warnings, None


def evidence_is_grounded(assessment: dict, trace: Trace) -> tuple[bool, str]:
    """Compatibility wrapper for callers that only need pass/fail."""
    _refs, _warnings, error = validate_evidence(assessment, trace)
    return error is None, error or ""


def requires_market_evidence(claim: str) -> bool:
    """Conservative extra backstop for explicit market-cause language.

    This is deliberately a heuristic, not semantic classification. The model
    must also declare market_cause; neither route is a proof of entailment.
    Statements about what is *recorded* can still use recorded_reason when
    they explicitly say so. New paraphrases belong in live evaluation.
    """
    cause = re.search(r"\b(caus\w*|because|due to|driven by|triggered by|attribut\w*)\b", claim, re.I)
    market = re.search(r"\b(intraday|sector[- ]wide|market[- ]wide|liquidity (?:shock|event)|market (?:move|shock))\b", claim, re.I)
    recorded = re.search(r"\b(recorded|logged|record states|log states)\b", claim, re.I)
    return bool(cause and market and not recorded)
