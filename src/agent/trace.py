"""Run tracing.

Every agent run is captured in full: the question, each tool call with its
arguments and result summary, token usage, latency, and the final answer.

Built before the loop rather than bolted on afterwards, for two reasons. It is
the observability story — the hand-rolled equivalent of what Langfuse provides
— and it is the substrate the eval harness runs against: an eval asserts on the
tool-call sequence a trace records, so the trace format decides what can be
tested.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces"
BASELINE_DIR = TRACE_DIR / "baseline"

# Bumped when the trace gains fields the predictor depends on. Traces from
# before per-turn usage, model identity and cache accounting existed are
# schema 1; they cannot train a router because they do not record what the
# router needs to reproduce.
TRACE_SCHEMA_VERSION = 2


@dataclass
class ToolCall:
    """One tool invocation within a run."""

    name: str
    arguments: dict[str, Any]
    summary: str
    rows: int | None
    duration_ms: int
    turn: int
    error: bool = False
    # Full tool output, kept so numeric-fidelity checks can verify every figure
    # the model states against everything the tool actually returned. Excluded
    # from the rendered trace, which would otherwise be unreadable.
    raw_result: Any = None
    provenance: dict | None = None   # full provenance; event counts and windows live here


@dataclass
class Trace:
    """The full record of one question, answered."""

    question: str
    schema_version: int = TRACE_SCHEMA_VERSION
    corpus: str | None = None            # "baseline" when generated for the predictor
    config: dict | None = None           # the inference configuration that produced this run
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    tool_calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0      # served from cache — billed at ~10%
    cache_write_tokens: int = 0     # written to cache this call — billed at 125%
    # Per-turn usage, so an exporter can emit one generation per API call
    # rather than one aggregate. Each entry: {turn, input, output, cache_read,
    # cache_write}.
    turn_usage: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    stop_reason: str | None = None
    error: str | None = None
    langfuse_trace_id: str | None = None   # set when exported; used for scoring
    model: str | None = None
    routing: dict | None = None             # the router's decision, if routed
    compressed_chars: int = 0               # history removed by compression

    _clock: float = field(default_factory=time.perf_counter, repr=False)

    # -- recording ---------------------------------------------------------

    def record_tool(self, name: str, arguments: dict, result, duration_ms: int, turn: int) -> None:
        provenance = getattr(result, "provenance", {}) or {}
        summary = getattr(result, "summary", "")
        self.tool_calls.append(ToolCall(
            name=name,
            arguments=arguments,
            summary=summary,
            rows=provenance.get("rows"),
            duration_ms=duration_ms,
            turn=turn,
            # A tool that returns nothing is not an error; one that could not
            # run is. The distinction matters when reviewing a trace.
            error=summary.startswith(("Unknown tool", "Invalid arguments")),
            raw_result=getattr(result, "data", None),
            provenance=dict(provenance),
        ))

    def record_usage(self, usage) -> None:
        if usage is None:
            return
        entry = {
            "turn": self.turns,
            "input": getattr(usage, "input_tokens", 0) or 0,
            "output": getattr(usage, "output_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }
        self.turn_usage.append(entry)
        self.input_tokens += entry["input"]
        self.output_tokens += entry["output"]
        self.cache_read_tokens += entry["cache_read"]
        self.cache_write_tokens += entry["cache_write"]

    def finish(self, answer: str, stop_reason: str | None = None) -> Trace:
        self.answer = answer
        self.stop_reason = stop_reason
        self.duration_ms = int((time.perf_counter() - self._clock) * 1000)
        return self

    # -- reading -----------------------------------------------------------

    @property
    def processed_tokens(self) -> int:
        """Every token the model handled, at face value: input + output +
        cache read + cache write. Distinct from billed cost, which weights
        each class differently. This is the canonical sum; nothing else
        should add these four up independently."""
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered tool names. This is what evals assert against."""
        return [call.name for call in self.tool_calls]

    def as_dict(self, include_raw: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_clock", None)
        payload["tool_sequence"] = self.tool_sequence
        if not include_raw:
            for call in payload["tool_calls"]:
                call.pop("raw_result", None)
                call.pop("provenance", None)
        return payload

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or TRACE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"trace_{self.run_id}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    def render(self) -> str:
        """Human-readable trace for the CLI's --trace flag."""
        lines = [
            f"question   {self.question}",
            f"run_id     {self.run_id}",
            f"model      {self.model or '-'}"
            + (f"   tier {self.routing['tier']} — {self.routing['reason']}" if self.routing else ""),
            f"turns      {self.turns}   tools {len(self.tool_calls)}   "
            f"tokens {self.input_tokens}in/{self.output_tokens}out"
            + (f" (+{self.cache_read_tokens} cached)" if self.cache_read_tokens else "")
            + (f" (-{self.compressed_chars // 4} compressed)" if self.compressed_chars else "")
            + f"   {self.duration_ms}ms",
            "",
        ]
        for i, call in enumerate(self.tool_calls, start=1):
            args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items()) or "-"
            marker = "!" if call.error else " "
            lines.append(f"{marker}{i}. {call.name}({args})")
            lines.append(f"     -> {call.summary}  [{call.duration_ms}ms]")
        if self.error:
            lines += ["", f"ERROR: {self.error}"]
        return "\n".join(lines)
