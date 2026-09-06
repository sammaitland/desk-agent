"""Export traces to Langfuse.

The hand-rolled `Trace` came first and stays: it is the source of truth for
evals, it needs no external service, and knowing what it captures is what makes
Langfuse's abstractions legible rather than magical. This module maps a
finished Trace onto Langfuse's model — one span for the run, one generation per
API turn, one span per tool call — and sends it.

Design rules:

  * **Additive.** The local JSON trace is written regardless. Langfuse is a
    second destination, not a replacement.
  * **No-op when unconfigured.** If LANGFUSE_PUBLIC_KEY is absent, nothing
    happens and nothing warns. A dev machine without Langfuse must behave
    identically to one with it.
  * **Never fatal.** If Langfuse is down or the export fails, the agent's
    answer is unaffected; a warning is logged and the run continues. An
    observability failure must not become an availability failure.
  * **Injectable.** The client is passed in, so tests exercise the mapping
    against a fake without a Langfuse instance — the same pattern as the
    scripted API client in the agent loop.

Built against langfuse 4.x (OpenTelemetry-based, rewritten March 2026).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from src.agent.trace import Trace

log = logging.getLogger(__name__)

MODEL_DEFAULT = "claude-sonnet-4-6"


class LangfuseLike(Protocol):
    """The slice of the Langfuse client this exporter uses."""

    def start_as_current_observation(self, *, name: str, as_type: str, **kwargs): ...
    def create_score(self, *, name: str, value: float, trace_id: str, **kwargs): ...
    def flush(self) -> None: ...


def configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def build_client():
    """Construct a real Langfuse client from environment variables.

    Imported lazily so the package installs and runs without langfuse present;
    it is an optional dependency.
    """
    from langfuse import Langfuse

    return Langfuse()


def export_trace(trace: Trace, client: LangfuseLike | None = None,
                 model: str | None = None) -> str | None:
    """Send a finished Trace to Langfuse. Returns the Langfuse trace id, or None.

    Mapping:
        Trace            -> root span   (input=question, output=answer)
        each API turn    -> generation  (model, usage incl. cache tokens)
        each tool call   -> span        (input=arguments, output=summary)

    Tool spans are nested under the generation for the turn that requested
    them, so the Langfuse tree reads as the loop actually ran: model asked,
    tools answered, model asked again.
    """
    if client is None:
        if not configured():
            return None
        try:
            client = build_client()
        except Exception as exc:  # missing package, bad config
            log.warning("Langfuse unavailable, trace not exported: %s", exc)
            return None

    # The model that ran is on the trace. Only fall back to the default when a
    # trace predates model recording.
    resolved = model or trace.model or MODEL_DEFAULT
    try:
        return _export(trace, client, resolved)
    except Exception as exc:
        log.warning("Langfuse export failed for run %s: %s", trace.run_id, exc)
        return None


def _export(trace: Trace, client: LangfuseLike, model: str) -> str:
    calls_by_turn: dict[int, list] = {}
    for call in trace.tool_calls:
        calls_by_turn.setdefault(call.turn, []).append(call)

    with client.start_as_current_observation(
        name="desk-agent",
        as_type="span",
        input={"question": trace.question},
        metadata={
            "run_id": trace.run_id,
            "turns": trace.turns,
            "stop_reason": trace.stop_reason,
            "tool_sequence": trace.tool_sequence,
            "error": trace.error,
        },
    ) as root:
        for entry in trace.turn_usage:
            turn = entry["turn"]
            with client.start_as_current_observation(
                name=f"turn-{turn}",
                as_type="generation",
                model=model,
            ) as generation:
                generation.update(
                    usage_details={
                        "input": entry["input"],
                        "output": entry["output"],
                        "cache_read_input_tokens": entry["cache_read"],
                        "cache_creation_input_tokens": entry["cache_write"],
                    },
                    metadata={"turn": turn},
                )
                for call in calls_by_turn.get(turn, []):
                    with client.start_as_current_observation(
                        name=call.name,
                        as_type="span",
                        input=call.arguments,
                        metadata={"rows": call.rows, "duration_ms": call.duration_ms,
                                  "error": call.error},
                    ) as tool_span:
                        tool_span.update(output=call.summary,
                                         level="ERROR" if call.error else "DEFAULT")

        root.update(output={"answer": trace.answer})
        trace_id = root.trace_id

    client.flush()
    return trace_id


def score_trace(trace_id: str, checks: list[dict], client: LangfuseLike | None = None) -> int:
    """Attach eval check results to a Langfuse trace as scores.

    Each check becomes a boolean score named after the check. This is what
    turns Langfuse from a trace viewer into an eval dashboard: filter traces by
    a failing score name and every regression of that kind is one click away.

    Returns the number of scores written.
    """
    if client is None:
        if not configured():
            return 0
        try:
            client = build_client()
        except Exception as exc:
            log.warning("Langfuse unavailable, scores not written: %s", exc)
            return 0

    written = 0
    try:
        for check in checks:
            client.create_score(
                name=check["name"].replace(" ", "_")[:60],
                value=1.0 if check["passed"] else 0.0,
                trace_id=trace_id,
                data_type="BOOLEAN",
                comment=check.get("detail") or None,
            )
            written += 1
        client.flush()
    except Exception as exc:
        log.warning("Langfuse scoring failed for %s: %s", trace_id, exc)
    return written


def _fake_for_tests():
    """A minimal recording client, so the mapping can be asserted without Langfuse.

    Lives here rather than in tests/ so it stays in sync with the Protocol.
    """
    from contextlib import contextmanager

    class _Obs:
        def __init__(self, parent, kwargs):
            self.kwargs, self.updates, self.children = kwargs, [], []
            self.trace_id = "trace-fake"
            self.parent = parent

        def update(self, **kw):
            self.updates.append(kw)

    class _Fake:
        def __init__(self):
            self.root: list[_Obs] = []
            self.scores: list[dict] = []
            self.flushed = 0
            self._stack: list[_Obs] = []

        @contextmanager
        def start_as_current_observation(self, **kwargs):
            obs = _Obs(self._stack[-1] if self._stack else None, kwargs)
            (self._stack[-1].children if self._stack else self.root).append(obs)
            self._stack.append(obs)
            try:
                yield obs
            finally:
                self._stack.pop()

        def create_score(self, **kw):
            self.scores.append(kw)

        def flush(self):
            self.flushed += 1

    return _Fake()
