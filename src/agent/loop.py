"""The agent loop.

The whole mechanism is one cycle:

    send messages + tools -> response
    if stop_reason == 'tool_use':  run the tools, append results, repeat
    if stop_reason == 'end_turn':  return the text

Written directly against the Messages API rather than through an orchestration
framework. The abstractions those frameworks provide are thin, and knowing
exactly what is on the wire is worth more than the lines of code they save —
particularly when debugging why a model chose the wrong tool.

The `client` is injected rather than constructed here. That is what lets the
eval harness run scripted conversations without API calls, and it keeps the
loop itself free of transport concerns.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from sqlalchemy.engine import Connection

from src.agent.prompt import build_system_prompt
from src.agent.trace import Trace
from src.routing.compression import compress_history, strip_private_keys
from src.tools import TOOL_SCHEMAS, dispatch

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000
MAX_TURNS = 8  # a confused model must not be able to spin indefinitely


class MessagesClient(Protocol):
    """The slice of the Anthropic SDK this loop actually uses."""

    @property
    def messages(self) -> Any: ...


def build_client(api_key: str | None = None):
    """Construct a real Anthropic client. Imported lazily so tests need no key."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _blotter_context(conn: Connection) -> str:
    """Tell the model the boundaries of its own data.

    Without this it discovers the date range by trial and error, wasting a turn
    and sometimes concluding the blotter is empty when it queried the wrong year.
    """
    from src.tools.base import blotter_date_range

    first, last = blotter_date_range(conn)
    if not first:
        return "The blotter is empty."
    return (f"The blotter covers {first} to {last}. When a question says "
            f"'yesterday' or 'recently' without a date, interpret it relative "
            f"to {last}, the most recent trading day held.")


def _cached_system(system: str) -> list[dict[str, Any]]:
    """Wrap the system prompt as a cacheable content block.

    The system prompt and tool schemas are ~3,000 tokens resent on every turn
    of every query. Marking them with cache_control lets the API serve them
    from cache at a fraction of the cost; only the conversation itself is
    charged at full rate. The cache key is the exact prefix, so the runtime
    context (blotter date range) must stay stable within a session for the
    cache to hit — which it does.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _cached_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the last tool definition as a cache breakpoint.

    cache_control on the final tool caches the whole tools array as one
    prefix. Tools are processed before the system prompt in the cache
    ordering, so this and _cached_system together cover the entire fixed
    portion of every request.
    """
    tools = [dict(t) for t in schemas]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _tool_result_block(tool_use_id: str, result, evidence_call: int | None = None) -> dict[str, Any]:
    """Serialise a ToolResult into the block shape the API expects.

    `content` must be a string, so the structured result is JSON-encoded. The
    summary is placed first in the payload because it is what the model reads
    when deciding whether it has enough to answer.
    """
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps({**result.as_dict(), **({"evidence_call": evidence_call}
                               if evidence_call is not None else {})}, default=str),
    }


def run_agent(
    question: str,
    conn: Connection,
    client: MessagesClient | None = None,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
    save_trace: bool = False,
    compress: bool = False,
    routing: dict | None = None,
    corpus: str | None = None,
    trace_dir=None,
    system_prompt: str | None = None,
) -> Trace:
    """Answer one question, returning the full trace.

    The trace rather than the string is returned deliberately: the answer alone
    hides how it was reached, and how it was reached is what gets evaluated.
    """
    client = client or build_client()
    trace = Trace(question=question)
    trace.model = model
    trace.routing = routing
    trace.config = {"model": model, "max_turns": max_turns, "caching": True,
                    "compress": compress, "routed": routing is not None}
    trace.corpus = corpus
    # The critic reuses this loop with its own instructions. Same tools, same
    # caching, same tracing — different job.
    base = system_prompt if system_prompt is not None else build_system_prompt(_blotter_context(conn))
    system = _cached_system(base)
    tools = _cached_tools(TOOL_SCHEMAS)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    try:
        for turn in range(1, max_turns + 1):
            trace.turns = turn
            if compress:
                messages, removed = compress_history(messages, current_turn=turn)
                trace.compressed_chars += removed
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=strip_private_keys(messages),
            )
            trace.record_usage(getattr(response, "usage", None))

            if response.stop_reason != "tool_use":
                text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                return _finish(trace, text.strip(), response.stop_reason, save_trace, trace_dir)

            # Echo the assistant turn back verbatim, then answer every tool_use
            # block it contained in a single following user message.
            messages.append({"role": "assistant", "content": _serialise(response.content)})

            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                started = time.perf_counter()
                result = dispatch(block.name, dict(block.input), conn)
                elapsed = int((time.perf_counter() - started) * 1000)
                trace.record_tool(block.name, dict(block.input), result, elapsed, turn)
                results.append({**_tool_result_block(block.id, result, len(trace.tool_calls)), "_turn": turn})

            messages.append({"role": "user", "content": results})

        # Turn budget exhausted: report it rather than presenting a partial
        # investigation as a finished answer.
        return _finish(
            trace,
            f"Stopped after {max_turns} turns without reaching an answer. "
            f"Tools called: {', '.join(trace.tool_sequence) or 'none'}.",
            "max_turns",
            save_trace, trace_dir,
        )

    except Exception as exc:  # transport, auth, rate limit
        trace.error = f"{type(exc).__name__}: {exc}"
        return _finish(trace, f"The request failed: {trace.error}", "error", save_trace, trace_dir)


def run_agent_routed(
    question: str,
    conn: Connection,
    client: MessagesClient | None = None,
    router=None,
    save_trace: bool = False,
) -> Trace:
    """Route first, then run with the chosen tier.

    The decision is recorded on the trace, so an eval can tell which tier
    answered and whether the router's prediction matched what actually ran.
    """
    from src.routing.router import Router

    router = router or Router()
    decision = router.decide(question)
    return run_agent(
        question, conn, client=client,
        model=decision.tier.model,
        max_turns=decision.tier.max_turns,
        compress=decision.tier.compress,
        routing=decision.as_dict(),
        save_trace=save_trace,
    )


def _serialise(content) -> list[dict[str, Any]]:
    """Convert SDK content blocks back into plain dicts for the next request."""
    blocks = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            blocks.append({"type": "tool_use", "id": block.id,
                           "name": block.name, "input": dict(block.input)})
    return blocks


def _finish(trace: Trace, answer: str, stop_reason: str | None, save: bool,
            trace_dir=None) -> Trace:
    trace.finish(answer, stop_reason)
    if save:
        trace.save(trace_dir)
    # Second destination, opt-in via environment. Never affects the answer.
    from src.observability.langfuse_export import configured, export_trace

    if configured():
        # Attribute generations to the model that actually ran — a routed Haiku
        # call logged as Sonnet would misprice every cost figure downstream.
        trace.langfuse_trace_id = export_trace(trace, model=trace.model)
    return trace
