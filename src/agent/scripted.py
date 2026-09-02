"""A scripted stand-in for the Anthropic client.

Replays a fixed sequence of responses so the loop can be exercised without API
calls. This is not only a test convenience: it is what makes the eval harness
cheap to run on every change, and it isolates loop bugs from model behaviour —
if a scripted run fails, the fault is in the loop, not the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    content: list
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


def tool_turn(name: str, arguments: dict, text: str = "", call_id: str = "toolu_1") -> Response:
    """A response in which the model requests one tool."""
    blocks: list = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.append(ToolUseBlock(id=call_id, name=name, input=arguments))
    return Response(content=blocks, stop_reason="tool_use",
                    usage=Usage(input_tokens=1200, output_tokens=90))


def parallel_tool_turn(calls: list[tuple[str, dict]], text: str = "") -> Response:
    """A response requesting several tools at once — the API permits this."""
    blocks: list = [TextBlock(text=text)] if text else []
    for i, (name, arguments) in enumerate(calls, start=1):
        blocks.append(ToolUseBlock(id=f"toolu_p{i}", name=name, input=arguments))
    return Response(content=blocks, stop_reason="tool_use",
                    usage=Usage(input_tokens=1400, output_tokens=140))


def final_turn(text: str) -> Response:
    """A response that ends the conversation."""
    return Response(content=[TextBlock(text=text)], stop_reason="end_turn",
                    usage=Usage(input_tokens=2100, output_tokens=220))


class ScriptedMessages:
    def __init__(self, script: list[Response]):
        self._script = list(script)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        # Deep-copy the messages: the loop mutates its list in place, so
        # storing the reference would make every recorded request show the
        # final state. Accurate per-turn request logs matter for evals.
        import copy
        self.requests.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        if not self._script:
            return final_turn("(script exhausted)")
        return self._script.pop(0)


class ScriptedClient:
    """Drop-in replacement for anthropic.Anthropic in tests and evals."""

    def __init__(self, script: list[Response]):
        self.messages = ScriptedMessages(script)

    @property
    def requests(self) -> list[dict]:
        return self.messages.requests


class FailingClient:
    """Simulates a transport failure, to check the loop degrades gracefully."""

    class _Messages:
        def create(self, **kwargs):
            raise ConnectionError("simulated API failure")

    def __init__(self):
        self.messages = self._Messages()
