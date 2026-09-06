"""Context compression for the agent loop.

The model is stateless, so every turn resends the whole conversation — and
tool results dominate it. A 50-row `detect_anomalies` payload is paid for when
it arrives and again on every turn that follows. Most of that is dead weight:
once the model has read a result and written its next turn, the conclusion is
in its own text and the raw rows are no longer load-bearing.

This is layer one of the standard three-layer cascade — compress tool outputs,
then sliding-window older turns, then summarise — and only layer one is
implemented. It is the cheapest and the safest, because the summary line every
tool already returns was written to be exactly this: the result, in one line,
with provenance.

**The rule:** a tool result is left intact on the turn immediately after it
arrives, so the model can reason over the full payload. From the turn after
that, it is replaced by its summary. If the model later needs the detail, it
can call the tool again — which is cheaper than resending the detail on every
turn in case it might.

Token-level compressors (LLMLingua and kin) are deliberately not used. They
mangle the structured content agents act on — a 2026 study found 17 of 17
agent test cells collapsed under them. The tool-result summary is the right
unit of meaning.
"""

from __future__ import annotations

import json
from typing import Any

COMPRESSED_MARKER = "[compressed]"


def compress_history(messages: list[dict[str, Any]], current_turn: int,
                     keep_recent: int = 1) -> tuple[list[dict[str, Any]], int]:
    """Replace stale tool results with their summaries.

    `messages` is the loop's message list. Tool results live in user messages
    as lists of `tool_result` blocks whose `content` is a JSON string of the
    tool's envelope. Each block is tagged with the turn it arrived on (the loop
    sets `_turn` when it appends the block; the API ignores unknown keys, and
    the tag is stripped before sending).

    A block is compressed when it is older than `keep_recent` turns. The
    summary replaces the full envelope; provenance is kept because it is short
    and is what the model needs to report scope honestly.

    Returns the new message list and the number of characters removed — the
    latter is recorded on the trace so the saving is visible.
    """
    removed = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            out.append(message)
            continue

        blocks = []
        for block in message["content"]:
            if (block.get("type") != "tool_result"
                    or block.get("_compressed")
                    or block.get("_turn") is None
                    or current_turn - block["_turn"] <= keep_recent):
                blocks.append(block)
                continue

            original = block.get("content", "")
            try:
                envelope = json.loads(original)
                summary = envelope.get("summary", "")
                provenance = envelope.get("provenance", {})
            except (json.JSONDecodeError, AttributeError):
                summary, provenance = original[:200], {}

            compact = json.dumps({
                "summary": summary,
                "provenance": provenance,
                "note": f"{COMPRESSED_MARKER} full result released after use; "
                        f"call the tool again if detail is needed",
            })
            removed += max(len(original) - len(compact), 0)
            blocks.append({**block, "content": compact, "_compressed": True})
        out.append({**message, "content": blocks})
    return out, removed


def strip_private_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the loop's bookkeeping keys before sending to the API.

    `_turn` and `_compressed` are ours; the API rejects unknown keys on
    tool_result blocks in strict mode, so they must not leave the process.
    """
    clean = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in block.items() if not k.startswith("_")}
                       for block in content]
        clean.append({**message, "content": content})
    return clean
