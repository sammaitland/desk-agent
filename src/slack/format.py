"""Markdown to Slack mrkdwn conversion.

Slack does not use standard Markdown. It uses `mrkdwn`, which differs in ways
that matter for this agent's output:

  * bold is *single* asterisks, not double — `**x**` renders literally
  * italic is _underscores_
  * there are no headers; `### Heading` renders as literal hashes
  * there are NO TABLES, and the agent produces them regularly

The table problem is the substantive one. A Markdown table pasted into Slack is
unreadable pipe soup, so tables are converted to fixed-width text inside a code
block, which Slack renders in monospace and therefore aligns correctly.
"""

from __future__ import annotations

import re

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_HEADER = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_HRULE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# **bold** or *bold*, and _italic_ or __italic__, but only when the marker
# wraps a span rather than sitting inside a word.
_EMPHASIS = re.compile(r"(?<![\w])([*_]{1,2})(?=\S)(.+?)(?<=\S)\1(?![\w])", re.DOTALL)


def _strip_emphasis(text: str) -> str:
    """Remove emphasis markers without touching identifiers like VGT_AAPL_L."""
    previous = None
    while previous != text:
        previous = text
        text = _EMPHASIS.sub(r"\2", text)
    return text.replace("`", "")


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _render_table(rows: list[list[str]]) -> str:
    """Fixed-width table inside a code block — Slack's only aligned option."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    # Strip Markdown emphasis markers, but ONLY where they are actually acting
    # as emphasis. An earlier version removed every underscore, which silently
    # mangled position tags: VGT_AAPL_NVDA_L became VGTAAPLNVDAL. Underscores
    # inside identifiers are data, not formatting.
    rows = [[_strip_emphasis(cell) for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(width)]

    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(width)))
    return "```\n" + "\n".join(lines) + "\n```"


def _convert_tables(text: str) -> str:
    output, buffer = [], []

    def flush() -> None:
        if buffer:
            output.append(_render_table(buffer))
            buffer.clear()

    for line in text.split("\n"):
        if _TABLE_ROW.match(line):
            if not _TABLE_DIVIDER.match(line):   # divider row carries no data
                buffer.append(_split_row(line))
            continue
        flush()
        output.append(line)
    flush()
    return "\n".join(output)


def to_mrkdwn(text: str) -> str:
    """Convert the agent's Markdown answer into something Slack renders well."""
    text = _convert_tables(text)
    text = _LINK.sub(r"<\2|\1>", text)          # Slack's link syntax is inverted
    text = _HEADER.sub(r"*\1*", text)           # no headers: bold the line instead
    text = _BOLD.sub(r"*\1*", text)             # ** -> *
    text = _HRULE.sub("", text)                 # horizontal rules render as noise
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int = 2900) -> str:
    """Slack rejects text blocks over 3000 characters.

    Cuts on a paragraph boundary where possible so the message does not end
    mid-sentence, and says plainly that it was cut rather than leaving the
    reader to wonder.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind("\n\n")
    if boundary > limit * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + "\n\n_(truncated — ask a narrower question for the full detail)_"
