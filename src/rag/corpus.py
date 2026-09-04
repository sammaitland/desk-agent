"""Document corpus for retrieval.

Loads Markdown files and splits them into chunks along heading boundaries.
Each chunk carries its source file and heading path, so a retrieved passage can
be cited — "ARCHITECTURE.md › Execution › Order routing" — rather than surfaced
as anonymous text. That citation is what keeps retrieval governed: the agent
can say where a claim came from, and a reader can check it.

Chunking by heading rather than by fixed token count is a deliberate choice for
this corpus. Technical documentation is already organised into sections that
each address one topic; cutting across them at an arbitrary token boundary
splits the very units of meaning retrieval is trying to find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_MAX_CHARS = 1800   # beyond this a section is split on paragraph breaks
_MIN_CHARS = 80     # below this a chunk is too thin to be worth indexing


@dataclass
class Chunk:
    """One retrievable unit of documentation."""

    text: str
    source: str            # file name
    heading_path: list[str] = field(default_factory=list)
    index: int = 0         # position within the source file

    @property
    def citation(self) -> str:
        path = " › ".join(self.heading_path) if self.heading_path else "(top)"
        return f"{self.source} › {path}"

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def _split_long(text: str) -> list[str]:
    """Split an over-long section on paragraph breaks, keeping paragraphs whole."""
    if len(text) <= _MAX_CHARS:
        return [text]
    parts, current = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > _MAX_CHARS and current:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split one Markdown document into heading-scoped chunks."""
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal counter
        body = "\n".join(buffer).strip()
        buffer.clear()
        if len(body) < _MIN_CHARS:
            return
        path = [h for _, h in heading_stack]
        for part in _split_long(body):
            # Prefix the heading path so the chunk is self-describing when
            # retrieved on its own — the model sees context, not a fragment.
            prefixed = (" › ".join(path) + "\n\n" + part) if path else part
            chunks.append(Chunk(text=prefixed, source=source,
                                heading_path=list(path), index=counter))
            counter += 1

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buffer.append(line)
            continue
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return chunks


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    """Load every Markdown file in the docs directory into chunks."""
    directory = directory or DOCS_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks
