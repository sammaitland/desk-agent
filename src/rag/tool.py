"""The documentation search tool.

Exposes retrieval to the agent through the same envelope every other tool
uses: `data`, `provenance`, `summary`. The agent does not know or care that
this tool is backed by retrieval rather than SQL — it sees ranked passages
with citations and a confidence score, and composes an answer from them.

That uniformity is the design point. The blotter tools answer "what happened";
this one answers "why is the system built this way". Two retrieval mechanisms
suited to two data shapes — structured rows, unstructured prose — behind one
agent, one loop, one provenance discipline. RAG alongside the tool layer, not
instead of it.

The retriever is built once and cached at module level: indexing is the
expensive step, and the corpus does not change within a session.
"""

from __future__ import annotations

from functools import lru_cache

from src.rag.retriever import Retriever, build_retriever
from src.tools.base import ToolResult, empty

# Below this score a hit is noise. Lexical and embedding scores are on
# different scales; each backend's floor is set inside its search method, and
# this is a second guard applied uniformly.
MIN_SCORE = 0.05


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return build_retriever()


def reset_retriever() -> None:
    """Drop the cached index. For tests, and after the docs directory changes."""
    _retriever.cache_clear()


def search_documentation(query: str, k: int = 4) -> ToolResult:
    """Search the trading system's design documentation.

    Answers questions about WHY the system is built the way it is — the
    rationale behind a filter, what a threshold protects against, how the
    calibration and implementation pipelines relate, what a term means in this
    system's vocabulary. Returns the most relevant passages with a citation
    (file and section) for each, so the answer can say where it came from.

    This is the counterpart to the blotter tools, which answer what HAPPENED.
    Use this when the question is about design, rationale or definitions;
    use the blotter tools when it is about data. Many questions need both.
    """
    query = (query or "").strip()
    if not query:
        return empty("No query supplied.")

    retriever = _retriever()
    hits = [h for h in retriever.search(query, k=k) if h.score >= MIN_SCORE]
    if not hits:
        return empty(
            f"No documentation passages matched '{query}'. The corpus covers "
            f"the trading system's architecture and the desk agent's design; "
            f"the question may be outside it.",
            query=query, backend=retriever.name,
        )

    top = hits[0]
    return ToolResult(
        data=[h.as_dict() for h in hits],
        provenance={
            "query": query,
            "backend": retriever.name,
            "rows": len(hits),
            "corpus_chunks": len(retriever.chunks),
            "top_score": round(float(top.score), 4),
            "sources": sorted({h.chunk.source for h in hits}),
        },
        summary=(f"{len(hits)} passages for '{query}'; best match "
                 f"{top.chunk.citation} (score {top.score:.2f})."),
    )
