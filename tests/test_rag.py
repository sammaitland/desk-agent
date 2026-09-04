"""Tests for the retrieval layer.

Three things are protected: chunking preserves structure and citations,
retrieval ranks the right passage first for questions with a clear answer in
the corpus, and the tool returns the standard envelope so the agent cannot
tell it apart from a SQL-backed tool.

The retrieval-quality tests use the lexical backend, which is deterministic.
The embedding backend is exercised only if sentence-transformers is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.corpus import Chunk, chunk_markdown, load_corpus
from src.rag.retriever import LexicalRetriever, build_retriever
from src.rag.tool import reset_retriever, search_documentation

SAMPLE = """# System

## Execution

### Order routing

Limit orders are used by default. If the quoted spread exceeds the cap the
order is routed to market instead, because a resting limit in a wide market
is unlikely to fill and the opportunity cost exceeds the spread saving.

### Stop losses

A stop is placed on the short leg at the alpha-deterioration threshold. When
it triggers, the long leg is orphaned and closed at market by the orphan
handler on the next run.

## Screening

The trend filter rejects pairs where either leg is trending, because a
trending leg violates the mean-reversion assumption the strategy depends on.

```
# this heading inside a code block must not split the chunk
```
"""


# --- chunking -------------------------------------------------------------

def test_chunks_follow_headings():
    chunks = chunk_markdown(SAMPLE, "sample.md")
    citations = [c.citation for c in chunks]
    assert "sample.md › System › Execution › Order routing" in citations
    assert "sample.md › System › Execution › Stop losses" in citations
    assert "sample.md › System › Screening" in citations


def test_headings_inside_code_blocks_are_ignored():
    chunks = chunk_markdown(SAMPLE, "sample.md")
    assert not any("this heading inside" in c.citation for c in chunks)


def test_chunk_text_is_self_describing():
    """A retrieved chunk carries its heading path, so the model sees context."""
    chunks = chunk_markdown(SAMPLE, "sample.md")
    routing = next(c for c in chunks if "Order routing" in c.citation)
    assert routing.text.startswith("System › Execution › Order routing")


def test_thin_sections_are_dropped():
    chunks = chunk_markdown("# A\n\ntiny\n\n# B\n\n" + "x" * 200, "t.md")
    assert len(chunks) == 1 and chunks[0].heading_path == ["B"]


def test_long_sections_split_on_paragraphs():
    long = "# H\n\n" + "\n\n".join("paragraph " * 60 for _ in range(6))
    chunks = chunk_markdown(long, "long.md")
    assert len(chunks) > 1
    assert all(len(c.text) <= 2200 for c in chunks)


def test_corpus_loads_from_docs():
    chunks = load_corpus()
    assert len(chunks) > 20
    assert len({c.source for c in chunks}) >= 3


# --- retrieval ------------------------------------------------------------

@pytest.fixture
def retriever():
    return LexicalRetriever(chunk_markdown(SAMPLE, "sample.md"))


def test_ranks_the_right_section_first(retriever):
    hits = retriever.search("why are orders routed to market when the spread is wide", k=3)
    assert hits and "Order routing" in hits[0].chunk.citation


def test_distinguishes_nearby_topics(retriever):
    hits = retriever.search("what does the trend filter reject", k=1)
    assert hits and "Screening" in hits[0].chunk.citation


def test_scores_are_descending(retriever):
    hits = retriever.search("stop loss short leg threshold", k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_no_match_returns_empty(retriever):
    assert retriever.search("zzqx nonsense token", k=3) == []


def test_build_retriever_falls_back_to_lexical(monkeypatch):
    """Requesting embeddings without the dependency must degrade, not fail."""
    import builtins
    real_import = builtins.__import__

    def no_st(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_st)
    with pytest.warns(UserWarning, match="sentence-transformers not installed"):
        r = build_retriever(chunk_markdown(SAMPLE, "s.md"), backend="embedding")
    assert isinstance(r, LexicalRetriever)


# --- the tool -------------------------------------------------------------

def test_tool_returns_the_envelope():
    reset_retriever()
    result = search_documentation("alpha formula beta index return")
    assert result.summary and result.data
    assert {"query", "backend", "rows", "top_score", "sources"} <= set(result.provenance)


def test_tool_hits_carry_citations():
    reset_retriever()
    for hit in search_documentation("calibration pipeline").data:
        assert " › " in hit["citation"] and hit["score"] > 0


def test_tool_handles_empty_query():
    assert search_documentation("").data == []


def test_tool_reports_a_miss_honestly():
    reset_retriever()
    result = search_documentation("qzxv wvut plorf")
    assert result.data == [] and "may be outside it" in result.summary


def test_tool_is_registered_with_the_agent():
    from src.tools import PURE_TOOLS, TOOL_SCHEMAS

    assert "search_documentation" in PURE_TOOLS
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "search_documentation")
    assert schema["input_schema"]["required"] == ["query"]
    assert "WHY" in schema["description"]


def test_dispatch_routes_to_the_tool():
    from src.tools import dispatch

    result = dispatch("search_documentation", {"query": "look-ahead bias"})
    assert result.data and "look-ahead" in result.data[0]["text"].lower()


# --- optional: embedding backend --------------------------------------------

def test_embedding_backend_if_available():
    pytest.importorskip("sentence_transformers")
    from src.rag.retriever import EmbeddingRetriever

    r = EmbeddingRetriever(chunk_markdown(SAMPLE, "s.md"))
    hits = r.search("sending an order straight to the market rather than resting it", k=1)
    # Paraphrase with no shared vocabulary — the case embeddings exist for.
    assert hits and "Order routing" in hits[0].chunk.citation
