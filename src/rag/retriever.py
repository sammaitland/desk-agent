"""Retrieval over the documentation corpus.

Two backends behind one interface:

  * `LexicalRetriever` — TF-IDF over the chunks. No model download, no
    external service, deterministic. Good when queries share vocabulary with
    the documents, which for technical documentation they usually do.
  * `EmbeddingRetriever` — sentence-transformer embeddings with cosine
    similarity. Finds passages that mean the same thing in different words.
    Costs a one-time model download (~80 MB) and a few seconds to index.

The interface is the point. Which backend is right depends on the corpus and
the queries, and that is an empirical question — so the choice is a
configuration, and swapping it does not touch the tool or the agent.

Both return the same shape: ranked chunks with a score, so a caller can report
how confident the retrieval was rather than presenting the top hit as fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.rag.corpus import Chunk, load_corpus


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "score": round(float(self.score), 4),
            "text": self.chunk.text,
        }


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 4) -> list[Hit]: ...


class LexicalRetriever:
    """TF-IDF with cosine similarity. Zero dependencies beyond scikit-learn."""

    name = "lexical (tf-idf)"

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        # Unigrams and bigrams: "spread cap" and "limit order" are the units
        # of meaning in this domain, not "spread" and "cap" separately.
        self._vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                           stop_words="english")
        self._matrix = self._vectoriser.fit_transform(c.text for c in chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        vector = self._vectoriser.transform([query])
        scores = (self._matrix @ vector.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], scores[i]) for i in order if scores[i] > 0]


class EmbeddingRetriever:
    """Dense retrieval via sentence-transformers. Imported lazily."""

    name = "embedding (all-MiniLM-L6-v2)"

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self._model = SentenceTransformer(model_name)
        self.name = f"embedding ({model_name})"
        texts = [c.text for c in chunks]
        self._matrix = self._model.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 1))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._matrix @ vector
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], scores[i]) for i in order if scores[i] > 0.1]


def build_retriever(chunks: list[Chunk] | None = None, backend: str | None = None) -> Retriever:
    """Construct the configured retriever.

    Backend comes from RAG_BACKEND (`lexical` or `embedding`), defaulting to
    lexical so the system works with no model download. If `embedding` is
    requested but sentence-transformers is not installed, fall back to lexical
    and say so rather than failing — retrieval quality degrading is better than
    the documentation tool disappearing.
    """
    chunks = load_corpus() if chunks is None else chunks
    backend = (backend or os.getenv("RAG_BACKEND", "lexical")).lower()
    if backend == "embedding":
        try:
            return EmbeddingRetriever(chunks)
        except ImportError:
            import warnings
            warnings.warn("sentence-transformers not installed; using lexical retrieval. "
                          "pip install sentence-transformers to enable embeddings.")
    return LexicalRetriever(chunks)
