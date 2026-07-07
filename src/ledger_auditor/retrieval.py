"""Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with
Reciprocal Rank Fusion (RRF).

Why hybrid: contract questions mix exact terms ("security deposit") that BM25
nails with paraphrases ("how much did I put down when I moved in?") that only
dense retrieval catches. RRF fuses the two rankings without score calibration.

The dense backend is pluggable:
  - SentenceTransformerBackend (all-MiniLM-L6-v2) when sentence-transformers
    is installed — the default for real use.
  - TfidfBackend (scikit-learn, char+word n-grams) as a dependency-light
    fallback used in CI so tests never need a model download.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9$%.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ------------------------------------------------------------- dense backends
class TfidfBackend:
    """Dependency-light dense-ish backend (word+char n-gram TF-IDF, cosine)."""
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        self._vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ])
        self._matrix = None

    def fit(self, texts: list[str]):
        self._matrix = self._vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self._vec.transform([query])
        return cosine_similarity(qv, self._matrix)[0]


class SentenceTransformerBackend:
    """Real dense embeddings. Requires `pip install .[dense]`."""
    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._emb = None

    def fit(self, texts: list[str]):
        self._emb = self._model.encode(texts, normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        q = self._model.encode([query], normalize_embeddings=True)
        return (self._emb @ q.T).ravel()


def default_dense_backend():
    try:
        return SentenceTransformerBackend()
    except Exception:
        return TfidfBackend()


# ------------------------------------------------------------------ retriever
class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense_backend=None, rrf_k: int = 60):
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [c.text for c in chunks]
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.dense = dense_backend or default_dense_backend()
        self.dense.fit(texts)

    # individual rankers (exposed separately so the eval can benchmark them)
    def bm25_ranking(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1])

    def dense_ranking(self, query: str) -> list[int]:
        return list(np.argsort(self.dense.scores(query))[::-1])

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               doc_filter: str | None = None) -> list[RetrievalResult]:
        if mode == "bm25":
            order, fused = self.bm25_ranking(query), None
        elif mode == "dense":
            order, fused = self.dense_ranking(query), None
        else:  # hybrid: reciprocal rank fusion
            fused = np.zeros(len(self.chunks))
            for ranking in (self.bm25_ranking(query), self.dense_ranking(query)):
                for rank, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (self.rrf_k + rank + 1)
            order = list(np.argsort(fused)[::-1])

        results = []
        for idx in order:
            chunk = self.chunks[idx]
            if doc_filter and chunk.doc_id != doc_filter:
                continue
            score = float(fused[idx]) if fused is not None else 0.0
            results.append(RetrievalResult(chunk=chunk, score=score, rank=len(results) + 1))
            if len(results) >= k:
                break
        return results
