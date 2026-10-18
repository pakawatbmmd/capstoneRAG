from __future__ import annotations

from typing import Callable, Protocol

from .types import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class NoRerank:
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        return hits[:k]


class ScoreFnReranker:
    """Reorders hits by `score_fn(query, texts)`, e.g. a cross-encoder or a Jev Noul."""

    def __init__(self, score_fn: Callable[[str, list[str]], list[float]]):
        self.score_fn = score_fn

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.score_fn(query, [h.chunk.embed_text() for h in hits])
        if len(scores) != len(hits):
            raise ValueError(f"score_fn returned {len(scores)} scores for {len(hits)} texts")
        rescored = [Hit(h.chunk, float(s)) for h, s in zip(hits, scores)]
        return sorted(rescored, key=lambda h: -h.score)[:k]
