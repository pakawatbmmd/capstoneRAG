from __future__ import annotations

from typing import Callable, Protocol

from .types import GateDecision, Hit


class Gate(Protocol):
    def decide(self, query: str, hits: list[Hit]) -> GateDecision: ...


class AlwaysAnswer:
    def decide(self, query: str, hits: list[Hit]) -> GateDecision:
        return GateDecision(answerable=bool(hits), score=hits[0].score if hits else 0.0)


class ThresholdGate:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def decide(self, query: str, hits: list[Hit]) -> GateDecision:
        score = hits[0].score if hits else 0.0
        return GateDecision(answerable=bool(hits) and score >= self.threshold, score=score)


class ProbabilityGate:
    def __init__(self, prob_fn: Callable[[str, list[str]], float], threshold: float = 0.5):
        self.prob_fn = prob_fn
        self.threshold = threshold

    def decide(self, query: str, hits: list[Hit]) -> GateDecision:
        if not hits:
            return GateDecision(answerable=False, score=0.0)
        p = self.prob_fn(query, [h.chunk.embed_text() for h in hits])
        return GateDecision(answerable=p >= self.threshold, score=p)
