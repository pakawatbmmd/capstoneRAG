"""One abstention policy per RAG type: guideline-backed content or exactly ABSTAIN_PHRASE."""

from __future__ import annotations

from .gate import ProbabilityGate, ThresholdGate
from .pipeline import RAGPipeline
from .types import RAGResult
from .verify import Verifier

_ABSTAIN_LAYERS = {
    "gate": "answerability gate",
    "llm_abstained": "LLM: not in the guidelines",
    "verification_failed": "verifier: claims not supported",
}


def guarded_pipeline(
    retriever,
    *,
    tau: float,
    reranker=None,
    generator=None,
    jev=None,
    nli_judge=None,
    retrieve_k: int = 20,
    top_k: int = 3,
) -> RAGPipeline:
    """Jev decides answerability and verifies claims; without Jev, a score threshold and an NLI judge do."""
    if jev is not None:
        gate, judge = ProbabilityGate(jev.answerable, tau), jev
    else:
        gate, judge = ThresholdGate(tau), nli_judge

    verifier = None
    if generator is not None:
        if judge is None:
            raise ValueError("a generator needs `jev` or `nli_judge` to verify its claims")
        verifier = Verifier(judge=judge)

    return RAGPipeline(
        retriever,
        reranker=reranker,
        gate=gate,
        generator=generator,
        verifier=verifier,
        retrieve_k=retrieve_k,
        top_k=top_k,
        extractive_fallback=False,
    )


def abstain_layer(result: RAGResult) -> str:
    """Name the layer that blocked the answer, or "" if the result is an answer."""
    return _ABSTAIN_LAYERS.get(result.reason, "") if result.abstained else ""
