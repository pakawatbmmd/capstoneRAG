from __future__ import annotations

from .gate import AlwaysAnswer
from .rerank import NoRerank
from .types import ABSTAIN_PHRASE, OMISSION_NOTE, Claim, RAGResult

_RED_FLAGS = {"contradicted", "number_mismatch", "fabricated"}


def _unique_citations(claims: list[Claim]) -> list[str]:
    return list(dict.fromkeys(c.chunk_id for c in claims))


class RAGPipeline:
    def __init__(
        self,
        retriever,
        reranker=None,
        gate=None,
        generator=None,
        verifier=None,
        retrieve_k: int = 20,
        top_k: int = 3,
        extractive_fallback: bool = True,
    ):
        self.retriever = retriever
        self.reranker = NoRerank() if reranker is None else reranker
        self.gate = AlwaysAnswer() if gate is None else gate
        self.generator = generator
        self.verifier = verifier
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.extractive_fallback = extractive_fallback

    def answer(self, question: str) -> RAGResult:
        hits = self.reranker.rerank(
            question, self.retriever.search(question, self.retrieve_k), self.top_k
        )
        retrieved_ids = [h.chunk.id for h in hits]

        def result(answer, abstained, citations, reason, verdicts=(), omitted=0):
            return RAGResult(
                question=question,
                answer=answer,
                abstained=abstained,
                retrieved_ids=retrieved_ids,
                citations=citations,
                reason=reason,
                verdicts=list(verdicts),
                omitted=omitted,
            )

        def abstain(reason, verdicts=(), omitted=0):
            return result(ABSTAIN_PHRASE, True, [], reason, verdicts, omitted)

        if not hits or not self.gate.decide(question, hits).answerable:
            return abstain("gate")

        top = hits[0].chunk
        if self.generator is None:
            return result(top.text, False, [top.id], "extractive")

        draft = self.generator.generate(question, hits)
        if draft.abstained:
            return abstain("llm_abstained")
        if self.verifier is None:
            return result(draft.answer, False, _unique_citations(draft.claims), "unverified")

        verdicts = self.verifier.verify(draft.claims, {h.chunk.id: h.chunk for h in hits})
        accepted = [v.claim for v in verdicts if v.accepted]
        omitted = len(verdicts) - len(accepted)
        if accepted and not any(v.verdict in _RED_FLAGS for v in verdicts):
            answer = " ".join(c.text for c in accepted)
            if omitted:
                return result(
                    f"{answer} {OMISSION_NOTE}", False, _unique_citations(accepted), "partial", verdicts, omitted
                )
            return result(answer, False, _unique_citations(accepted), "verified", verdicts)
        if self.extractive_fallback:
            answer = f'Guideline text [{top.id}]: "{top.text}"'
            return result(answer, False, [top.id], "fallback_extractive", verdicts, omitted)
        return abstain("verification_failed", verdicts, omitted)
