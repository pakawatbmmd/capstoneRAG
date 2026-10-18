"""Heavy-model adapters. Third-party imports are deferred so importing this module is free."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class CrossEncoderScorer:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, text) for text in texts])
        return [float(s) for s in scores]


class HFPipelineLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 1024):
        from transformers import pipeline

        self.generator = pipeline(
            "text-generation", model=model_name, device_map="auto", dtype="auto"
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        output = self.generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return output[0]["generated_text"][-1]["content"]


_NLI_TO_RELATION = {
    "entailment": "supports",
    "contradiction": "contradicts",
    "neutral": "says_nothing",
}


class NLIJudge:
    """Local CitationJudge: an NLI cross-encoder reading (passage, claim) as (premise, hypothesis)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", model=None):
        self.model_name = model_name
        self._model = model

    def judge(self, claim_text: str, chunk_text: str) -> tuple[str, float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        probs = np.asarray(
            self._model.predict([(chunk_text, claim_text)], apply_softmax=True)
        )[0]
        best = int(np.argmax(probs))
        label = str(self._model.config.id2label[best]).lower()
        return _NLI_TO_RELATION[label], float(probs[best])


class JevJudge:
    """TypeSafe System One (Jev) as reranker scorer, answerability gate and citation judge."""

    def __init__(self, client=None, model: str = "jev-1.13.0", noul_cls=None, choice_cls=None):
        self._client = client
        self.model = model
        self._noul_cls = noul_cls
        self._choice_cls = choice_cls

    def _ensure_sdk(self) -> None:
        if self._client is None or self._noul_cls is None or self._choice_cls is None:
            import typesafe_sdk

            if self._client is None:
                self._client = typesafe_sdk.TypeSafeClient()
            if self._noul_cls is None:
                self._noul_cls = typesafe_sdk.Noul
            if self._choice_cls is None:
                self._choice_cls = typesafe_sdk.Choice

    def _ask(self, state: dict, questions: dict):
        return self._client.system_one(state=state, questions=questions, model=self.model).answers

    def relevance(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self._ensure_sdk()
        keys = [f"p{i}" for i in range(len(texts))]
        questions = {
            key: self._noul_cls(
                instructions=(
                    f"Does the passage `passages.{key}` contain information that helps "
                    "answer the user's `question`?"
                )
            )
            for key in keys
        }
        answers = self._ask({"question": query, "passages": dict(zip(keys, texts))}, questions)
        return [answers[key].noul for key in keys]

    def answerable(self, query: str, texts: list[str]) -> float:
        self._ensure_sdk()
        question = self._noul_cls(
            instructions=(
                "Do the `passages` together contain enough information to fully answer "
                "the user's `question`?"
            )
        )
        answers = self._ask({"question": query, "passages": texts}, {"answerable": question})
        return answers["answerable"].noul

    def judge(self, claim_text: str, chunk_text: str) -> tuple[str, float]:
        self._ensure_sdk()
        question = self._choice_cls(
            instructions="How does the `passage` relate to the `claim`?",
            criteria={
                "supports": "The passage states or clearly implies the claim.",
                "contradicts": "The passage says something that conflicts with the claim.",
                "says_nothing": "The passage does not address the claim.",
            },
        )
        answers = self._ask({"claim": claim_text, "passage": chunk_text}, {"relation": question})
        answer = answers["relation"]
        return answer.choice, answer.confidence
