"""Every RAG system in the comparison, built from a persistent store (store/<name>/), each behind its guardrail.

    name            retrieval                                   rerank            answer      guard -> "Not in the guidelines"
    bm25            BM25                                        -                 extractive  score τ
    vector          all-MiniLM-L6-v2                            -                 extractive  cosine τ
    hybrid          BM25 + bge-m3 (RRF)                         bge-reranker-v2   extractive  reranker τ
    graph           entity graph + Personalized PageRank        -                 extractive  PPR τ
    hybrid_graph    BM25 + bge-m3 + graph (RRF)                 bge-reranker-v2   extractive  reranker τ
    jev             BM25 + bge-m3 + graph (RRF)                 Jev               extractive  Jev answerability
    hybrid_graph_llm  as hybrid_graph                           bge-reranker-v2   Qwen3-4B    reranker τ + LLM + NLI verifier
    jev_llm         as jev                                      Jev               Qwen3-4B    Jev answerability + LLM + Jev verifier

Jev systems send guideline text to the TypeSafe API. NICE content outside the UK is licensed for personal
study only, so they are built only with allow_external=True.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guideline_rag import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LLMGenerator,
    ScoreFnReranker,
    guarded_pipeline,
    load_chunks,
)
from guideline_rag.graph import GraphRetriever, LexiconExtractor, load_lexicon

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
KEY_FILE = Path.home() / ".config" / "capstone-rag" / "typesafe.env"
SYSTEMS = ["bm25", "vector", "hybrid", "graph", "hybrid_graph", "jev", "hybrid_graph_llm", "jev_llm"]
NEEDS_MODELS = {"vector", "hybrid", "hybrid_graph", "jev", "hybrid_graph_llm", "jev_llm"}
EXTERNAL = {"jev", "jev_llm"}

# Untuned starting points; tune_taus() in evaluate_systems.py overwrites them from the dev set (store/<name>/taus.json).
DEFAULT_TAUS = {"bm25": 8.0, "vector": 0.45, "hybrid": 0.1, "graph": 0.05, "hybrid_graph": 0.1, "jev": 0.5,
                "hybrid_graph_llm": 0.1, "jev_llm": 0.5}


class Cached:
    """Memoise search()/rerank() so threshold sweeps and repeated systems reuse model calls."""

    def __init__(self, inner):
        self.inner = inner
        self.memo = {}

    def search(self, query, k):
        key = ("search", query, k)
        if key not in self.memo:
            self.memo[key] = self.inner.search(query, k)
        return self.memo[key]

    def rerank(self, query, hits, k):
        key = ("rerank", query, tuple(h.chunk.id for h in hits), k)
        if key not in self.memo:
            self.memo[key] = self.inner.rerank(query, hits, k)
        return self.memo[key]


class CachedJev:
    """Jev judge with memoised calls (cost and latency), same interface as JevJudge."""

    def __init__(self, jev):
        self.jev = jev
        self.memo = {}

    def _call(self, name, *args):
        key = (name, args)
        if key not in self.memo:
            self.memo[key] = getattr(self.jev, name)(*(list(a) if isinstance(a, tuple) else a for a in args))
        return self.memo[key]

    def relevance(self, query, texts):
        return self._call("relevance", query, tuple(texts))

    def answerable(self, query, texts):
        return self._call("answerable", query, tuple(texts))

    def judge(self, claim_text, chunk_text):
        return self._call("judge", claim_text, chunk_text)


def load_typesafe_key():
    if "TYPESAFE_API_KEY" not in os.environ and KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                os.environ["TYPESAFE_API_KEY"] = value.strip()


def load_taus(store: Path) -> dict:
    path = store / "taus.json"
    taus = dict(DEFAULT_TAUS)
    if path.exists():
        taus.update(json.loads(path.read_text(encoding="utf-8")))
    return taus


class Components:
    """Shared, lazily built retrievers and models so several systems reuse one GPU copy of each model."""

    def __init__(self, store: Path, lexicon_path: Path):
        self.store = store
        self.chunks = load_chunks(store / "chunks.jsonl")
        self.lexicon = load_lexicon(lexicon_path)
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            self._cache[name] = getattr(self, f"_make_{name}")()
        return self._cache[name]

    def _make_bm25(self):
        return BM25Retriever(self.chunks)

    def _make_graph(self):
        return GraphRetriever(self.chunks, LexiconExtractor(self.lexicon), lexicon=self.lexicon)

    def _make_minilm(self):
        from guideline_rag.adapters import SentenceTransformerEmbedder

        name = "sentence-transformers/all-MiniLM-L6-v2"
        return DenseRetriever(self.chunks, SentenceTransformerEmbedder(name),
                              store_dir=self.store / "dense_minilm", embedder_name=name)

    def _make_bge(self):
        from guideline_rag.adapters import SentenceTransformerEmbedder

        name = "BAAI/bge-m3"
        return DenseRetriever(self.chunks, SentenceTransformerEmbedder(name),
                              store_dir=self.store / "dense_bge-m3", embedder_name=name)

    def _make_hybrid(self):
        return Cached(HybridRetriever([self.get("bm25"), self.get("bge")]))

    def _make_hybrid_graph(self):
        return Cached(HybridRetriever([self.get("bm25"), self.get("bge"), self.get("graph")]))

    def _make_cross(self):
        from guideline_rag.adapters import CrossEncoderScorer

        return Cached(ScoreFnReranker(CrossEncoderScorer("BAAI/bge-reranker-v2-m3")))

    def _make_nli(self):
        from guideline_rag.adapters import NLIJudge

        return NLIJudge()

    def _make_llm(self):
        from guideline_rag.adapters import HFPipelineLLM

        return LLMGenerator(HFPipelineLLM(LLM_MODEL))

    def _make_jev(self):
        from guideline_rag.adapters import JevJudge

        load_typesafe_key()
        return CachedJev(JevJudge())

    def _make_jev_rerank(self):
        return Cached(ScoreFnReranker(self.get("jev").relevance))


def build_system(c: Components, name: str, tau: float, allow_external: bool = False):
    if name in EXTERNAL and not allow_external:
        raise PermissionError(f"'{name}' sends guideline text to the TypeSafe API; pass allow_external=True")
    if name == "bm25":
        return guarded_pipeline(c.get("bm25"), tau=tau)
    if name == "vector":
        return guarded_pipeline(c.get("minilm"), tau=tau)
    if name == "graph":
        return guarded_pipeline(c.get("graph"), tau=tau)
    if name == "hybrid":
        return guarded_pipeline(c.get("hybrid"), tau=tau, reranker=c.get("cross"))
    if name == "hybrid_graph":
        return guarded_pipeline(c.get("hybrid_graph"), tau=tau, reranker=c.get("cross"))
    if name == "jev":
        return guarded_pipeline(c.get("hybrid_graph"), tau=tau, reranker=c.get("jev_rerank"), jev=c.get("jev"))
    if name == "hybrid_graph_llm":
        return guarded_pipeline(c.get("hybrid_graph"), tau=tau, reranker=c.get("cross"),
                                generator=c.get("llm"), nli_judge=c.get("nli"))
    if name == "jev_llm":
        return guarded_pipeline(c.get("hybrid_graph"), tau=tau, reranker=c.get("jev_rerank"),
                                generator=c.get("llm"), jev=c.get("jev"))
    raise ValueError(f"unknown system {name!r}; choose from {SYSTEMS}")
