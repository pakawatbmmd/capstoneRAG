"""The chosen pipeline, hybrid_llm, behind one call: load the models once, then ask(question) -> API response dict.

hybrid_llm = BM25 + bge-m3 (RRF) -> bge-reranker top 3 -> answerability gate (tau) -> Qwen3-4B JSON claims
             -> verifier (verbatim quote, numbers in quote, NLI support >= auto_accept) -> answer or "Not in the guidelines"
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guideline_rag import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LLMGenerator,
    ScoreFnReranker,
    abstain_layer,
    guarded_pipeline,
    load_chunks,
)

PIPELINE = "hybrid_llm"
MODELS = {
    "embedder": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "generator": "Qwen/Qwen3-4B-Instruct-2507",
    "verifier": "cross-encoder/nli-deberta-v3-base",
}
DEFAULTS = {"tau": 0.004, "auto_accept": 0.7}  # tuned on the train split (store/nice/taus.json, auto_accept.json)
TOP_K = 3
DISCLAIMER = (
    "Decision support only. Answers come from NICE guidelines and must be checked against the cited "
    "recommendation; clinical judgement applies."
)


def _status(result) -> str:
    if result.abstained:
        return "not_in_guidelines"
    return "partial" if result.reason == "partial" else "answered"


class Engine:
    """Real models on the GPU. Construct cheaply, call load() once (slow), then ask() from any thread."""

    mock = False

    def __init__(self, store: Path = ROOT / "store" / "nice", guidelines: Path = ROOT / "data" / "nice" / "guidelines.json"):
        self.store = Path(store)
        meta = json.loads(Path(guidelines).read_text(encoding="utf-8"))
        self.codes = meta["codes"]  # code -> source name
        self.code_of = {name: code for code, name in self.codes.items()}
        self.tau = self._setting("taus.json", "tau")
        self.auto_accept = self._setting("auto_accept.json", "auto_accept")
        self.chunks, self.by_id = [], {}
        self.pipeline = None
        self.status, self.error = "loading", None
        self._lock = threading.Lock()

    def _setting(self, filename: str, key: str) -> float:
        path = self.store / filename
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8")).get(PIPELINE, DEFAULTS[key]))
        return DEFAULTS[key]

    def load(self) -> None:
        try:
            chunks_file = self.store / "chunks.jsonl"
            if not chunks_file.exists():
                raise FileNotFoundError(
                    f"{chunks_file} not found: build the store with "
                    "`python examples/ingest.py --name nice Guidelines/*.pdf`, or run with RAG_MOCK=1"
                )
            self.chunks = load_chunks(chunks_file)
            self.by_id = {c.id: c for c in self.chunks}
            from guideline_rag.adapters import CrossEncoderScorer, HFPipelineLLM, NLIJudge, SentenceTransformerEmbedder

            dense = DenseRetriever(self.chunks, SentenceTransformerEmbedder(MODELS["embedder"]),
                                   store_dir=self.store / "dense_bge-m3", embedder_name=MODELS["embedder"])
            self.pipeline = guarded_pipeline(
                HybridRetriever([BM25Retriever(self.chunks), dense]),
                tau=self.tau,
                reranker=ScoreFnReranker(CrossEncoderScorer(MODELS["reranker"])),
                generator=LLMGenerator(HFPipelineLLM(MODELS["generator"])),
                nli_judge=NLIJudge(MODELS["verifier"]),
                auto_accept=self.auto_accept,
                top_k=TOP_K,
            )
            self.status = "ready"
        except Exception as exc:  # reported through /api/health instead of crashing the server
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def ref(self, chunk_id: str) -> str:
        chunk = self.by_id.get(chunk_id)
        if chunk is None:
            return chunk_id
        code = self.code_of.get(chunk.source, chunk.source)
        return f"{code}-{chunk.metadata['rec']}" if chunk.metadata.get("rec") else f"{code}-{chunk.id}"

    def citation(self, chunk_id: str) -> dict:
        chunk = self.by_id[chunk_id]
        return {
            "ref": self.ref(chunk_id),
            "guideline_code": self.code_of.get(chunk.source, chunk.source),
            "guideline": chunk.source,
            "section": chunk.section.replace(" > ", " › "),
            "recommendation": chunk.metadata.get("rec"),
            "page": chunk.page,
            "version": chunk.version,
            "text": chunk.text,
        }

    def ask(self, question: str) -> dict:
        with self._lock:
            started = time.perf_counter()
            result = self.pipeline.answer(question)
            latency_ms = round((time.perf_counter() - started) * 1000)
        claims = [
            {
                "text": v.claim.text,
                "quote": v.claim.quote,
                "ref": self.ref(v.claim.chunk_id),
                "verdict": v.verdict,
                "confidence": None if v.confidence is None else round(float(v.confidence), 3),
                "shown": v.accepted and not result.abstained,
            }
            for v in result.verdicts
        ]
        return {
            "question": question,
            "status": _status(result),
            "answer": result.answer,
            "abstained": result.abstained,
            "blocked_by": abstain_layer(result) or None,
            "citations": [self.citation(c) for c in result.citations if c in self.by_id],
            "claims": claims,
            "omitted_claims": result.omitted,
            "retrieved": [self.ref(c) for c in result.retrieved_ids],
            "latency_ms": latency_ms,
            "pipeline": PIPELINE,
            "mock": self.mock,
            "disclaimer": DISCLAIMER,
        }

    def guidelines(self) -> list[dict]:
        out = []
        for code, name in self.codes.items():
            mine = [c for c in self.chunks if c.source == name]
            out.append({
                "code": code,
                "name": name,
                "version": mine[0].version if mine else None,
                "chunks": len(mine),
                "recommendations": sum(1 for c in mine if c.metadata.get("rec")),
            })
        return out

    def health(self) -> dict:
        return {
            "status": self.status,
            "pipeline": PIPELINE,
            "mock": self.mock,
            "models": MODELS,
            "settings": {"tau": self.tau, "auto_accept": self.auto_accept, "top_k": TOP_K},
            "chunks": len(self.chunks),
            "error": self.error,
        }


class MockEngine(Engine):
    """Canned, licence-free responses with no models, GPU or store, for building and testing the frontend.

    Questions mentioning "reposition" get an answered response, "dressing" a partial one, anything else
    "Not in the guidelines". Shapes match the real engine exactly.
    """

    mock = True
    _GUIDELINES = [
        {"code": "CG179", "name": "Pressure ulcers", "version": "23 April 2014", "chunks": 101, "recommendations": 95},
        {"code": "NG249", "name": "Fall", "version": "29 April 2025", "chunks": 69, "recommendations": 40},
        {"code": "CG32", "name": "Nutrition support", "version": "4 August 2017", "chunks": 97, "recommendations": 91},
        {"code": "CG183", "name": "Drug allergy", "version": "3 September 2014", "chunks": 47, "recommendations": 34},
    ]

    def __init__(self):
        super().__init__(store=ROOT / "store" / "__mock__")
        self.chunks = [None] * 314

    def load(self) -> None:
        self.status = "ready"

    def guidelines(self) -> list[dict]:
        return [dict(g) for g in self._GUIDELINES]

    def ask(self, question: str) -> dict:
        q = question.lower()
        base = {"question": question, "latency_ms": 5, "pipeline": PIPELINE, "mock": True, "disclaimer": DISCLAIMER}
        citation = {
            "ref": "CG179-1.1.9", "guideline_code": "CG179", "guideline": "Pressure ulcers",
            "section": "Recommendations › 1.1 Prevention: adults › Repositioning", "recommendation": "1.1.9",
            "page": 13, "version": "23 April 2014",
            "text": "[Mock mode: the recommendation's guideline text appears here in real mode.]",
        }
        claim = {"text": "Adults at high risk of pressure ulcers should change position at least every 4 hours.",
                 "quote": "[mock quote]", "ref": "CG179-1.1.9", "verdict": "verified", "confidence": 0.97, "shown": True}
        if "reposition" in q:
            return {**base, "status": "answered", "answer": claim["text"], "abstained": False, "blocked_by": None,
                    "citations": [citation], "claims": [claim], "omitted_claims": 0,
                    "retrieved": ["CG179-1.1.9", "CG179-1.1.8", "CG179-1.2.8"]}
        if "dressing" in q:
            weak = {**claim, "text": "[mock unverifiable claim]", "verdict": "unsupported", "confidence": 0.41, "shown": False}
            return {**base, "status": "partial",
                    "answer": claim["text"] + " (Some statements could not be verified against the guideline and were left out.)",
                    "abstained": False, "blocked_by": None, "citations": [citation], "claims": [claim, weak],
                    "omitted_claims": 1, "retrieved": ["CG179-1.1.9", "CG179-1.4.12", "CG179-1.4.13"]}
        return {**base, "status": "not_in_guidelines", "answer": "Not in the guidelines", "abstained": True,
                "blocked_by": "LLM: not in the guidelines", "citations": [], "claims": [], "omitted_claims": 0,
                "retrieved": ["CG183-1.2.3", "CG183-1.2.1", "CG183-1.2.6"]}
