"""RagService: the one object the web app talks to. Wraps examples/systems.py; never changes guideline_rag."""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from systems import EXTERNAL, SYSTEMS, Components, build_system, load_taus  # noqa: E402

from guideline_rag import abstain_layer  # noqa: E402

DEFAULT = "hybrid_llm"  # the chosen pipeline (docs/report.html, final verdict)
FALLBACK = "hybrid"  # best pipeline that runs without a GPU
LLM_SYSTEMS = {"hybrid_llm", "hybrid_graph_llm", "jev_llm"}
DESCRIPTIONS = {
    "bm25": "BM25 keyword search, score τ guard",
    "vector": "all-MiniLM-L6-v2 dense search, cosine τ guard (lab09 baseline)",
    "vector_bge": "bge-m3 dense search, cosine τ guard",
    "hybrid": "BM25 + bge-m3 (RRF), bge-reranker, reranker τ guard",
    "graph": "Entity graph + Personalized PageRank, PPR τ guard",
    "hybrid_graph": "BM25 + bge-m3 + graph (RRF), bge-reranker, reranker τ guard",
    "hybrid_llm": "Recommended: hybrid + Qwen3-4B answer + NLI claim verifier",
    "hybrid_graph_llm": "hybrid_graph + Qwen3-4B answer + NLI claim verifier (graph adds nothing measurable)",
    "jev": "hybrid_graph retrieval, Jev answerability",
    "jev_llm": "Jev + Qwen3-4B + Jev verifier",
}
UNKNOWN_DESCRIPTION = "Backend system not described by this frontend yet"
NO_GPU = "Needs a CUDA GPU for Qwen3-4B; none was found on this machine."
NOT_IN_FRONTEND = "Not supported by this frontend yet."
EXTERNAL_REASON = "Sends guideline text to the external TypeSafe API (NICE licence); disabled."
BLOCKED_BY = {"gate": "answerability gate", "llm_abstained": "LLM: not in the guidelines",
              "verification_failed": "verifier: claims not supported"}


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _load_auto_accept(store: Path) -> dict:
    path = store / "auto_accept.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class RagService:
    def __init__(self, components, taus: dict, builder=build_system, results_dir: Path | None = None,
                 auto_accept: dict | None = None, gpu: bool | None = None,
                 gold_path: Path | None = ROOT / "Test+answer" / "questions_nurse40.json"):
        self.c = components
        self.taus = taus
        self.builder = builder
        self.results_dir = results_dir
        self.auto_accept = auto_accept or {}
        self.gpu = gpu_available() if gpu is None else gpu
        self.gold_path = gold_path
        self.by_id = {ch.id: ch for ch in components.chunks}
        self._pipelines = {}
        self._lock = threading.Lock()

    @classmethod
    def from_store(cls, store: Path, lexicon: Path, results_dir: Path | None = None) -> "RagService":
        return cls(Components(store, lexicon), load_taus(store), results_dir=results_dir,
                   auto_accept=_load_auto_accept(store))

    def disabled_reason(self, name: str) -> str:
        if name in EXTERNAL:
            return EXTERNAL_REASON
        if name not in DESCRIPTIONS:
            return NOT_IN_FRONTEND
        if name in LLM_SYSTEMS and not self.gpu:
            return NO_GPU
        return ""

    def enabled(self) -> list[str]:
        return [n for n in SYSTEMS if not self.disabled_reason(n)]

    def default_system(self) -> str:
        return DEFAULT if DEFAULT in self.enabled() else FALLBACK

    def systems(self) -> list[dict]:
        default = self.default_system()
        return [{"name": n, "description": DESCRIPTIONS.get(n, UNKNOWN_DESCRIPTION), "tau": self.taus.get(n),
                 "enabled": not self.disabled_reason(n), "disabled_reason": self.disabled_reason(n),
                 "default": n == default} for n in SYSTEMS]

    def warm(self) -> None:
        """Build every enabled pipeline now so the first question is not slow."""
        for name in self.enabled():
            self._pipeline(name)

    def _pipeline(self, name: str):
        reason = self.disabled_reason(name) if name in SYSTEMS or name in DESCRIPTIONS else f"unknown system {name!r}"
        if reason:
            raise ValueError(reason)
        if name not in self._pipelines:
            self._pipelines[name] = self.builder(self.c, name, self.taus[name],
                                                 auto_accept=self.auto_accept.get(name, 0.8))
        return self._pipelines[name]

    def _passage(self, cid: str) -> dict:
        ch = self.by_id[cid]
        return {"id": ch.id, "text": ch.text, "source": ch.source, "section": ch.section, "page": ch.page}

    def ask(self, question: str, system: str) -> dict:
        with self._lock:  # Cached memo dicts in systems.py are not thread-safe
            pipeline = self._pipeline(system)
            t0 = time.perf_counter()
            res = pipeline.answer(question)
            seconds = round(time.perf_counter() - t0, 2)
        return {
            "system": system, "tau": self.taus[system], "question": question,
            "answer": res.answer, "abstained": res.abstained, "blocked_by": abstain_layer(res), "reason": res.reason,
            "citations": [self._passage(c) for c in res.citations],
            "top3": [self._passage(c) for c in res.retrieved_ids[:3]],
            "verdicts": [{"claim": v.claim.text, "verdict": v.verdict, "confidence": v.confidence} for v in res.verdicts],
            "omitted": res.omitted, "seconds": float(seconds),
        }

    def results(self) -> dict:
        """Saved evaluation from results/<store>/: the train metrics table and per-question predictions by split."""
        out = {"ablation_md": "", "table_split": "", "predictions": {}}
        if self.results_dir is None or not self.results_dir.exists():
            return out
        tables = [self._split_table(self.results_dir / name, "train") for name in ("metrics.md", "metrics_nograph.md")]
        header, rows = None, []
        for table in tables:
            if table:
                header = header or table[:2]
                rows += [r for r in table[2:] if r not in rows]
        if header:
            out["ablation_md"] = "\n".join(header + rows) + "\n"
            out["table_split"] = "train"
        questions = {}
        if self.gold_path and Path(self.gold_path).exists():
            questions = {q["id"]: q["question"] for q in json.loads(Path(self.gold_path).read_text(encoding="utf-8"))}
        for name in ("per_question.csv", "per_question_nograph.csv"):
            path = self.results_dir / name
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    by_system = out["predictions"].setdefault(row["split"], {}).setdefault(row["system"], [])
                    if any(p["id"] == row["qid"] for p in by_system):
                        continue
                    abstained = row["abstained"] == "True"
                    by_system.append({
                        "id": row["qid"], "question": questions.get(row["qid"], ""), "answer": row["answer"],
                        "blocked_by": BLOCKED_BY.get(row["reason"], "") if abstained else "",
                        "top3": json.loads(row["retrieved_ids"])[:3],
                    })
        return out

    @staticmethod
    def _split_table(path: Path, split: str) -> list[str]:
        """Markdown table lines under the '### <split>' heading of an evaluate_systems.py metrics file."""
        if not path.exists():
            return []
        lines, inside = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):
                inside = re.match(rf"### {split}\b", line) is not None
            elif inside and line.startswith("|"):
                lines.append(line)
        return lines
