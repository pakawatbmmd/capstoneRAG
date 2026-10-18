"""Evaluate every RAG system on the team's gold question set: tune τ on train, check val, run test once.

    python examples/evaluate_systems.py                                   # NICE, Test+answer/questions_nurse40.json
    python examples/evaluate_systems.py --systems hybrid_graph hybrid_graph_llm

Gold file: a JSON list of {"id", "split" (train|val|test), "guideline" (NICE code), "answerable",
"gold_ids" (["CG179-1.1.9", ...]; [] = the right output is "Not in the guidelines"), "risk" (standard|high), ...}.
Scoring is at recommendation level: a chunk counts as "<code>-<rec>", so any window of a split
recommendation retrieves it.

Metrics per split:
    Recall@3        answerable questions: share of gold recommendations in the top 3
    Guideline@1     all questions: the top-1 retrieved chunk comes from the question's guideline
    Abstention acc. all questions: abstained exactly when the question is not answerable
    Course score    0.5 * Recall@3 + 0.5 * Abstention acc. (lab09 metric)
    Unsafe          not-answerable questions that were answered anyway (lower is better)
    Unsafe (high)   the same, restricted to high-risk questions
    Verified claims LLM claims that passed verification
"""

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

from systems import ROOT, SYSTEMS, Components, build_system, load_taus

from guideline_rag import EvalItem, evaluate

GRIDS = {
    "bm25": [x / 2 for x in range(0, 61)],
    "graph": [x / 400 for x in range(0, 81)],
}
UNIT_GRID = sorted({x / 100 for x in range(0, 101, 2)} | {x / 1000 for x in range(0, 50, 2)})
AUTO_ACCEPT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]  # verifier confidence needed to accept a claim (LLM systems)
LLM_SYSTEMS = {"hybrid_graph_llm", "jev_llm"}
DEFAULT_SYSTEMS = ["bm25", "vector", "vector_bge", "hybrid", "graph", "hybrid_graph", "hybrid_graph_llm"]


ACCEPT = {}  # system -> tuned verifier auto_accept
FIRST_SECONDS = {}  # (system, question) -> latency of the first, uncached answer; later sweeps hit model caches


class RefPipeline:
    """Report retrieved chunks and citations as '<code>-<rec>' references, and time each answer."""

    def __init__(self, pipeline, ref_of, name):
        self.pipeline = pipeline
        self.ref_of = ref_of
        self.name = name
        self.raw = {}

    def answer(self, question):
        t0 = time.perf_counter()
        res = self.pipeline.answer(question)
        FIRST_SECONDS.setdefault((self.name, question), time.perf_counter() - t0)
        self.raw[question] = res
        return dataclasses.replace(
            res,
            retrieved_ids=[self.ref_of[i] for i in res.retrieved_ids],
            citations=[self.ref_of[i] for i in res.citations],
        )


def mean(values):
    return sum(values) / len(values) if values else None


def score(c, name, tau, records, ref_of, auto_accept=0.8):
    pipe = RefPipeline(build_system(c, name, tau, auto_accept=auto_accept), ref_of, name)
    items = [EvalItem(r["id"], r["question"], r["gold"]) for r in records]
    summary = evaluate(pipe, items)
    by_id = {r["id"]: r for r in records}
    for row in summary["rows"]:
        q = by_id[row["qid"]]
        top1 = row["retrieved_ids"][0] if row["retrieved_ids"] else ""
        row.update(guideline=q["guideline"], risk=q["risk"], split=q["split"],
                   guideline_at_1=top1.split("-", 1)[0] == q["guideline"],
                   seconds=round(FIRST_SECONDS[(name, q["question"])], 2),
                   claims=[[v.verdict, None if v.confidence is None else round(v.confidence, 3),
                            ref_of.get(v.claim.chunk_id, v.claim.chunk_id), v.claim.text, v.claim.quote]
                           for v in pipe.raw[q["question"]].verdicts])
    rows = summary["rows"]
    high = [r for r in rows if not r["gold_ids"] and r["risk"] == "high"]
    summary.update(
        guideline_at_1=mean([float(r["guideline_at_1"]) for r in rows]),
        unsafe_high_risk=mean([float(r["unsafe"]) for r in high]),
        seconds=mean([r["seconds"] for r in rows]),
    )
    return summary


def tune(c, name, records, ref_of):
    """Best (τ, auto_accept) by train course score; ties go to the higher, more cautious values."""
    grid = GRIDS.get(name, UNIT_GRID)
    accepts = AUTO_ACCEPT_GRID if name in LLM_SYSTEMS else [0.8]
    best = max((score(c, name, t, records, ref_of, a)["course_score"], t, a) for t in grid for a in accepts)
    return best[1], best[2]


def fmt(v, pct=False):
    return "-" if v is None else (f"{100 * v:.0f}%" if pct else f"{v:.2f}")


def table(results, split):
    head = ("| System | τ | Recall@3 | Guideline@1 | Abstention acc. | Course score | Unsafe | Unsafe (high-risk) "
            "| Verified claims | s/question (first call) |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for name, (tau, s) in results.items():
        lines.append(
            f"| {name} | {tau:g}{'' if name not in LLM_SYSTEMS else f' / accept {ACCEPT[name]:g}'} | {fmt(s['recall_at_k'])} | {fmt(s['guideline_at_1'])} | {fmt(s['abstention_accuracy'])} "
            f"| **{fmt(s['course_score'])}** | {fmt(s['unsafe_answer_rate'], True)} | {fmt(s['unsafe_high_risk'], True)} "
            f"| {fmt(s['verified_claim_rate'], True)} | {s['seconds']:.2f} |"
        )
    n = len(next(iter(results.values()))[1]["rows"])
    return f"### {split} (n={n})\n\n" + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="nice")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "nice" / "lexicon.json")
    parser.add_argument("--gold", type=Path, default=ROOT / "Test+answer" / "questions_nurse40.json")
    parser.add_argument("--guidelines", type=Path, default=ROOT / "data" / "nice" / "guidelines.json")
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=SYSTEMS)
    args = parser.parse_args()

    store = ROOT / "store" / args.store
    out = ROOT / "results" / args.store
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.guidelines.read_text(encoding="utf-8"))
    code_of = {source: code for code, source in meta["codes"].items()}
    c = Components(store, args.lexicon)
    ref_of = {
        ch.id: f"{code_of[ch.source]}-{ch.metadata['rec']}" if ch.metadata.get("rec")
        else f"{code_of[ch.source]}-text-{ch.id}"
        for ch in c.chunks
    }
    known = set(ref_of.values())
    records = json.loads(args.gold.read_text(encoding="utf-8"))
    for r in records:
        r["gold"] = sorted({meta["aliases"].get(g, g) for g in r["gold_ids"]})
        missing = [g for g in r["gold"] if g not in known]
        if missing:
            raise ValueError(f"{r['id']}: gold {missing} not in the store")
    splits = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}

    taus = load_taus(store)
    for name in args.systems:
        taus[name], ACCEPT[name] = tune(c, name, splits["train"], ref_of)
        print(f"τ[{name}] = {taus[name]:g}, auto_accept = {ACCEPT[name]:g}  (tuned on {len(splits['train'])} train questions)")
    (store / "taus.json").write_text(json.dumps({s: taus[s] for s in args.systems}, indent=1), encoding="utf-8")
    (store / "auto_accept.json").write_text(json.dumps(ACCEPT, indent=1), encoding="utf-8")

    report = []
    per_question = []
    for split in ("train", "val", "test"):
        results = {name: (taus[name], score(c, name, taus[name], splits[split], ref_of, ACCEPT[name])) for name in args.systems}
        report.append(table(results, split + (" — τ tuned here, optimistic" if split == "train" else "")))
        for name, (_, s) in results.items():
            for row in s["rows"]:
                per_question.append({"system": name, **{k: row[k] for k in (
                    "split", "qid", "guideline", "risk", "gold_ids", "retrieved_ids", "abstained", "reason",
                    "recall", "guideline_at_1", "abstain_correct", "unsafe", "seconds", "answer", "claims")}})
    all_results = {name: (taus[name], score(c, name, taus[name], records, ref_of, ACCEPT[name])) for name in args.systems}
    report.append(table(all_results, "all 40 (train+val+test)"))

    text = "\n\n".join(report)
    print("\n" + text)
    (out / "metrics.md").write_text(text + "\n", encoding="utf-8")
    with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_question[0]))
        w.writeheader()
        for row in per_question:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in row.items()})
    print(f"\n-> {out / 'metrics.md'}\n-> {out / 'per_question.csv'}")


if __name__ == "__main__":
    main()
