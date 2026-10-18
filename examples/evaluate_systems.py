"""Run the RAG systems on a question set; with gold labels, tune τ on dev, score val, print the ablation table.

    # NICE, no gold yet: answer every question with every system, save predictions for later scoring
    python examples/evaluate_systems.py --store nice --questions data/nice/questions.json

    # NICE with gold: tune τ on the dev split only, then report dev and val (val is used once)
    python examples/evaluate_systems.py --store nice --questions data/nice/questions.json --gold data/nice/gold.json

    # Jev side study on the licence-clean L1 synthetic data (gold = our dev set)
    python examples/evaluate_systems.py --store l1 --lexicon data/rag/lexicon.json \
        --questions data/rag/dev_questions.json --systems hybrid_graph jev hybrid_graph_llm jev_llm --allow-external

Gold file (NICE): {"N01": ["Pressure ulcers 1.1.2", ...], "N02": [], ...}. A reference is "<guideline> <rec number>",
or a chunk id; [] means the correct output is "Not in the guidelines". A questions file whose items carry
"gold_ids" (the L1 dev set) needs no separate gold file.
"""

import argparse
import json
import time
from pathlib import Path

from systems import ROOT, SYSTEMS, Components, build_system, load_taus

from guideline_rag import EvalItem, abstain_layer, ablation_table_markdown, evaluate

GRIDS = {
    "bm25": [x / 2 for x in range(0, 41)],
    "vector": [x / 40 for x in range(0, 41)],
    "graph": [x / 200 for x in range(0, 41)],
    "jev": [x / 20 for x in range(0, 21)],
}
DEFAULT_GRID = [x / 40 for x in range(0, 41)]
GATE_SOURCE = {"hybrid_graph_llm": "hybrid_graph", "jev_llm": "jev"}  # LLM systems reuse the retrieval system's τ


def ref_to_ids(ref: str, chunks) -> list[str]:
    ids = [c.id for c in chunks if c.id == ref]
    if ids:
        return ids
    source, _, rec = ref.rpartition(" ")
    ids = [c.id for c in chunks if c.source == source and c.metadata.get("rec") == rec]
    if not ids:
        raise ValueError(f"gold reference {ref!r} matches no chunk")
    return ids


def load_items(questions_path: Path, gold_path: Path | None, chunks):
    records = json.loads(questions_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path else None
    items = {}
    for r in records:
        if "gold_ids" in r:
            labels = r["gold_ids"]
        elif gold is not None and r["id"] in gold:
            labels = [i for ref in gold[r["id"]] for i in ref_to_ids(ref, chunks)]
        else:
            labels = None
        items[r["id"]] = (r, labels)
    return items


def predict(system, records, by_id):
    out = []
    for r in records:
        t0 = time.perf_counter()
        res = system.answer(r["question"])
        ref = lambda cid: f"{by_id[cid].source} {by_id[cid].metadata.get('rec') or by_id[cid].section} (p.{by_id[cid].page})"
        out.append({
            "id": r["id"], "question": r["question"], "answer": res.answer, "abstained": res.abstained,
            "blocked_by": abstain_layer(res), "reason": res.reason,
            "citations": [ref(c) for c in res.citations], "top3": [ref(c) for c in res.retrieved_ids],
            "verdicts": [(v.verdict, v.confidence) for v in res.verdicts],
            "seconds": round(time.perf_counter() - t0, 2),
        })
    return out


def tune(c, name, items, allow_external):
    grid = GRIDS.get(name, DEFAULT_GRID)
    scored = [(evaluate(build_system(c, name, t, allow_external), items)["course_score"], t) for t in grid]
    return max(scored)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="nice")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "nice" / "lexicon.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "nice" / "questions.json")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--systems", nargs="+", default=["bm25", "vector", "hybrid", "graph", "hybrid_graph", "hybrid_graph_llm"],
                        choices=SYSTEMS)
    parser.add_argument("--allow-external", action="store_true", help="allow Jev systems (TypeSafe API)")
    args = parser.parse_args()

    store = ROOT / "store" / args.store
    out_dir = ROOT / "results" / args.store
    out_dir.mkdir(parents=True, exist_ok=True)
    c = Components(store, args.lexicon)
    by_id = {ch.id: ch for ch in c.chunks}
    items = load_items(args.questions, args.gold, c.chunks)
    taus = load_taus(store)

    labelled = {qid: EvalItem(qid, r["question"], labels) for qid, (r, labels) in items.items() if labels is not None}
    if labelled:
        dev = [it for qid, it in labelled.items() if items[qid][0].get("split", "dev") == "dev"]
        val = [it for qid, it in labelled.items() if items[qid][0].get("split") == "val"]
        tuned = set()
        for name in args.systems:
            if GATE_SOURCE.get(name) in tuned:
                taus[name] = taus[GATE_SOURCE[name]]
            else:
                taus[name] = tune(c, name, dev, args.allow_external)
            tuned.add(name)
            print(f"τ[{name}] = {taus[name]} (tuned on {len(dev)} dev questions)")
        (store / "taus.json").write_text(json.dumps({s: taus[s] for s in args.systems}, indent=1), encoding="utf-8")
        for split_name, split_items in (("dev (tuning set, optimistic)", dev), ("val (held out)", val)):
            if not split_items:
                continue
            rows = []
            for name in args.systems:
                summary = evaluate(build_system(c, name, taus[name], args.allow_external), split_items)
                summary.pop("rows")
                rows.append({"config": name, **summary})
            table = ablation_table_markdown(rows)
            print(f"\n## {args.store} — {split_name}, n={len(split_items)}\n\n{table}")
            (out_dir / f"ablation_{split_name.split()[0]}.md").write_text(table + "\n", encoding="utf-8")

    records = [r for r, _ in items.values()]
    predictions = {}
    for name in args.systems:
        predictions[name] = predict(build_system(c, name, taus[name], args.allow_external), records, by_id)
        n_abs = sum(p["abstained"] for p in predictions[name])
        secs = sum(p["seconds"] for p in predictions[name]) / len(records)
        print(f"{name:17s} τ={taus[name]:<6} answered {len(records) - n_abs:2d} / abstained {n_abs:2d}   {secs:5.2f}s per question")
    path = out_dir / "predictions.json"
    path.write_text(json.dumps(predictions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\npredictions -> {path}")


if __name__ == "__main__":
    main()
