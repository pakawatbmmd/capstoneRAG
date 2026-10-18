"""Ask the guideline store a question with any RAG system and see the answer, its sources and which guard fired.

    python examples/ask.py "How often should a person at high risk of pressure ulcers be repositioned?"
    python examples/ask.py --system graph "Should we use a falls risk prediction tool?"
    python examples/ask.py --system jev_llm --allow-external "..."     # sends guideline text to TypeSafe
"""

import argparse
import json
from pathlib import Path

from systems import ROOT, SYSTEMS, Components, build_system, load_taus

from guideline_rag import abstain_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--system", default="hybrid_graph_llm", choices=SYSTEMS)
    parser.add_argument("--store", default="nice")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "nice" / "lexicon.json")
    parser.add_argument("--allow-external", action="store_true", help="allow Jev (TypeSafe API) systems")
    args = parser.parse_args()

    store = ROOT / "store" / args.store
    c = Components(store, args.lexicon)
    tau = load_taus(store)[args.system]
    result = build_system(c, args.system, tau, allow_external=args.allow_external).answer(args.question)
    by_id = {ch.id: ch for ch in c.chunks}

    print(f"\n[{args.system}, τ={tau}]  {result.answer}\n")
    if result.abstained:
        print(f"blocked by: {abstain_layer(result)}")
    for cid in result.citations:
        ch = by_id[cid]
        print(f"  source: {ch.source} | {ch.section} | p.{ch.page} | {ch.metadata.get('rec', '')} ({ch.version})")
    for v in result.verdicts:
        print(f"  claim [{v.verdict}{'' if v.confidence is None else f' {v.confidence:.2f}'}]: {v.claim.text}")
    print("  retrieved:", ", ".join(f"{by_id[i].source} {by_id[i].metadata.get('rec', by_id[i].section)}" for i in result.retrieved_ids))
    if "graph" in args.system:
        print("  graph:", json.dumps(c.get("graph").explain(args.question)["query_entities"]))


if __name__ == "__main__":
    main()
