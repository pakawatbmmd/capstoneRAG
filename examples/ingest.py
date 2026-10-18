"""Ingest guideline files into a persistent store inside this project: store/<name>/.

    python examples/ingest.py --name nice Guidelines/*.pdf
    python examples/ingest.py --name l1 --lexicon data/rag/lexicon.json data/rag/kb.json

Writes
    store/<name>/chunks.jsonl              every chunk (id = content hash, so edits get new ids)
    store/<name>/dense_bge-m3/vectors.npz  vector index; re-running only embeds new or changed chunks
    store/<name>/graph.json                entity graph (inspect in Gephi / import into Neo4j)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guideline_rag import DenseRetriever, chunk_markdown, chunk_nice_guideline, load_kb_json, load_pdf_pages, save_chunks
from guideline_rag.graph import GraphRetriever, LexiconExtractor, load_lexicon

EMBEDDER = "BAAI/bge-m3"


def load_any(path: Path, source: str, version: str):
    if path.suffix.lower() == ".json":
        return load_kb_json(path, source=source)
    if path.suffix.lower() == ".pdf":
        return chunk_nice_guideline(load_pdf_pages(path), source=source, version=version)
    return chunk_markdown(path.read_text(encoding="utf-8"), source=source, version=version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--name", required=True, help="store folder name, e.g. thai-dm-2023")
    parser.add_argument("--version", default="", help="guideline version/date, kept on every chunk")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "nice" / "lexicon.json")
    parser.add_argument("--no-dense", action="store_true", help="skip the embedding model (offline)")
    args = parser.parse_args()

    out = ROOT / "store" / args.name
    chunks = [c for f in args.files for c in load_any(f, source=f.stem, version=args.version)]
    save_chunks(chunks, out / "chunks.jsonl")
    for f in args.files:
        mine = [c for c in chunks if c.source == f.stem]
        recs = sum(1 for c in mine if c.metadata.get("kind") == "recommendation")
        print(f"  {f.stem}: {len(mine)} chunks ({recs} recommendations), version {mine[0].version if mine else '-'}")
    print(f"{len(chunks)} chunks -> {out / 'chunks.jsonl'}")

    if not args.no_dense:
        from guideline_rag.adapters import SentenceTransformerEmbedder

        dense = DenseRetriever(
            chunks, SentenceTransformerEmbedder(EMBEDDER),
            store_dir=out / "dense_bge-m3", embedder_name=EMBEDDER,
        )
        print(f"vector store: embedded {dense.n_encoded} new/changed chunks, reused {len(chunks) - dense.n_encoded}")

    lexicon = load_lexicon(args.lexicon)
    graph = GraphRetriever(chunks, LexiconExtractor(lexicon), lexicon=lexicon)
    graph.export_json(out / "graph.json")
    print(f"graph -> {out / 'graph.json'}")


if __name__ == "__main__":
    main()
