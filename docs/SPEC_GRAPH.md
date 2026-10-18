# guideline_rag.graph: GraphRAG retriever spec (locked)

This is a second RAG type for the ablation. It uses an **entity graph with Personalized PageRank (PPR)**, the HippoRAG-style "local" GraphRAG. That suits factoid guideline Q&A, because it links a query's entities to passages through drug classes and synonyms that word overlap misses. `GraphRetriever` follows the same Retriever protocol as `BM25Retriever` (`search(query, k) -> list[Hit]`), so it plugs into `RAGPipeline` directly and can be fused with BM25 or dense retrievers via the existing `HybridRetriever`.

Module: `guideline_rag/graph.py`. It depends only on numpy and the standard library, and imports `Chunk`, `Hit` from `guideline_rag.types`.

## Lexicon

```json
{
  "entities": {"ace inhibitor": ["ace inhibitor", "acei", "lisinopril", "ramipril"], "...": ["..."]},
  "is_a":     {"apixaban": ["doac", "anticoagulant"], "...": ["..."]}
}
```

- `load_lexicon(path) -> dict`: reads UTF-8 JSON and returns `{"entities": dict[str, list[str]], "is_a": dict[str, list[str]]}`. A missing `"is_a"` key becomes `{}`. All keys and aliases are lowercased and stripped. The canonical name is always added to its own alias list if absent.

## Entity extractors (plug point)

Protocol: `extract(text: str) -> list[str]`, which returns canonical entity names, unique, sorted ascending.

- `LexiconExtractor(lexicon)`: `lexicon` is the dict from `load_lexicon`.
  - Matching is on `text.lower()`. An alias matches as a whole word or phrase, optionally followed by a plural `s` or `es`. Regex per alias: `(?<![a-z0-9])` + `re.escape(alias)` + `(?:s|es)?(?![a-z0-9])`.
  - Every internal whitespace run in an alias matches any whitespace run (`\s+`), so a phrase still matches across line breaks.
  - A match on any alias yields the canonical entity. Several aliases of one entity count once.
  - `is_a` is **not** applied here. Hierarchy lives in the graph.
- `LLMEntityExtractor(llm)`: `llm` is any `Callable[[str], str]`.
  - The prompt contains the text and asks for a JSON list of the medical entities (conditions, drugs, drug classes, tests, procedures) as lowercase strings.
  - Parsing finds the first `[...]` JSON array in the reply, keeps only string items, lowercases and strips them, drops empty ones, and returns them unique and sorted.
  - An unparsable reply returns `[]`.

## GraphRetriever

`GraphRetriever(chunks, extractor, lexicon=None, damping=0.5, iterations=30, fallback=None)`

**Graph built at init.** It is undirected and weighted:
- one **chunk node** per chunk (keyed by `chunk.id`)
- one **entity node** per entity name that appears either in `extractor.extract(chunk.embed_text())` for some chunk, or in any `is_a` entry when `lexicon` is given
- a **mention edge** chunk–entity with weight 1.0 for every entity extracted from that chunk
- an **is_a edge** entity–parent with weight 1.0 for every pair in `lexicon["is_a"]` (child → each parent). Both ends become entity nodes.
- Entity and chunk namespaces never collide: entity `"p01"` and chunk `"P01"` are different nodes.

**Entity IDF:** `idf(e) = ln(1 + N / max(df(e), 1))`, where N = the number of chunks and df(e) = the number of chunks with a mention edge to e.

**`search(query, k) -> list[Hit]`:**
1. `q_ents` = entities in `extractor.extract(query)` that exist as entity nodes.
2. If `q_ents` is empty: return `fallback.search(query, k)` if a fallback is set, else `[]`.
3. Personalization vector `p`: on each entity node e in `q_ents`, `idf(e)`; normalised to sum 1. Every other node is 0.
4. Run PPR by power iteration, `iterations` times, starting from `r = p`:
   `r_next[v] = (1 - damping) * p[v] + damping * Σ_u r[u] * w(u, v) / W(u)`, where W(u) = the sum of u's edge weights.
   A node with W(u) = 0 (isolated) sends its mass back by teleport: its `r[u]` is added to `(1 - damping)`'s share, distributed as `damping * r[u] * p`. This keeps total mass at 1.
5. The chunk score is `r` at that chunk's node. Return Hits for chunks with score > 0, sorted descending, ties broken by original chunk order, at most `k`.

**`explain(query) -> dict`**, for the demo slide:
- `query_entities`: sorted `q_ents`
- `top_entities`: up to 5 entity nodes (entity nodes only, never chunks) with the highest final `r` and r > 0, as `[(name, score), ...]` sorted by descending score, then name
- `paths`: for each returned chunk id (top 3), the sorted list of entities that link it to a query entity. These are entities it mentions that are either in `q_ents` or are one is_a hop away from one of them, in either direction.

  `explain` with no query entities returns `{"query_entities": [], "top_entities": [], "paths": {}}`.

**Properties the tests rely on:**
- A chunk mentioning a query entity directly outranks a chunk reachable only through an is_a hop or a shared second entity, when everything else is symmetric.
- A rarer query entity (higher idf) pulls more mass than a common one.
- Chunks in a disconnected component from all query entities get score 0 and are not returned.
- The total mass of `r` stays 1.0 (within 1e-9) after every iteration.
- It is deterministic: same inputs give the same output.

## Usage (plug-in)

```python
lex = load_lexicon("data/rag/lexicon.json")
graph = GraphRetriever(chunks, LexiconExtractor(lex), lexicon=lex, fallback=BM25Retriever(chunks))
RAGPipeline(graph, gate=ThresholdGate(t))                        # GraphRAG alone
RAGPipeline(HybridRetriever([BM25Retriever(chunks), graph]))     # BM25 + graph, fused with RRF
```

Note that the PPR scores are probabilities, not BM25 or cosine scores, so the `ThresholdGate` τ must be tuned separately per retriever on the dev set.

## Non-goals

No Microsoft-GraphRAG community summaries (those target "global" sensemaking questions, not factoid lookup), no graph database, no relation-typed triples beyond `is_a`, no LLM extraction in tests beyond a fake callable.

---

# v1.1 addition: graph export (persisted in this project folder)

- `GraphRetriever.export_json(path)` writes UTF-8 JSON and creates parent directories:
  ```json
  {"nodes": [{"id": "chunk:P01", "type": "chunk", "label": "P01"}, {"id": "entity:metformin", "type": "entity", "label": "metformin"}],
   "edges": [{"source": "chunk:P01", "target": "entity:metformin", "type": "mentions", "weight": 1.0},
             {"source": "entity:apixaban", "target": "entity:doac", "type": "is_a", "weight": 1.0}]}
  ```
  - Node ids are prefixed `chunk:` or `entity:`.
  - Nodes are sorted by id. Edges are sorted by (type, source, target).
  - Each undirected edge is written once: `mentions` as chunk → entity, `is_a` as child → parent.
- `load_graph_json(path) -> dict`: the inverse read, returning the same dict.
- The graph is rebuilt from chunks at init in milliseconds, so the export is for inspection and visualisation (Gephi/Neo4j import) and reproducibility, not for speed.
