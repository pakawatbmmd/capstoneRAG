# capstoneRAG: Guideline Q&A with guarded RAG

Capstone project for **3099704 AI for Digital Health** (Faculty of Medicine, Chulalongkorn University, 2026).

A nurse or doctor asks a question at the point of care. The system answers **only** from clinical guidelines, cites the recommendation it used (guideline, section, recommendation number, page, version), and otherwise replies exactly **"Not in the guidelines"**.

The project compares six RAG designs on the same guideline store, from a keyword baseline to an LLM with claim-level verification:

| System | Retrieval | Rerank | Answer | Guard → "Not in the guidelines" |
|---|---|---|---|---|
| `bm25` (baseline) | BM25 | – | extractive | score τ |
| `vector` (lab09 baseline) | all-MiniLM-L6-v2 | – | extractive | cosine τ |
| `hybrid` | BM25 + BAAI/bge-m3 (RRF) | bge-reranker-v2-m3 | extractive | reranker τ |
| `graph` | entity graph + Personalized PageRank | – | extractive | PPR τ |
| `hybrid_graph` | BM25 + bge-m3 + graph (RRF) | bge-reranker-v2-m3 | extractive | reranker τ |
| **`hybrid_graph_llm`** (proposed) | as `hybrid_graph` | bge-reranker-v2-m3 | **Qwen3-4B-Instruct-2507** | τ + LLM abstain + **NLI claim check** (nli-deberta-v3-base) |
| `jev`, `jev_llm` (side study) | as `hybrid_graph` | Jev (TypeSafe) | extractive / Qwen3-4B | Jev answerability + Jev citation check |

Every model runs locally on one laptop GPU. On an RTX 4090 Laptop (16 GB), the full stack uses 13.9 GiB. Measure your own machine with `examples/check_models.py`.

## How an answer is guarded

1. **Answerability gate.** If the top retrieval or reranker score is below τ (or Jev's probability is below τ), the system abstains. τ is tuned on the dev split only.
2. **The LLM must answer in JSON:** a list of claims, each with a verbatim quote and the chunk id it cites. It may itself reply "Not in the guidelines".
3. **Each claim is verified.**
   - The cited chunk must exist.
   - The quote must be at least 4 tokens and appear verbatim in that chunk.
   - Every number in the claim must appear in the quote.
   - An NLI model (or Jev) must judge that the passage **supports** the claim.
4. **Red flags block the whole answer.** If any claim is fabricated, contradicted or has a number mismatch, the output is "Not in the guidelines". If some claims are merely unverifiable, only the verified ones are shown, with an omission note.

## Chunking

- Guidelines are split on their own structure: **one chunk per numbered recommendation**, with a section header such as `Pressure ulcers > Recommendations > 1.1 Prevention: adults > Repositioning`.
- Long sections are cut into 300-word windows with a 40-word overlap.
- Page furniture, tables of contents, boilerplate and the duplicated "Key priorities" section are dropped.
- Chunk ids are content hashes. Re-ingesting an updated guideline re-embeds **only the changed chunks**, which is the basis for auto-updating guidelines.

## Usage

```bash
# 1. environment (CUDA 13 wheels shown; pick yours from pytorch.org)
python -m venv ~/.venvs/capstone-rag
~/.venvs/capstone-rag/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
~/.venvs/capstone-rag/Scripts/python -m pip install -e ".[models,jev,pdf]"

# 2. put your guideline PDFs in Guidelines/ (not included, see Licence), then build the store
python examples/ingest.py --name nice Guidelines/*.pdf

# 3. ask
python examples/ask.py "How often should adults at high risk of pressure ulcers be repositioned?"

# 4. evaluate: predictions now; with a gold file, τ is tuned on dev and scored on val
python examples/evaluate_systems.py --store nice --questions data/nice/questions.json [--gold data/nice/gold.json]
```

Jev systems need `TYPESAFE_API_KEY` (read from `~/.config/capstone-rag/typesafe.env`, never stored in the repo) and the `--allow-external` flag.

## Metrics

- **Course score:** 0.5 × Recall@3 + 0.5 × abstention accuracy.
- **Unsafe-answer rate:** the share of out-of-scope questions that were answered.
- **Verified-claim rate:** the share of LLM claims that passed verification.

## Layout

| Path | Contents |
|---|---|
| `guideline_rag/` | the library. Its core needs only numpy; heavy models are lazy adapters |
| `examples/` | `ingest.py`, `ask.py`, `systems.py`, `evaluate_systems.py`, `check_models.py` |
| `docs/SPEC.md`, `docs/SPEC_GRAPH.md` | the specification the code was built against, versions 1.0 to 1.4 |
| `data/nice/` | the medical lexicon for GraphRAG, plus the team's evaluation questions |
| `data/rag/` | our dev questions and lexicon for the course L1 synthetic set |

## Licence and data

- The guideline PDFs, the derived store and the predictions are **not included**. NICE content may be used outside the UK only for personal study and research, and may not be redistributed or used in retrieval systems without NICE's agreement (see the [NICE terms](https://www.nice.org.uk/terms-and-conditions)).
- The course L1 dataset and course materials are not included either.
- No patient data is used.

## Disclosure of AI-assistant use

This code, its specification and its tests were written with an AI coding assistant (Claude Code), under the direction of the project team. The team reviewed the design decisions, the clinical framing and the results.
