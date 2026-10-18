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

## Summary report

Open [`docs/report.html`](docs/report.html) in a browser (download it, then double-click). It covers the pipeline diagrams, the guardrails, how vectors and the graph are stored, the models, the metrics by split, every question with its real answer and each pipeline's output, the final verdict and a glossary. [`docs/report_public.html`](docs/report_public.html) is the same report without the LLM's answer sentences.

**Verdict:** against the vector-RAG baseline, the guarded LLM pipeline wins: 33/40 answers correct vs 22/40, 0% unsafe answers vs 20%. A confirmatory run showed the graph component adds nothing, so the recommended deployment is `hybrid_llm` (hybrid retrieval + reranker + Qwen3-4B + NLI verifier).

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

# 4. evaluate against the gold set: tune τ on train, report train / val / test
python examples/evaluate_systems.py --gold Test+answer/questions_nurse40.json
```

Jev systems need `TYPESAFE_API_KEY` (read from `~/.config/capstone-rag/typesafe.env`, never stored in the repo) and the `--allow-external` flag.

## Metrics

- **Course score:** 0.5 × Recall@3 + 0.5 × abstention accuracy.
- **Unsafe-answer rate:** the share of out-of-scope questions that were answered.
- **Verified-claim rate:** the share of LLM claims that passed verification.

## Results (NICE, 40 gold questions: 24 train / 8 val / 8 test; 20 answerable, 20 near-miss not documented)

τ is tuned on train only. Below are all 40 questions (train is optimistic) and the sealed test split:

| System | Recall@3 | Guideline@1 | Abstention acc. | Course score (all 40) | Course score (test) | Unsafe answers |
|---|---|---|---|---|---|---|
| bm25 | 0.53 | 0.75 | 0.75 | 0.64 | 0.50 | 20% |
| vector (MiniLM, lab09) | 0.68 | 1.00 | 0.75 | 0.71 | 0.50 | 20% |
| vector_bge | 0.78 | 1.00 | 0.62 | 0.70 | 0.50 | 15% |
| hybrid | 0.85 | 1.00 | 0.62 | 0.74 | 0.50 | 0% |
| graph | 0.30 | 0.82 | 0.55 | 0.43 | 0.38 | 10% |
| hybrid_graph | 0.85 | 1.00 | 0.62 | 0.74 | 0.50 | 0% |
| **hybrid_graph_llm** | **0.85** | **1.00** | **0.82** | **0.84** | **0.62** | **0%** |
| hybrid_graph_llm + fix 1 (post-hoc) | 0.85 | 1.00 | 0.85 | 0.85 | 0.62 | 0% |

What the results show:
- **Proposed system:** `hybrid_graph_llm` is correct on 33/40 questions. It answered 13, all correct with a citation, and gave **no unsafe answers**. All 7 errors are over-cautious abstentions: 4 correct drafts rejected by the verifier, and 3 retrieval misses.
- **Baselines answer what they shouldn't:** BM25 and vector answer 15–20% of the "not documented" questions.
- **The graph adds nothing measurable** on this corpus. `hybrid` and `hybrid_graph` differ only in third-place retrievals that were blocked anyway.
- **Fix 1 (post-hoc):** applied after the error analysis, so its val and test numbers are post-hoc, not a fresh held-out result. It does three things:
  - normalises the non-breaking hyphens (U+2011) in quotes
  - tunes the verifier's `auto_accept` on train together with τ (0.8 → 0.7)
  - logs every claim

  It fixes 1 train question (34/40 correct) and leaves val and test unchanged. The remaining rejections come from the small NLI model, which judges near-verbatim clinical fragments as "unsupported", and from correctly refused paraphrased quotes.
- **Caveat:** the splits are small (test n=8). The 95% intervals overlap, so the test difference between `hybrid_graph_llm` and `hybrid` rests on 2 questions.

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
