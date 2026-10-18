# guideline_rag — RAG core spec (locked)

A pluggable RAG core for guideline Q&A. Every stage is an interface, so you can swap in any guideline source, embedder, reranker (cross-encoder or Jev), gate, LLM or verifier. The core runs offline with only `numpy`. Heavy models live in `adapters.py` behind lazy imports.

- Python >= 3.10 (works on Colab). Core runtime dependency: `numpy` only.
- Tests: `pytest`, no network, no model downloads, no `typesafe_sdk`/`torch`/`sentence_transformers` needed.
- Run tests: `python -m pytest -q` from the project root.

## Package layout

```
guideline_rag/
  __init__.py     re-exports the public names below
  types.py        Chunk, Hit, Claim, Draft, ClaimVerdict, GateDecision, RAGResult, ABSTAIN_PHRASE
  loaders.py      load_kb_json, chunk_markdown, chunk_id, load_pdf_text (pypdf, lazy)
  retrieval.py    tokenize, BM25Retriever, DenseRetriever, HybridRetriever
  rerank.py       NoRerank, ScoreFnReranker
  gate.py         AlwaysAnswer, ThresholdGate, ProbabilityGate
  generate.py     build_prompt, parse_llm_output, LLMGenerator
  verify.py       normalize, quote_in_chunk, claim_numbers, chunk_numbers, numbers_supported, Verifier
  pipeline.py     RAGPipeline
  evaluate.py     EvalItem, load_eval_set, recall_at_k, evaluate, run_ablation, ablation_table_markdown, to_submission
  adapters.py     SentenceTransformerEmbedder, CrossEncoderScorer, HFPipelineLLM, JevJudge  (lazy heavy imports)
```

## types.py

```python
ABSTAIN_PHRASE = "Not in the guidelines"

@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""        # guideline name, e.g. "Thai DM CPG 2023"
    section: str = ""       # heading path, e.g. "Treatment > Metformin"
    page: int | None = None
    version: str = ""
    metadata: dict = field(default_factory=dict)
    def header(self) -> str:      # " > ".join of non-empty [source, section]; "" if both empty
    def embed_text(self) -> str:  # header() + "\n" + text if header() else text

@dataclass
class Hit:
    chunk: Chunk
    score: float

@dataclass
class Claim:
    text: str
    quote: str
    chunk_id: str

@dataclass
class Draft:
    answer: str
    claims: list[Claim]
    abstained: bool

@dataclass
class GateDecision:
    answerable: bool
    score: float
    reason: str = ""

@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: str              # "verified" | "contradicted" | "unsupported" | "fabricated" | "number_mismatch"
    confidence: float | None = None
    needs_review: bool = False

@dataclass
class RAGResult:
    question: str
    answer: str
    abstained: bool
    retrieved_ids: list[str]  # ids of the final top_k hits, in rank order
    citations: list[str]      # chunk ids the answer relies on, unique, in first-seen order
    reason: str               # see pipeline.py
    verdicts: list[ClaimVerdict] = field(default_factory=list)
```

## loaders.py

- `chunk_id(source, section, text) -> str`: the first 12 hex chars of `sha1(f"{source}|{section}|{text}".encode("utf-8"))`. It is deterministic and content-addressed, so the same text gives the same id and any edit gives a new id. This is what auto-update relies on.
- `load_kb_json(path, source="") -> list[Chunk]`: reads a JSON list of objects (the L1 format `[{"id": "P01", "text": "..."}]`). It keeps the given `id` and `text`. The optional keys `section`, `page` and `version` map to fields; every other key goes into `metadata`. `source` applies to all chunks. It reads UTF-8 and preserves file order.
- `chunk_markdown(text, source, version="", max_words=250, overlap_words=30) -> list[Chunk]`:
  - Headings are lines matching `^#{1,6} `. The section path is the heading stack joined with `" > "`, e.g. `"Diabetes > Treatment"`, and a deeper heading replaces any equal or deeper level.
  - Body text under a heading is split into blocks on blank lines. Consecutive blocks are packed into one chunk while the total word count is `<= max_words`.
  - A single non-table block longer than `max_words` is split into word windows of `max_words`, each starting `max_words - overlap_words` words after the previous one.
  - A **table block** (every non-empty line starts with `|`) is never split, even if it is longer than `max_words`. It is emitted as its own chunk.
  - Heading lines themselves are not part of chunk text. Whitespace-only chunks are dropped.
  - Text before the first heading has section `""`.
  - Each chunk gets `id = chunk_id(source, section, text)`, plus `source`, `section` and `version`.
- `load_pdf_text(path) -> str`: joins the page texts with `"\n\n"` using `pypdf`, imported lazily. Not unit-tested.

## retrieval.py

- `tokenize(text) -> list[str]`: `re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())`. So `"eGFR 2.5 mg"` gives `["egfr", "2.5", "mg"]`.
- Retriever protocol: `search(query: str, k: int) -> list[Hit]`, sorted by score descending. Ties keep the original chunk order. It returns at most `k` hits.
- `BM25Retriever(chunks, k1=1.5, b=0.75)`: Okapi BM25 over `tokenize(chunk.embed_text())`. IDF is `ln((N - df + 0.5) / (df + 0.5) + 1)`. It only returns hits with score > 0. An empty or no-match query returns `[]`.
- `DenseRetriever(chunks, embedder, query_prefix="", passage_prefix="")`: `embedder.encode(list[str]) -> np.ndarray (n, d)`. At init it encodes `passage_prefix + chunk.embed_text()` for all chunks. Vectors are L2-normalised, and the score is cosine similarity. `search` encodes `query_prefix + query`. The prefixes are there for e5-style models.
- `HybridRetriever(retrievers, rrf_k=60, candidate_k=50)`: calls each retriever's `search(query, candidate_k)`. Fused score = sum over retrievers of `1 / (rrf_k + rank)`, with rank starting at 1. Chunks are identified by `chunk.id`. The returned Hit holds the Chunk object and the fused score. Ties are broken by first appearance (retriever order, then rank).

## rerank.py

- Reranker protocol: `rerank(query, hits, k) -> list[Hit]`.
- `NoRerank()`: returns `hits[:k]` unchanged.
- `ScoreFnReranker(score_fn)`: `score_fn(query, list_of_texts) -> list[float]`, where the texts are `hit.chunk.embed_text()`. It returns new Hits carrying the new scores, sorted descending (stable), truncated to `k`. Empty hits give `[]` without calling `score_fn`. This is the plug point for a cross-encoder or the Jev Noul.

## gate.py (answerability)

- Gate protocol: `decide(query, hits) -> GateDecision`.
- `AlwaysAnswer()`: `answerable = bool(hits)`, `score = hits[0].score` if hits else `0.0`.
- `ThresholdGate(threshold)`: `answerable = bool(hits) and hits[0].score >= threshold`. Score is the top score, or 0.0 when there are no hits. This is the lab's `ABSTAIN_BELOW`.
- `ProbabilityGate(prob_fn, threshold=0.5)`: `prob_fn(query, list_of_texts) -> float` (P that the passages answer the question; the Jev Noul). `answerable = p >= threshold`. Empty hits give not answerable with score 0.0, and `prob_fn` is **not** called.

## generate.py

- `build_prompt(question, hits) -> str`: must contain the question, every passage as a line `[<chunk.id>] <chunk.embed_text()>`, and the exact `ABSTAIN_PHRASE`. It instructs the model to answer only from the passages and to reply with JSON only: `{"answer": str, "claims": [{"text": str, "quote": str, "chunk_id": str}]}`. Each quote must be copied verbatim from the passage it cites. If the passages do not answer the question, the model replies `{"answer": "Not in the guidelines", "claims": []}`.
- `parse_llm_output(raw) -> Draft`:
  - Strips a ```` ```json ... ``` ```` or ```` ``` ... ``` ```` fence if present, then parses the first `{...}` JSON object found in the text, even if surrounded by prose.
  - Missing `claims` means `[]`. A claim missing `text`, `quote` or `chunk_id` gets `""` for that field. Non-dict claim entries are skipped.
  - If no valid JSON is found, it returns `Draft(answer=raw.strip(), claims=[], abstained=...)`.
  - `abstained = ABSTAIN_PHRASE.lower() in answer.lower()`, in both the JSON and the fallback case.
- `LLMGenerator(llm, prompt_builder=build_prompt)`: `llm` is any `Callable[[str], str]`, which makes it pluggable with Qwen, Claude, GPT or Gemini. `generate(question, hits) -> Draft` = `parse_llm_output(llm(prompt_builder(question, hits)))`.

## verify.py

- `normalize(text)`: casefold; curly quotes `“ ” ‘ ’` become `" '`; en/em dashes `– —` become `-`; whitespace collapsed to a single space; stripped.
- `quote_in_chunk(quote, chunk_text) -> bool`: `normalize(quote)` is a substring of `normalize(chunk_text)`. An empty or whitespace-only quote gives False.
- `claim_numbers(text) -> list[float]`: strict extraction. It matches `\d+(?:\.\d+)?` **not immediately preceded by a letter or digit**, so the digits inside `HbA1c`, `SGLT2`, `CHA2DS2`, `G3b` are ignored, while `5mg`, `2.5`, `130/80` → `[130.0, 80.0]` are extracted.
- `chunk_numbers(text) -> list[float]`: loose extraction. All `\d+(?:\.\d+)?` runs, including those inside words.
- `numbers_supported(claim_text, chunk_text) -> bool`: every number in `claim_numbers(claim_text)` is in `set(chunk_numbers(chunk_text))`, compared as floats. A claim without numbers gives True.
- CitationJudge protocol: `judge(claim_text, chunk_text) -> tuple[str, float]`, where relation is `"supports" | "contradicts" | "says_nothing"` and confidence is in [0, 1].
- `Verifier(judge=None, auto_accept=0.8)`: `verify(claims, chunks_by_id: dict[str, Chunk]) -> list[ClaimVerdict]`, one per claim, in order. The first rule that matches wins:
  1. `claim.chunk_id` not in `chunks_by_id` → `fabricated`, confidence None, needs_review False
  2. not `quote_in_chunk(claim.quote, chunk.text)` → `fabricated`
  3. not `numbers_supported(claim.text, chunk.text)` → `number_mismatch`
  4. `judge is None` → `verified`, confidence None, needs_review False
  5. otherwise the relation maps `supports→verified`, `contradicts→contradicted`, `says_nothing→unsupported`. The confidence is recorded and `needs_review = confidence < auto_accept`. The judge is called with `(claim.text, chunk.text)` and is **not called** when rules 1–3 already decided.

## pipeline.py

`RAGPipeline(retriever, reranker=None, gate=None, generator=None, verifier=None, retrieve_k=20, top_k=3, extractive_fallback=True)`. `None` reranker means `NoRerank()`, and `None` gate means `AlwaysAnswer()`.

`answer(question) -> RAGResult`:
1. `hits = reranker.rerank(question, retriever.search(question, retrieve_k), top_k)`, then `retrieved_ids = [h.chunk.id for h in hits]`. This is always filled, even when the pipeline abstains.
2. If `gate.decide` is not answerable → `answer=ABSTAIN_PHRASE, abstained=True, citations=[], reason="gate"`.
3. If `generator is None` → extractive: `answer = hits[0].chunk.text`, `citations=[hits[0].chunk.id]`, `abstained=False`, `reason="extractive"`.
4. Otherwise `draft = generator.generate(question, hits)`. If `draft.abstained` → ABSTAIN_PHRASE, `abstained=True`, `reason="llm_abstained"`.
5. If `verifier is None` → `answer=draft.answer`, citations = unique `chunk_id`s of the claims (first-seen order), `reason="unverified"`.
6. Otherwise `verdicts = verifier.verify(draft.claims, {h.chunk.id: h.chunk for h in hits})`. A claim is *accepted* iff `verdict == "verified" and not needs_review`.
   - All claims accepted and there is at least 1 claim → `answer=draft.answer`, citations from the claims, `reason="verified"`.
   - Some, but not all, accepted → `answer = " ".join(accepted claim texts)`, citations from the accepted claims, `reason="partial"`.
   - None accepted (including zero claims): if `extractive_fallback` → `answer = f'Guideline text [{hits[0].chunk.id}]: "{hits[0].chunk.text}"'`, `citations=[hits[0].chunk.id]`, `abstained=False`, `reason="fallback_extractive"`. Otherwise → ABSTAIN_PHRASE, `abstained=True`, `reason="verification_failed"`.
   - `verdicts` is stored on the result in every step-6 branch.

## evaluate.py

- `EvalItem(qid: str, question: str, gold_ids: list[str])`: an empty `gold_ids` means the question is out of scope and the correct behaviour is to abstain.
- `load_eval_set(path) -> list[EvalItem]`: reads a JSON list of `{"id", "question", "gold_ids"}`.
- `recall_at_k(retrieved_ids, gold_ids, k=3) -> float | None`: `|gold ∩ retrieved[:k]| / |gold|`, or None if gold is empty.
- `evaluate(pipeline, items, k=3) -> dict` with keys:
  - `rows`: one dict per item: `qid, question, gold_ids, retrieved_ids, answer, abstained, reason, recall` (None for out-of-scope), `abstain_correct` (bool: `abstained == (len(gold_ids) == 0)`)
  - `recall_at_k`: mean recall over in-scope items (None if there are none)
  - `abstention_accuracy`: mean of `abstain_correct` over all items (None if there are no items)
  - `course_score`: `0.5 * recall_at_k + 0.5 * abstention_accuracy` (a None value counts as 0.0)
  - `verified_claim_rate`: accepted claims / all claims across rows (None if there are no claims). Accepted is defined as in the pipeline.
  - `n_in_scope`, `n_out_of_scope`
- `run_ablation(configs: dict[str, RAGPipeline], items, k=3) -> list[dict]`: one row per config, in dict order: `{"config": name, **summary}` where summary is every `evaluate` key except `rows`.
- `ablation_table_markdown(rows) -> str`: a markdown table with header `| Config | Recall@3 | Abstention acc. | Course score | Verified claims |`. Floats are formatted with 2 decimals and None as `-`. One line per row, in order.
- `to_submission(pairs: list[tuple[str, RAGResult]]) -> dict`: `{qid: {"passages": result.retrieved_ids[:3], "answer": result.answer}}`, the lab09 leaderboard format.

## adapters.py (thin, lazy, not unit-tested except JevJudge with a fake client)

- `SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")`: `.encode(texts)`.
- `CrossEncoderScorer(model_name="BAAI/bge-reranker-v2-m3")`: `__call__(query, texts) -> list[float]`.
- `HFPipelineLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=300)`: `__call__(prompt) -> str`, chat template, greedy decoding.
- `JevJudge(client=None, model="jev-1.13.0", noul_cls=None, choice_cls=None)`. The defaults are `typesafe_sdk.TypeSafeClient()`, `typesafe_sdk.Noul` and `typesafe_sdk.Choice`, imported lazily on first use. The model is pinned to a version, not an alias.
  - `relevance(query, texts) -> list[float]`: **one** `client.system_one(state=..., questions=..., model=...)` request with state `{"question": query, "passages": {"p0": t0, "p1": t1, ...}}` and one Noul per passage, keyed `"p0".."pN"`. The instructions reference `` `question` `` and `` `passages.pI` ``. It returns `[response.answers[f"p{i}"].noul ...]`. Empty texts give `[]` without a call.
  - `answerable(query, texts) -> float`: one request, one Noul keyed `"answerable"`, with state `{"question": query, "passages": [...]}`. It returns `.noul`.
  - `judge(claim_text, chunk_text) -> (relation, confidence)`: one request with state `{"claim", "passage"}` and a Choice keyed `"relation"` with criteria `supports / contradicts / says_nothing`. It returns `(answer.choice, answer.confidence)`.
  - Plug-in: `ScoreFnReranker(jev.relevance)`, `ProbabilityGate(jev.answerable, τ)`, `Verifier(judge=jev)`.

## Non-goals (YAGNI)

No vector DB service, no web UI, no async, no caching layer in the core (cache in the notebook), no Thai tokenizer for BM25 (dense multilingual embeddings cover Thai), no auto-update crawler yet (content-hash ids make it possible later).

---

# v1.1 changes (these SUPERSEDE the matching rules above)

Sources: code review findings B1–B3, M1–M5, plus persistence (the user wants the vector store and graph saved in this project folder).

## verify.py
- Number extraction: a thousands-grouped number `\d{1,3}(?:,\d{3})+(?:\.\d+)?` is matched first and read with the commas removed (`"1,000"` → `1000.0`). This applies to both `claim_numbers` (still not preceded by a letter or digit) and `chunk_numbers`.
- `Verifier(judge=None, auto_accept=0.8, min_quote_tokens=4)`. Rules, first match wins:
  1. `claim.chunk_id` not in `chunks_by_id` → `fabricated`
  2. `claim.text` is empty or whitespace → `fabricated`
  3. `len(tokenize(claim.quote)) < min_quote_tokens` (using `retrieval.tokenize`) → `weak_quote`
  4. not `quote_in_chunk(claim.quote, chunk.text)` → `fabricated`
  5. not `numbers_supported(claim.text, claim.quote)` → `number_mismatch`. Numbers are now checked against the **quote**, not the whole chunk.
  6. `judge is None` → `quote_found` (confidence None, needs_review False). **This is not "verified" and is never accepted.**
  7. Otherwise, `relation = str(relation).strip().lower()`, mapped `supports→verified`, `contradicts→contradicted`, `says_nothing→unsupported`.
     - An unknown relation → `unsupported` with `needs_review=True`.
     - Confidence None → `needs_review=True`.
     - Otherwise `needs_review = confidence < auto_accept`.
     - The judge is not called when rules 1–5 decide.
- The verdict set is now `verified | contradicted | unsupported | fabricated | number_mismatch | weak_quote | quote_found`. Accepted is still `verdict == "verified" and not needs_review`.

## generate.py
- `parse_llm_output(raw)`:
  - Candidate texts are the content of every fenced block (in order), then the full raw text.
  - In each candidate, every `{` position is tried with `json.JSONDecoder().raw_decode`. The first decoded **dict that has key `"answer"` with a `str` value** is used, so a claim-only inner dict or truncated JSON is never mistaken for the answer object.
  - If none is found, the fallback is `Draft(answer=raw.strip(), claims=[], abstained=ABSTAIN_PHRASE.lower() in raw.lower())`.
  - In **every** path, if `answer.strip() == ""` then `abstained = True`.

## pipeline.py
- Defaults use `is None` (not truthiness): `reranker is None → NoRerank()`, `gate is None → AlwaysAnswer()`.
- Step 2: if `hits` is empty **or** the gate is not answerable → abstain with `reason="gate"`.
- Step 6, all-accepted branch: `answer = " ".join(c.text for c in claims)`. The LLM's free-text `draft.answer` is no longer shown on the verified path, so uncited content cannot slip through. The other branches are unchanged.

## retrieval.py
- `BM25Retriever`: if the corpus average document length is 0, `search` returns `[]`.
- `DenseRetriever(chunks, embedder, query_prefix="", passage_prefix="", store_dir=None, embedder_name="")`:
  - With no chunks, `encode` is not called at init and `search` returns `[]`.
  - If the query vector has norm 0, `search` returns `[]`.
  - Attribute `n_encoded`: how many passage texts were encoded at init.
  - **Persistence:** if `store_dir` is set, `<store_dir>/vectors.npz` holds arrays `ids` (str) and `vectors` (float32, L2-normalised), and `<store_dir>/vectors_meta.json` holds `{"embedder_name", "passage_prefix", "dim"}`.
    - At init, if both files exist and the meta's `embedder_name` and `passage_prefix` equal the current ones, vectors are reused for chunk ids already present.
    - Only the missing chunks are encoded, in one `encode` call, or no call if none are missing.
    - The store is then rewritten with exactly the current chunks' ids and vectors (removed ids are dropped).
    - On a meta mismatch, everything is re-encoded and the store is overwritten.
    - The directory is created if it is missing.

## rerank.py
- `ScoreFnReranker`: raises `ValueError` if `score_fn` returns a different number of scores than texts.

## store.py (new)
- `save_chunks(chunks, path)`: creates parent directories; JSONL, one chunk per line with all Chunk fields; UTF-8, `ensure_ascii=False`.
- `load_chunks(path) -> list[Chunk]`: the exact inverse, so a round trip gives equal Chunks.

## adapters.py
- `HFPipelineLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=600)`: passes `dtype="auto"` to `transformers.pipeline`.
- `NLIJudge(model_name="cross-encoder/nli-deberta-v3-base", model=None)`: a **local** CitationJudge for laptop-only runs with no Jev.
  - `model` is injectable; by default it is a lazily created `sentence_transformers.CrossEncoder(model_name)`.
  - `judge(claim_text, chunk_text)` calls `model.predict([(chunk_text, claim_text)], apply_softmax=True)`, which returns probabilities for one pair.
  - Labels come from `model.config.id2label` (index → label, lowercased): `entailment→supports`, `contradiction→contradicts`, `neutral→says_nothing`.
  - It returns `(mapped label of the argmax, float(max probability))`.

---

# v1.2 additions: NICE guideline chunker + "Not in the guidelines" guardrail

## loaders.py: `load_pdf_pages` and `chunk_nice_guideline`

- `load_pdf_pages(path) -> list[str]`: one extracted text per page (pypdf, lazy import). Not unit-tested.
- `chunk_nice_guideline(pages, source, version="", max_words=300, overlap_words=40) -> list[Chunk]`. `pages[i]` is page i+1. Processing is line by line, keeping track of the page each line came from.

**1. Drop boilerplate lines** (after `strip()`):
- lines starting with `©` and containing `NICE`
- lines containing `conditions#notice-of-rights`
- `^Page \d+ of\s*$` and `^\d+$`
- table-of-contents lines with dot leaders: `\.{4,}\s*\d+$`
- empty lines

**2. Version.** If the `version` argument is `""`, it is taken from the first line matching `Last updated:\s*(.+)`. Failing that, the first line matching `Published:\s*(.+)` is used. The captured text is stripped, e.g. `"4 August 2017"`. Every chunk gets this version.

**3. Top-level sections.** A line whose stripped text **equals** one of these titles starts a new top-level section, and is not itself chunk text:
`Overview`, `Who is it for?`, `Your responsibility`, `Key priorities for implementation`, `Recommendations`, `Terms used in this guideline`, `Recommendations for research`, `Rationale and impact`, `Context`, `Update information`.
- A line **starting with** `Finding more information` also starts a top-level section titled `Finding more information`.
- Lines before the first top-level title belong to the section `Front matter`.

**4. Dropped sections.** Nothing is emitted for `Front matter`, `Your responsibility` (boilerplate), `Key priorities for implementation` (a verbatim duplicate of recommendations, which would crowd the top-k), `Finding more information` or `Update information`.

**5. Inside `Recommendations`:**
- A line matching `^(\d+\.\d+)\s+(\S.*)$` that is **not** `^\d+\.\d+\.\d+` is a subsection heading. The section becomes `f"Recommendations > {num} {title}"`, and the heading line is not chunk text.
- A line matching `^(\d+\.\d+\.\d+)\s+` starts a **recommendation chunk**. It runs until the next recommendation, subsection heading or top-level title. Its metadata is `{"kind": "recommendation", "rec": "<number>"}`, and its text starts with the number, e.g. `"1.2.3 When a person ..."`.
- Text lines inside `Recommendations` but outside any recommendation (for example a note under a subsection heading, before its first recommendation) form one chunk per contiguous run, with metadata `{"kind": "text"}`.

**6. Other kept sections** (`Overview`, `Who is it for?`, `Terms used in this guideline`, `Recommendations for research`, `Rationale and impact`, `Context`):
- A line matching `^Recommendations? \d+\.\d+\.\d+` with at most 12 words is a sub-heading. The section becomes `f"{top} > {line}"` and the line is not chunk text.
- Text is grouped per (section, sub-heading) run, with metadata `{"kind": "text"}`.

**7. Building each chunk:**
- Lines are joined with a single space and whitespace runs are collapsed.
- A chunk longer than `max_words` is split into word windows of `max_words` with stride `max_words - overlap_words`. Every window keeps the same section and metadata.
- Chunks whose text is empty are dropped.
- `page` is the page of the chunk's first line (the window's first word for split chunks).
- `id = chunk_id(source, section, text)`.
- Chunks are output in document order.

## guardrails.py (new): one abstention policy per RAG type

Every pipeline answers either with guideline-backed content or with exactly `ABSTAIN_PHRASE` ("Not in the guidelines", the lab09 metric phrase).

`guarded_pipeline(retriever, *, tau, reranker=None, generator=None, jev=None, nli_judge=None, retrieve_k=20, top_k=3) -> RAGPipeline`

- **With Jev** (`jev is not None`): the gate is `ProbabilityGate(jev.answerable, tau)`, so Jev decides whether the passages answer the question. If a generator is given, the verifier is `Verifier(judge=jev)`.
- **Without Jev:** the gate is `ThresholdGate(tau)`, applied to the top retrieval or reranker score. If a generator is given, `nli_judge` is required (otherwise raise `ValueError`), and the verifier is `Verifier(judge=nli_judge)`.
- In both cases `extractive_fallback=False`. Unverifiable LLM output therefore becomes `ABSTAIN_PHRASE` (reason `verification_failed`) instead of a quoted passage.
- `reranker`, `retrieve_k` and `top_k` are passed through. With no generator, the pipeline is retrieval-only: an extractive answer when the gate passes, `ABSTAIN_PHRASE` otherwise.

`abstain_layer(result: RAGResult) -> str` names the layer that blocked, for the demo and the error analysis:
- `"gate"` → `"answerability gate"`
- `"llm_abstained"` → `"LLM: not in the guidelines"`
- `"verification_failed"` → `"verifier: claims not supported"`
- `""` when `result.abstained` is False

`__init__.py` re-exports `load_pdf_pages`, `chunk_nice_guideline`, `guarded_pipeline` and `abstain_layer`.

---

# v1.3 changes: second review fixes (SUPERSEDE matching rules above)

## retrieval.py: DenseRetriever persistence is keyed by content, not by id
- The cache key per chunk is `sha1((passage_prefix + chunk.embed_text()).encode("utf-8")).hexdigest()`.
- `vectors.npz` stores the arrays `keys` (str), `ids` (str, informational) and `vectors` (float32).
- At init, a vector is reused only when its **key** matches. Editing a chunk's text or source therefore forces re-encoding, even if its id is unchanged. Two chunks with the same id but different text each get their own vector. `n_encoded` = the number of chunks whose key was not in the store.
- `store_dir` set with an empty `embedder_name` → raise `ValueError`.
- Writes are atomic: write to a temporary file in the same directory, then `os.replace`. A store that fails to load (corrupt npz or meta) is treated as empty and fully rebuilt.
- The internal matrix is float32.

## generate.py
- In parsed claims, `chunk_id` is stripped of whitespace and surrounding `[` `]`, so `"[P1]"` becomes `"P1"`.
- If the stripped raw output starts with `{` or a code fence but no valid answer object was found (truncated or invalid JSON), the result is `Draft(answer="", claims=[], abstained=True)`. A JSON fragment is never shipped as an answer.

## adapters.py
- `HFPipelineLLM` default `max_new_tokens=1024`.

## verify.py
- **Confidence check:** a confidence counts as valid only if it is an `int` or `float`, not a `bool`, and finite. Any other value (None, NaN, inf, bool, str) → `needs_review=True`.
- **Judge context:** the judge receives a **context window**, not the whole chunk. If `chunk.text` has at most `judge_context_words` words (default 200), it gets the whole text. Otherwise it gets a window of `judge_context_words` words, centred on the word where the quote starts (located in the normalised text) and clamped to the text bounds. This keeps the premise within an NLI model's 512-token limit.
- The signature becomes `Verifier(judge=None, auto_accept=0.8, min_quote_tokens=4, judge_context_words=200)`.

## pipeline.py: a failed caveat can no longer disappear silently
- A **red-flag** verdict is `contradicted`, `number_mismatch` or `fabricated`. If **any** claim has one, the whole draft is rejected: `fallback_extractive` when `extractive_fallback` is True, else `ABSTAIN_PHRASE` with `reason="verification_failed"`.
- Partial answers are only allowed when the non-accepted claims are all non-red-flag (`unsupported`, `weak_quote`, `quote_found`, or `needs_review`). The answer is then `" ".join(accepted texts) + " " + OMISSION_NOTE`, where `OMISSION_NOTE = "(Some statements could not be verified against the guideline and were left out.)"`, exported from `types.py`. The reason stays `"partial"`.
- `RAGResult` gains `omitted: int = 0`, the number of claims not accepted, set in every step-6 branch.

## graph.py
- `load_lexicon`: an `entities` value that is not a list, or an `is_a` value that is not a list → `TypeError`.

## v1.3.1: chunk_nice_guideline subsection headings (amends v1.2 rule 5; found on the real NICE PDFs)
- A subsection heading is `^([1-9]\d*\.\d+)\s+([A-Z].*)$` (not `^\d+\.\d+\.\d+`). **The number cannot start with 0 and the title must start with an uppercase letter**, so a wrapped dose line such as `0.6 mmol/kg/day) and ...` is body text, not a heading.
- **Wrapped title:** if the next kept line after a subsection heading starts with a lowercase letter, it continues the title. Any trailing `-` or `‑` (U+2011) on the heading line is removed, then the two are joined with one space. At most one continuation line is joined, and it is not chunk text. Example: `1.2 Documenting and sharing information with` + `other healthcare professionals` → section `Recommendations > 1.2 Documenting and sharing information with other healthcare professionals`.

---

# v1.4: page-furniture removal, unnumbered sub-headings, unsafe-answer metric

## chunk_nice_guideline: more boilerplate lines are dropped (adds to v1.2 rule 1)
On the real PDFs, 105 of 325 chunks carried page furniture. These stripped lines are now also dropped:
- `^Page \d+( of( \d+)?)?$` (for example `Page 12`, `Page 12 of`, `Page 12 of 31`)
- `^of \d+$` (for example `of 31`)
- running title footers: lines ending in `\((CG|NG|QS|PH)\d+\)$` (for example `Drug allergy: diagnosis and management (CG183)`)

## chunk_nice_guideline: unnumbered sub-headings (new rule 5b, inside `Recommendations`)
A line is an **unnumbered sub-heading** when all of these hold:
- It is not a recommendation line and not a subsection heading.
- It has at most 8 words, starts with an uppercase letter, does not end with `.` `,` `;` or `:`, and does not start with `•`.
- The **next** kept line is a recommendation line.
- The **previous** kept line ended with `.`, or was itself a heading: a top-level title, a subsection heading, the subsection title's continuation line, or another unnumbered sub-heading. A bullet list item without a full stop therefore cannot precede one.

When it applies:
- The sub-heading is not chunk text. It flushes the current run.
- Following chunks get the section `f"Recommendations > {num} {title} > {label}"`.
- The label is replaced by the next unnumbered sub-heading. It is cleared by the next subsection heading or top-level title.

For example, `1.1 Prevention: adults`, then `Repositioning`, then `1.1.8 …` gives section `Recommendations > 1.1 Prevention: adults > Repositioning`.

## evaluate.py: unsafe-answer rate
- `evaluate` summary key `unsafe_answer_rate`: among out-of-scope items (empty `gold_ids`), the fraction where the system did **not** abstain. It is None if there are no out-of-scope items. The per-row key is `unsafe` (bool, True only for an out-of-scope item that was answered).
- `ablation_table_markdown` header becomes `| Config | Recall@3 | Abstention acc. | Course score | Verified claims | Unsafe answers |`, with the same number formatting as before.
