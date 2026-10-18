# Guarded Guideline Q&A: API contract for the frontend

The backend serves the chosen pipeline, **`hybrid_llm`**: BM25 + bge-m3 hybrid retrieval, bge-reranker, a local Qwen3-4B generator and an NLI claim verifier. It answers only from the four NICE guidelines in `store/nice/`. Otherwise it replies exactly `Not in the guidelines`.

- Base URL (local): `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs` (OpenAPI; the schema is at `/openapi.json`)
- All bodies are JSON (UTF-8). CORS is open by default; restrict it with `RAG_CORS_ORIGINS`.

## Run it

```bash
# real models (GPU laptop, ~40 s to load; /api/health turns "ready")
uvicorn server.app:app --port 8000

# mock mode: canned answers, no models or GPU, for building the UI
RAG_MOCK=1 uvicorn server.app:app --port 8000
```

On Windows PowerShell, use `$env:RAG_MOCK="1"; uvicorn server.app:app --port 8000`.

## `GET /api/health`

Poll it until `status` is `"ready"`. While it reports `"loading"`, `/api/ask` returns **503**.

```json
{
  "status": "ready",
  "pipeline": "hybrid_llm",
  "mock": false,
  "models": {"embedder": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3",
             "generator": "Qwen/Qwen3-4B-Instruct-2507", "verifier": "cross-encoder/nli-deberta-v3-base"},
  "settings": {"tau": 0.004, "auto_accept": 0.7, "top_k": 3},
  "chunks": 314,
  "error": null
}
```

`status` is one of `loading`, `ready` or `error` (with `error` set to a message).

## `GET /api/guidelines`

```json
[
  {"code": "CG179", "name": "Pressure ulcers", "version": "23 April 2014", "chunks": 101, "recommendations": 95},
  {"code": "NG249", "name": "Fall", "version": "29 April 2025", "chunks": 69, "recommendations": 40}
]
```

## `POST /api/ask`

Request:

```json
{"question": "How often should an adult at high risk of pressure ulcers be repositioned?"}
```

`question` is 3–1000 characters after trimming. Anything else returns **422**.

Response: **answered**

```json
{
  "question": "How often should an adult at high risk of pressure ulcers be repositioned?",
  "status": "answered",
  "answer": "Encourage adults at high risk of a pressure ulcer to change position at least every 4 hours.",
  "abstained": false,
  "blocked_by": null,
  "citations": [
    {"ref": "CG179-1.1.9", "guideline_code": "CG179", "guideline": "Pressure ulcers",
     "section": "Recommendations › 1.1 Prevention: adults › Repositioning",
     "recommendation": "1.1.9", "page": 13, "version": "23 April 2014",
     "text": "1.1.9 Encourage adults who have been assessed as being at high risk ..."}
  ],
  "claims": [
    {"text": "...", "quote": "...", "ref": "CG179-1.1.9", "verdict": "verified", "confidence": 0.98, "shown": true}
  ],
  "omitted_claims": 0,
  "retrieved": ["CG179-1.1.9", "CG179-1.1.8", "CG179-1.2.8"],
  "latency_ms": 3120,
  "pipeline": "hybrid_llm",
  "mock": false,
  "disclaimer": "Decision support only. Answers come from NICE guidelines and must be checked against the cited recommendation; clinical judgement applies."
}
```

Response: **not in the guidelines** (still HTTP 200)

```json
{
  "question": "What colour allergy wristband should a patient with a drug allergy wear?",
  "status": "not_in_guidelines",
  "answer": "Not in the guidelines",
  "abstained": true,
  "blocked_by": "LLM: not in the guidelines",
  "citations": [], "claims": [], "omitted_claims": 0,
  "retrieved": ["CG183-1.2.3", "CG183-1.2.1", "CG183-1.2.6"],
  "latency_ms": 2050, "pipeline": "hybrid_llm", "mock": false,
  "disclaimer": "..."
}
```

### Fields the UI should use

| Field | Meaning | Suggested UI |
|---|---|---|
| `status` | `answered`, `partial` (some claims could not be verified and were left out) or `not_in_guidelines` | Main state: answer card, answer card with warning, or a clear "Not in the guidelines" message |
| `answer` | Text to show. For `partial` it ends with the omission note. For `not_in_guidelines` it is exactly `Not in the guidelines`. | Main text |
| `citations[]` | The recommendations the answer rests on, with page, version and source text | Source chips: "CG179 1.1.9 · p.13"; expand to show `text` |
| `blocked_by` | Which guardrail refused: `answerability gate`, `LLM: not in the guidelines` or `verifier: claims not supported` | Small "why" line under a refusal |
| `claims[]` | Every drafted claim with its verification verdict; `shown` marks the ones in `answer` | Optional "how this was checked" panel |
| `retrieved` | Top-3 recommendation refs the system read | Optional debug panel |
| `latency_ms` | Server time for this answer (about 2–4 s on the GPU) | Spinner while waiting |
| `disclaimer` | Always present | Footer under every answer |

### Errors

| HTTP | When | Body |
|---|---|---|
| 422 | Question missing, too short or too long | FastAPI validation error |
| 503 | Models still loading, or failed to load | `{"detail": "loading"}` or `{"detail": "error: ..."}` |
| 500 | Unexpected failure while answering | `{"detail": "..."}` |

Requests are processed one at a time, because the GPU holds one copy of each model. A second request waits for the first.

## Licence note

`citations[].text` contains NICE guideline text. That's fine for local use and study, but before exposing this API publicly, read the NICE terms (redistribution outside the UK needs NICE's agreement).
