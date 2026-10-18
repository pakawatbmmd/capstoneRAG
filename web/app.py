"""Local-only web frontend for the guideline RAG systems.

    PYTHONUTF8=1 python -m web.app            # NICE store, default system hybrid_llm
    -> http://127.0.0.1:8793
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from web.service import ROOT, RagService

STATIC = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str
    system: str = "hybrid_llm"

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


def create_app(service: RagService) -> FastAPI:
    app = FastAPI(title="Guideline RAG", docs_url=None, redoc_url=None)

    @app.get("/api/systems")
    def systems():
        return service.systems()

    @app.post("/api/ask")
    def ask(req: AskRequest):
        try:
            return service.ask(req.question, req.system)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/results")
    def results():
        return service.results()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="nice")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data" / "nice" / "lexicon.json")
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    service = RagService.from_store(ROOT / "store" / args.store, args.lexicon, results_dir=ROOT / "results" / args.store)
    print(f"loading models for {len(service.enabled())} systems (GPU: {service.gpu}); default {service.default_system()}...", flush=True)
    service.warm()
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)  # local only: licensed course data


if __name__ == "__main__":
    main()
