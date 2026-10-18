"""FastAPI app serving hybrid_llm. Contract: docs/API.md. Interactive docs: /docs.

    uvicorn server.app:app --port 8000              # real models (GPU); /api/health turns "ready" after loading
    RAG_MOCK=1 uvicorn server.app:app --port 8000   # canned responses, no models, for frontend work

Environment:
    RAG_MOCK=1            use MockEngine
    RAG_CORS_ORIGINS      comma-separated allowed origins (default "*")
    RAG_STORE             path to the guideline store (default store/nice)
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from server.engine import ROOT, Engine, MockEngine

engine = MockEngine() if os.environ.get("RAG_MOCK") == "1" else Engine(store=Path(os.environ.get("RAG_STORE", ROOT / "store" / "nice")))


class AskRequest(BaseModel):
    question: str = Field(description="The nurse's question, 3–1000 characters")

    @field_validator("question")
    @classmethod
    def trimmed_length(cls, value: str) -> str:
        value = value.strip()
        if not 3 <= len(value) <= 1000:
            raise ValueError("question must be 3–1000 characters after trimming")
        return value


class Citation(BaseModel):
    ref: str
    guideline_code: str
    guideline: str
    section: str
    recommendation: str | None
    page: int | None
    version: str
    text: str


class ClaimOut(BaseModel):
    text: str
    quote: str
    ref: str
    verdict: str
    confidence: float | None
    shown: bool


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "partial", "not_in_guidelines"]
    answer: str
    abstained: bool
    blocked_by: str | None
    citations: list[Citation]
    claims: list[ClaimOut]
    omitted_claims: int
    retrieved: list[str]
    latency_ms: int
    pipeline: str
    mock: bool
    disclaimer: str


class Health(BaseModel):
    status: Literal["loading", "ready", "error"]
    pipeline: str
    mock: bool
    models: dict[str, str]
    settings: dict[str, float]
    chunks: int
    error: str | None


class Guideline(BaseModel):
    code: str
    name: str
    version: str | None
    chunks: int
    recommendations: int


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=engine.load, name="load-models", daemon=True).start()
    yield


app = FastAPI(
    title="Guarded Guideline Q&A API",
    version="1.0.0",
    description="Answers nurses' questions only from NICE guidelines (pipeline hybrid_llm); otherwise 'Not in the guidelines'.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("RAG_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
def health():
    return engine.health()


@app.get("/api/guidelines", response_model=list[Guideline])
def guidelines():
    return engine.guidelines()


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if engine.status != "ready":
        raise HTTPException(503, detail="loading" if engine.status == "loading" else f"error: {engine.error}")
    try:
        return engine.ask(request.question)
    except Exception as exc:
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}") from exc
