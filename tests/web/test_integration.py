"""Runs the real NICE store. Skips cleanly when the store (licensed, not in git) or a GPU is missing."""
from pathlib import Path

import pytest

from web.service import ROOT, RagService, gpu_available

STORE = ROOT / "store" / "nice"
LEXICON = ROOT / "data" / "nice" / "lexicon.json"
pytestmark = pytest.mark.skipif(not (STORE / "chunks.jsonl").exists(), reason="store/nice not ingested (NICE PDFs are not in git)")

IN_SCOPE = "How often should an adult at high risk of pressure ulcers be repositioned?"
OUT_OF_SCOPE = "What is the capital of France?"


@pytest.fixture(scope="module")
def svc():
    return RagService.from_store(STORE, LEXICON, results_dir=ROOT / "results" / "nice")


def test_hybrid_in_scope_and_out_of_scope(svc):
    off = svc.ask(OUT_OF_SCOPE, "hybrid")
    assert off["abstained"] is True and off["answer"] == "Not in the guidelines" and off["blocked_by"]
    assert len(off["top3"]) == 3


@pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU for Qwen3-4B")
def test_hybrid_llm_answers_with_citation_and_refuses_off_topic(svc):
    ok = svc.ask(IN_SCOPE, "hybrid_llm")
    assert ok["abstained"] is False and ok["citations"]
    assert any(c["source"] == "Pressure ulcers" for c in ok["citations"])
    off = svc.ask(OUT_OF_SCOPE, "hybrid_llm")
    assert off["abstained"] is True and off["answer"] == "Not in the guidelines" and len(off["top3"]) == 3


def test_saved_results_have_real_splits(svc):
    out = svc.results()
    assert set(out["predictions"]) == {"train", "val", "test"}
    assert "hybrid_llm" in out["predictions"]["test"]
