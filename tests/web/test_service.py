import csv
import json

from guideline_rag.types import Chunk, Claim, ClaimVerdict, RAGResult
from web.service import DESCRIPTIONS, LLM_SYSTEMS, RagService, SYSTEMS

CHUNKS = [Chunk(id="P1", text="Repeat lipid panel 4-12 weeks after a statin.", source="kb"),
          Chunk(id="P2", text="Offer varenicline to smokers with COPD.", source="kb"),
          Chunk(id="P3", text="Reposition high-risk patients every 4 hours.", source="kb")]


class FakeComponents:
    chunks = CHUNKS


class FakePipeline:
    def __init__(self, result):
        self.result = result

    def answer(self, question):
        return self.result


def make(result, built=None, gpu=True, results_dir=None):
    built = built if built is not None else []

    def builder(c, name, tau, allow_external=False, auto_accept=0.8):
        built.append((name, tau, auto_accept))
        return FakePipeline(result)

    taus = {"bm25": 8.0, "hybrid_graph": 0.25, "hybrid_llm": 0.004, "hybrid_graph_llm": 0.004}
    return RagService(FakeComponents(), taus, builder=builder, auto_accept={"hybrid_llm": 0.7},
                      gpu=gpu, results_dir=results_dir)


ANSWER = RAGResult(question="q", answer="Repeat at 4-12 weeks.", abstained=False,
                   retrieved_ids=["P1", "P2", "P3"], citations=["P1"], reason="",
                   verdicts=[ClaimVerdict(Claim("Repeat at 4-12 weeks.", "4-12 weeks", "P1"), "verified", 0.93)])
BLOCKED = RAGResult(question="q", answer="Not in the guidelines", abstained=True,
                    retrieved_ids=["P2", "P3", "P1"], citations=[], reason="gate")


def test_roster_is_the_latest_backend_roster_with_descriptions():
    names = [s["name"] for s in make(ANSWER).systems()]
    assert names == SYSTEMS
    assert {"vector_bge", "hybrid_llm"} <= set(names)
    assert all(DESCRIPTIONS.get(n) for n in names)


def test_unknown_backend_name_does_not_crash_systems(monkeypatch):
    monkeypatch.setattr("web.service.SYSTEMS", SYSTEMS + ["future_system"])
    systems = {s["name"]: s for s in make(ANSWER).systems()}
    assert systems["future_system"]["enabled"] is False and systems["future_system"]["description"]


def test_llm_systems_enabled_with_gpu_jev_never():
    systems = {s["name"]: s for s in make(ANSWER, gpu=True).systems()}
    assert systems["hybrid_llm"]["enabled"] and systems["hybrid_graph_llm"]["enabled"]
    for name in ("jev", "jev_llm"):
        assert systems[name]["enabled"] is False and "external" in systems[name]["disabled_reason"].lower()
    assert systems["bm25"]["tau"] == 8.0


def test_llm_systems_disabled_without_gpu():
    systems = {s["name"]: s for s in make(ANSWER, gpu=False).systems()}
    for name in LLM_SYSTEMS - {"jev_llm"}:  # jev_llm is external first: that reason takes priority
        assert systems[name]["enabled"] is False and "GPU" in systems[name]["disabled_reason"]
    assert "external" in systems["jev_llm"]["disabled_reason"].lower()
    assert systems["hybrid"]["enabled"] is True


def test_default_system_is_hybrid_llm_with_gpu_else_hybrid():
    assert make(ANSWER, gpu=True).default_system() == "hybrid_llm"
    assert make(ANSWER, gpu=False).default_system() == "hybrid"
    flags = {s["name"]: s["default"] for s in make(ANSWER, gpu=True).systems()}
    assert [n for n, d in flags.items() if d] == ["hybrid_llm"]


def test_llm_pipeline_built_with_tuned_auto_accept():
    built = []
    make(ANSWER, built).ask("q", "hybrid_llm")
    assert built == [("hybrid_llm", 0.004, 0.7)]


def test_ask_answer_shape():
    out = make(ANSWER).ask("When to recheck lipids?", "bm25")
    assert out["system"] == "bm25" and out["tau"] == 8.0
    assert out["answer"] == "Repeat at 4-12 weeks." and out["abstained"] is False and out["blocked_by"] == ""
    assert out["citations"] == [{"id": "P1", "text": CHUNKS[0].text, "source": "kb", "section": "", "page": None}]
    assert [p["id"] for p in out["top3"]] == ["P1", "P2", "P3"] and out["top3"][1]["text"] == CHUNKS[1].text
    assert out["verdicts"] == [{"claim": "Repeat at 4-12 weeks.", "verdict": "verified", "confidence": 0.93}]
    assert isinstance(out["seconds"], float)


def test_ask_abstain_names_guard():
    out = make(BLOCKED).ask("What is the capital of France?", "hybrid_graph")
    assert out["abstained"] is True and out["answer"] == "Not in the guidelines"
    assert out["blocked_by"] == "answerability gate" and out["reason"] == "gate"
    assert out["citations"] == [] and len(out["top3"]) == 3


def test_ask_rejects_disabled_and_unknown():
    svc = make(ANSWER, gpu=False)
    for name in ("hybrid_llm", "jev", "nope"):
        try:
            svc.ask("q", name)
        except ValueError:
            continue
        raise AssertionError(f"{name} should be rejected")


def test_pipelines_are_cached():
    built = []
    svc = make(ANSWER, built)
    svc.ask("a", "bm25")
    svc.ask("b", "bm25")
    assert built == [("bm25", 8.0, 0.8)]


def _write_results(tmp_path):
    (tmp_path / "metrics.md").write_text(
        "### train — τ tuned here, optimistic (n=2)\n\n| System | τ | Course score |\n|---|---|---|\n| bm25 | 17.5 | 0.60 |\n\n"
        "### val (n=1)\n\n| System | τ | Course score |\n|---|---|---|\n| bm25 | 17.5 | 0.88 |\n", encoding="utf-8")
    (tmp_path / "metrics_nograph.md").write_text(
        "### train — τ tuned here, optimistic (n=2)\n\n| System | τ | Course score |\n|---|---|---|\n| hybrid_llm | 0.004 | 0.88 |\n",
        encoding="utf-8")
    fields = ["system", "split", "qid", "abstained", "reason", "retrieved_ids", "answer"]
    rows = [
        {"system": "bm25", "split": "train", "qid": "PU01", "abstained": "False", "reason": "extractive",
         "retrieved_ids": '["CG179-1.1.9", "CG179-1.1.8"]', "answer": "x"},
        {"system": "bm25", "split": "val", "qid": "DA10", "abstained": "True", "reason": "gate",
         "retrieved_ids": '["CG183-1.2.6"]', "answer": "Not in the guidelines"},
    ]
    for name, data in (("per_question.csv", rows),
                       ("per_question_nograph.csv", [{**rows[0], "system": "hybrid_llm"}])):
        with open(tmp_path / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([{"id": "PU01", "question": "Reposition how often?"},
                                {"id": "DA10", "question": "Observe how long?"}]), encoding="utf-8")
    return gold


def test_results_use_real_splits_and_train_table(tmp_path):
    gold = _write_results(tmp_path)
    svc = make(ANSWER, results_dir=tmp_path)
    svc.gold_path = gold
    out = svc.results()
    assert out["table_split"] == "train"
    assert "| bm25 | 17.5 | 0.60 |" in out["ablation_md"] and "| hybrid_llm | 0.004 | 0.88 |" in out["ablation_md"]
    assert "0.88 |\n" not in out["ablation_md"].split("hybrid_llm")[0]  # val rows are not mixed into the train table
    assert set(out["predictions"]) == {"train", "val"}
    row = out["predictions"]["val"]["bm25"][0]
    assert row == {"id": "DA10", "question": "Observe how long?", "answer": "Not in the guidelines",
                   "blocked_by": "answerability gate", "top3": ["CG183-1.2.6"]}
    assert out["predictions"]["train"]["hybrid_llm"][0]["id"] == "PU01"


def test_results_empty_when_missing(tmp_path):
    out = make(ANSWER, results_dir=tmp_path / "missing").results()
    assert out == {"ablation_md": "", "table_split": "", "predictions": {}}
