from fastapi.testclient import TestClient

from tests.web.test_service import ANSWER, BLOCKED, _write_results, make
from web.app import AskRequest, create_app


def client(result=ANSWER, results_dir=None, gpu=True, gold=None):
    svc = make(result, gpu=gpu, results_dir=results_dir)
    if gold:
        svc.gold_path = gold
    return TestClient(create_app(svc))


def test_index_served_with_nice_examples():
    r = client().get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert "pressure ulcer" in r.text.lower() and "statin" not in r.text.lower()


def test_systems_endpoint_has_latest_roster_and_default():
    systems = client().get("/api/systems").json()
    names = [s["name"] for s in systems]
    assert {"hybrid_llm", "vector_bge", "hybrid_graph_llm"} <= set(names)
    assert [s["name"] for s in systems if s["default"]] == ["hybrid_llm"]


def test_request_default_system_is_hybrid_llm():
    assert AskRequest(question="abc").system == "hybrid_llm"


def test_ask_answer():
    r = client().post("/api/ask", json={"question": "When to recheck lipids?", "system": "bm25"})
    body = r.json()
    assert r.status_code == 200
    assert set(body) >= {"answer", "abstained", "blocked_by", "reason", "citations", "top3", "verdicts", "seconds"}


def test_ask_blocked():
    body = client(BLOCKED).post("/api/ask", json={"question": "Capital of France?", "system": "hybrid_graph"}).json()
    assert body["abstained"] is True and body["blocked_by"] == "answerability gate"


def test_ask_validation():
    c = client(gpu=False)
    assert c.post("/api/ask", json={"question": "  ", "system": "bm25"}).status_code == 422
    assert c.post("/api/ask", json={"question": "q", "system": "jev"}).status_code == 400
    assert c.post("/api/ask", json={"question": "q", "system": "hybrid_llm"}).status_code == 400


def test_results_endpoint(tmp_path):
    gold = _write_results(tmp_path)
    body = client(results_dir=tmp_path, gold=gold).get("/api/results").json()
    assert body["table_split"] == "train" and "bm25" in body["ablation_md"]
    assert body["predictions"]["val"]["bm25"][0]["id"] == "DA10"
