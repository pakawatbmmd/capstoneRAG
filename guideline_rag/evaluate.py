from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import RAGResult


@dataclass
class EvalItem:
    qid: str
    question: str
    gold_ids: list[str]


def load_eval_set(path: str | Path) -> list[EvalItem]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(r["id"], r["question"], list(r["gold_ids"])) for r in records]


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 3) -> float | None:
    if not gold_ids:
        return None
    gold = set(gold_ids)
    return len(gold & set(retrieved_ids[:k])) / len(gold)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(pipeline, items: list[EvalItem], k: int = 3) -> dict:
    rows = []
    n_claims = n_accepted = 0
    for item in items:
        res = pipeline.answer(item.question)
        n_claims += len(res.verdicts)
        n_accepted += sum(v.accepted for v in res.verdicts)
        rows.append(
            {
                "qid": item.qid,
                "question": item.question,
                "gold_ids": item.gold_ids,
                "retrieved_ids": res.retrieved_ids,
                "answer": res.answer,
                "abstained": res.abstained,
                "reason": res.reason,
                "recall": recall_at_k(res.retrieved_ids, item.gold_ids, k),
                "abstain_correct": res.abstained == (len(item.gold_ids) == 0),
                "unsafe": not item.gold_ids and not res.abstained,
            }
        )

    recall = _mean([r["recall"] for r in rows if r["recall"] is not None])
    abstention = _mean([float(r["abstain_correct"]) for r in rows])
    n_in_scope = sum(1 for item in items if item.gold_ids)
    unsafe = _mean([float(r["unsafe"]) for r in rows if not r["gold_ids"]])
    return {
        "rows": rows,
        "recall_at_k": recall,
        "abstention_accuracy": abstention,
        "course_score": 0.5 * (recall or 0.0) + 0.5 * (abstention or 0.0),
        "verified_claim_rate": n_accepted / n_claims if n_claims else None,
        "n_in_scope": n_in_scope,
        "n_out_of_scope": len(items) - n_in_scope,
        "unsafe_answer_rate": unsafe,
    }


def run_ablation(configs: dict, items: list[EvalItem], k: int = 3) -> list[dict]:
    table = []
    for name, pipeline in configs.items():
        summary = evaluate(pipeline, items, k)
        summary.pop("rows")
        table.append({"config": name, **summary})
    return table


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def ablation_table_markdown(rows: list[dict]) -> str:
    lines = [
        "| Config | Recall@3 | Abstention acc. | Course score | Verified claims | Unsafe answers |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [
            r["config"],
            _fmt(r["recall_at_k"]),
            _fmt(r["abstention_accuracy"]),
            _fmt(r["course_score"]),
            _fmt(r["verified_claim_rate"]),
            _fmt(r["unsafe_answer_rate"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_submission(pairs: list[tuple[str, RAGResult]]) -> dict:
    return {
        qid: {"passages": result.retrieved_ids[:3], "answer": result.answer}
        for qid, result in pairs
    }
