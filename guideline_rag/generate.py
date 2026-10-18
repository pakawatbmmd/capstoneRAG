from __future__ import annotations

import json
import re
from typing import Callable

from .types import ABSTAIN_PHRASE, Claim, Draft, Hit

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_PROMPT = """Answer the question using ONLY the passages below. Do not use outside knowledge.

Reply with JSON only, in exactly this shape:
{{"answer": str, "claims": [{{"text": str, "quote": str, "chunk_id": str}}]}}

Each claim is one statement from your answer. Its "quote" must be copied verbatim from the passage it cites, and "chunk_id" is the id in square brackets before that passage.

If the passages do not answer the question, reply exactly:
{{"answer": "{abstain}", "claims": []}}

Passages:
{passages}

Question: {question}
JSON:"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n".join(f"[{h.chunk.id}] {h.chunk.embed_text()}" for h in hits)
    return _PROMPT.format(abstain=ABSTAIN_PHRASE, passages=passages, question=question)


def _answer_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj
    return None


def _clean_chunk_id(value) -> str:
    return str(value or "").strip().strip("[]").strip()


def _is_abstention(answer: str) -> bool:
    return not answer.strip() or ABSTAIN_PHRASE.lower() in answer.lower()


def parse_llm_output(raw: str) -> Draft:
    for candidate in [*_FENCE.findall(raw), raw]:
        obj = _answer_object(candidate)
        if obj is not None:
            raw_claims = obj.get("claims")
            claims = [
                Claim(
                    text=str(c.get("text") or ""),
                    quote=str(c.get("quote") or ""),
                    chunk_id=_clean_chunk_id(c.get("chunk_id")),
                )
                for c in (raw_claims if isinstance(raw_claims, list) else [])
                if isinstance(c, dict)
            ]
            return Draft(answer=obj["answer"], claims=claims, abstained=_is_abstention(obj["answer"]))
    stripped = raw.strip()
    if stripped.startswith(("{", "```")):
        return Draft(answer="", claims=[], abstained=True)
    return Draft(answer=stripped, claims=[], abstained=_is_abstention(raw))


class LLMGenerator:
    """`llm` is any callable str -> str (Qwen, Claude, GPT, Gemini, ...)."""

    def __init__(
        self,
        llm: Callable[[str], str],
        prompt_builder: Callable[[str, list[Hit]], str] = build_prompt,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        return parse_llm_output(self.llm(self.prompt_builder(question, hits)))
