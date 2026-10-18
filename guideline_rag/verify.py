from __future__ import annotations

import math
import re
from typing import Protocol

from .retrieval import tokenize
from .types import Chunk, Claim, ClaimVerdict

_PUNCTUATION = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "‐": "-", "‑": "-", "‒": "-", "−": "-"}
)
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_STRICT_NUMBER = re.compile(rf"(?<![A-Za-z0-9]){_NUMBER}")
_LOOSE_NUMBER = re.compile(_NUMBER)
_RELATION_TO_VERDICT = {
    "supports": "verified",
    "contradicts": "contradicted",
    "says_nothing": "unsupported",
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().translate(_PUNCTUATION).split())


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    normalized = normalize(quote)
    return bool(normalized) and normalized in normalize(chunk_text)


def claim_numbers(text: str) -> list[float]:
    """Numbers in a claim, ignoring digits embedded in identifiers like HbA1c or SGLT2."""
    return [float(m.replace(",", "")) for m in _STRICT_NUMBER.findall(text)]


def chunk_numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _LOOSE_NUMBER.findall(text)]


def numbers_supported(claim_text: str, chunk_text: str) -> bool:
    return set(claim_numbers(claim_text)) <= set(chunk_numbers(chunk_text))


def _is_valid_confidence(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _judge_context(chunk_text: str, quote: str, max_words: int) -> str:
    """The chunk itself if short, else `max_words` words centred on where the quote starts."""
    words = chunk_text.split()
    if len(words) <= max_words:
        return chunk_text
    normalized = normalize(chunk_text)
    quote_word = normalized.count(" ", 0, normalized.find(normalize(quote)))
    start = min(max(quote_word - max_words // 2, 0), len(words) - max_words)
    return " ".join(words[start : start + max_words])


class CitationJudge(Protocol):
    def judge(self, claim_text: str, chunk_text: str) -> tuple[str, float]: ...


class Verifier:
    def __init__(
        self,
        judge: CitationJudge | None = None,
        auto_accept: float = 0.8,
        min_quote_tokens: int = 4,
        judge_context_words: int = 200,
    ):
        self.judge = judge
        self.auto_accept = auto_accept
        self.min_quote_tokens = min_quote_tokens
        self.judge_context_words = judge_context_words

    def verify(self, claims: list[Claim], chunks_by_id: dict[str, Chunk]) -> list[ClaimVerdict]:
        return [self._verify_one(claim, chunks_by_id) for claim in claims]

    def _verify_one(self, claim: Claim, chunks_by_id: dict[str, Chunk]) -> ClaimVerdict:
        chunk = chunks_by_id.get(claim.chunk_id)
        if chunk is None or not claim.text.strip():
            return ClaimVerdict(claim, "fabricated")
        if len(tokenize(claim.quote)) < self.min_quote_tokens:
            return ClaimVerdict(claim, "weak_quote")
        if not quote_in_chunk(claim.quote, chunk.text):
            return ClaimVerdict(claim, "fabricated")
        if not numbers_supported(claim.text, claim.quote):
            return ClaimVerdict(claim, "number_mismatch")
        if self.judge is None:
            return ClaimVerdict(claim, "quote_found")

        context = _judge_context(chunk.text, claim.quote, self.judge_context_words)
        relation, confidence = self.judge.judge(claim.text, context)
        verdict = _RELATION_TO_VERDICT.get(str(relation).strip().lower())
        if verdict is None:
            return ClaimVerdict(claim, "unsupported", confidence=confidence, needs_review=True)
        needs_review = not _is_valid_confidence(confidence) or confidence < self.auto_accept
        return ClaimVerdict(claim, verdict, confidence=confidence, needs_review=needs_review)
