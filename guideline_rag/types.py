from __future__ import annotations

from dataclasses import dataclass, field

ABSTAIN_PHRASE = "Not in the guidelines"
OMISSION_NOTE = "(Some statements could not be verified against the guideline and were left out.)"


@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""
    section: str = ""
    page: int | None = None
    version: str = ""
    metadata: dict = field(default_factory=dict)

    def header(self) -> str:
        return " > ".join(part for part in (self.source, self.section) if part)

    def embed_text(self) -> str:
        header = self.header()
        return f"{header}\n{self.text}" if header else self.text


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
    verdict: str  # verified, contradicted, unsupported, fabricated, number_mismatch, weak_quote, quote_found
    confidence: float | None = None
    needs_review: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict == "verified" and not self.needs_review


@dataclass
class RAGResult:
    question: str
    answer: str
    abstained: bool
    retrieved_ids: list[str]
    citations: list[str]
    reason: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    omitted: int = 0
