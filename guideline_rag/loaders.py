from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .types import Chunk

_HEADING = re.compile(r"^(#{1,6}) (.*)$")
_KB_FIELDS = {"id", "text", "section", "page", "version"}


def chunk_id(source: str, section: str, text: str) -> str:
    """Content-addressed id: any edit to the text yields a new id."""
    return hashlib.sha1(f"{source}|{section}|{text}".encode("utf-8")).hexdigest()[:12]


def load_kb_json(path: str | Path, source: str = "") -> list[Chunk]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Chunk(
            id=rec["id"],
            text=rec["text"],
            source=source,
            section=rec.get("section", ""),
            page=rec.get("page"),
            version=rec.get("version", ""),
            metadata={k: v for k, v in rec.items() if k not in _KB_FIELDS},
        )
        for rec in records
    ]


def _is_table(block: str) -> bool:
    return all(line.lstrip().startswith("|") for line in block.splitlines() if line.strip())


def _split_blocks(lines: list[str]) -> list[str]:
    blocks, current = [], []
    for line in lines + [""]:
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current).strip())
            current = []
    return blocks


def _window_spans(n_words: int, max_words: int, overlap_words: int) -> list[tuple[int, int]]:
    stride = max(1, max_words - overlap_words)
    spans = []
    for start in range(0, n_words, stride):
        spans.append((start, min(start + max_words, n_words)))
        if start + max_words >= n_words:
            break
    return spans


def _windows(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[a:b]) for a, b in _window_spans(len(words), max_words, overlap_words)]


def _section_texts(lines: list[str], max_words: int, overlap_words: int) -> list[str]:
    texts: list[str] = []
    pending: list[str] = []
    pending_words = 0

    def flush() -> None:
        nonlocal pending, pending_words
        if pending:
            texts.append("\n\n".join(pending))
        pending, pending_words = [], 0

    for block in _split_blocks(lines):
        n_words = len(block.split())
        if _is_table(block):
            flush()
            texts.append(block)
        elif n_words > max_words:
            flush()
            texts.extend(_windows(block, max_words, overlap_words))
        else:
            if pending_words + n_words > max_words:
                flush()
            pending.append(block)
            pending_words += n_words
    flush()
    return texts


def chunk_markdown(
    text: str,
    source: str,
    version: str = "",
    max_words: int = 250,
    overlap_words: int = 30,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def emit() -> None:
        section = " > ".join(title for _, title in stack)
        for chunk_text in _section_texts(body, max_words, overlap_words):
            chunks.append(
                Chunk(
                    id=chunk_id(source, section, chunk_text),
                    text=chunk_text,
                    source=source,
                    section=section,
                    version=version,
                )
            )

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            emit()
            body = []
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            body.append(line)
    emit()
    return chunks


_NICE_TOP_TITLES = {
    "Overview",
    "Who is it for?",
    "Your responsibility",
    "Key priorities for implementation",
    "Recommendations",
    "Terms used in this guideline",
    "Recommendations for research",
    "Rationale and impact",
    "Context",
    "Update information",
}
_NICE_DROPPED = {
    "Front matter",
    "Your responsibility",
    "Key priorities for implementation",
    "Finding more information",
    "Update information",
}
_NICE_FINDING_MORE = "Finding more information"
_NICE_BOILERPLATE = re.compile(
    r"^Page \d+( of( \d+)?)?$|^of \d+$|^\d+$|\.{4,}\s*\d+$|\((?:CG|NG|QS|PH)\d+\)$"
)
_NICE_SUBHEADING_MAX_WORDS = 8
_NICE_SUBSECTION = re.compile(r"^([1-9]\d*\.\d+)\s+([A-Z].*)$")
_NICE_REC = re.compile(r"^(\d+\.\d+\.\d+)\s+")
_NICE_RATIONALE_HEADING = re.compile(r"^Recommendations? \d+\.\d+\.\d+")
_NICE_RATIONALE_HEADING_MAX_WORDS = 12
_LAST_UPDATED = re.compile(r"Last updated:\s*(.+)")
_PUBLISHED = re.compile(r"Published:\s*(.+)")


def _is_nice_boilerplate(line: str) -> bool:
    return (
        not line
        or (line.startswith("©") and "NICE" in line)
        or "conditions#notice-of-rights" in line
        or bool(_NICE_BOILERPLATE.search(line))
    )


def _is_subheading(line: str) -> bool:
    return (
        len(line.split()) <= _NICE_SUBHEADING_MAX_WORDS
        and line[:1].isupper()
        and not line.endswith((".", ",", ";", ":"))
        and not line.startswith("•")
    )


def _nice_version(lines: list[str]) -> str:
    for pattern in (_LAST_UPDATED, _PUBLISHED):
        for line in lines:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
    return ""


def chunk_nice_guideline(
    pages: list[str],
    source: str,
    version: str = "",
    max_words: int = 300,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Chunk a NICE guideline: one chunk per recommendation, boilerplate and duplicates dropped."""
    lines = [
        (page_no, stripped)
        for page_no, text in enumerate(pages, start=1)
        for stripped in map(str.strip, text.splitlines())
        if not _is_nice_boilerplate(stripped)
    ]
    version = version or _nice_version([line for _, line in lines])

    chunks: list[Chunk] = []
    top, sub = "Front matter", ""
    run: list[tuple[str, int]] = []
    metadata: dict = {}
    title_may_continue = False
    label = ""
    prev_was_heading = False

    def flush() -> None:
        section = " > ".join(part for part in (top, sub, label) if part)
        for start, end in _window_spans(len(run), max_words, overlap_words):
            text = " ".join(word for word, _ in run[start:end])
            chunks.append(
                Chunk(
                    id=chunk_id(source, section, text),
                    text=text,
                    source=source,
                    section=section,
                    page=run[start][1],
                    version=version,
                    metadata=dict(metadata),
                )
            )
        run.clear()

    def start_run(page_no: int, line: str, new_metadata: dict) -> None:
        nonlocal metadata
        flush()
        metadata = new_metadata
        run.extend((word, page_no) for word in line.split())

    for i, (page_no, line) in enumerate(lines):
        after_heading, prev_was_heading = prev_was_heading, False
        if title_may_continue and line[:1].islower():
            sub = f"{sub.rstrip(' -‑')} {line}"
            title_may_continue = False
            prev_was_heading = True
            continue
        title_may_continue = False
        if line in _NICE_TOP_TITLES or line.startswith(_NICE_FINDING_MORE):
            flush()
            top = line if line in _NICE_TOP_TITLES else _NICE_FINDING_MORE
            sub = label = ""
            prev_was_heading = True
            continue
        if top in _NICE_DROPPED:
            continue

        if top == "Recommendations":
            rec = _NICE_REC.match(line)
            heading = _NICE_SUBSECTION.match(line)
            if rec:
                start_run(page_no, line, {"kind": "recommendation", "rec": rec.group(1)})
                continue
            if heading:
                flush()
                sub = f"{heading.group(1)} {heading.group(2)}"
                label = ""
                title_may_continue = True
                prev_was_heading = True
                continue
            next_is_rec = i + 1 < len(lines) and bool(_NICE_REC.match(lines[i + 1][1]))
            prev_ends_sentence = i > 0 and lines[i - 1][1].endswith(".")
            if next_is_rec and (after_heading or prev_ends_sentence) and _is_subheading(line):
                flush()
                label = line.rstrip(" -‑")
                prev_was_heading = True
                continue
        elif (
            _NICE_RATIONALE_HEADING.match(line)
            and len(line.split()) <= _NICE_RATIONALE_HEADING_MAX_WORDS
        ):
            flush()
            sub = line
            continue

        if not run:
            metadata = {"kind": "text"}
        run.extend((word, page_no) for word in line.split())
    flush()
    return chunks


def load_pdf_pages(path: str | Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def load_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
