from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import Chunk


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]
