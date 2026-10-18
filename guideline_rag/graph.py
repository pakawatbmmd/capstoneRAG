"""Entity-graph retriever: chunk-entity-hierarchy graph scored by Personalized PageRank."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .types import Chunk, Hit

_EXPLAIN_ENTITIES = 5
_EXPLAIN_CHUNKS = 3

_LLM_PROMPT = """List the medical entities (conditions, drugs, drug classes, tests, procedures) mentioned in the text below.

Reply with a JSON array of lowercase strings only, for example ["metformin", "type 2 diabetes"].

Text:
{text}

JSON:"""


def _clean(value: str) -> str:
    return value.strip().lower()


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _require_lists(section: str, mapping: dict) -> dict[str, list]:
    for key, values in mapping.items():
        if not isinstance(values, list):
            raise TypeError(f"lexicon {section}[{key!r}] must be a list, got {type(values).__name__}")
    return mapping


def load_lexicon(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    entities: dict[str, list[str]] = {}
    for name, aliases in _require_lists("entities", raw.get("entities") or {}).items():
        canonical = _clean(name)
        merged = entities.setdefault(canonical, [])
        merged.extend(a for a in map(_clean, aliases) if a)
        entities[canonical] = _dedupe(merged + [canonical])

    is_a: dict[str, list[str]] = {}
    for child, parents in _require_lists("is_a", raw.get("is_a") or {}).items():
        key = _clean(child)
        is_a[key] = _dedupe(is_a.get(key, []) + [p for p in map(_clean, parents) if p])
    return {"entities": entities, "is_a": is_a}


class LexiconExtractor:
    def __init__(self, lexicon: dict):
        self._patterns = {
            name: re.compile("|".join(self._alias_pattern(a) for a in aliases if a.split()))
            for name, aliases in lexicon["entities"].items()
            if any(a.split() for a in aliases)
        }

    @staticmethod
    def _alias_pattern(alias: str) -> str:
        words = r"\s+".join(re.escape(word) for word in alias.lower().split())
        return rf"(?<![a-z0-9]){words}(?:s|es)?(?![a-z0-9])"

    def extract(self, text: str) -> list[str]:
        lowered = text.lower()
        return sorted(name for name, pattern in self._patterns.items() if pattern.search(lowered))


class LLMEntityExtractor:
    def __init__(self, llm: Callable[[str], str]):
        self.llm = llm

    def extract(self, text: str) -> list[str]:
        reply = self.llm(_LLM_PROMPT.format(text=text))
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\[", reply):
            try:
                items, _ = decoder.raw_decode(reply, match.start())
            except ValueError:
                continue
            cleaned = {_clean(item) for item in items if isinstance(item, str)}
            return sorted(cleaned - {""})
        return []


class GraphRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        extractor,
        lexicon: dict | None = None,
        damping: float = 0.5,
        iterations: int = 30,
        fallback=None,
    ):
        self.chunks = chunks
        self.extractor = extractor
        self.damping = damping
        self.iterations = iterations
        self.fallback = fallback

        self._chunk_entities = [sorted(set(extractor.extract(c.embed_text()))) for c in chunks]
        is_a = (lexicon.get("is_a") or {}) if lexicon else {}
        self._is_a_pairs = sorted(
            {(child, parent) for child, parents in is_a.items() for parent in parents if child != parent}
        )

        names = {e for ents in self._chunk_entities for e in ents}
        names.update(is_a)
        names.update(parent for parents in is_a.values() for parent in parents)
        self._entity_names = sorted(names)
        n_chunks = len(chunks)
        self._entity_node = {name: n_chunks + i for i, name in enumerate(self._entity_names)}
        self._n_nodes = n_chunks + len(self._entity_names)

        self._is_a_neighbours: dict[str, set[str]] = {}
        for child, parent in self._is_a_pairs:
            self._is_a_neighbours.setdefault(child, set()).add(parent)
            self._is_a_neighbours.setdefault(parent, set()).add(child)

        edges = [
            (ci, self._entity_node[e]) for ci, ents in enumerate(self._chunk_entities) for e in ents
        ]
        mention_df = np.bincount(
            np.array([entity for _, entity in edges], dtype=np.int64), minlength=self._n_nodes
        )
        edges += [(self._entity_node[c], self._entity_node[p]) for c, p in self._is_a_pairs]
        pairs = np.array(edges, dtype=np.int64).reshape(-1, 2)
        self._src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        self._dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        degree = np.bincount(self._src, minlength=self._n_nodes).astype(float)
        self._dangling = degree == 0
        self._inv_degree = np.divide(1.0, degree, out=np.zeros(self._n_nodes), where=~self._dangling)
        self._idf = np.log1p(n_chunks / np.maximum(mention_df, 1))

    def _query_entities(self, query: str) -> list[str]:
        return sorted(e for e in set(self.extractor.extract(query)) if e in self._entity_node)

    def _ppr(self, query_entities: list[str]) -> np.ndarray:
        teleport = np.zeros(self._n_nodes)
        for name in query_entities:
            node = self._entity_node[name]
            teleport[node] = self._idf[node]
        total = teleport.sum()
        if total <= 0:
            return teleport
        teleport /= total

        d = self.damping
        rank = teleport.copy()
        for _ in range(self.iterations):
            spread = np.bincount(
                self._dst, weights=(rank * self._inv_degree)[self._src], minlength=self._n_nodes
            )
            rank = (1 - d) * teleport + d * spread + d * rank[self._dangling].sum() * teleport
        return rank

    def _ranked_chunks(self, rank: np.ndarray) -> np.ndarray:
        scores = rank[: len(self.chunks)]
        positive = np.flatnonzero(scores > 0)
        return positive[np.argsort(-scores[positive], kind="stable")]

    def search(self, query: str, k: int) -> list[Hit]:
        query_entities = self._query_entities(query)
        if not query_entities:
            return self.fallback.search(query, k) if self.fallback is not None else []
        rank = self._ppr(query_entities)
        return [Hit(self.chunks[i], float(rank[i])) for i in self._ranked_chunks(rank)[:k]]

    def explain(self, query: str) -> dict:
        query_entities = self._query_entities(query)
        if not query_entities:
            return {"query_entities": [], "top_entities": [], "paths": {}}
        rank = self._ppr(query_entities)

        entity_rank = rank[len(self.chunks) :]
        positive = np.flatnonzero(entity_rank > 0)
        by_score = positive[np.argsort(-entity_rank[positive], kind="stable")]
        top_entities = [
            (self._entity_names[i], float(entity_rank[i])) for i in by_score[:_EXPLAIN_ENTITIES]
        ]

        linking = set(query_entities)
        for name in query_entities:
            linking |= self._is_a_neighbours.get(name, set())
        paths = {
            self.chunks[i].id: sorted(linking.intersection(self._chunk_entities[i]))
            for i in self._ranked_chunks(rank)[:_EXPLAIN_CHUNKS]
        }
        return {"query_entities": query_entities, "top_entities": top_entities, "paths": paths}

    def export_json(self, path: str | Path) -> None:
        nodes = [{"id": f"chunk:{c.id}", "type": "chunk", "label": c.id} for c in self.chunks]
        nodes += [{"id": f"entity:{e}", "type": "entity", "label": e} for e in self._entity_names]
        mentions = [
            (f"chunk:{chunk.id}", f"entity:{entity}")
            for chunk, ents in zip(self.chunks, self._chunk_entities)
            for entity in ents
        ]
        is_a = [(f"entity:{child}", f"entity:{parent}") for child, parent in self._is_a_pairs]
        edges = [
            {"source": source, "target": target, "type": kind, "weight": 1.0}
            for kind, pairs in (("is_a", is_a), ("mentions", mentions))
            for source, target in pairs
        ]
        document = {
            "nodes": sorted(nodes, key=lambda n: n["id"]),
            "edges": sorted(edges, key=lambda e: (e["type"], e["source"], e["target"])),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)


def load_graph_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
