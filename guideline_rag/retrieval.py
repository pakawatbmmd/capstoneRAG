from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import IO, Callable, Protocol

import numpy as np

from .types import Chunk, Hit

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _atomic_write(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write via a temp file in the same directory, then `os.replace` it into place."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


def _top_hits(chunks: list[Chunk], scores: list[float], k: int, positive_only: bool) -> list[Hit]:
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return [Hit(chunks[i], float(scores[i])) for i in order[:k]]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(tokenize(c.embed_text())) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avgdl = sum(self._lengths) / len(chunks) if chunks else 0.0
        doc_freq = Counter(term for tf in self._term_freqs for term in tf)
        n = len(chunks)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1) for term, df in doc_freq.items()
        }

    def _score(self, query_terms: list[str], doc: int) -> float:
        term_freqs, length = self._term_freqs[doc], self._lengths[doc]
        norm = 1 - self.b + self.b * length / self._avgdl
        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term)
            if tf:
                score += self._idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return score

    def search(self, query: str, k: int) -> list[Hit]:
        terms = tokenize(query)
        if not terms or self._avgdl == 0:
            return []
        scores = [self._score(terms, i) for i in range(len(self.chunks))]
        return _top_hits(self.chunks, scores, k, positive_only=True)


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        query_prefix: str = "",
        passage_prefix: str = "",
        store_dir: str | Path | None = None,
        embedder_name: str = "",
    ):
        if store_dir is not None and not embedder_name:
            raise ValueError("embedder_name is required when store_dir is set")
        self.chunks = chunks
        self.embedder = embedder
        self.query_prefix = query_prefix
        store = Path(store_dir) if store_dir is not None else None
        texts = [passage_prefix + c.embed_text() for c in chunks]
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        cached = self._load_store(store, embedder_name, passage_prefix) if store else {}

        missing = [i for i, key in enumerate(keys) if key not in cached]
        self.n_encoded = len(missing)
        fresh: dict[int, np.ndarray] = {}
        if missing:
            fresh = dict(zip(missing, np.asarray(embedder.encode([texts[i] for i in missing]))))
        rows = [fresh[i] if i in fresh else cached[key] for i, key in enumerate(keys)]
        self._matrix = (
            self._unit(np.vstack(rows)) if rows else np.zeros((0, 0), dtype=np.float32)
        )

        if store and rows:
            self._save_store(store, embedder_name, passage_prefix, keys)

    @staticmethod
    def _unit(vectors) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    @staticmethod
    def _load_store(store: Path, embedder_name: str, passage_prefix: str) -> dict[str, np.ndarray]:
        """Cached vectors by content key; a missing, mismatched or corrupt store is empty."""
        vectors_path, meta_path = store / "vectors.npz", store / "vectors_meta.json"
        if not (vectors_path.is_file() and meta_path.is_file()):
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("embedder_name") != embedder_name
                or meta.get("passage_prefix") != passage_prefix
            ):
                return {}
            with np.load(vectors_path, allow_pickle=False) as data:
                keys, vectors = data["keys"], data["vectors"]
            if vectors.ndim != 2 or len(keys) != len(vectors):
                return {}
            return {str(k): v for k, v in zip(keys, vectors)}
        except (OSError, ValueError, KeyError, EOFError, AttributeError, zipfile.BadZipFile):
            return {}

    def _save_store(
        self, store: Path, embedder_name: str, passage_prefix: str, keys: list[str]
    ) -> None:
        store.mkdir(parents=True, exist_ok=True)
        meta = {
            "embedder_name": embedder_name,
            "passage_prefix": passage_prefix,
            "dim": int(self._matrix.shape[1]),
        }
        _atomic_write(
            store / "vectors.npz",
            lambda f: np.savez(
                f,
                keys=np.array(keys, dtype=str),
                ids=np.array([c.id for c in self.chunks], dtype=str),
                vectors=self._matrix,
            ),
        )
        _atomic_write(store / "vectors_meta.json", lambda f: f.write(json.dumps(meta).encode("utf-8")))

    def search(self, query: str, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        query_vec = np.asarray(self.embedder.encode([self.query_prefix + query]), dtype=float)[0]
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        scores = (self._matrix @ (query_vec / norm)).tolist()
        return _top_hits(self.chunks, scores, k, positive_only=False)


class HybridRetriever:
    """Reciprocal rank fusion over several retrievers."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, candidate_k: int = 50):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, k: int) -> list[Hit]:
        chunks: dict[str, Chunk] = {}
        scores: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.candidate_k), start=1):
                cid = hit.chunk.id
                chunks.setdefault(cid, hit.chunk)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=lambda cid: -scores[cid])
        return [Hit(chunks[cid], scores[cid]) for cid in ranked[:k]]
