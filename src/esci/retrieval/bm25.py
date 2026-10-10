"""BM25 index over the product corpus, built with bm25s."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

META_FILE = "esci_bm25_meta.json"


@dataclass(frozen=True)
class TokenizerSpec:
    """How text is split into tokens. The same spec must be used for index and queries."""

    stopwords: bool = False
    stem: bool = False
    token_pattern: str = r"(?u)\b\w\w+\b"

    def _kwargs(self) -> dict:
        return {
            "lower": True,
            "stopwords": "en" if self.stopwords else None,
            "stemmer": Stemmer.Stemmer("english") if self.stem else None,
            "token_pattern": self.token_pattern,
            "show_progress": False,
        }

    def tokenize_corpus(self, texts: Sequence[str]):
        return bm25s.tokenize(list(texts), return_ids=True, **self._kwargs())

    def tokenize_queries(self, queries: Sequence[str]) -> list[list[str]]:
        return bm25s.tokenize(list(queries), return_ids=False, **self._kwargs())


def build_index(
    texts: Sequence[str],
    spec: TokenizerSpec,
    k1: float = 1.5,
    b: float = 0.75,
    method: str = "lucene",
) -> tuple[bm25s.BM25, float]:
    """Build an index. Returns the retriever and the build time in seconds."""
    start = time.perf_counter()
    tokens = spec.tokenize_corpus(texts)
    retriever = bm25s.BM25(k1=k1, b=b, method=method)
    retriever.index(tokens, show_progress=False)
    return retriever, time.perf_counter() - start


def search(
    retriever: bm25s.BM25,
    spec: TokenizerSpec,
    queries: Sequence[str],
    k: int,
    n_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (doc_ids, scores), each shaped (n_queries, k), best first."""
    query_tokens = spec.tokenize_queries(queries)
    docs, scores = retriever.retrieve(query_tokens, k=k, n_threads=n_threads, show_progress=False)
    return np.asarray(docs), np.asarray(scores)


def save_index(
    retriever: bm25s.BM25,
    spec: TokenizerSpec,
    directory: Path,
    params: dict,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    retriever.save(str(directory))
    meta = {"tokenizer": asdict(spec), "params": params}
    (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(directory: Path) -> tuple[bm25s.BM25, TokenizerSpec, dict]:
    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    retriever = bm25s.BM25.load(str(directory), load_corpus=False)
    return retriever, TokenizerSpec(**meta["tokenizer"]), meta["params"]


def index_size_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
