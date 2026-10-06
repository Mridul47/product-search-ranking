"""Read and write retrieval runs.

A run file is a parquet table with one row per returned document. Its columns
are query_id, doc_id and rank (1 is best). Every retrieval and ranking script
writes this format, and scripts/evaluate.py reads it. That way any setup is
scored the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUN_COLUMNS = ("query_id", "doc_id", "rank")


class RunError(RuntimeError):
    """Raised when a run file is missing or malformed."""


def _join(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def write_run(run: dict[int, np.ndarray], path: Path) -> Path:
    """Write a run to parquet. A query with no results writes no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = [np.asarray(order, dtype=np.int64) for order in run.values()]
    ids = [np.full(d.size, qid, dtype=np.int64) for qid, d in zip(run, docs, strict=True)]
    ranks = [np.arange(1, d.size + 1, dtype=np.int64) for d in docs]
    frame = pd.DataFrame({"query_id": _join(ids), "doc_id": _join(docs), "rank": _join(ranks)})
    frame.to_parquet(path, index=False)
    return path


def read_run(path: Path) -> dict[int, np.ndarray]:
    """Load a run as query_id -> doc_ids, best first."""
    if not path.is_file():
        raise RunError(f"run file not found: {path}")
    frame = pd.read_parquet(path)
    missing = [c for c in RUN_COLUMNS if c not in frame.columns]
    if missing:
        raise RunError(f"run file is missing columns: {missing}")
    if frame.duplicated(["query_id", "rank"]).any():
        raise RunError("run file has two documents at the same rank for one query")

    frame = frame.sort_values(["query_id", "rank"], kind="stable")
    return {
        int(qid): group["doc_id"].to_numpy(dtype=np.int64)
        for qid, group in frame.groupby("query_id", sort=False)
    }
