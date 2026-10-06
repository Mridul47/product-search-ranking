"""Reference rankings that set the ceiling and the floor for every metric.

oracle_run is the ceiling, so it must score 1.0 on every metric. The random
and title-length runs are the floor. Judged-set NDCG is high even for random
ordering, because most judged products are exact or substitute. A model's
score only means something when it is read next to these runs.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from esci.eval.metrics import Run
from esci.eval.qrels import Qrels


def _judged_arrays(judged: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    docs = np.fromiter(judged.keys(), dtype=np.int64, count=len(judged))
    gains = np.fromiter(judged.values(), dtype=float, count=len(judged))
    return docs, gains


def oracle_run(qrels: Qrels) -> Run:
    """Judged products ordered by gain, best first."""
    run: Run = {}
    for qid in qrels.query_ids:
        docs, gains = _judged_arrays(qrels.gains[qid])
        run[qid] = docs[np.argsort(-gains, kind="stable")]
    return run


def random_judged_run(qrels: Qrels, seed: int) -> Run:
    """Judged products in random order. The floor for judged-set evaluation."""
    rng = np.random.default_rng(seed)
    run: Run = {}
    for qid in qrels.query_ids:
        docs, _ = _judged_arrays(qrels.gains[qid])
        run[qid] = rng.permutation(docs)
    return run


def title_length_run(qrels: Qrels, title_length: np.ndarray) -> Run:
    """Judged products ordered by title length, longest first.

    title_length is indexed by doc_id. The direction is fixed here, before any
    result exists, so it cannot be chosen after seeing which way scores higher.
    """
    run: Run = {}
    for qid in qrels.query_ids:
        docs, _ = _judged_arrays(qrels.gains[qid])
        run[qid] = docs[np.argsort(-title_length[docs], kind="stable")]
    return run


def random_corpus_run(query_ids: Iterable[int], n_docs: int, depth: int, seed: int) -> Run:
    """Random documents drawn from the whole corpus. The end-to-end floor.

    The expected Recall@depth is depth / n_docs, so the measured value should
    sit very close to that.
    """
    if depth > n_docs:
        raise ValueError(f"depth {depth} is larger than the corpus ({n_docs} documents)")
    rng = np.random.default_rng(seed)
    return {
        qid: rng.choice(n_docs, size=depth, replace=False).astype(np.int64)
        for qid in sorted(query_ids)
    }
