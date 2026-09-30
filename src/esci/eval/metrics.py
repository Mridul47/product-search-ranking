"""Ranking metrics.

Every setup in the project is scored by this module, so a mistake here would
affect every reported number. The implementations are deliberately plain, and
the tests check them against hand-computed values and against ranx.

Two evaluation modes (decision D8):

- judged: only judged products are ranked, as in the KDD Cup task. Comparable
  with published ESCI numbers.
- end-to-end: the real pipeline output is scored, and retrieved products with
  no judgement get gain 0. This is a lower bound, because some unjudged
  products are relevant.

The mode is decided by what is passed in as the run, not by a flag here.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

import numpy as np

from esci.eval.qrels import Qrels

Run = dict[int, np.ndarray]  # query_id -> doc_ids, best first
_EMPTY = np.empty(0, dtype=np.int64)


def dcg(gains: np.ndarray) -> float:
    """Discounted cumulative gain with the standard log2(rank + 1) discount."""
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(ranked_gains: np.ndarray, ideal_gains: np.ndarray, k: int) -> float:
    """NDCG@k for one query.

    ranked_gains holds the gains of the returned documents in rank order.
    ideal_gains holds every gain available for the query, in any order. The
    ideal ranking is built here by sorting.
    """
    ideal = np.sort(ideal_gains)[::-1][:k]
    best = dcg(ideal)
    if best == 0.0:
        # No gain exists anywhere, so no ranking can do better than another.
        return 0.0
    return dcg(ranked_gains[:k]) / best


def recall_at_k(retrieved: np.ndarray, relevant: AbstractSet[int], k: int) -> float:
    """Share of relevant documents found in the top k.

    Relevant means exact products only (decision D6). Callers must not pass
    queries with no relevant documents (decision D7).
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant documents")
    hits = sum(1 for doc in retrieved[:k] if int(doc) in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: np.ndarray, relevant: AbstractSet[int]) -> float:
    """1 / rank of the first relevant document, or 0 if none is retrieved."""
    if not relevant:
        raise ValueError("reciprocal rank is undefined for a query with no relevant documents")
    for position, doc in enumerate(retrieved, start=1):
        if int(doc) in relevant:
            return 1.0 / position
    return 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_run(
    run: Run,
    qrels: Qrels,
    ndcg_cutoffs: tuple[int, ...] = (5, 10),
    recall_cutoffs: tuple[int, ...] = (100,),
) -> dict[str, float]:
    """Score a run and return the metrics reported for every setup (D11).

    Queries present in qrels but missing from the run are scored as empty
    results rather than skipped, so a retriever cannot improve its average by
    returning nothing for hard queries. A document appearing twice in one
    query's results is an error, because it would be counted twice.
    """
    query_ids = qrels.query_ids
    if not query_ids:
        raise ValueError("qrels contain no queries")

    ndcg_scores: dict[int, list[float]] = {k: [] for k in ndcg_cutoffs}
    recall_scores: dict[int, list[float]] = {k: [] for k in recall_cutoffs}
    rr_scores: list[float] = []

    for qid in query_ids:
        judged = qrels.gains[qid]
        retrieved = np.asarray(run.get(qid, _EMPTY), dtype=np.int64)
        if np.unique(retrieved).size != retrieved.size:
            raise ValueError(f"run for query {qid} contains the same document more than once")

        # Unjudged retrieved documents contribute gain 0 (decision D8).
        ranked_gains = np.array([judged.get(int(doc), 0.0) for doc in retrieved], dtype=float)
        ideal_gains = np.fromiter(judged.values(), dtype=float, count=len(judged))
        for k in ndcg_cutoffs:
            ndcg_scores[k].append(ndcg_at_k(ranked_gains, ideal_gains, k))

        relevant = qrels.exact[qid]
        if relevant:
            for k in recall_cutoffs:
                recall_scores[k].append(recall_at_k(retrieved, relevant, k))
            rr_scores.append(reciprocal_rank(retrieved, relevant))

    results = {f"ndcg@{k}": _mean(scores) for k, scores in ndcg_scores.items()}
    results.update({f"recall@{k}": _mean(scores) for k, scores in recall_scores.items()})
    results["mrr"] = _mean(rr_scores)
    results["n_queries"] = float(len(query_ids))
    results["n_queries_recall"] = float(len(rr_scores))
    return results
