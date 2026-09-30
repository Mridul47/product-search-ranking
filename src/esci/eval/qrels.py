"""Relevance judgements in the form the metrics expect.

A qrels object maps each query to the gain of every judged product. Retrieval
runs are scored against it. Both are keyed by the integer ids used elsewhere
(query_id, doc_id) rather than strings, because the arrays involved are large
and integer lookups are much cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from esci.logging_utils import get_logger

logger = get_logger(__name__)

EXACT_LABEL = "E"


class QrelsError(RuntimeError):
    """Raised when judgements cannot be built from the given inputs."""


@dataclass(frozen=True)
class Qrels:
    """Judged products and their gains, per query.

    gains[qid] maps doc_id to relevance gain. exact[qid] is the set of exact
    products. Recall and MRR count exact products only (D6) and skip queries
    whose set is empty (D7). Storing the set explicitly means that rule does
    not depend on what the gain values happen to be.
    """

    gains: dict[int, dict[int, float]]
    exact: dict[int, frozenset[int]]

    @property
    def query_ids(self) -> list[int]:
        return sorted(self.gains)

    def eligible_for_recall(self) -> list[int]:
        """Queries with at least one exact product (decision D7)."""
        return sorted(qid for qid, docs in self.exact.items() if docs)

    def judged_docs(self, query_id: int) -> np.ndarray:
        """Doc ids judged for this query, used for judged-set evaluation."""
        return np.fromiter(self.gains.get(query_id, {}), dtype=np.int64)


def build_qrels(
    examples: pd.DataFrame,
    doc_id_by_product: pd.Series,
    gains: dict[str, float],
) -> Qrels:
    """Build Qrels from in-scope examples and the corpus doc_id mapping.

    doc_id_by_product is indexed by product_id and holds the doc_id. Judgements
    for products missing from the corpus are dropped: a product that is not in
    the index can never be retrieved, so counting it would put a ceiling on
    recall that no method could reach.
    """
    required = {"query_id", "product_id", "esci_label"}
    missing = required - set(examples.columns)
    if missing:
        raise QrelsError(f"examples frame is missing columns: {sorted(missing)}")

    unknown_labels = set(examples["esci_label"].unique()) - set(gains)
    if unknown_labels:
        raise QrelsError(f"labels with no configured gain: {sorted(unknown_labels)}")

    frame = examples.loc[:, ["query_id", "product_id", "esci_label"]].copy()
    frame["doc_id"] = frame["product_id"].map(doc_id_by_product)

    dropped = int(frame["doc_id"].isna().sum())
    if dropped:
        logger.warning("dropping %d judgements whose product is not in the corpus", dropped)
        frame = frame.dropna(subset=["doc_id"])
    if frame.empty:
        raise QrelsError("no judgements are left after matching products to the corpus")

    frame["doc_id"] = frame["doc_id"].astype(np.int64)
    frame["gain"] = frame["esci_label"].map(gains).astype(float)

    gains_by_query: dict[int, dict[int, float]] = {}
    exact_by_query: dict[int, frozenset[int]] = {}
    for qid, group in frame.groupby("query_id", sort=False):
        docs = group["doc_id"].to_numpy()
        gain_values = group["gain"].tolist()
        gains_by_query[int(qid)] = dict(zip(docs.tolist(), gain_values, strict=True))
        is_exact = (group["esci_label"] == EXACT_LABEL).to_numpy()
        exact_by_query[int(qid)] = frozenset(docs[is_exact].tolist())

    eligible = sum(1 for docs in exact_by_query.values() if docs)
    logger.info(
        "built qrels for %d queries (%d eligible for recall)", len(gains_by_query), eligible
    )
    return Qrels(gains=gains_by_query, exact=exact_by_query)
