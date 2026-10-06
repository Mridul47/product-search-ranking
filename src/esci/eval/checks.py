"""Integrity checks on judgements and runs.

verify_qrels compares the qrels with the raw examples by going through the
corpus row positions. This path does not depend on the product_id to doc_id
mapping used to build the qrels. An oracle run cannot do this job. It is built
from the same qrels, so it scores 1.0 even if the mapping is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from esci.eval.metrics import Run
from esci.eval.qrels import EXACT_LABEL, Qrels
from esci.logging_utils import get_logger

logger = get_logger(__name__)


class CheckError(RuntimeError):
    """Raised when judgements or a run fail an integrity check."""


def verify_qrels(
    qrels: Qrels,
    examples: pd.DataFrame,
    corpus: pd.DataFrame,
    gains: dict[str, float],
) -> dict[str, int]:
    """Check every qrels entry against the raw examples. Raises on any mismatch."""
    if not np.array_equal(corpus["doc_id"].to_numpy(), np.arange(len(corpus))):
        raise CheckError("corpus doc_id must equal the row position in the corpus")
    products = corpus["product_id"].to_numpy()

    raw = examples.loc[examples["query_id"].isin(list(qrels.gains))]
    raw = raw.loc[raw["product_id"].isin(corpus["product_id"])]
    raw = raw.drop_duplicates(["query_id", "product_id"])

    keys = zip(raw["query_id"].tolist(), raw["product_id"].tolist(), strict=True)
    labels = dict(zip(keys, raw["esci_label"].tolist(), strict=True))

    n_entries = 0
    disagree = 0
    for qid, judged in qrels.gains.items():
        for doc, gain in judged.items():
            n_entries += 1
            label = labels.get((qid, products[doc]))
            if label is None or gains[label] != gain:
                disagree += 1
    if disagree:
        raise CheckError(f"{disagree} of {n_entries} qrels entries disagree with the raw examples")
    if n_entries != len(raw):
        raise CheckError(f"qrels hold {n_entries} judgements but the examples hold {len(raw)}")

    exact_raw = raw.loc[raw["esci_label"] == EXACT_LABEL].groupby("query_id").size()
    wrong_exact = sum(
        1
        for qid, docs in qrels.exact.items()
        if len(docs) != int(exact_raw.get(qid, 0)) or not docs.issubset(qrels.gains[qid])
    )
    if wrong_exact:
        raise CheckError(f"{wrong_exact} queries have exact-product sets that do not match")

    return {
        "queries": len(qrels.gains),
        "judgements": n_entries,
        "exact_judgements": sum(len(docs) for docs in qrels.exact.values()),
        "queries_without_exact": sum(1 for docs in qrels.exact.values() if not docs),
    }


def assert_judged_only(run: Run, qrels: Qrels) -> None:
    """A judged-set run may only contain products that have a judgement.

    Queries that are not in the qrels are ignored here, because scoring only
    looks at queries that have judgements.
    """
    for qid, docs in run.items():
        judged = qrels.gains.get(qid)
        if judged is None:
            continue
        extra = [int(d) for d in docs if int(d) not in judged]
        if extra:
            raise CheckError(
                f"query {qid} has {len(extra)} documents that are not judged, "
                "so this run is not a judged-set run"
            )
