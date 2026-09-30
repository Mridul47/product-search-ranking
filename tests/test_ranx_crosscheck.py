"""Cross-check the metric implementations against ranx on random data."""

import numpy as np
import pytest

from esci.eval.metrics import evaluate_run
from esci.eval.qrels import Qrels

ranx = pytest.importorskip("ranx")

GAIN_BY_LABEL = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
LABELS = list(GAIN_BY_LABEL)
LABEL_PROBS = [0.3, 0.35, 0.05, 0.3]


def _random_problem(seed: int, n_queries: int = 60, n_judged: int = 20, depth: int = 25):
    rng = np.random.default_rng(seed)
    n_unjudged = 10
    gains: dict[int, dict[int, float]] = {}
    exact: dict[int, frozenset[int]] = {}
    run: dict[int, np.ndarray] = {}

    for qid in range(n_queries):
        docs = np.arange(qid * n_judged, (qid + 1) * n_judged)
        labels = rng.choice(LABELS, size=n_judged, p=LABEL_PROBS)
        labels[0] = "E"  # every query has an exact product, so recall and MRR apply
        gains[qid] = {int(d): GAIN_BY_LABEL[str(x)] for d, x in zip(docs, labels, strict=True)}
        exact[qid] = frozenset(int(d) for d, x in zip(docs, labels, strict=True) if x == "E")

        start = 1_000_000 + qid * n_unjudged
        unjudged = np.arange(start, start + n_unjudged)
        pool = np.concatenate([docs, unjudged])
        run[qid] = rng.permutation(pool)[:depth]

    return Qrels(gains=gains, exact=exact), run


def _to_ranx(qrels: Qrels, run: dict[int, np.ndarray]):
    graded = {
        str(qid): {str(doc): int(round(gain * 100)) for doc, gain in docs.items()}
        for qid, docs in qrels.gains.items()
    }
    binary = {
        str(qid): {str(doc): int(doc in qrels.exact[qid]) for doc in docs}
        for qid, docs in qrels.gains.items()
    }
    scores = {
        str(qid): {str(int(doc)): float(len(order) - i) for i, doc in enumerate(order)}
        for qid, order in run.items()
    }
    return ranx.Qrels.from_dict(graded), ranx.Qrels.from_dict(binary), ranx.Run.from_dict(scores)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_ranx(seed):
    qrels, run = _random_problem(seed)
    ours = evaluate_run(run, qrels)

    graded, binary, ranx_run = _to_ranx(qrels, run)
    ndcg = ranx.evaluate(graded, ranx_run, ["ndcg@5", "ndcg@10"])
    ranking = ranx.evaluate(binary, ranx_run, ["recall@100", "mrr"])

    assert ours["ndcg@5"] == pytest.approx(ndcg["ndcg@5"], abs=1e-6)
    assert ours["ndcg@10"] == pytest.approx(ndcg["ndcg@10"], abs=1e-6)
    assert ours["recall@100"] == pytest.approx(ranking["recall@100"], abs=1e-6)
    assert ours["mrr"] == pytest.approx(ranking["mrr"], abs=1e-6)
