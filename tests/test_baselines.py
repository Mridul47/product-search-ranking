import numpy as np
import pytest

from esci.eval import baselines as bl
from esci.eval.metrics import evaluate_run
from esci.eval.qrels import Qrels


@pytest.fixture
def qrels() -> Qrels:
    return Qrels(
        gains={
            1: {10: 1.0, 11: 0.1, 12: 0.0, 13: 1.0},
            2: {20: 0.1, 21: 0.01, 22: 1.0},
        },
        exact={1: frozenset({10, 13}), 2: frozenset({22})},
    )


def test_oracle_scores_perfectly(qrels):
    result = evaluate_run(bl.oracle_run(qrels), qrels)
    for key in ("ndcg@5", "ndcg@10", "recall@100", "mrr"):
        assert result[key] == pytest.approx(1.0)


def test_oracle_puts_exact_products_first(qrels):
    run = bl.oracle_run(qrels)
    assert set(run[1][:2].tolist()) == {10, 13}
    assert run[2][0] == 22


def test_random_judged_run_keeps_the_same_documents(qrels):
    run = bl.random_judged_run(qrels, seed=0)
    for qid, judged in qrels.gains.items():
        assert sorted(run[qid].tolist()) == sorted(judged)


def test_random_judged_run_is_reproducible_and_seed_dependent():
    big = Qrels(gains={1: {d: 0.1 for d in range(50)}}, exact={1: frozenset()})
    a = bl.random_judged_run(big, seed=0)
    b = bl.random_judged_run(big, seed=0)
    c = bl.random_judged_run(big, seed=1)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_title_length_orders_longest_first_with_stable_ties(qrels):
    lengths = np.zeros(30, dtype=int)
    lengths[11] = 50
    lengths[10] = 20
    lengths[12] = 20
    lengths[13] = 5
    run = bl.title_length_run(qrels, lengths)
    assert run[1].tolist() == [11, 10, 12, 13]
    assert run[2].tolist() == [20, 21, 22]


def test_random_corpus_run_shape_range_and_uniqueness():
    run = bl.random_corpus_run([3, 1, 2], n_docs=1000, depth=100, seed=0)
    assert sorted(run) == [1, 2, 3]
    for docs in run.values():
        assert docs.size == 100
        assert np.unique(docs).size == 100
        assert docs.min() >= 0
        assert docs.max() < 1000


def test_random_corpus_run_does_not_depend_on_query_order():
    a = bl.random_corpus_run([3, 1, 2], n_docs=1000, depth=10, seed=0)
    b = bl.random_corpus_run([1, 2, 3], n_docs=1000, depth=10, seed=0)
    assert all(np.array_equal(a[q], b[q]) for q in a)


def test_random_corpus_run_rejects_depth_larger_than_corpus():
    with pytest.raises(ValueError, match="depth"):
        bl.random_corpus_run([1], n_docs=5, depth=10, seed=0)
