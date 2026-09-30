import numpy as np
import pytest

from esci.eval import metrics as m
from esci.eval.qrels import Qrels

LOG3 = np.log2(3.0)
# NDCG of the ranking [S, E] when the ideal ranking is [E, S]
SWAPPED = (0.1 + 1.0 / LOG3) / (1.0 + 0.1 / LOG3)


def test_dcg_known_value():
    assert m.dcg(np.array([1.0, 0.1])) == pytest.approx(1.0 + 0.1 / LOG3)


def test_dcg_of_nothing_is_zero():
    assert m.dcg(np.empty(0)) == 0.0


def test_ndcg_perfect_ranking_is_one():
    gains = np.array([1.0, 0.1])
    assert m.ndcg_at_k(gains, gains, k=10) == pytest.approx(1.0)


def test_ndcg_swapped_ranking():
    ranked = np.array([0.1, 1.0])
    ideal = np.array([1.0, 0.1])
    assert m.ndcg_at_k(ranked, ideal, k=10) == pytest.approx(SWAPPED)


def test_ndcg_ideal_uses_gains_that_were_not_retrieved():
    ranked = np.array([1.0])
    ideal = np.array([1.0, 1.0])
    expected = 1.0 / (1.0 + 1.0 / LOG3)
    assert m.ndcg_at_k(ranked, ideal, k=2) == pytest.approx(expected)


def test_ndcg_respects_the_cutoff():
    ranked = np.array([0.0, 0.0, 1.0])
    ideal = np.array([1.0])
    assert m.ndcg_at_k(ranked, ideal, k=2) == 0.0
    assert m.ndcg_at_k(ranked, ideal, k=3) == pytest.approx(0.5)  # 1 / log2(4)


def test_ndcg_is_zero_when_no_gain_exists():
    assert m.ndcg_at_k(np.array([0.0, 0.0]), np.array([0.0, 0.0]), k=5) == 0.0


def test_recall_counts_hits_in_the_top_k():
    retrieved = np.array([5, 6, 7])
    relevant = {5, 7, 9}
    assert m.recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert m.recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)
    assert m.recall_at_k(retrieved, relevant, k=100) == pytest.approx(2 / 3)


def test_recall_rejects_an_empty_relevant_set():
    with pytest.raises(ValueError, match="undefined"):
        m.recall_at_k(np.array([1]), set(), k=10)


def test_reciprocal_rank_uses_the_first_hit():
    retrieved = np.array([3, 4, 5])
    assert m.reciprocal_rank(retrieved, {5}) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(retrieved, {4, 5}) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_zero_without_a_hit():
    assert m.reciprocal_rank(np.array([1, 2]), {9}) == 0.0


def test_reciprocal_rank_rejects_an_empty_relevant_set():
    with pytest.raises(ValueError, match="undefined"):
        m.reciprocal_rank(np.array([1]), set())


@pytest.fixture
def qrels() -> Qrels:
    return Qrels(
        gains={1: {10: 1.0, 11: 0.1, 12: 0.0}, 2: {20: 0.1, 21: 0.0}},
        exact={1: frozenset({10}), 2: frozenset()},
    )


def test_evaluate_run_hand_computed(qrels):
    run = {1: np.array([11, 10]), 2: np.array([20])}
    result = m.evaluate_run(run, qrels)
    # query 1 is the swapped ranking, query 2 is perfect
    assert result["ndcg@5"] == pytest.approx((SWAPPED + 1.0) / 2)
    assert result["ndcg@10"] == pytest.approx((SWAPPED + 1.0) / 2)
    # query 2 has no exact product, so only query 1 counts for recall and MRR
    assert result["recall@100"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["n_queries"] == 2
    assert result["n_queries_recall"] == 1


def test_query_missing_from_the_run_scores_zero(qrels):
    result = m.evaluate_run({2: np.array([20])}, qrels)
    assert result["ndcg@5"] == pytest.approx(0.5)
    assert result["recall@100"] == 0.0
    assert result["mrr"] == 0.0


def test_unjudged_documents_score_zero(qrels):
    run = {1: np.array([99, 10]), 2: np.array([20])}
    result = m.evaluate_run(run, qrels)
    query_one = (1.0 / LOG3) / (1.0 + 0.1 / LOG3)
    assert result["ndcg@5"] == pytest.approx((query_one + 1.0) / 2)
    assert result["mrr"] == pytest.approx(0.5)


def test_duplicate_documents_are_rejected(qrels):
    run = {1: np.array([10, 10]), 2: np.array([20])}
    with pytest.raises(ValueError, match="more than once"):
        m.evaluate_run(run, qrels)


def test_oracle_run_scores_perfectly(qrels):
    run = {
        qid: np.array(sorted(judged, key=judged.get, reverse=True))
        for qid, judged in qrels.gains.items()
    }
    result = m.evaluate_run(run, qrels)
    assert result["ndcg@5"] == pytest.approx(1.0)
    assert result["ndcg@10"] == pytest.approx(1.0)
    assert result["recall@100"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)


def test_empty_qrels_raise():
    with pytest.raises(ValueError, match="no queries"):
        m.evaluate_run({}, Qrels(gains={}, exact={}))


def test_recall_is_nan_when_no_query_is_eligible():
    qrels = Qrels(gains={1: {5: 0.1}}, exact={1: frozenset()})
    result = m.evaluate_run({1: np.array([5])}, qrels)
    assert np.isnan(result["recall@100"])
    assert np.isnan(result["mrr"])
