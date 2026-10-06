import numpy as np
import pandas as pd
import pytest

from esci.eval import checks as ck
from esci.eval.qrels import build_qrels

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


@pytest.fixture
def corpus() -> pd.DataFrame:
    return pd.DataFrame({"doc_id": [0, 1, 2, 3], "product_id": ["A", "B", "C", "D"]})


@pytest.fixture
def examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_id": ["A", "B", "C", "A", "D"],
            "esci_label": ["E", "S", "I", "S", "C"],
        }
    )


def _mapping(corpus):
    return corpus.set_index("product_id")["doc_id"]


def test_verify_passes_on_consistent_data(examples, corpus):
    qrels = build_qrels(examples, _mapping(corpus), GAINS)
    stats = ck.verify_qrels(qrels, examples, corpus, GAINS)
    assert stats == {
        "queries": 2,
        "judgements": 5,
        "exact_judgements": 1,
        "queries_without_exact": 1,
    }


def test_verify_catches_a_wrong_doc_id_mapping(examples, corpus):
    swapped = pd.Series({"A": 1, "B": 0, "C": 2, "D": 3})
    qrels = build_qrels(examples, swapped, GAINS)
    with pytest.raises(ck.CheckError, match="disagree"):
        ck.verify_qrels(qrels, examples, corpus, GAINS)


def test_verify_catches_missing_judgements(examples, corpus):
    qrels = build_qrels(examples.iloc[:4], _mapping(corpus), GAINS)
    with pytest.raises(ck.CheckError, match="hold"):
        ck.verify_qrels(qrels, examples, corpus, GAINS)


def test_verify_rejects_a_corpus_whose_doc_id_is_not_the_row_position(examples, corpus):
    corpus["doc_id"] = [0, 2, 1, 3]
    qrels = build_qrels(examples, _mapping(corpus), GAINS)
    with pytest.raises(ck.CheckError, match="row position"):
        ck.verify_qrels(qrels, examples, corpus, GAINS)


def test_judged_only_accepts_a_judged_run(examples, corpus):
    qrels = build_qrels(examples, _mapping(corpus), GAINS)
    ck.assert_judged_only({1: np.array([0, 1]), 2: np.array([3, 0])}, qrels)


def test_judged_only_rejects_unjudged_documents(examples, corpus):
    qrels = build_qrels(examples, _mapping(corpus), GAINS)
    with pytest.raises(ck.CheckError, match="not judged"):
        ck.assert_judged_only({1: np.array([0, 3])}, qrels)


def test_judged_only_ignores_queries_without_qrels(examples, corpus):
    qrels = build_qrels(examples, _mapping(corpus), GAINS)
    ck.assert_judged_only({99: np.array([1])}, qrels)
