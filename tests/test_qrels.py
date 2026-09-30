import numpy as np
import pandas as pd
import pytest

from esci.eval import qrels as qr

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


@pytest.fixture
def examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 3],
            "product_id": ["A", "B", "C", "A", "D", "Z"],
            "esci_label": ["E", "S", "I", "S", "C", "E"],
        }
    )


@pytest.fixture
def doc_ids() -> pd.Series:
    # product Z is deliberately absent from the corpus
    return pd.Series({"A": 0, "B": 1, "C": 2, "D": 3})


def test_gains_and_exact_sets(examples, doc_ids):
    qrels = qr.build_qrels(examples, doc_ids, GAINS)
    assert qrels.gains[1] == {0: 1.0, 1: 0.1, 2: 0.0}
    assert qrels.gains[2] == {0: 0.1, 3: 0.01}
    assert qrels.exact[1] == frozenset({0})
    assert qrels.exact[2] == frozenset()


def test_products_missing_from_corpus_are_dropped(examples, doc_ids, caplog):
    with caplog.at_level("WARNING"):
        qrels = qr.build_qrels(examples, doc_ids, GAINS)
    assert qrels.query_ids == [1, 2]
    assert "not in the corpus" in caplog.text


def test_eligible_for_recall_excludes_queries_without_exact(examples, doc_ids):
    qrels = qr.build_qrels(examples, doc_ids, GAINS)
    assert qrels.eligible_for_recall() == [1]


def test_exact_set_does_not_depend_on_gain_values(examples, doc_ids):
    changed = {"E": 5.0, "S": 1.0, "C": 0.5, "I": 0.0}
    qrels = qr.build_qrels(examples, doc_ids, changed)
    assert qrels.exact[1] == frozenset({0})
    assert qrels.exact[2] == frozenset()


def test_judged_docs(examples, doc_ids):
    qrels = qr.build_qrels(examples, doc_ids, GAINS)
    assert sorted(qrels.judged_docs(1).tolist()) == [0, 1, 2]
    assert qrels.judged_docs(999).size == 0
    assert qrels.judged_docs(1).dtype == np.int64


def test_unknown_label_raises(examples, doc_ids):
    examples.loc[0, "esci_label"] = "X"
    with pytest.raises(qr.QrelsError, match="no configured gain"):
        qr.build_qrels(examples, doc_ids, GAINS)


def test_missing_columns_raises(doc_ids):
    with pytest.raises(qr.QrelsError, match="missing columns"):
        qr.build_qrels(pd.DataFrame({"query_id": [1]}), doc_ids, GAINS)


def test_all_judgements_dropped_raises(examples):
    nothing = pd.Series({"NOPE": 0})
    with pytest.raises(qr.QrelsError, match="no judgements are left"):
        qr.build_qrels(examples, nothing, GAINS)
