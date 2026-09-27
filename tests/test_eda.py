import numpy as np
import pandas as pd

from esci.data import eda


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 3],
            "query": ["red shoes"] * 3 + ["blue mug"] * 2 + ["Red Shoes"],
            "product_id": list("abcdef"),
            "esci_label": ["E", "E", "S", "S", "I", "C"],
        }
    )


def test_exact_per_query_includes_queries_with_none():
    counts = eda.exact_per_query(_examples())
    assert counts.to_dict() == {1: 2, 2: 0, 3: 0}


def test_exact_per_query_handles_no_exact_labels_at_all():
    frame = _examples()
    frame["esci_label"] = "S"
    assert eda.exact_per_query(frame).sum() == 0


def test_describe_series_percentiles():
    summary = eda.describe_series(pd.Series(range(1, 101)))
    assert summary["count"] == 100
    assert summary["min"] == 1
    assert summary["max"] == 100
    assert summary["p50"] == 50.5


def test_duplicate_queries_are_detected_case_insensitively():
    splits = {"train": np.array([1, 2]), "test": np.array([3])}
    result = eda.duplicate_queries_across_splits(_examples(), splits)
    values = dict(zip(result["metric"], result["value"], strict=True))
    # "red shoes" in train and "Red Shoes" in test are the same string
    assert values["unique query strings"] == 2
    assert values["strings in more than one split"] == 1
    assert values["percent"] == 50.0


def test_no_duplicates_reported_when_splits_are_disjoint():
    frame = _examples()
    frame.loc[frame.query_id == 3, "query"] = "green hat"
    splits = {"train": np.array([1, 2]), "test": np.array([3])}
    result = eda.duplicate_queries_across_splits(frame, splits)
    values = dict(zip(result["metric"], result["value"], strict=True))
    assert values["strings in more than one split"] == 0
