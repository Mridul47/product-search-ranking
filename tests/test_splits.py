import numpy as np
import pandas as pd
import pytest

from esci.data import splits as sp


@pytest.fixture
def examples() -> pd.DataFrame:
    rows = []
    for qid in range(100):
        split = "train" if qid < 80 else "test"
        for i in range(3):
            rows.append(
                {
                    "query_id": qid,
                    "product_id": f"p{qid}_{i}",
                    "product_locale": "us",
                    "small_version": 1,
                    "split": split,
                    "esci_label": ["E", "S", "I"][i],
                }
            )
    # rows that must be filtered out
    rows.append(
        {
            "query_id": 999,
            "product_id": "x",
            "product_locale": "es",
            "small_version": 1,
            "split": "train",
            "esci_label": "E",
        }
    )
    rows.append(
        {
            "query_id": 998,
            "product_id": "y",
            "product_locale": "us",
            "small_version": 0,
            "split": "train",
            "esci_label": "E",
        }
    )
    return pd.DataFrame(rows)


def test_filter_keeps_only_configured_scope(examples):
    subset = sp.filter_examples(examples, "us", 1)
    assert set(subset["product_locale"]) == {"us"}
    assert set(subset["small_version"]) == {1}
    assert subset["query_id"].nunique() == 100


def test_filter_rejects_empty_scope(examples):
    with pytest.raises(sp.SplitError, match="no rows"):
        sp.filter_examples(examples, "jp", 1)


def test_filter_rejects_missing_columns():
    with pytest.raises(sp.SplitError, match="missing columns"):
        sp.filter_examples(pd.DataFrame({"query_id": [1]}), "us", 1)


def test_splits_do_not_overlap(examples):
    subset = sp.filter_examples(examples, "us", 1)
    result = sp.split_query_ids(subset, val_fraction=0.1, seed=42)
    train, val, test = result["train"], result["val"], result["test"]
    assert np.intersect1d(train, val).size == 0
    assert np.intersect1d(train, test).size == 0
    assert np.intersect1d(val, test).size == 0


def test_split_sizes_match_fraction(examples):
    subset = sp.filter_examples(examples, "us", 1)
    result = sp.split_query_ids(subset, val_fraction=0.1, seed=42)
    assert len(result["val"]) == 8
    assert len(result["train"]) == 72
    assert len(result["test"]) == 20


def test_train_and_val_cover_all_train_queries(examples):
    subset = sp.filter_examples(examples, "us", 1)
    result = sp.split_query_ids(subset, val_fraction=0.1, seed=42)
    recovered = np.union1d(result["train"], result["val"])
    expected = np.sort(subset.loc[subset["split"] == "train", "query_id"].unique())
    assert np.array_equal(recovered, expected)


def test_same_seed_gives_same_split(examples):
    subset = sp.filter_examples(examples, "us", 1)
    a = sp.split_query_ids(subset, 0.1, seed=42)
    b = sp.split_query_ids(subset, 0.1, seed=42)
    c = sp.split_query_ids(subset, 0.1, seed=7)
    assert np.array_equal(a["val"], b["val"])
    assert not np.array_equal(a["val"], c["val"])


def test_bad_val_fraction_rejected(examples):
    subset = sp.filter_examples(examples, "us", 1)
    with pytest.raises(sp.SplitError, match="val_fraction"):
        sp.split_query_ids(subset, val_fraction=0.9, seed=42)


def test_round_trip_through_disk(examples, tmp_path):
    subset = sp.filter_examples(examples, "us", 1)
    result = sp.split_query_ids(subset, 0.1, seed=42)
    sp.write_splits(result, tmp_path)
    loaded = sp.read_splits(tmp_path)
    for name in sp.SPLIT_NAMES:
        assert np.array_equal(result[name], loaded[name])


def test_read_missing_splits_raises(tmp_path):
    with pytest.raises(sp.SplitError, match="Run scripts/make_splits.py"):
        sp.read_splits(tmp_path)


def test_summary_counts_are_consistent(examples):
    subset = sp.filter_examples(examples, "us", 1)
    result = sp.split_query_ids(subset, 0.1, seed=42)
    summary = sp.summarise(subset, result)
    assert list(summary["split"]) == ["train", "val", "test"]
    assert summary["queries"].sum() == 100
    assert summary["judgements"].sum() == len(subset)
