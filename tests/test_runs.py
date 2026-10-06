import numpy as np
import pandas as pd
import pytest

from esci.eval import runs as rn


def test_round_trip(tmp_path):
    run = {1: np.array([5, 3, 9]), 2: np.array([7])}
    path = rn.write_run(run, tmp_path / "nested" / "run.parquet")
    loaded = rn.read_run(path)
    assert sorted(loaded) == [1, 2]
    assert np.array_equal(loaded[1], run[1])
    assert np.array_equal(loaded[2], run[2])


def test_rows_are_ordered_by_rank_on_read(tmp_path):
    path = tmp_path / "run.parquet"
    frame = pd.DataFrame({"query_id": [1, 1, 1], "doc_id": [30, 10, 20], "rank": [3, 1, 2]})
    frame.to_parquet(path, index=False)
    assert rn.read_run(path)[1].tolist() == [10, 20, 30]


def test_empty_run_round_trips(tmp_path):
    path = rn.write_run({}, tmp_path / "empty.parquet")
    assert rn.read_run(path) == {}


def test_duplicate_rank_is_rejected(tmp_path):
    path = tmp_path / "run.parquet"
    pd.DataFrame({"query_id": [1, 1], "doc_id": [1, 2], "rank": [1, 1]}).to_parquet(path)
    with pytest.raises(rn.RunError, match="same rank"):
        rn.read_run(path)


def test_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "run.parquet"
    pd.DataFrame({"query_id": [1]}).to_parquet(path)
    with pytest.raises(rn.RunError, match="missing columns"):
        rn.read_run(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(rn.RunError, match="not found"):
        rn.read_run(tmp_path / "nope.parquet")
