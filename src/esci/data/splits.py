"""Query-level train, validation and test splits.

Test comes from the dataset's own split column.
Validation is carved out of the train queries with a fixed seed.
Splitting on query IDs rather than rows keeps every judgement for a query on one side,
so a model never sees part of a query's judgements during training and then gets scored on the rest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from esci.logging_utils import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_COLUMNS = ("query_id", "product_locale", "small_version", "split")


class SplitError(RuntimeError):
    """Raised when the split cannot be built from the given examples."""


def filter_examples(examples: pd.DataFrame, locale: str, small_version: int) -> pd.DataFrame:
    """Restrict the examples table to the configured locale and version."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in examples.columns]
    if missing:
        raise SplitError(f"examples table is missing columns: {missing}")

    mask = (examples["product_locale"] == locale) & (examples["small_version"] == small_version)
    subset = examples.loc[mask]
    if subset.empty:
        raise SplitError(f"no rows for locale={locale!r}, small_version={small_version}")

    logger.info(
        "filtered examples to %d rows (%d queries) for locale=%s small_version=%d",
        len(subset),
        subset["query_id"].nunique(),
        locale,
        small_version,
    )
    return subset


def split_query_ids(
    examples: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return sorted query IDs for each split.

    Validation is a random sample of the train queries. Sampling IDs rather than
    rows is what prevents a query from landing in both train and validation.
    """
    if not 0.0 < val_fraction < 0.5:
        raise SplitError(f"val_fraction must be between 0 and 0.5, got {val_fraction}")

    by_split = examples.groupby("split")["query_id"].unique()
    for name in ("train", "test"):
        if name not in by_split.index:
            raise SplitError(f"examples contain no {name!r} split")

    train_pool = np.sort(np.asarray(by_split.loc["train"]))
    test_ids = np.sort(np.asarray(by_split.loc["test"]))

    overlap = np.intersect1d(train_pool, test_ids)
    if overlap.size:
        raise SplitError(f"{overlap.size} query IDs appear in both train and test upstream")

    n_val = int(round(len(train_pool) * val_fraction))
    if n_val == 0:
        raise SplitError("val_fraction is too small: it selects zero queries")

    rng = np.random.default_rng(seed)
    val_ids = np.sort(rng.choice(train_pool, size=n_val, replace=False))
    train_ids = np.sort(np.setdiff1d(train_pool, val_ids))

    logger.info(
        "split queries: train=%d val=%d test=%d (seed=%d)",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        seed,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_splits(splits: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    """Write one parquet file of query IDs per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}_query_ids.parquet"
        pd.DataFrame({"query_id": splits[name]}).to_parquet(path, index=False)
        logger.info("wrote %d query ids to %s", len(splits[name]), path)
        written[name] = path
    return written


def read_splits(out_dir: Path) -> dict[str, np.ndarray]:
    """Load previously written splits. Every downstream script uses this."""
    splits: dict[str, np.ndarray] = {}
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}_query_ids.parquet"
        if not path.is_file():
            raise SplitError(f"split file not found: {path}. Run scripts/make_splits.py first.")
        splits[name] = pd.read_parquet(path)["query_id"].to_numpy()
    return splits


def summarise(examples: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-split counts of queries, judgements and label mix, for the EDA notes."""
    rows = []
    for name in SPLIT_NAMES:
        subset = examples[examples["query_id"].isin(splits[name])]
        label_counts = subset["esci_label"].value_counts()
        n_queries = subset["query_id"].nunique()
        rows.append(
            {
                "split": name,
                "queries": n_queries,
                "judgements": len(subset),
                "judgements_per_query": round(len(subset) / max(n_queries, 1), 2),
                **{f"n_{label}": int(label_counts.get(label, 0)) for label in ("E", "S", "C", "I")},
            }
        )
    return pd.DataFrame(rows)
