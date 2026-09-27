"""Shared analysis helpers.

Only calculations reused outside the EDA notebook live here, so they can be
tested once and relied on later:

- exact_per_query: decision D7 excludes queries with no exact product from
  Recall and MRR, so the evaluation harness needs this same computation.
- describe_series: the percentile summary used for length distributions now
  and for latency distributions in the benchmarking phase.
- duplicate_queries_across_splits: a leakage check. A broken leakage check
  reports no leakage and is believed, so it needs a test.

One-off tables and plots stay in the notebook.
"""

from __future__ import annotations

import pandas as pd

ESCI_LABELS = ("E", "S", "C", "I")
LABEL_NAMES = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}


def exact_per_query(examples: pd.DataFrame) -> pd.Series:
    """Number of exact matches per query, including queries that have none.

    Recall@k is capped by how many exact products a query actually has.
    A query with 3 exact products can score at most 3, however good the
    retriever is. Queries with zero exact products are dropped from
    Recall and MRR (see decision D7).
    """
    exact = examples[examples["esci_label"] == "E"].groupby("query_id").size()
    return exact.reindex(examples["query_id"].unique(), fill_value=0).rename("exact_products")


def describe_series(series: pd.Series) -> pd.Series:
    """The percentile summary of a numeric series."""
    return pd.Series(
        {
            "count": int(series.size),
            "mean": round(float(series.mean()), 2),
            "min": int(series.min()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p90": float(series.quantile(0.90)),
            "p99": float(series.quantile(0.99)),
            "max": int(series.max()),
        }
    )


def duplicate_queries_across_splits(
    examples: pd.DataFrame,
    splits: dict[str, object],
) -> pd.DataFrame:
    """Query strings that appear in more than one split.

    We split by query_id, so the same wording can end up on both sides
    under different IDs. This is mild leakage: the model could memorise
    a training query and then be scored on it at test time.
    Matching is case-insensitive, with whitespace trimmed.
    """
    lookup = examples.drop_duplicates("query_id").set_index("query_id")["query"]
    frames = [
        pd.DataFrame({"query": lookup.reindex(ids).to_numpy(), "split": name})
        for name, ids in splits.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["normalised"] = combined["query"].str.lower().str.strip()

    per_query = combined.groupby("normalised")["split"].nunique()
    shared = per_query[per_query > 1]
    return pd.DataFrame(
        {
            "metric": ["unique query strings", "strings in more than one split", "percent"],
            "value": [
                int(per_query.size),
                int(shared.size),
                round(100 * shared.size / max(per_query.size, 1), 2),
            ],
        }
    )
