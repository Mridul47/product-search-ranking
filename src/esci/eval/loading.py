"""Load one split's corpus, qrels and query text, the same way evaluate.py does."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from esci.config import Config
from esci.data.corpus import read_corpus
from esci.data.download import DATASET_FILES, DATASET_SUBDIR
from esci.data.splits import filter_examples, read_splits
from esci.eval.checks import CheckError, verify_qrels
from esci.eval.qrels import Qrels, build_qrels
from esci.logging_utils import get_logger

logger = get_logger("loading")

EXAMPLE_COLUMNS = [
    "query_id",
    "query",
    "product_id",
    "product_locale",
    "small_version",
    "split",
    "esci_label",
]


@dataclass(frozen=True)
class SplitData:
    corpus: pd.DataFrame
    qrels: Qrels
    queries: dict[int, str]


def load_split(cfg: Config, split: str) -> SplitData:
    examples_path = cfg.paths.raw_dir / "esci-data" / DATASET_SUBDIR / DATASET_FILES[0]
    if not examples_path.is_file():
        raise CheckError(f"{examples_path} not found. Run scripts/download_data.py")

    raw = pd.read_parquet(examples_path, columns=EXAMPLE_COLUMNS)
    examples = filter_examples(raw, cfg.data.locale, cfg.data.small_version)
    splits = read_splits(cfg.paths.processed_dir / "splits")
    corpus = read_corpus(cfg.paths.processed_dir)

    subset = examples.loc[examples["query_id"].isin(splits[split])]
    doc_ids = corpus.set_index("product_id")["doc_id"]
    qrels = build_qrels(subset, doc_ids, cfg.relevance.gains)
    stats = verify_qrels(qrels, subset, corpus, cfg.relevance.gains)
    logger.info("qrels verified against the raw examples: %s", stats)

    text = subset.drop_duplicates("query_id").set_index("query_id")["query"]
    queries = {int(q): str(text.loc[q]) for q in qrels.gains}
    return SplitData(corpus=corpus, qrels=qrels, queries=queries)
