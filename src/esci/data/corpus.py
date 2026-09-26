"""Build the retrieval corpus from the raw ESCI product catalogue.

Decision D5: the corpus is every US product that appears in the small-version
examples, train and test combined. This keeps the index small enough to embed
on free GPU quota, and the distractors are still realistic.

The corpus carries an integer doc_id. FAISS returns integer positions, and
integers are far cheaper than strings for the candidate arrays used in
re-ranking, so the mapping is fixed once here and reused everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from esci.data.text import TEXT_FIELDS, build_text_column, text_diagnostics
from esci.logging_utils import get_logger

logger = get_logger(__name__)

PRODUCT_COLUMNS = ("product_id", "product_locale", *TEXT_FIELDS)
CORPUS_COLUMNS = ("doc_id", "product_id", "product_title", "product_brand", "product_text")
CORPUS_FILENAME = "corpus.parquet"


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be built from the given inputs."""


def products_in_scope(examples: pd.DataFrame) -> pd.Index:
    """Unique product IDs judged for any in-scope query."""
    if "product_id" not in examples.columns:
        raise CorpusError("examples frame has no product_id column")
    ids = pd.Index(examples["product_id"].unique(), name="product_id")
    logger.info("%d unique products appear in the in-scope examples", len(ids))
    return ids


def build_corpus(products: pd.DataFrame, examples: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Filter the catalogue to the in-scope products and build their text.

    Products are sorted by product_id before doc_id is assigned, so the same
    inputs always produce the same mapping. Without that, an index built today
    would not line up with candidate files written yesterday.
    """
    wanted = products_in_scope(examples)

    catalogue = products.loc[products["product_locale"] == locale]
    corpus = catalogue.loc[catalogue["product_id"].isin(wanted)].copy()

    if corpus.empty:
        raise CorpusError(f"no products matched locale={locale!r} and the in-scope product ids")

    duplicates = int(corpus["product_id"].duplicated().sum())
    if duplicates:
        logger.warning("dropping %d duplicate product_id rows from the catalogue", duplicates)
        corpus = corpus.drop_duplicates(subset="product_id", keep="first")

    # Counted after de-duplication, on unique ids. Counting rows here would let
    # a duplicate row mask a genuinely absent product.
    missing = len(wanted) - corpus["product_id"].nunique()
    if missing:
        # A judged product with no catalogue entry can never be retrieved, so
        # recall has a ceiling below 1.0. Recorded rather than silently ignored.
        logger.warning(
            "%d of %d judged products (%.2f%%) are missing from the catalogue",
            missing,
            len(wanted),
            100 * missing / len(wanted),
        )

    corpus = corpus.sort_values("product_id", kind="stable").reset_index(drop=True)
    corpus["product_text"] = build_text_column(corpus)
    corpus.insert(0, "doc_id", range(len(corpus)))

    stats = text_diagnostics(corpus["product_text"])
    logger.info("corpus text stats: %s", stats)
    if stats["n_empty"]:
        # Kept, not dropped: removing judged products would change what the
        # metrics mean. They are unreachable by search and counted here.
        logger.warning("%d products have empty text and can never be retrieved", stats["n_empty"])

    return corpus.loc[:, list(CORPUS_COLUMNS)]


def write_corpus(corpus: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CORPUS_FILENAME
    corpus.to_parquet(path, index=False)
    logger.info("wrote %d products to %s", len(corpus), path)
    return path


def read_corpus(out_dir: Path) -> pd.DataFrame:
    """Load the corpus. Every retrieval and ranking script starts here."""
    path = out_dir / CORPUS_FILENAME
    if not path.is_file():
        raise CorpusError(f"corpus not found: {path}. Run scripts/build_corpus.py first.")
    return pd.read_parquet(path)
