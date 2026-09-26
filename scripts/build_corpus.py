"""Entry point: build the retrieval corpus from the raw ESCI files."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from esci.config import load_config
from esci.data.corpus import PRODUCT_COLUMNS, CorpusError, build_corpus, write_corpus
from esci.data.download import DATASET_FILES, DATASET_SUBDIR
from esci.data.splits import filter_examples
from esci.logging_utils import get_logger, setup_logging

logger = get_logger("build_corpus")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    dataset_dir = cfg.paths.raw_dir / "esci-data" / DATASET_SUBDIR
    examples_path = dataset_dir / DATASET_FILES[0]
    products_path = dataset_dir / DATASET_FILES[1]
    for path in (examples_path, products_path):
        if not path.is_file():
            logger.error("%s not found. Run scripts/download_data.py", path)
            return 1

    logger.info("reading examples")
    examples = pd.read_parquet(examples_path)

    # Only the listed columns are read: the description column alone is most of
    # the 1.1 GB file and is not part of the text representation (decision D5).
    logger.info("reading products (selected columns only)")
    start = time.perf_counter()
    products = pd.read_parquet(products_path, columns=list(PRODUCT_COLUMNS))
    logger.info("read %d catalogue rows in %.1fs", len(products), time.perf_counter() - start)

    try:
        in_scope = filter_examples(examples, cfg.data.locale, cfg.data.small_version)
        corpus = build_corpus(products, in_scope, cfg.data.locale)
        path = write_corpus(corpus, cfg.paths.processed_dir)
    except CorpusError as exc:
        logger.error("could not build corpus: %s", exc)
        return 1

    logger.info("corpus ready at %s (%d products)", path, len(corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
