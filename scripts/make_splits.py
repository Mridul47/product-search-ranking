"""Entry point: build and save train, validation and test query ID splits."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from esci.config import load_config
from esci.data.download import DATASET_FILES, DATASET_SUBDIR
from esci.data.splits import SplitError, filter_examples, split_query_ids, summarise, write_splits
from esci.logging_utils import get_logger, setup_logging

logger = get_logger("make_splits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    examples_path = cfg.paths.raw_dir / "esci-data" / DATASET_SUBDIR / DATASET_FILES[0]
    if not examples_path.is_file():
        logger.error("examples file not found at %s. Run scripts/download_data.py", examples_path)
        return 1

    logger.info("reading %s", examples_path)
    examples = pd.read_parquet(examples_path)

    try:
        subset = filter_examples(examples, cfg.data.locale, cfg.data.small_version)
        splits = split_query_ids(subset, cfg.data.val_fraction, cfg.project.seed)
        write_splits(splits, cfg.paths.processed_dir / "splits")
    except SplitError as exc:
        logger.error("could not build splits: %s", exc)
        return 1

    summary = summarise(subset, splits)
    logger.info("split summary:\n%s", summary.to_string(index=False))

    summary_path = cfg.paths.processed_dir / "splits" / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
