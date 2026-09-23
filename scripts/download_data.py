"""Entry point: download the ESCI dataset into the configured data directory."""

from __future__ import annotations

import argparse
import sys

from esci.config import load_config
from esci.data.download import DownloadError, download_dataset
from esci.logging_utils import get_logger, setup_logging

logger = get_logger("download_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--force", action="store_true", help="re-download even if data exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = download_dataset(raw_dir, force=args.force)
    except DownloadError as exc:
        logger.error("download failed: %s", exc)
        return 1

    logger.info("dataset ready, manifest at %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
