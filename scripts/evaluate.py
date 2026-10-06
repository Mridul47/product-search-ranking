"""Score a reference baseline or a run file on one split.

Every score uses the same qrels construction, the same integrity checks and
the same metrics. It is then logged to Weights & Biases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from esci.config import load_config
from esci.data.corpus import CorpusError, read_corpus
from esci.data.download import DATASET_FILES, DATASET_SUBDIR
from esci.data.splits import SplitError, filter_examples, read_splits
from esci.eval.baselines import (
    oracle_run,
    random_corpus_run,
    random_judged_run,
    title_length_run,
)
from esci.eval.checks import CheckError, assert_judged_only, verify_qrels
from esci.eval.metrics import evaluate_run
from esci.eval.qrels import QrelsError, build_qrels
from esci.eval.results import RESULTS_DIR, write_result
from esci.eval.runs import RunError, read_run
from esci.eval.tracking import log_run
from esci.logging_utils import get_logger, setup_logging

logger = get_logger("evaluate")

BASELINES = {
    "oracle": "judged",
    "random": "judged",
    "title_length": "judged",
    "random_corpus": "end_to_end",
}
EXAMPLE_COLUMNS = [
    "query_id",
    "product_id",
    "product_locale",
    "small_version",
    "split",
    "esci_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--setup", choices=sorted(BASELINES), help="reference baseline")
    source.add_argument("--run", type=Path, help="run file from a retrieval or ranking script")
    parser.add_argument("--mode", choices=["judged", "end_to_end"], help="required with --run")
    parser.add_argument("--name", help="W&B run name, defaults to setup_mode_split")
    parser.add_argument("--seed", type=int, default=None, help="defaults to the config seed")
    parser.add_argument("--depth", type=int, default=100, help="documents per query, random_corpus")
    parser.add_argument("--no-wandb", action="store_true", help="skip logging to W&B")
    args = parser.parse_args()
    if args.run is not None and args.mode is None:
        parser.error("--mode is required with --run")
    return args


def build_run(setup: str, qrels, corpus, depth: int, seed: int):
    if setup == "oracle":
        return oracle_run(qrels)
    if setup == "random":
        return random_judged_run(qrels, seed)
    if setup == "title_length":
        lengths = corpus["product_title"].fillna("").str.len().to_numpy()
        return title_length_run(qrels, lengths)
    return random_corpus_run(qrels.query_ids, len(corpus), depth, seed)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    seed = cfg.project.seed if args.seed is None else args.seed

    examples_path = cfg.paths.raw_dir / "esci-data" / DATASET_SUBDIR / DATASET_FILES[0]
    if not examples_path.is_file():
        logger.error("%s not found. Run scripts/download_data.py", examples_path)
        return 1

    try:
        raw = pd.read_parquet(examples_path, columns=EXAMPLE_COLUMNS)
        examples = filter_examples(raw, cfg.data.locale, cfg.data.small_version)
        splits = read_splits(cfg.paths.processed_dir / "splits")
        corpus = read_corpus(cfg.paths.processed_dir)

        subset = examples.loc[examples["query_id"].isin(splits[args.split])]
        doc_ids = corpus.set_index("product_id")["doc_id"]
        qrels = build_qrels(subset, doc_ids, cfg.relevance.gains)
        stats = verify_qrels(qrels, subset, corpus, cfg.relevance.gains)
        logger.info("qrels verified against the raw examples: %s", stats)

        if args.run is not None:
            run, mode, setup = read_run(args.run), args.mode, args.run.stem
        else:
            run = build_run(args.setup, qrels, corpus, args.depth, seed)
            mode, setup = BASELINES[args.setup], args.setup

        covered = len(set(run) & set(qrels.gains))
        if covered == 0:
            raise CheckError("the run has no queries in common with this split")
        logger.info("run covers %d of %d queries", covered, len(qrels.gains))
        if mode == "judged":
            assert_judged_only(run, qrels)

        metrics = evaluate_run(run, qrels)
    except (SplitError, CorpusError, QrelsError, CheckError, RunError) as exc:
        logger.error("evaluation failed: %s", exc)
        return 1

    print(f"\n{setup} | mode={mode} | split={args.split} | {len(qrels.gains)} queries")
    for key, value in metrics.items():
        shown = f"{value:.0f}" if key.startswith("n_") else f"{value:.4f}"
        print(f"  {key:<18}{shown}")

        name = args.name or f"{setup}_{mode}_{args.split}"
    params = {"setup": setup, "mode": mode, "split": args.split, "seed": seed}
    if args.setup == "random_corpus":
        params["depth"] = args.depth

    url = None
    wandb_failed = False
    if not args.no_wandb:
        try:
            url = log_run(cfg, name, metrics, params, tags=["phase3", args.split, mode])
            logger.info("logged to %s", url)
        except Exception as exc:
            wandb_failed = True
            logger.error("scores were not logged to W&B: %s", exc)

    path = write_result(RESULTS_DIR, name, cfg, metrics, params, url)
    logger.info("wrote result record to %s", path)
    return 1 if wandb_failed else 0


if __name__ == "__main__":
    sys.exit(main())
