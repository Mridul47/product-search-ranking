"""Run BM25 on one split: retrieve, score end to end and judged, time it, log it.

Index build time, index size and per-query latency are measured on CPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from esci.config import load_config
from esci.data.corpus import CorpusError
from esci.data.splits import SplitError
from esci.eval.checks import CheckError, assert_judged_only
from esci.eval.loading import load_split
from esci.eval.metrics import evaluate_run
from esci.eval.qrels import QrelsError
from esci.eval.results import RESULTS_DIR, write_result
from esci.eval.runs import RunError, write_run
from esci.eval.tracking import log_run
from esci.logging_utils import get_logger, setup_logging
from esci.retrieval.bm25 import (
    TokenizerSpec,
    build_index,
    index_size_bytes,
    save_index,
    search,
)

logger = get_logger("run_bm25")

TOKENIZERS = {
    "plain": TokenizerSpec(),
    "stop": TokenizerSpec(stopwords=True),
    "stem": TokenizerSpec(stem=True),
    "stop_stem": TokenizerSpec(stopwords=True, stem=True),
}
RECALL_DEPTHS = (100, 200, 500, 1000)
WARMUP_QUERIES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="required to touch the test split")
    parser.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="plain")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--depth", type=int, default=1000, help="documents retrieved per query")
    parser.add_argument("--latency-queries", type=int, default=500)
    parser.add_argument("--save", action="store_true", help="keep the index in artifacts/bm25/")
    parser.add_argument("--dry-run", action="store_true", help="no W&B run, no result record")
    return parser.parse_args()


def recall_at(run, qrels, k: int) -> float:
    """Recall over exact products. Queries with no exact product are skipped (D7)."""
    values = []
    for qid, exact in qrels.exact.items():
        if not exact:
            continue
        got = run.get(qid)
        hits = len(exact.intersection(got[:k].tolist())) if got is not None else 0
        values.append(hits / len(exact))
    return float(np.mean(values))


def judged_run(retriever, spec, qrels, queries, seed: int):
    """Rank only the judged products of each query by BM25 score. Ties break at random."""
    rng = np.random.default_rng(seed)
    run = {}
    for qid in sorted(qrels.gains):
        docs = np.fromiter(qrels.gains[qid].keys(), dtype=np.int64)
        tokens = spec.tokenize_queries([queries[qid]])[0]
        scores = retriever.get_scores(tokens)[docs] if tokens else np.zeros(len(docs))
        order = np.lexsort((rng.random(len(docs)), -scores))
        run[qid] = docs[order]
    return run


def report(name: str, mode: str, split: str, metrics: dict) -> None:
    print(f"\n{name} | mode={mode} | split={split}")
    for key, value in metrics.items():
        shown = f"{value:.0f}" if key.startswith("n_") else f"{value:.4f}"
        print(f"  {key:<22}{shown}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    seed = cfg.project.seed

    if args.split == "test" and not args.final:
        logger.error("the test split is reserved for the final table. Pass --final to use it.")
        return 1

    spec = TOKENIZERS[args.tokenizer]
    tag = f"{args.tokenizer}_k1{args.k1:g}_b{args.b:g}"
    params = {
        "tokenizer": args.tokenizer,
        "k1": args.k1,
        "b": args.b,
        "depth": args.depth,
        "split": args.split,
        "seed": seed,
    }

    try:
        data = load_split(cfg, args.split)
        corpus, qrels = data.corpus, data.qrels
        if not (corpus["doc_id"].to_numpy() == np.arange(len(corpus))).all():
            raise CheckError("doc_id is not the corpus row position")

        # build, save, measure size
        texts = corpus["product_text"].fillna("").tolist()
        retriever, build_seconds = build_index(texts, spec, k1=args.k1, b=args.b)
        logger.info("built index in %.1f s", build_seconds)
        artifact_dir = cfg.paths.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if args.save:
            index_dir = artifact_dir / "bm25" / tag
        else:
            index_dir = Path(tempfile.mkdtemp(dir=artifact_dir, prefix="bm25_tmp_"))
        save_index(retriever, spec, index_dir, params)
        size_mb = index_size_bytes(index_dir) / 1e6
        if not args.save:
            shutil.rmtree(index_dir)
        logger.info("index size %.0f MB", size_mb)

        # retrieve. Queries with no tokens get no results. Zero-score documents are dropped.
        qids = sorted(data.queries)
        qtexts = [data.queries[q] for q in qids]
        tokens = spec.tokenize_queries(qtexts)
        keep = [i for i, t in enumerate(tokens) if t]
        n_empty = len(qids) - len(keep)
        if n_empty:
            logger.warning("%d queries have no tokens and return nothing", n_empty)
        docs, scores = search(
            retriever, spec, [qtexts[i] for i in keep], k=args.depth, n_threads=-1
        )
        run = {}
        for row, i in enumerate(keep):
            run[qids[i]] = docs[row][scores[row] > 0].astype(np.int64)

        run_path = artifact_dir / "runs" / f"bm25_{tag}_{args.split}.parquet"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        write_run(run, run_path)
        logger.info("wrote run file %s", run_path)

        # end-to-end metrics
        e2e = evaluate_run(run, qrels)
        mine = recall_at(run, qrels, 100)
        if abs(mine - e2e["recall@100"]) > 1e-9:
            raise CheckError(f"recall@100 mismatch: {mine} against {e2e['recall@100']}")
        for k in RECALL_DEPTHS[1:]:
            if k <= args.depth:
                e2e[f"recall@{k}"] = recall_at(run, qrels, k)

        # latency, one query at a time on one thread, tokenising included
        sample = [qtexts[i] for i in keep[: args.latency_queries]]
        for q in sample[:WARMUP_QUERIES]:
            search(retriever, spec, [q], k=args.depth, n_threads=0)
        times = []
        for q in sample:
            start = time.perf_counter()
            search(retriever, spec, [q], k=args.depth, n_threads=0)
            times.append((time.perf_counter() - start) * 1000)
        e2e["build_seconds"] = build_seconds
        e2e["index_mb"] = size_mb
        e2e["latency_p50_ms"] = float(np.percentile(times, 50))
        e2e["latency_p95_ms"] = float(np.percentile(times, 95))

        # judged-set metrics
        jrun = judged_run(retriever, spec, qrels, data.queries, seed)
        assert_judged_only(jrun, qrels)
        judged = evaluate_run(jrun, qrels)
    except (SplitError, CorpusError, QrelsError, CheckError, RunError) as exc:
        logger.error("run failed: %s", exc)
        return 1

    params["n_empty_queries"] = n_empty
    failed = False
    for mode, metrics in (("end_to_end", e2e), ("judged", judged)):
        name = f"bm25_{tag}_{mode}_{args.split}"
        report(f"bm25 {tag}", mode, args.split, metrics)
        if args.dry_run:
            continue
        url = None
        try:
            url = log_run(
                cfg, name, metrics, {**params, "mode": mode}, tags=["phase4", args.split, mode]
            )
            logger.info("logged to %s", url)
        except Exception as exc:
            failed = True
            logger.error("scores were not logged to W&B: %s", exc)
        path = write_result(RESULTS_DIR, name, cfg, metrics, {**params, "mode": mode}, url)
        logger.info("wrote result record to %s", path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
