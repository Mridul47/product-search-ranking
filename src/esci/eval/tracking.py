"""Experiment tracking.

Every evaluated setup is logged to Weights & Biases along with the config
fingerprint. This lets you trace a reported number back to the settings that
produced it. wandb is imported inside the function. Modules that never log a
run, and CI, therefore do not need it installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from esci.config import Config
from esci.logging_utils import get_logger

logger = get_logger(__name__)


def run_parameters(cfg: Config, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Settings recorded with every run. Filesystem paths are left out on purpose."""
    params: dict[str, Any] = {
        "config_fingerprint": cfg.fingerprint(),
        "seed": cfg.project.seed,
        "locale": cfg.data.locale,
        "small_version": cfg.data.small_version,
        "val_fraction": cfg.data.val_fraction,
        "corpus_rule": cfg.data.corpus_rule,
        "gains": dict(cfg.relevance.gains),
    }
    params.update(extra or {})
    return params


def log_run(
    cfg: Config,
    name: str,
    metrics: Mapping[str, float],
    params: Mapping[str, Any] | None = None,
    tags: Iterable[str] = (),
    enabled: bool = True,
) -> str | None:
    """Log one evaluated setup. Returns the run URL, or None when disabled."""
    if not enabled:
        logger.info("tracking disabled, not logging %s", name)
        return None

    import wandb

    with wandb.init(
        project=cfg.project.name,
        name=name,
        config=run_parameters(cfg, params),
        tags=list(tags),
    ) as run:
        run.log(dict(metrics))
        url = run.url
    return url
