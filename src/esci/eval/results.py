"""A small, committed record of every evaluated setup.

Weights and Biases keeps the full history online. A reader of the repo should
still be able to see the numbers without an account. Each evaluated setup
writes one JSON file under results/. The file holds the metrics, the settings
that produced them and the git commit of the code. This means a number can be
traced back, and every change to it shows up in the repo history.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from esci.config import PROJECT_ROOT, Config

RESULTS_DIR = PROJECT_ROOT / "results"


def git_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Current commit, and whether tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def _clean(value: float) -> float | None:
    """JSON has no NaN, so undefined metrics are stored as null."""
    return None if math.isnan(value) else value


def write_result(
    results_dir: Path,
    name: str,
    cfg: Config,
    metrics: Mapping[str, float],
    params: Mapping[str, Any],
    wandb_url: str | None,
) -> Path:
    """Write one result record. Re-running a setup replaces its file."""
    record = {
        "name": name,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "params": dict(params),
        "metrics": {key: _clean(value) for key, value in metrics.items()},
        "wandb_url": wandb_url,
        "git": git_state(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
