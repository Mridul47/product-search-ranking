"""Download the Amazon Shopping Queries (ESCI) dataset.

The files live in amazon-science/esci-data and are tracked with Git LFS. We
clone the repository with LFS smudging disabled, which fetches pointers only
and keeps the clone fast, then pull just the three files the project needs.

We also write a manifest with the upstream commit, file sizes and SHA-256
digests, so you can tell later exactly which version of the data an experiment
used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from esci.logging_utils import get_logger

logger = get_logger(__name__)

ESCI_REPO_URL = "https://github.com/amazon-science/esci-data.git"
DATASET_SUBDIR = "shopping_queries_dataset"
DATASET_FILES = (
    "shopping_queries_dataset_examples.parquet",
    "shopping_queries_dataset_products.parquet",
    "shopping_queries_dataset_sources.csv",
)
MANIFEST_NAME = "esci_manifest.json"
LFS_POINTER_PREFIX = b"version https://git-lfs"
_POINTER_PROBE_BYTES = 64


class DownloadError(RuntimeError):
    """Raised when the dataset cannot be fetched or fails verification."""


def _run(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run a command and return stdout, raising DownloadError on failure."""
    logger.debug("running: %s", " ".join(args))
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:  # command not installed
        raise DownloadError(f"{args[0]} is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadError(f"{' '.join(args)} failed:\n{exc.stderr.strip()}") from exc
    return result.stdout.strip()


def check_prerequisites() -> None:
    """Fail early with a clear message if git or git-lfs is missing."""
    for tool, args in (("git", ["git", "--version"]), ("git-lfs", ["git", "lfs", "version"])):
        if shutil.which("git") is None:
            raise DownloadError("git is not installed or not on PATH")
        _run(args)
        logger.debug("%s is available", tool)


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large parquet files fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    """True if the file is an unresolved Git LFS pointer rather than real content.

    Git writes a small text pointer when LFS is missing or the pull failed. The
    file looks present but is a few hundred bytes of metadata, so this check
    prevents a confusing parquet parse error later.
    """
    with path.open("rb") as fh:
        return fh.read(_POINTER_PROBE_BYTES).startswith(LFS_POINTER_PREFIX)


def clone_pointers_only(repo_dir: Path) -> None:
    """Shallow-clone esci-data without downloading LFS content."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("cloning %s (pointers only)", ESCI_REPO_URL)
    _run(
        ["git", "clone", "--depth", "1", ESCI_REPO_URL, str(repo_dir)],
        env={"GIT_LFS_SKIP_SMUDGE": "1"},
    )


def pull_dataset_files(repo_dir: Path) -> None:
    """Fetch the LFS content for the dataset files only."""
    logger.info("pulling LFS content for %s", DATASET_SUBDIR)
    _run(["git", "lfs", "pull", "--include", f"{DATASET_SUBDIR}/*"], cwd=repo_dir)


def verify_files(repo_dir: Path) -> dict[str, dict[str, object]]:
    """Check every expected file exists and holds real content, and hash it."""
    records: dict[str, dict[str, object]] = {}
    for name in DATASET_FILES:
        path = repo_dir / DATASET_SUBDIR / name
        if not path.is_file():
            raise DownloadError(f"expected file is missing after download: {path}")
        if path.suffix == ".parquet" and is_lfs_pointer(path):
            raise DownloadError(
                f"{name} is still a Git LFS pointer. Run 'git lfs install' and try again."
            )
        size = path.stat().st_size
        logger.info("verifying %s (%.1f MB)", name, size / 1e6)
        records[name] = {"bytes": size, "sha256": sha256(path)}
    return records


def write_manifest(repo_dir: Path, raw_dir: Path, files: dict[str, dict[str, object]]) -> Path:
    """Record the upstream commit and file digests next to the data."""
    manifest = {
        "source": ESCI_REPO_URL,
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo_dir),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    path = raw_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote manifest to %s", path)
    return path


def download_dataset(raw_dir: Path, force: bool = False) -> Path:
    """Download and verify the ESCI dataset. Returns the manifest path.

    Existing data is left alone unless force is set, so re-running the script is
    cheap and safe.
    """
    repo_dir = raw_dir / "esci-data"
    manifest_path = raw_dir / MANIFEST_NAME

    if manifest_path.is_file() and not force:
        logger.info("dataset already present, skipping download (use --force to redo)")
        return manifest_path

    check_prerequisites()

    if repo_dir.exists() and force:
        logger.info("removing existing clone at %s", repo_dir)
        shutil.rmtree(repo_dir, ignore_errors=True)

    if not repo_dir.exists():
        clone_pointers_only(repo_dir)

    pull_dataset_files(repo_dir)
    files = verify_files(repo_dir)
    return write_manifest(repo_dir, raw_dir, files)
