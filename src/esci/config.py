"""Typed project configuration loaded from YAML.

Every script loads settings through load_config() so paths, seeds and
relevance gains come from one place instead of being hardcoded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
ESCI_LABELS = ("E", "S", "C", "I")


class _Strict(BaseModel):
    """Base model: unknown keys are an error and settings are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSettings(_Strict):
    name: str
    seed: int = 42


class PathSettings(_Strict):
    data_dir: Path
    artifact_dir: Path

    @field_validator("data_dir", "artifact_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        # Relative paths are resolved against the repo root, not the current
        # working directory, so scripts behave the same wherever they run from.
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


class DataSettings(_Strict):
    locale: Literal["us", "es", "jp"]
    small_version: Literal[0, 1]
    val_fraction: float = Field(gt=0.0, lt=0.5)
    corpus_rule: Literal["small_version_products", "full_catalogue"]


class RelevanceSettings(_Strict):
    gains: dict[str, float]
    lgbm_labels: dict[str, int]

    @model_validator(mode="after")
    def _check_consistency(self) -> RelevanceSettings:
        expected = set(ESCI_LABELS)
        if set(self.gains) != expected or set(self.lgbm_labels) != expected:
            raise ValueError(f"gains and lgbm_labels must have exactly the keys {sorted(expected)}")
        if sorted(self.lgbm_labels.values()) != list(range(len(ESCI_LABELS))):
            raise ValueError("lgbm_labels must use each integer 0..3 exactly once")
        ordered = self.label_gain
        if ordered != sorted(ordered):
            raise ValueError("a higher lgbm label must never have a lower gain")
        return self

    @property
    def label_gain(self) -> list[float]:
        """Gains indexed by LightGBM label, for LGBMRanker(label_gain=...)."""
        by_label = sorted(self.lgbm_labels, key=self.lgbm_labels.__getitem__)
        return [self.gains[k] for k in by_label]


class LoggingSettings(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Config(_Strict):
    project: ProjectSettings
    paths: PathSettings
    data: DataSettings
    relevance: RelevanceSettings
    logging: LoggingSettings = LoggingSettings()

    def fingerprint(self) -> str:
        """Short hash of the experiment settings, logged with every run.

        Paths are excluded so the same experiment gives the same hash on
        any machine.
        """
        payload = json.dumps(self.model_dump(mode="json", exclude={"paths"}), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config. Env vars DATA_DIR and ARTIFACT_DIR override paths."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    paths = dict(raw.get("paths") or {})
    for key, env_var in (("data_dir", "DATA_DIR"), ("artifact_dir", "ARTIFACT_DIR")):
        if os.getenv(env_var):
            paths[key] = os.environ[env_var]
    raw["paths"] = paths

    return Config.model_validate(raw)
