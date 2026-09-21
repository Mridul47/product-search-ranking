from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from esci.config import DEFAULT_CONFIG_PATH, load_config


@pytest.fixture
def base_raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads():
    cfg = load_config()
    assert cfg.data.locale == "us"
    assert cfg.data.small_version == 1
    assert cfg.relevance.gains == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def test_label_gain_is_ordered_for_lightgbm():
    assert load_config().relevance.label_gain == [0.0, 0.01, 0.1, 1.0]


def test_paths_are_absolute():
    cfg = load_config()
    assert cfg.paths.data_dir.is_absolute()
    assert cfg.paths.raw_dir.parent == cfg.paths.data_dir


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert load_config().paths.data_dir == tmp_path


def test_unknown_key_is_rejected(tmp_path, base_raw):
    base_raw["data"]["locael"] = "us"  # typo on purpose
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, base_raw))


def test_inconsistent_labels_are_rejected(tmp_path, base_raw):
    base_raw["relevance"]["lgbm_labels"] = {"I": 0, "C": 1, "S": 3, "E": 2}
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, base_raw))


def test_bad_val_fraction_is_rejected(tmp_path, base_raw):
    base_raw["data"]["val_fraction"] = 0.9
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, base_raw))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_fingerprint_tracks_settings_not_paths(monkeypatch, tmp_path, base_raw):
    a = load_config()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    b = load_config()
    assert a.fingerprint() == b.fingerprint()

    base_raw["project"]["seed"] = 7
    c = load_config(_write(tmp_path, base_raw))
    assert c.fingerprint() != a.fingerprint()