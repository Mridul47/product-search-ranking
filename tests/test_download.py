import json

import pytest

from esci.data import download as dl


def test_sha256_matches_known_value(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert dl.sha256(path) == expected


def test_lfs_pointer_is_detected(tmp_path):
    pointer = tmp_path / "p.parquet"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n")
    real = tmp_path / "r.parquet"
    real.write_bytes(b"PAR1somethingbinary")
    assert dl.is_lfs_pointer(pointer)
    assert not dl.is_lfs_pointer(real)


def _make_dataset(raw_dir):
    data_dir = raw_dir / "esci-data" / dl.DATASET_SUBDIR
    data_dir.mkdir(parents=True)
    for name in dl.DATASET_FILES:
        (data_dir / name).write_bytes(b"PAR1 fake content")
    return data_dir


def test_verify_returns_size_and_digest(tmp_path):
    _make_dataset(tmp_path)
    files = dl.verify_files(tmp_path / "esci-data")
    assert set(files) == set(dl.DATASET_FILES)
    for record in files.values():
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64


def test_verify_rejects_missing_file(tmp_path):
    data_dir = _make_dataset(tmp_path)
    (data_dir / dl.DATASET_FILES[0]).unlink()
    with pytest.raises(dl.DownloadError, match="missing"):
        dl.verify_files(tmp_path / "esci-data")


def test_verify_rejects_lfs_pointer(tmp_path):
    data_dir = _make_dataset(tmp_path)
    (data_dir / dl.DATASET_FILES[0]).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(dl.DownloadError, match="pointer"):
        dl.verify_files(tmp_path / "esci-data")


def test_existing_manifest_skips_download(tmp_path, monkeypatch):
    manifest = tmp_path / dl.MANIFEST_NAME
    manifest.write_text(json.dumps({"files": {}}), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("download should not run when a manifest exists")

    monkeypatch.setattr(dl, "clone_pointers_only", fail)
    monkeypatch.setattr(dl, "pull_dataset_files", fail)
    assert dl.download_dataset(tmp_path) == manifest
