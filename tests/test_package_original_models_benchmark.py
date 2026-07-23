import sys
import zipfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from package_original_models_benchmark import (INCLUDED_DIRECTORIES,
                                               create_archive)


def test_archive_contains_only_portable_report_directories(tmp_path):
    root = tmp_path / "run"
    for name in INCLUDED_DIRECTORIES:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text(name, encoding="utf-8")
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "large.pth").write_bytes(b"checkpoint")
    (root / "features" / "wavelet_nn").mkdir(parents=True)
    (root / "features" / "wavelet_nn" / "cache.npz").write_bytes(b"cache")

    output = create_archive(root, root / "report.zip")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert names == {"{}/artifact.txt".format(name) for name in INCLUDED_DIRECTORIES}
    assert not any(name.startswith("checkpoints/") or name.startswith("features/")
                   for name in names)


def test_archive_rejects_incomplete_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        create_archive(tmp_path, tmp_path / "report.zip")
