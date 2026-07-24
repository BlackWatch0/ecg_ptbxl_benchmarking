import sys
import zipfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from package_original_models_benchmark import create_archive


def test_archive_contains_complete_experiment_root(tmp_path):
    root = tmp_path / "run"
    for name in ("config", "metrics", "predictions", "training_logs", "final_report",
                 "runtime_logs", "manifest", "checkpoints", "features/wavelet_nn"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text(name, encoding="utf-8")
    (root / "checkpoints" / "large.pth").write_bytes(b"checkpoint")
    (root / "features" / "wavelet_nn" / "cache.npz").write_bytes(b"cache")

    output = create_archive(root, root / "report.zip")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "checkpoints/large.pth" in names
    assert "features/wavelet_nn/cache.npz" in names
    assert "manifest/artifact.txt" in names
