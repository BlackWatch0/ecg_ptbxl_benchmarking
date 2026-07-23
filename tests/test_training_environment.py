import sys
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from check_training_environment import version_in_range, version_tuple


def test_version_parsing_handles_build_suffixes():
    assert version_tuple("2.5.1+cu121") == (2, 5, 1)
    assert version_tuple("1.26.4") == (1, 26, 4)


def test_version_range_is_lower_inclusive_and_upper_exclusive():
    assert version_in_range("2.5.1", (2, 5), (2, 6))
    assert not version_in_range("2.6.0", (2, 5), (2, 6))
    assert not version_in_range("2.4.9", (2, 5), (2, 6))


def test_aws_environment_pins_gpu_torch_and_cpu_tensorflow():
    root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load(
        (root / "environments" / "ecg-training.yml").read_text(encoding="utf-8"))
    dependencies = environment["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = next(item["pip"] for item in dependencies if isinstance(item, dict))
    assert "pytorch=2.5.1" in conda_dependencies
    assert "pytorch-cuda=12.1" in conda_dependencies
    assert "cuda-version=12.1" in conda_dependencies
    assert "numpy=1.26.4" in conda_dependencies
    assert pip_dependencies == ["tensorflow-cpu==2.15.1"]
    assert "nodefaults" in environment["channels"]
