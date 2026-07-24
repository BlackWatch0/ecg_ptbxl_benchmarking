"""Validate a completed original-model benchmark result contract."""

import argparse
import json
from pathlib import Path

from models.original_model_factory import BENCHMARK_MODEL_NAMES
from utils.experiment_artifacts import ArtifactValidationError, validate_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=list(BENCHMARK_MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args(argv)
    try:
        report = validate_experiment(args.input_root, args.models, args.seeds, strict=True)
    except ArtifactValidationError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
