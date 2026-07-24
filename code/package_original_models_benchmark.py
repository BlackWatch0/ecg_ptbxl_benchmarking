"""Validate and package the complete original-model benchmark experiment."""

import argparse
import json
import os
import zipfile
from pathlib import Path

from models.original_model_factory import BENCHMARK_MODEL_NAMES, canonical_model_name
from utils.experiment_artifacts import (ArtifactValidationError, validate_experiment,
                                        verify_archive, write_experiment_status)


def _run_metadata(root):
    config_path = root / "config" / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Missing resolved config: {}".format(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = [canonical_model_name(name) for name in config.get("models", BENCHMARK_MODEL_NAMES)]
    return config, models, [int(seed) for seed in config.get("seeds", [42])]


def create_archive(input_root, output_file):
    input_root = Path(input_root).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for path in sorted(input_root.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.resolve() in (output_file, temporary):
                    continue
                archive.write(path, path.relative_to(input_root).as_posix())
        os.replace(str(temporary), str(output_file))
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file


def package_experiment(input_root, output_file):
    input_root = Path(input_root).expanduser().resolve()
    config, models, seeds = _run_metadata(input_root)
    report = validate_experiment(input_root, models, seeds, strict=True)
    commit = config.get("git_commit", "unknown")
    write_experiment_status(input_root, "packaging", input_root.name, commit, models, seeds, report)
    archive = create_archive(input_root, output_file)
    verify_archive(archive, input_root,
                   json.loads((input_root / "manifest" / "expected_artifacts.json").read_text())["files"])
    write_experiment_status(input_root, "success", input_root.name, commit, models, seeds, report)
    archive = create_archive(input_root, output_file)
    verify_archive(archive, input_root,
                   json.loads((input_root / "manifest" / "expected_artifacts.json").read_text())["files"])
    return archive


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_file = (Path(args.output_file).expanduser().resolve() if args.output_file else
                   input_root / "original_models_benchmark_report.zip")
    try:
        archive = package_experiment(input_root, output_file)
    except ArtifactValidationError as error:
        print(str(error))
        return 1
    print("Created and verified {}".format(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
