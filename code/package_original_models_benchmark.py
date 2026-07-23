"""Create the portable original-model benchmark report archive."""

import argparse
import os
import zipfile
from pathlib import Path


INCLUDED_DIRECTORIES = (
    "final_report",
    "metrics",
    "predictions",
    "training_logs",
    "config",
)


def create_archive(input_root, output_file):
    input_root = Path(input_root).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    missing = [name for name in INCLUDED_DIRECTORIES
               if not (input_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Cannot package incomplete benchmark; missing {}".format(", ".join(missing)))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for directory_name in INCLUDED_DIRECTORIES:
                directory = input_root / directory_name
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.relative_to(input_root).as_posix())
        os.replace(str(temporary), str(output_file))
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_file = (Path(args.output_file).expanduser().resolve()
                   if args.output_file else
                   input_root / "original_models_benchmark_report.zip")
    archive = create_archive(input_root, output_file)
    print("Created {}".format(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
