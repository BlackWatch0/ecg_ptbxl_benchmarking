import argparse
import sys

from .config import ConfigError, load_config
from .models import MODEL_NAMES
from .runner import TaskRunner


def build_parser():
    parser = argparse.ArgumentParser(description="Unified PTB-XL task manager")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models", help="List supported original benchmark models")
    validate = commands.add_parser("validate", help="Validate and resolve a YAML configuration")
    validate.add_argument("--config", required=True)
    run = commands.add_parser("run", help="Run configured tasks")
    run.add_argument("--config", required=True)
    run.add_argument("--task", nargs="+", action="append", default=[])
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "list-models":
        for name in MODEL_NAMES:
            print(name)
        return 0
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print("Valid configuration: {} task(s), {} model(s)".format(
                len(config["tasks"]), len(config["models"])))
            return 0
        task_names = [name for group in args.task for name in group]
        return TaskRunner(config, task_names=task_names, dry_run=args.dry_run).run()
    except (ConfigError, ValueError) as error:
        print("taskmanager: error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
