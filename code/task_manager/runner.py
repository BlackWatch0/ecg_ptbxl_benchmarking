import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "prepare_data": REPO_ROOT / "code" / "prepare_original_models_benchmark_data.py",
    "train": REPO_ROOT / "code" / "run_original_models_benchmark.py",
    "evaluate": REPO_ROOT / "code" / "run_original_models_benchmark.py",
    "report": REPO_ROOT / "code" / "build_original_models_benchmark_report.py",
    "package": REPO_ROOT / "code" / "package_original_models_benchmark.py",
}

VALUE_FLAGS = {
    "data_root": "--data-root",
    "data_config": "--data-config",
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "wavelet_epochs": "--wavelet-epochs",
    "wavelet_batch_size": "--wavelet-batch-size",
    "learning_rate": "--learning-rate",
    "crop_length": "--crop-length",
    "num_workers": "--num-workers",
    "device": "--device",
    "cache_dir": "--cache-dir",
    "noisy_manifest": "--noisy-manifest",
    "noisy_root": "--noisy-root",
    "denoised_manifest": "--denoised-manifest",
    "denoised_root": "--denoised-root",
}

BOOLEAN_FLAGS = {
    "official_raw_data": "--official-raw-data",
    "resume": "--resume",
    "no_mixed_precision": "--no-mixed-precision",
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _digest(config):
    content = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _selected_tasks(tasks, names):
    if not names:
        return tasks
    by_name = {task["name"]: task for task in tasks}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError("Unknown task(s): {}".format(", ".join(unknown)))
    included = set()

    def include(name):
        if name in included:
            return
        for dependency in by_name[name]["depends_on"]:
            include(dependency)
        included.add(name)

    for name in names:
        include(name)
    return [task for task in tasks if task["name"] in included]


def _merge_benchmark_options(config, task, model):
    options = {}
    layers = [config["global"], task["options"], config["models"].get(model, {})]
    for layer in layers:
        if "seed" in layer:
            options.pop("seeds", None)
        if "seeds" in layer:
            options.pop("seed", None)
        options.update({key: value for key, value in layer.items() if key != "fail_fast"})
    options.setdefault("output_dir", config["output_dir"])
    seed = options.pop("seed", None)
    seeds = options.pop("seeds", None)
    options["seeds"] = seeds if seeds is not None else [42 if seed is None else seed]
    return options


def _benchmark_command(task_type, model, seed, options):
    command = [sys.executable, str(RUNNERS[task_type]), "--models", model,
               "--seeds", str(seed)]
    for key, flag in VALUE_FLAGS.items():
        if key in options:
            command.extend([flag, str(options[key])])
    for key, flag in BOOLEAN_FLAGS.items():
        if options.get(key):
            command.append(flag)
    command.extend(["--output-dir", str(options["output_dir"])])
    if task_type == "train":
        command.append("--skip-test-evaluation")
    else:
        command.append("--evaluate-only")
    return command


def _single_command(config, task):
    options = task["options"]
    if task["type"] == "prepare_data":
        command = [sys.executable, str(RUNNERS["prepare_data"])]
        workspace = options.get("workspace", str(Path(config["output_dir"]) / "data_workspace"))
        output_dir = options.get(
            "output_dir", str(Path(config["output_dir"]) / "config" / "prepared_data"))
        command.extend(["--workspace", workspace, "--output-dir", output_dir])
        for archive in options.get("archive", []):
            command.extend(["--archive", archive])
        for root in options.get("search_root", []):
            command.extend(["--search-root", root])
        return command
    input_root = options.get("input_root", config["output_dir"])
    if task["type"] == "package":
        command = [sys.executable, str(RUNNERS["package"]),
                   "--input-root", input_root]
        if "output_file" in options:
            command.extend(["--output-file", options["output_file"]])
        return command
    command = [sys.executable, str(RUNNERS["report"]), "--input-root", input_root]
    if "output_dir" in options:
        command.extend(["--output-dir", options["output_dir"]])
    seeds = options.get("expected_seeds")
    if seeds is None:
        seed = config["global"].get("seed")
        seeds = config["global"].get("seeds", [42 if seed is None else seed])
    command.append("--expected-seeds")
    command.extend(str(seed) for seed in seeds)
    if options.get("excluded_wavelet_status"):
        command.extend(["--excluded-wavelet-status", options["excluded_wavelet_status"]])
    return command


class TaskRunner:
    def __init__(self, config, task_names=None, dry_run=False):
        self.config = config
        self.tasks = _selected_tasks(config["tasks"], task_names)
        self.dry_run = dry_run
        self.output = Path(config["output_dir"])
        self.runtime_status_path = self.output / "task_status.json"
        self.status_path = self.output / ("task_plan.json" if dry_run else "task_status.json")
        self.digest = _digest(config)
        self.resume = bool(config["global"].get("resume", False))
        self.fail_fast = config["global"].get("fail_fast", True)
        self.previous = self._load_previous()
        self.status = {
            "version": 1,
            "config": config["config_path"],
            "config_digest": self.digest,
            "dry_run": dry_run,
            "started_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "tasks": {},
        }

    def _load_previous(self):
        if not self.resume or not self.runtime_status_path.exists():
            return {}
        try:
            previous = json.loads(self.runtime_status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if previous.get("config_digest") != self.digest:
            return {}
        return previous

    def _write_status(self):
        self.status["updated_at"] = _now()
        _atomic_json(self.status_path, self.status)

    def _write_resolved(self):
        resolved = dict(self.config)
        _atomic_text(
            self.output / "config" / "resolved_config.yaml",
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=False),
        )

    def _was_completed(self, task_name, run_name):
        run = self.previous.get("tasks", {}).get(task_name, {}).get("runs", {}).get(run_name, {})
        return run.get("status") in ("completed", "skipped")

    def _run_command(self, task, run_name, command, resume):
        safe_name = run_name.replace("/", "_").replace("\\", "_")
        log_directory = "task_plan_logs" if self.dry_run else "task_logs"
        log_path = self.output / log_directory / task["name"] / "{}.log".format(safe_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "planned" if self.dry_run else "running",
            "command": command,
            "log": str(log_path),
            "started_at": _now(),
        }
        self.status["tasks"][task["name"]]["runs"][run_name] = record
        self._write_status()
        if not self.dry_run and resume and self._was_completed(task["name"], run_name):
            record["status"] = "skipped"
            record["reason"] = "already completed with the same resolved config"
            record["finished_at"] = _now()
            self._write_status()
            return True
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: {}\n".format(json.dumps(command)))
            log.flush()
            if self.dry_run:
                return True
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
            except OSError as error:
                log.write("\nFailed to start process: {}\n".format(error))
                record["status"] = "failed"
                record["error"] = str(error)
                record["finished_at"] = _now()
                self._write_status()
                return False
        record["returncode"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        record["finished_at"] = _now()
        self._write_status()
        return completed.returncode == 0

    def _execute_task(self, task):
        entry = {
            "type": task["type"],
            "depends_on": task["depends_on"],
            "status": "running",
            "started_at": _now(),
            "runs": {},
        }
        self.status["tasks"][task["name"]] = entry
        self._write_status()
        succeeded = True
        if task["type"] in ("train", "evaluate"):
            for model in task["models"]:
                options = _merge_benchmark_options(self.config, task, model)
                for seed in options["seeds"]:
                    run_name = "{}_seed_{}".format(model, seed)
                    command = _benchmark_command(task["type"], model, seed, options)
                    current = self._run_command(
                        task, run_name, command, options.get("resume", False))
                    succeeded = current and succeeded
                    if not current and self.fail_fast:
                        break
                if not succeeded and self.fail_fast:
                    break
        else:
            command = _single_command(self.config, task)
            succeeded = self._run_command(task, task["type"], command, self.resume)
        entry["status"] = "planned" if self.dry_run else ("completed" if succeeded else "failed")
        entry["finished_at"] = _now()
        self._write_status()
        return succeeded

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_resolved()
        self._write_status()
        failed = False
        for task in self.tasks:
            failed_dependencies = [
                dependency for dependency in task["depends_on"]
                if self.status["tasks"].get(dependency, {}).get("status") in ("failed", "skipped")
            ]
            if failed_dependencies:
                self.status["tasks"][task["name"]] = {
                    "type": task["type"],
                    "depends_on": task["depends_on"],
                    "status": "skipped",
                    "reason": "failed dependencies: {}".format(", ".join(failed_dependencies)),
                    "runs": {},
                    "finished_at": _now(),
                }
                self._write_status()
                continue
            if failed and self.fail_fast:
                self.status["tasks"][task["name"]] = {
                    "type": task["type"],
                    "depends_on": task["depends_on"],
                    "status": "skipped",
                    "reason": "fail_fast stopped execution",
                    "runs": {},
                    "finished_at": _now(),
                }
                self._write_status()
                continue
            failed = not self._execute_task(task) or failed
        self.status["status"] = "planned" if self.dry_run else ("failed" if failed else "completed")
        self.status["finished_at"] = _now()
        self._write_status()
        return 1 if failed else 0
