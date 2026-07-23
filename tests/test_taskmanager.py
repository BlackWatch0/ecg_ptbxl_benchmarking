import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from task_manager.config import ConfigError, load_config
from task_manager.runner import TaskRunner, _is_foreign_absolute_path


def write_config(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def base_config(tmp_path):
    return {
        "version": 1,
        "output_dir": "artifacts",
        "global": {"seed": 7, "resume": True, "fail_fast": False},
        "models": {
            "xresnet1d101": {"epochs": 2},
            "lstm": {"batch_size": 4},
        },
        "model_groups": {"small": ["xresnet1d101", "lstm"]},
        "tasks": [
            {"name": "train", "type": "train", "models": "small"},
            {"name": "evaluate", "type": "evaluate", "depends_on": ["train"],
             "models": ["@small"]},
            {"name": "report", "type": "report", "depends_on": ["evaluate"]},
        ],
    }


def test_config_resolves_paths_groups_and_topological_order(tmp_path):
    value = base_config(tmp_path)
    value["global"]["data_root"] = "datasets/clean"
    value["tasks"] = [value["tasks"][2], value["tasks"][1], value["tasks"][0]]
    config = load_config(write_config(tmp_path / "tasks.yaml", value))
    assert config["output_dir"] == str((tmp_path / "artifacts").resolve())
    assert config["global"]["data_root"] == str((tmp_path / "datasets/clean").resolve())
    assert config["model_groups"]["small"] == ["xresnet1d101", "lstm"]
    assert [task["name"] for task in config["tasks"]] == ["train", "evaluate", "report"]


def test_strict_schema_rejects_unknown_duplicate_and_cycles(tmp_path):
    value = base_config(tmp_path)
    value["global"]["typo"] = True
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(write_config(tmp_path / "unknown.yaml", value))

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("output_dir: one\noutput_dir: two\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate YAML key"):
        load_config(duplicate)

    value = base_config(tmp_path)
    value["tasks"][0]["depends_on"] = ["report"]
    with pytest.raises(ConfigError, match="cycle detected"):
        load_config(write_config(tmp_path / "cycle.yaml", value))

    value = base_config(tmp_path)
    value["model_groups"] = {"first": ["second"], "second": ["first"]}
    with pytest.raises(ConfigError, match="Model group cycle detected"):
        load_config(write_config(tmp_path / "group-cycle.yaml", value))


def test_dry_run_writes_commands_status_and_logs_without_training(tmp_path):
    value = base_config(tmp_path)
    value["models"]["xresnet1d101"]["seed"] = 11
    config = load_config(write_config(tmp_path / "tasks.yaml", value))
    result = TaskRunner(config, task_names=["evaluate"], dry_run=True).run()
    assert result == 0
    output = Path(config["output_dir"])
    status = json.loads((output / "task_plan.json").read_text(encoding="utf-8"))
    assert status["status"] == "planned"
    assert set(status["tasks"]) == {"train", "evaluate"}
    train_runs = status["tasks"]["train"]["runs"]
    assert set(train_runs) == {"xresnet1d101_seed_11", "lstm_seed_7"}
    for run in train_runs.values():
        assert run["command"][0] == sys.executable
        assert "--skip-test-evaluation" in run["command"]
        assert "--evaluate-only" not in run["command"]
        assert Path(run["log"]).is_file()
    for run in status["tasks"]["evaluate"]["runs"].values():
        assert "--evaluate-only" in run["command"]
        seed_index = run["command"].index("--seeds")
        model = run["command"][run["command"].index("--models") + 1]
        expected = "11" if model == "xresnet1d101" else "7"
        assert run["command"][seed_index + 1] == expected
    assert (output / "config" / "resolved_config.yaml").is_file()


def test_dry_run_does_not_replace_runtime_status(tmp_path):
    value = base_config(tmp_path)
    config = load_config(write_config(tmp_path / "tasks.yaml", value))
    output = Path(config["output_dir"])
    output.mkdir(parents=True)
    runtime = output / "task_status.json"
    runtime.write_text('{"status": "completed"}\n', encoding="utf-8")
    assert TaskRunner(config, task_names=["train"], dry_run=True).run() == 0
    assert json.loads(runtime.read_text(encoding="utf-8"))["status"] == "completed"
    assert (output / "task_plan.json").is_file()


def test_execution_uses_one_safe_subprocess_per_model(tmp_path, monkeypatch):
    value = base_config(tmp_path)
    value["global"]["resume"] = False
    config = load_config(write_config(tmp_path / "tasks.yaml", value))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("task_manager.runner.subprocess.run", fake_run)
    assert TaskRunner(config, task_names=["train"]).run() == 0
    assert len(calls) == 2
    assert all(call[0][0] == sys.executable for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert all("--skip-test-evaluation" in call[0] for call in calls)


def test_wavelet_process_is_cpu_only(tmp_path, monkeypatch):
    value = base_config(tmp_path)
    value["global"].update({"resume": False, "device": "cuda"})
    value["models"] = ["wavelet_nn"]
    value["tasks"] = [{"name": "train", "type": "train"}]
    config = load_config(write_config(tmp_path / "wavelet.yaml", value))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("task_manager.runner.subprocess.run", fake_run)
    assert TaskRunner(config).run() == 0
    command, kwargs = calls[0]
    assert command[command.index("--device") + 1] == "cpu"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["env"]["TF_CPP_MIN_LOG_LEVEL"] == "2"


def test_resume_skips_completed_run_without_truncating_log(tmp_path, monkeypatch):
    value = base_config(tmp_path)
    value["models"] = ["xresnet1d101"]
    value["tasks"] = [{"name": "train", "type": "train"}]
    config = load_config(write_config(tmp_path / "tasks.yaml", value))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("task_manager.runner.subprocess.run", fake_run)
    assert TaskRunner(config).run() == 0
    log_path = Path(config["output_dir"]) / "task_logs" / "train" / "xresnet1d101_seed_7.log"
    log_path.write_text("preserve this log\n", encoding="utf-8")
    assert TaskRunner(config).run() == 0
    assert len(calls) == 1
    assert log_path.read_text(encoding="utf-8") == "preserve this log\n"


def test_prepare_requires_a_local_archive_or_search_root(tmp_path):
    value = {
        "version": 1,
        "output_dir": "artifacts",
        "tasks": [{"name": "prepare", "type": "prepare_data"}],
    }
    with pytest.raises(ConfigError, match="archive or search_root"):
        load_config(write_config(tmp_path / "prepare.yaml", value))


def test_foreign_absolute_paths_are_not_rewritten(tmp_path):
    value = {
        "version": 1,
        "output_dir": "/mnt/ecg/runs/example",
        "global": {"data_root": "C:\\ecg\\data"},
        "models": ["xresnet1d101"],
        "tasks": [{"name": "train", "type": "train"}],
    }
    config = load_config(write_config(tmp_path / "paths.yaml", value))
    assert config["output_dir"] == "/mnt/ecg/runs/example"
    assert config["global"]["data_root"] == "C:\\ecg\\data"


def test_foreign_paths_validate_but_cannot_run_on_the_wrong_os(tmp_path):
    assert _is_foreign_absolute_path("/mnt/ecg/run", platform_name="nt")
    assert _is_foreign_absolute_path("C:\\ecg\\run", platform_name="posix")
    assert not _is_foreign_absolute_path("/mnt/ecg/run", platform_name="posix")
    value = {
        "version": 1,
        "output_dir": "/mnt/ecg/run",
        "models": ["xresnet1d101"],
        "tasks": [{"name": "train", "type": "train"}],
    }
    config = load_config(write_config(tmp_path / "foreign.yaml", value))
    if sys.platform == "win32":
        with pytest.raises(ValueError, match="another operating system"):
            TaskRunner(config, dry_run=True)


@pytest.mark.parametrize("name", [
    "original_models_benchmark.yaml",
    "original_models_evaluate.yaml",
    "prepare_original_data.yaml",
    "aws_original_models.example.yaml",
])
def test_repository_task_configs_validate(name):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "taskmanager" / name)
    assert config["version"] == 1
    assert config["tasks"]
