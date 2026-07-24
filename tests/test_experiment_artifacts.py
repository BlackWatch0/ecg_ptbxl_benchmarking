import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from utils import experiment_artifacts as artifacts
from package_original_models_benchmark import package_experiment


def _write_contract(root, model="toy", seed=42):
    artifacts.prepare_experiment_root(root)
    config = root / "config"
    for name in ("resolved_config.yaml", "mlb.pkl", "standard_scaler.pkl"):
        (config / name).write_bytes(b"value")
    for name in ("resolved_config.json", "data_integrity.json", "dataset_manifest.json",
                 "artifact_status.json"):
        artifacts.json_dump(config / name, {})
    artifacts.json_dump(config / "split_manifest.json", {
        "validation_records": 2, "test_records": 2,
    })
    prefix = "seed_{}".format(seed)
    history = pd.DataFrame([{
        "epoch": 1, "train_loss": .5, "valid_loss": .4, "train_accuracy": .8,
        "valid_accuracy": .7, "learning_rate": .01, "epoch_duration_seconds": 1,
        "best_epoch_so_far": 1,
    }])
    history_path = root / "training_logs" / model / (prefix + ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    artifacts.write_runtime_log(root, model, seed, "completed")
    checkpoint = root / "checkpoints" / model / prefix
    checkpoint.mkdir(parents=True)
    (checkpoint / "best_model.pth").write_bytes(b"best")
    (checkpoint / "last_model.pth").write_bytes(b"last")
    artifacts.json_dump(checkpoint / "checkpoint_metadata.json", {})
    metric = root / "metrics" / model
    metric.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "_per_class", "_complexity", "_threshold_search"):
        pd.DataFrame({"value": [1]}).to_csv(metric / (prefix + suffix + ".csv"), index=False)
    artifacts.json_dump(metric / (prefix + "_validation_metrics.json"), {})
    ids = np.array([1, 2])
    values = np.zeros((2, 5), dtype=float)
    prediction = np.zeros((2, 5), dtype=int)
    prediction_dir = root / "predictions" / model / prefix
    artifacts.save_predictions(prediction_dir / "validation_predictions.csv", ids, values, values, prediction, .5)
    artifacts.save_predictions(prediction_dir / "validation_predictions_threshold_0_5.csv", ids, values, values, prediction, .5)
    for condition in artifacts.TEST_CONDITIONS:
        artifacts.json_dump(metric / "{}_{}_integrity.json".format(prefix, condition), {})
        artifacts.save_predictions(prediction_dir / "test_predictions_{}.csv".format(condition), ids, values, values, prediction, .5)
        for strategy in artifacts.THRESHOLD_FILES:
            artifacts.save_predictions(prediction_dir / "test_predictions_{}_{}.csv".format(condition, strategy), ids, values, values, prediction, .5)
    report = root / "final_report"
    for name in artifacts.REQUIRED_REPORT_FILES:
        if name.endswith(".json"):
            artifacts.json_dump(report / name, {})
        else:
            (report / name).write_text("value", encoding="utf-8")
    figure_names = list(artifacts.REQUIRED_FIGURES) + [
        "training_loss_{}".format(model),
        "training_validation_accuracy_{}_{}".format(model, prefix),
    ]
    for name in figure_names:
        for extension in ("png", "pdf"):
            (report / "figures" / (name + "." + extension)).write_bytes(b"figure")


def test_validator_accepts_complete_contract_and_writes_manifests(tmp_path):
    root = tmp_path / "run"
    _write_contract(root)
    report = artifacts.validate_experiment(root, ["toy"], [42])
    assert report["artifact_validation_passed"]
    assert (root / "manifest" / "expected_artifacts.json").is_file()


def test_validator_rejects_missing_wavelet_validation_artifact(tmp_path):
    root = tmp_path / "run"
    _write_contract(root, model="wavelet_nn")
    (root / "metrics" / "wavelet_nn" / "seed_42_validation_metrics.json").unlink()
    with pytest.raises(artifacts.ArtifactValidationError, match="validation_metrics"):
        artifacts.validate_experiment(root, ["wavelet_nn"], [42])


def test_package_contains_complete_validated_root(tmp_path):
    root = tmp_path / "run"
    _write_contract(root, model="xresnet1d101")
    artifacts.json_dump(root / "config" / "resolved_config.json", {
        "models": ["xresnet1d101"], "seeds": [42], "git_commit": "test",
    })
    archive = package_experiment(root, root / "bundle.zip")
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "checkpoints/xresnet1d101/seed_42/best_model.pth" in names
    assert "runtime_logs/xresnet1d101_seed_42.log" in names
    assert "manifest/experiment_status.json" in names
