import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.compare_experiments import compare_experiments
from evaluation.config import (DataConfig, ModelConfig, ScenarioConfig,
                               config_from_mapping)
from evaluation.data import NPZDataAdapter
from evaluation.metrics import (ThresholdManager, apply_thresholds,
                                bootstrap_confidence_intervals,
                                compute_metrics)
from evaluation.pipeline import run_standard_evaluation
from evaluation.prediction_export import export_sample_predictions
from evaluation.robustness import compute_robustness_summary


def fixture_arrays():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    probabilities = np.array([[.9, .1], [.2, .8], [.8, .7], [.1, .2]])
    return labels, probabilities


def test_metrics_and_thresholds_on_manual_data():
    labels, probabilities = fixture_arrays()
    result = compute_metrics(labels, probabilities, ["A", "B"], [.5, .6])
    assert result["overall"]["macro_roc_auc"] == pytest.approx(1.0)
    assert result["overall"]["macro_f1"] == pytest.approx(1.0)
    assert result["overall"]["hamming_loss"] == 0
    assert result["per_class"].tp.tolist() == [2, 2]
    assert apply_thresholds(probabilities, [.5, .6]).tolist() == labels.tolist()


def test_threshold_file_and_test_provenance(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"source_split": "validation",
                                "thresholds": {"A": .4, "B": .6}}))
    manager = ThresholdManager("load_from_file", path=path)
    assert manager.resolve(2, ["A", "B"]).tolist() == [.4, .6]
    with pytest.raises(ValueError, match="test split"):
        ThresholdManager("fixed_global", source_split="test")
    path.write_text(json.dumps({"thresholds": {"A": .4, "B": .6}}))
    with pytest.raises(ValueError, match="source_split"):
        manager.resolve(2, ["A", "B"])


def test_class_without_positives_returns_nan_and_warning():
    labels = np.array([[0, 1], [0, 0], [0, 1]])
    probabilities = np.array([[.1, .8], [.2, .1], [.3, .7]])
    with pytest.warns(Warning):
        result = compute_metrics(labels, probabilities, ["empty", "valid"])
    empty = result["per_class"].iloc[0]
    assert np.isnan(empty.roc_auc) and np.isnan(empty.pr_auc)
    assert not empty.valid_roc_auc and not empty.valid_pr_auc
    assert empty.warning


def test_bootstrap_is_stable():
    labels, probabilities = fixture_arrays()
    first = bootstrap_confidence_intervals(labels, probabilities, n_bootstrap=30,
                                           random_state=42,
                                           metric_names=["macro_f1"])
    second = bootstrap_confidence_intervals(labels, probabilities, n_bootstrap=30,
                                            random_state=42,
                                            metric_names=["macro_f1"])
    pd.testing.assert_frame_equal(first, second)


def test_prediction_export_preserves_sample_order(tmp_path):
    labels, probabilities = fixture_arrays()
    paths = export_sample_predictions(tmp_path, [30, 10, 20, 40], labels,
                                      probabilities, ["A", "B"], thresholds=.5,
                                      basename="sample_predictions")
    frame = pd.read_csv(paths["csv"])
    assert frame.sample_id.tolist() == [30, 10, 20, 40]
    with np.load(paths["npz"], allow_pickle=False) as archive:
        assert archive["sample_id"].tolist() == [30, 10, 20, 40]


def test_multi_scenario_prediction_export_is_reusable(tmp_path):
    labels, probabilities = fixture_arrays()
    ids = np.tile(np.arange(4), 2)
    paths = export_sample_predictions(
        tmp_path, ids, np.vstack([labels, labels]), np.vstack([probabilities, probabilities]),
        ["A", "B"], thresholds=.5,
        extra_columns={"condition": np.repeat(["clean", "noisy"], 4),
                       "snr": np.repeat(["none", "6"], 4)})
    with np.load(paths["npz"], allow_pickle=False) as archive:
        assert archive["condition"].tolist() == ["clean"] * 4 + ["noisy"] * 4
        assert archive["snr"].tolist() == ["none"] * 4 + ["6"] * 4


def test_clean_noisy_alignment_is_checked(tmp_path):
    labels, _ = fixture_arrays()
    clean = tmp_path / "clean.npz"
    noisy = tmp_path / "noisy.npz"
    np.savez(clean, labels=labels, sample_id=np.arange(4))
    np.savez(noisy, labels=labels, sample_id=np.array([0, 2, 1, 3]))
    config = DataConfig((ScenarioConfig("clean", str(clean), "clean"),
                         ScenarioConfig("noisy", str(noisy), "noisy", 6)),
                        require_ecg=False)
    with pytest.raises(ValueError, match="IDs/order"):
        NPZDataAdapter(config, 2).load_scenarios()


def test_single_and_late_fusion_cpu_adapters_do_not_update_parameters():
    torch = pytest.importorskip("torch")
    from evaluation.model_registry import TorchModelAdapter

    class Single(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.output = torch.nn.Linear(3, 2)
        def forward(self, value):
            return self.output(value)

    class Fusion(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.output = torch.nn.Linear(5, 2)
        def forward(self, ecg, features):
            return self.output(torch.cat([ecg, features], dim=1))

    single = Single(); before = [value.detach().clone() for value in single.parameters()]
    adapter = TorchModelAdapter(single, ModelConfig("single", num_classes=2,
                                                     input_channels=3,
                                                     allow_uninitialized=True), "cpu")
    assert adapter.predict_batch({"ecg": np.ones((4, 3), dtype=np.float32)}).probabilities.shape == (4, 2)
    assert not single.training
    assert all(torch.equal(old, new) for old, new in zip(before, single.parameters()))
    fusion = TorchModelAdapter(Fusion(), ModelConfig("fusion", num_classes=2,
                                                      input_channels=3,
                                                      call_mode="late_fusion",
                                                      allow_uninitialized=True), "cpu")
    output = fusion.predict_batch({"ecg": np.ones((4, 3), dtype=np.float32),
                                   "features": np.ones((4, 2), dtype=np.float32)})
    assert output.probabilities.shape == (4, 2)


def test_torch_adapter_rejects_uninitialized_model():
    torch = pytest.importorskip("torch")
    from evaluation.model_registry import TorchModelAdapter
    with pytest.raises(ValueError, match="requires a checkpoint"):
        TorchModelAdapter(torch.nn.Linear(3, 2), ModelConfig("unsafe", num_classes=2,
                                                             input_channels=3), "cpu")


def test_checkpoint_output_mismatch_is_clear(tmp_path):
    torch = pytest.importorskip("torch")
    from evaluation.model_registry import ClassOutputMismatch, load_checkpoint_strict
    model = torch.nn.Module(); model.output = torch.nn.Linear(3, 2)
    source = torch.nn.Module(); source.output = torch.nn.Linear(3, 3)
    path = tmp_path / "bad.pth"; torch.save({"model": source.state_dict()}, path)
    with pytest.raises(ClassOutputMismatch):
        load_checkpoint_strict(model, str(path), torch.device("cpu"), 2)


def test_precomputed_pipeline_runs_on_cpu(tmp_path):
    labels, probabilities = fixture_arrays()
    data_path = tmp_path / "clean.npz"
    prediction_path = tmp_path / "predictions.npz"
    np.savez(data_path, labels=labels, sample_id=np.arange(4))
    np.savez(prediction_path, probabilities=probabilities, sample_id=np.arange(4))
    output = tmp_path / "result"
    config = config_from_mapping({
        "run": {"experiment_name": "smoke", "output_dir": str(output),
                "overwrite": True, "dataset_name": "synthetic"},
        "model": {"name": "cached", "adapter": "precomputed_npz",
                  "precomputed_path": str(prediction_path), "num_classes": 2,
                  "input_channels": 12},
        "data": {"scenarios": [{"name": "clean", "path": str(data_path),
                                  "condition": "clean"}], "require_ecg": False,
                 "input_channels": 12},
        "analysis": {"class_names": ["A", "B"], "bootstrap": 10,
                     "calibration": True, "robustness": True},
        "output": {"save_predictions": True, "save_plots": False},
    })
    assert run_standard_evaluation(config) == output.resolve()
    assert (output / "metrics/overall_metrics.csv").is_file()
    assert (output / "predictions/sample_probabilities.npz").is_file()
    assert json.loads((output / "manifest.json").read_text())["artifact_count"] > 5


def test_compare_experiments_groups_seeds(tmp_path):
    roots = []
    for model, values in (("m1", [.8, .9]), ("m2", [.7, .75])):
        root = tmp_path / model; (root / "metrics").mkdir(parents=True)
        pd.DataFrame({"experiment_name": [model, model], "model_name": [model, model],
                      "seed": [1, 2], "condition": ["clean", "clean"],
                      "ecg_scenario": ["clean", "clean"],
                      "threshold_strategy": ["fixed_global", "fixed_global"],
                      "macro_roc_auc": values}).to_csv(root / "metrics/overall_metrics.csv", index=False)
        roots.append(root)
    result = compare_experiments(roots, tmp_path / "comparison",
                                 threshold_strategy="fixed_global")
    frame = pd.read_csv(result["model_comparison"])
    assert set(frame.experiment_name) == {"m1", "m2"}
    assert frame.macro_roc_auc_n.tolist() == [2, 2]


def test_robustness_auc_supports_current_numpy():
    result = compute_robustness_summary(.9, [.8, .7, .6], [12, 6, 0])
    assert np.isfinite(result["auc_over_snr"])
    assert result["clean_to_min_snr_absolute_drop"] == pytest.approx(.3)
    assert result["max_adjacent_drop"] == pytest.approx(.1)
