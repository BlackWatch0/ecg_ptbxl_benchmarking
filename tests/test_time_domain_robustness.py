import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from time_domain_robustness.analysis import aggregate_pairs, bootstrap_metrics, compute_metrics, confidence_intervals, matching_report, pair_condition, sample_errors
from time_domain_robustness.constants import FEATURE_COLUMNS
from time_domain_robustness.io import DuplicateKeyError, classify_file, load_data_root


def frame(offset=0.0, missing=False):
    rows = []
    for record, lead, beat in ((1, 0, 0), (1, 0, 1), (2, 1, 0)):
        row = {"RecordNumber": record, "LeadIndex": lead, "BeatIndex": beat, "Patient": "p{}".format(record)}
        row.update({feature: float(index + beat + offset + 1) for index, feature in enumerate(FEATURE_COLUMNS)})
        rows.append(row)
    result = pd.DataFrame(rows)
    if missing:
        result.loc[0, FEATURE_COLUMNS[0]] = np.nan
    return result


def write_condition(root, condition, data, snr=None):
    directory = root / condition / ("snr_{}db".format(snr) if snr is not None else "base")
    directory.mkdir(parents=True, exist_ok=True)
    data.to_csv(directory / "{}_features.csv".format(condition), index=False)


@pytest.fixture
def root(tmp_path):
    write_condition(tmp_path, "clean", frame(missing=True))
    write_condition(tmp_path, "noisy", frame(2), 5)
    write_condition(tmp_path, "denoised", frame(1), 5)
    return tmp_path


def test_discovery_matching_metrics_and_clustered_bootstrap(root):
    data, quality = load_data_root(root)
    assert set(data.Condition) == {"clean", "noisy", "denoised"}
    assert "Patient" in data.columns and set(quality.report_type) == {"file", "feature_quality"}
    report = matching_report(data, "noisy", 5)
    pairs = pair_condition(data, "noisy", 5)
    assert set(report.match_status) == {"matched"}
    assert len(aggregate_pairs(pairs, "record", "median")) == 2
    metrics = compute_metrics(pairs)
    feature = metrics[metrics.feature == FEATURE_COLUMNS[0]].iloc[0]
    assert feature.n_excluded == 1 and feature.nae > 0 and "__macro__" in set(metrics.feature)
    first = bootstrap_metrics(pairs, iterations=3, seed=8)
    assert first.equals(bootstrap_metrics(pairs, iterations=3, seed=8))
    assert "nae_ci_low" in confidence_intervals(metrics, first)
    assert len(sample_errors(pairs, limit=1)) == 1


def test_invalid_condition_and_duplicate_composite_keys_are_rejected(root, tmp_path):
    path = tmp_path / "other.csv"
    frame().to_csv(path, index=False)
    with pytest.raises(ValueError, match="Cannot infer"):
        classify_file(path, tmp_path)
    path.unlink()
    frame().iloc[:1].to_csv(root / "noisy" / "snr_5db" / "extra.csv", index=False)
    with pytest.raises(DuplicateKeyError):
        load_data_root(root)


def test_cli_writes_complete_output_contract(root, tmp_path):
    output = tmp_path / "output"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "code" / "run_time_domain_robustness.py"), "--data-root", str(root), "--output-dir", str(output), "--evaluation-level", "beat", "--aggregation", "mean", "--bootstraps", "2", "--seed", "3"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    expected = {"quality_report.csv", "matching_report.csv", "feature_metrics.csv", "macro_overall.csv", "bootstrap_samples.csv", "feature_ranking.csv", "denoising_improvement.csv", "sample_errors_top100.csv"}
    expected |= {"heatmap_{}_nae.{}".format(condition, extension) for condition in ("noisy", "denoised") for extension in ("png", "pdf")}
    expected |= {"snr_robustness.{}".format(extension) for extension in ("png", "pdf")}
    assert expected <= {path.name for path in output.iterdir()}
