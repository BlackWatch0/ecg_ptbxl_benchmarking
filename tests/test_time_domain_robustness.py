import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from time_domain_robustness.analysis import _bootstrap_draw_values, _bootstrap_weights, aggregate_pairs, bootstrap_inputs, bootstrap_metrics, compute_metrics, confidence_intervals, matching_report, pair_condition, sample_errors
from time_domain_robustness.constants import FEATURE_COLUMNS
from time_domain_robustness.io import DuplicateKeyError, classify_file, load_data_root
from time_domain_robustness_v2 import bootstrap_metrics as bootstrap_metrics_v2, clean_scale, compute_metrics as compute_metrics_v2, input_manifest, overlap_audit, pair_condition as pair_condition_v2


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


def test_batched_bootstrap_matches_duplicate_record_samples():
    rows = []
    for record, offset in ((1, 1.0), (2, 3.0)):
        for beat, clean in enumerate((10.0, 20.0, np.nan if record == 2 else 30.0)):
            row = {"RecordNumber": record, "LeadIndex": 0, "BeatIndex": beat, "comparison": "noisy", "SNR": 5}
            for feature in FEATURE_COLUMNS:
                row[feature + "_clean"] = clean
                row[feature + "_comparison"] = clean + offset if np.isfinite(clean) else np.nan
            rows.append(row)
    pairs = pd.DataFrame(rows)
    inputs = bootstrap_inputs(pairs)
    indices = np.array([[0, 0], [1, 0]])
    nae, raw, scaled = _bootstrap_draw_values(inputs, _bootstrap_weights(indices, len(inputs.record_ids)))
    for draw, selected_records in enumerate(indices):
        sampled = pd.concat([pairs[pairs.RecordNumber == inputs.record_ids[index]] for index in selected_records], ignore_index=True)
        naive = compute_metrics(sampled).set_index("feature")
        for column, expected in (("nae", nae[draw]), ("cosine_raw", raw[draw]), ("cosine_scaled", scaled[draw])):
            assert np.allclose(expected, naive.loc[list(FEATURE_COLUMNS), column].to_numpy(), equal_nan=True)


def test_bootstrap_defaults_to_macro_and_can_be_disabled(root, tmp_path):
    data, _ = load_data_root(root)
    pairs = pair_condition(data, "noisy", 5)
    default = bootstrap_metrics(pairs, iterations=2, seed=4, batch_size=1)
    detailed = bootstrap_metrics(pairs, iterations=2, seed=4, batch_size=2, per_feature=True)
    assert set(default.feature) == {"__macro__"}
    assert len(detailed) == 2 * (len(FEATURE_COLUMNS) + 1)
    output = tmp_path / "output"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "code" / "run_time_domain_robustness.py"), "--data-root", str(root), "--output-dir", str(output), "--evaluation-level", "beat", "--aggregation", "mean", "--bootstrap-iterations", "2", "--bootstrap-batch-size", "1", "--disable-bootstrap"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert pd.read_csv(output / "bootstrap_samples.csv").empty


def test_cli_writes_complete_output_contract(root, tmp_path):
    output = tmp_path / "output"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "code" / "run_time_domain_robustness.py"), "--data-root", str(root), "--output-dir", str(output), "--evaluation-level", "beat", "--aggregation", "mean", "--bootstraps", "2", "--seed", "3"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    expected = {"quality_report.csv", "matching_report.csv", "feature_metrics.csv", "macro_overall.csv", "bootstrap_samples.csv", "feature_ranking.csv", "denoising_improvement.csv", "sample_errors_top100.csv"}
    expected |= {"heatmap_{}_nae.{}".format(condition, extension) for condition in ("noisy", "denoised") for extension in ("png", "pdf")}
    expected |= {"snr_robustness.{}".format(extension) for extension in ("png", "pdf")}
    assert expected <= {path.name for path in output.iterdir()}


def test_v2_clean_scale_strict_macro_vectors_and_provenance(root):
    fallback = clean_scale(np.array([3.0, 3.0, np.nan]))
    assert fallback.value == 3.0 and fallback.method == "median_absolute_clean"
    data, _ = load_data_root(root)
    pairs = pair_condition_v2(data, "noisy", 5)
    metrics = compute_metrics_v2(pairs)
    macro = metrics[metrics.feature == "__macro_13d__"].iloc[0]
    assert macro.clean_scale_method == "strict_13_feature_macro"
    assert np.isfinite(macro.nmae)
    assert np.isfinite(macro.cosine_raw_13d) and np.isfinite(macro.cosine_scaled_13d)
    pairs.loc[0, FEATURE_COLUMNS[0] + "_comparison"] = np.nan
    assert np.isnan(compute_metrics_v2(pairs).query("feature == '__macro_13d__'").iloc[0].nmae)
    manifest = input_manifest(root)
    assert set(manifest.columns) == {"path", "condition", "snr_db", "bytes", "sha256"}
    assert manifest.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    audit = overlap_audit(data)
    assert set(audit.overlap_keys) == {3}
    with pytest.raises(ValueError, match="13 unique"):
        compute_metrics_v2(pair_condition_v2(data, "noisy", 5), FEATURE_COLUMNS[:-1])


def test_v2_bootstrap_and_cli_output_contract(root, tmp_path):
    data, _ = load_data_root(root)
    pairs = pair_condition_v2(data, "noisy", 5)
    draws = bootstrap_metrics_v2(pairs, iterations=3, seed=6)
    assert len(draws) == 3 and set(draws.feature) == {"__macro_13d__"}
    output = tmp_path / "output-v2"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "code" / "run_time_domain_robustness_v2.py"), "--data-root", str(root), "--output-dir", str(output), "--evaluation-level", "beat", "--aggregation", "mean", "--bootstrap-iterations", "2"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    expected = {"v2_input_manifest.csv", "v2_quality_report.csv", "v2_overlap_audit.csv", "v2_feature_metrics.csv", "v2_macro_metrics.csv", "v2_bootstrap_samples.csv", "v2_macro_nmae_by_snr.png", "v2_macro_nmae_by_snr.pdf"}
    expected |= {"v2_heatmap_{}_nmae.{}".format(condition, extension) for condition in ("noisy", "denoised") for extension in ("png", "pdf")}
    assert expected <= {path.name for path in output.iterdir()}
