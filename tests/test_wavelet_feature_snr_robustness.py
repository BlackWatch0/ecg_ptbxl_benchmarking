import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from wavelet_feature_snr_robustness import FEATURE_TYPES, LEADS, run_analysis


def _features():
    return [f"Lead_{lead}_{feature_type}" for lead in LEADS for feature_type in FEATURE_TYPES]


def _write_features(path: Path, records, values):
    frame = pd.DataFrame(values, columns=_features())
    frame.insert(0, "RecordName", records)
    frame.to_csv(path, index=False)


def test_runs_record_aligned_no_label_analysis(tmp_path):
    rng = np.random.default_rng(42)
    records = [f"{index:05d}_lr" for index in range(10)]
    clean = rng.normal(size=(10, 72))
    noisy = clean + rng.normal(scale=.5, size=(10, 72))
    denoised = clean + rng.normal(scale=.1, size=(10, 72))
    clean_path, noisy_path, denoised_path = (tmp_path / "clean.csv", tmp_path / "noisy.csv", tmp_path / "denoised.csv")
    _write_features(clean_path, records, clean)
    _write_features(noisy_path, list(reversed(records)), noisy[::-1])
    _write_features(denoised_path, records, denoised)
    output = tmp_path / "output"

    results = run_analysis({
        "clean_feature_csv": clean_path,
        "noisy_feature_csvs": {"0": noisy_path, "24": tmp_path / "not_present.csv"},
        "denoised_feature_csvs": {"0": denoised_path},
        "output_dir": output,
        "enable_bootstrap_ci": True,
        "bootstrap_iterations": 10,
        "random_seed": 42,
    })

    assert set(results["overall"]["dataset_type"]) == {"noisy", "denoised"}
    assert results["overall"].set_index("dataset_type").loc["denoised", "mean_nae"] < results["overall"].set_index("dataset_type").loc["noisy", "mean_nae"]
    assert (output / "data_validation_report.csv").is_file()
    assert (output / "clean_feature_statistics.csv").is_file()
    assert (output / "overall_metrics.csv").is_file()
    assert (output / "per_lead_metrics.csv").is_file()
    assert (output / "per_feature_type_metrics.csv").is_file()
    assert (output / "per_feature_metrics.csv").is_file()
    assert (output / "per_sample_metrics.csv").is_file()
    assert (output / "denoising_improvement.csv").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "figures" / "sample_nae_boxplot.png").is_file()
