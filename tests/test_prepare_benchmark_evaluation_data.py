import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluation import prepare_benchmark_evaluation_data as preparation


def test_prepare_includes_scaled_wavelet_features(tmp_path, monkeypatch):
    ids = np.asarray([10, 20])
    labels = pd.DataFrame({
        "ecg_id": ids,
        "true_NORM": [1, 0], "true_MI": [0, 1], "true_STTC": [0, 0],
        "true_CD": [0, 1], "true_HYP": [0, 0],
    })
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    clean = pd.DataFrame({"ecg_id": ids, "record_path": ["10.dat", "20.dat"]})
    altered = pd.DataFrame({
        "ecg_id": np.tile(ids, 1), "snr_db": [0, 0],
        "record_path": ["10.dat", "20.dat"],
    })
    manifests = {}
    for name, frame in (("clean", clean), ("noisy", altered), ("denoised", altered)):
        path = tmp_path / (name + ".csv")
        frame.to_csv(path, index=False)
        manifests[name] = str(path)
    config_path = tmp_path / "data.json"
    config_path.write_text(json.dumps({"manifests": manifests}), encoding="utf-8")

    ecg_scaler = StandardScaler().fit(np.arange(20, dtype=float).reshape(-1, 1))
    feature_scaler = StandardScaler().fit(
        np.stack([np.zeros(864), np.ones(864), np.full(864, 2.0)]))
    ecg_scaler_path = tmp_path / "ecg_scaler.pkl"
    feature_scaler_path = tmp_path / "feature_scaler.pkl"
    for path, scaler in ((ecg_scaler_path, ecg_scaler),
                         (feature_scaler_path, feature_scaler)):
        with path.open("wb") as handle:
            pickle.dump(scaler, handle)

    monkeypatch.setattr(
        preparation, "_read_waveform",
        lambda path: np.full((1000, 12), 10.0 if path.startswith("10") else 20.0,
                            dtype=np.float32))

    def extract(waveforms):
        return np.stack([np.full(864, index, dtype=np.float32)
                         for index in range(len(waveforms))])

    output = tmp_path / "scenarios"
    preparation.prepare(config_path, labels_path, ecg_scaler_path, output, 2, [0],
                        feature_scaler_path, extract)

    with np.load(output / "clean.npz", allow_pickle=False) as archive:
        assert archive["ecg"].shape == (2, 12, 1000)
        assert archive["features"].shape == (2, 864)
        assert np.isfinite(archive["features"]).all()
        assert np.array_equal(archive["sample_id"], ids)
    manifest = json.loads((output / "scenarios.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 2
    assert manifest["wavelet_features"]["feature_count"] == 864
    assert set(manifest["scenarios"]) == {"clean", "noisy_snr0", "denoised_snr0"}
