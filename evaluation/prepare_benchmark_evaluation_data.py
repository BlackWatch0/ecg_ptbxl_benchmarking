"""Prepare aligned NPZ scenarios from original benchmark manifests."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import wfdb


DEFAULT_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _read_waveform(path: str) -> np.ndarray:
    signal = wfdb.rdsamp(str(Path(path).with_suffix("")))[0]
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("Expected [T,12] WFDB signal at {}, got {}".format(path, signal.shape))
    return signal


def _standardize(records: Sequence[np.ndarray], scaler: object) -> np.ndarray:
    standardized = []
    for record in records:
        shape = record.shape
        standardized.append(scaler.transform(record.reshape(-1, 1)).reshape(shape))
    return np.asarray(standardized, dtype=np.float32)


def prepare(data_config: Path, labels_csv: Path, scaler_path: Path,
            output_dir: Path, sample_count: int, snr_list: Sequence[int],
            wavelet_scaler_path: Optional[Path] = None,
            wavelet_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None
            ) -> Dict[str, str]:
    """Create ID-aligned clean/noisy/denoised scenario NPZ files."""
    config = json.loads(data_config.read_text(encoding="utf-8"))
    manifests = {name: Path(path) for name, path in config["manifests"].items()}
    labels_frame = pd.read_csv(labels_csv)
    id_column = "ecg_id" if "ecg_id" in labels_frame else "record_id"
    label_columns = ["true_" + name for name in DEFAULT_CLASSES]
    missing = [column for column in [id_column] + label_columns if column not in labels_frame]
    if missing:
        raise ValueError("Label prediction CSV is missing {}".format(missing))
    labels_frame = labels_frame.drop_duplicates(id_column).set_index(id_column)
    clean = pd.read_csv(manifests["clean"])
    clean = clean[clean.ecg_id.isin(labels_frame.index)].sort_values("ecg_id")
    if len(clean) < sample_count:
        raise ValueError("Only {} aligned clean records are available".format(len(clean)))
    clean = clean.iloc[:sample_count]
    sample_ids = clean.ecg_id.to_numpy()
    labels = labels_frame.loc[sample_ids, label_columns].to_numpy(dtype=np.float32)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    if int(getattr(scaler, "n_features_in_", -1)) != 1:
        raise ValueError("ECG scaler must be the train-only single-feature standardizer")
    wavelet_scaler = None
    if wavelet_scaler_path is not None:
        with wavelet_scaler_path.open("rb") as handle:
            wavelet_scaler = pickle.load(handle)
        if int(getattr(wavelet_scaler, "n_features_in_", -1)) != 864:
            raise ValueError("Wavelet scaler must expect exactly 864 features")
        if wavelet_extractor is None:
            code_root = Path(__file__).resolve().parents[1] / "code"
            sys.path.insert(0, str(code_root))
            from models.wavelet import get_ecg_features
            wavelet_extractor = get_ecg_features
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    def save(name: str, condition: str, snr: float, rows: pd.DataFrame) -> None:
        rows = rows.set_index("ecg_id").loc[sample_ids]
        waveforms = _standardize([_read_waveform(path) for path in rows.record_path], scaler)
        arrays = {
            "ecg": np.transpose(waveforms, (0, 2, 1)), "labels": labels,
            "sample_id": sample_ids, "condition": np.asarray(condition),
            "snr": np.asarray(snr), "class_names": np.asarray(DEFAULT_CLASSES),
        }
        if wavelet_scaler is not None and wavelet_extractor is not None:
            features = np.asarray(wavelet_extractor(waveforms), dtype=np.float32)
            if features.shape != (len(sample_ids), 864) or not np.isfinite(features).all():
                raise ValueError("Invalid wavelet features for {}: {}".format(name, features.shape))
            scaled_features = wavelet_scaler.transform(features).astype(np.float32)
            if not np.isfinite(scaled_features).all():
                raise ValueError("Scaled wavelet features are non-finite for {}".format(name))
            arrays["features"] = scaled_features
        path = output_dir / (name + ".npz")
        np.savez_compressed(path, **arrays)
        outputs[name] = str(path.resolve())
        print("Prepared {}: records={}, shape={}".format(name, len(rows), tuple(waveforms.shape)))

    save("clean", "clean", np.nan, clean)
    for condition in ("noisy", "denoised"):
        frame = pd.read_csv(manifests[condition])
        for snr in snr_list:
            rows = frame[(frame.snr_db == snr) & frame.ecg_id.isin(sample_ids)]
            if len(rows) != len(sample_ids) or rows.ecg_id.nunique() != len(sample_ids):
                raise ValueError("{} SNR {} is not aligned for all selected IDs".format(condition, snr))
            label = "m{}".format(abs(snr)) if snr < 0 else str(snr)
            save("{}_snr{}".format(condition, label), condition, snr, rows)
    manifest = output_dir / "scenarios.json"
    manifest_payload = {"class_names": DEFAULT_CLASSES, "sample_count": sample_count,
                        "scenarios": outputs}
    if wavelet_scaler_path is not None:
        manifest_payload["wavelet_features"] = {
            "feature_count": 864, "scaler": str(wavelet_scaler_path.resolve())}
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--snr-list", nargs="+", type=int, default=[24, 12, 6, 0, -6])
    parser.add_argument("--wavelet-scaler", type=Path,
                        help="Train-only 864-feature scaler; enables Wavelet+NN features")
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive")
    prepare(args.data_config, args.labels_csv, args.scaler, args.output_dir,
            args.sample_count, args.snr_list, args.wavelet_scaler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
