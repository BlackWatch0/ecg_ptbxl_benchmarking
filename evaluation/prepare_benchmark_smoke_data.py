"""Prepare small aligned NPZ scenarios from original benchmark manifests."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

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
            output_dir: Path, sample_count: int, snr_list: Sequence[int]) -> Dict[str, str]:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    def save(name: str, condition: str, snr: float, rows: pd.DataFrame) -> None:
        rows = rows.set_index("ecg_id").loc[sample_ids]
        waveforms = _standardize([_read_waveform(path) for path in rows.record_path], scaler)
        path = output_dir / (name + ".npz")
        np.savez_compressed(path, ecg=np.transpose(waveforms, (0, 2, 1)), labels=labels,
                            sample_id=sample_ids, condition=np.asarray(condition),
                            snr=np.asarray(snr), class_names=np.asarray(DEFAULT_CLASSES))
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
    manifest.write_text(json.dumps({"class_names": DEFAULT_CLASSES, "sample_count": sample_count,
                                    "scenarios": outputs}, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--snr-list", nargs="+", type=int, default=[24, 12, 6, 0, -6])
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive")
    prepare(args.data_config, args.labels_csv, args.scaler, args.output_dir,
            args.sample_count, args.snr_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
