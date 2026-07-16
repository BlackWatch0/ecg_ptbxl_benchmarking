"""Discovery and validated loading for condition-labelled ECG feature tables."""

import re
from pathlib import Path

import pandas as pd

from .constants import FEATURE_COLUMNS, KEY_COLUMNS


class DuplicateKeyError(ValueError):
    """Raised when a condition has non-unique composite beat keys."""


CONDITIONS = ("clean", "noisy", "denoised")


def discover_feature_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Feature directory does not exist: {}".format(root))
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"})
    if not files:
        raise FileNotFoundError("No CSV or Parquet feature tables found below: {}".format(root))
    return files


def classify_file(path, root):
    text = "/".join(Path(path).relative_to(root).with_suffix("").parts).lower()
    condition = next((name for name in CONDITIONS if re.search(r"(^|[^a-z]){}([^a-z]|$)".format(name), text)), None)
    if condition is None:
        raise ValueError("Cannot infer clean/noisy/denoised condition from path: {}".format(path))
    match = re.search(r"(?:snr|db)[_-]?(-?\d+(?:\.\d+)?)", text)
    return condition, float(match.group(1)) if match else None


def _read(path):
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def load_data_root(root, features=FEATURE_COLUMNS):
    root = Path(root)
    frames, quality = [], []
    required = list(KEY_COLUMNS) + list(features)
    for path in discover_feature_files(root):
        condition, snr = classify_file(path, root)
        frame = _read(path)
        missing = [name for name in required if name not in frame]
        if missing:
            raise ValueError("{} is missing required columns: {}".format(path, ", ".join(missing)))
        frame = frame.copy()
        frame["Condition"], frame["SNR"], frame["SourceFile"] = condition, snr, str(path.relative_to(root))
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frames.append(frame)
        metadata = [name for name in frame.columns if name not in required + ["Condition", "SNR", "SourceFile"]]
        quality.append({"report_type": "file", "Condition": condition, "SNR": snr, "SourceFile": str(path.relative_to(root)), "rows": len(frame), "columns": len(frame.columns), "metadata_columns": ",".join(metadata)})
        quality.extend({"report_type": "feature_quality", "Condition": condition, "SNR": snr, "SourceFile": str(path.relative_to(root)), "feature": feature, "rows": len(frame), "missing_values": int(frame[feature].isna().sum())} for feature in features)
    result = pd.concat(frames, ignore_index=True, sort=False)
    duplicated = result.duplicated(["Condition", "SNR"] + list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["Condition", "SNR"] + list(KEY_COLUMNS)].head(5).to_dict("records")
        raise DuplicateKeyError("Duplicate condition/SNR/composite keys found: {}".format(examples))
    return result, pd.DataFrame(quality)


def load_feature_tables(root, features=FEATURE_COLUMNS):
    data, _ = load_data_root(root, features)
    if data.Condition.nunique() != 1:
        raise ValueError("load_feature_tables requires one condition; use load_data_root instead")
    return data.loc[:, list(KEY_COLUMNS) + list(features)]
