"""Version 2 time-domain robustness metrics and provenance reporting."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from time_domain_robustness.constants import FEATURE_COLUMNS, KEY_COLUMNS
from time_domain_robustness.io import classify_file, discover_feature_files, load_data_root


@dataclass(frozen=True)
class CleanScale:
    """A robust clean-reference scale and the deterministic fallback used."""

    value: float
    method: str
    p05: float
    p95: float


def _validate_features(features):
    if len(features) != 13 or len(set(features)) != 13:
        raise ValueError("v2 requires exactly 13 unique feature columns")


def clean_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return CleanScale(np.nan, "unavailable", np.nan, np.nan)
    p05, p95 = np.percentile(values, [5, 95])
    scale = p95 - p05
    if np.isfinite(scale) and scale > np.finfo(float).eps:
        return CleanScale(float(scale), "p95_p05", float(p05), float(p95))
    median_absolute = float(np.median(np.abs(values)))
    if np.isfinite(median_absolute) and median_absolute > np.finfo(float).eps:
        return CleanScale(median_absolute, "median_absolute_clean", float(p05), float(p95))
    return CleanScale(1.0, "unit", float(p05), float(p95))


def input_manifest(root):
    root = Path(root)
    rows = []
    for path in discover_feature_files(root):
        condition, snr = classify_file(path, root)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"path": path.relative_to(root).as_posix(), "condition": condition,
                     "snr_db": snr, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return pd.DataFrame(rows, columns=["path", "condition", "snr_db", "bytes", "sha256"])


def overlap_audit(data):
    keys = list(KEY_COLUMNS)
    rows = []
    clean = data[data.Condition.eq("clean")][keys]
    for candidate in data[data.Condition.ne("clean")][["Condition", "SNR"]].drop_duplicates().to_dict("records"):
        comparison, snr = candidate["Condition"], candidate["SNR"]
        selected = data.Condition.eq(comparison) & (data.SNR.isna() if pd.isna(snr) else data.SNR.eq(snr))
        other = data.loc[selected, keys]
        matched = clean.merge(other, on=keys, how="inner")
        rows.append({"comparison": comparison, "snr_db": snr, "clean_keys": len(clean),
                     "comparison_keys": len(other), "overlap_keys": len(matched),
                     "clean_only_keys": len(clean) - len(matched),
                     "comparison_only_keys": len(other) - len(matched),
                     "overlap_fraction_of_clean": len(matched) / len(clean) if len(clean) else np.nan,
                     "overlap_fraction_of_comparison": len(matched) / len(other) if len(other) else np.nan})
    return pd.DataFrame(rows)


def pair_condition(data, comparison, snr, features=FEATURE_COLUMNS):
    keys, columns = list(KEY_COLUMNS), list(features)
    clean = data[data.Condition.eq("clean")][keys + columns]
    selected = data.Condition.eq(comparison) & (data.SNR.isna() if pd.isna(snr) else data.SNR.eq(snr))
    other = data.loc[selected, keys + columns]
    paired = clean.merge(other, on=keys, how="inner", validate="one_to_one", suffixes=("_clean", "_comparison"))
    paired["comparison"], paired["snr_db"] = comparison, snr
    return paired


def _vector_cosines(data, features, scales):
    _validate_features(features)
    left = data[[feature + "_clean" for feature in features]].to_numpy(dtype=float)
    right = data[[feature + "_comparison" for feature in features]].to_numpy(dtype=float)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    raw = np.full(len(data), np.nan)
    scaled = np.full(len(data), np.nan)
    if valid.any():
        raw_left, raw_right = left[valid], right[valid]
        raw_denominator = np.linalg.norm(raw_left, axis=1) * np.linalg.norm(raw_right, axis=1)
        raw[valid] = np.divide(np.sum(raw_left * raw_right, axis=1), raw_denominator,
                               out=np.full(len(raw_left), np.nan), where=raw_denominator > 0)
        scale_values = np.asarray([scales[feature].value for feature in features])
        scaled_left, scaled_right = raw_left / scale_values, raw_right / scale_values
        scaled_denominator = np.linalg.norm(scaled_left, axis=1) * np.linalg.norm(scaled_right, axis=1)
        scaled[valid] = np.divide(np.sum(scaled_left * scaled_right, axis=1), scaled_denominator,
                                  out=np.full(len(scaled_left), np.nan), where=scaled_denominator > 0)
    return raw, scaled


def compute_metrics(data, features=FEATURE_COLUMNS, scales=None, level="beat", aggregation="mean"):
    if data.empty:
        return pd.DataFrame()
    _validate_features(features)
    scales = scales or {feature: clean_scale(data[feature + "_clean"]) for feature in features}
    rows = []
    nmae_values = []
    complete_13d = True
    for feature in features:
        left = data[feature + "_clean"].to_numpy(dtype=float)
        right = data[feature + "_comparison"].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        complete_13d = complete_13d and bool(np.isfinite(right).all())
        nmae = np.mean(np.abs(right[valid] - left[valid])) / scales[feature].value if valid.any() else np.nan
        nmae_values.append(nmae)
        rows.append({"comparison": data.comparison.iloc[0], "snr_db": data.snr_db.iloc[0],
                     "evaluation_level": level, "aggregation": aggregation, "feature": feature,
                     "n_total": len(data), "n_valid": int(valid.sum()), "n_excluded": int((~valid).sum()),
                     "clean_scale": scales[feature].value, "clean_scale_method": scales[feature].method,
                     "clean_p05": scales[feature].p05, "clean_p95": scales[feature].p95, "nmae": nmae})
    raw, scaled = _vector_cosines(data, features, scales)
    strict_nmae = (np.mean(nmae_values)
                   if complete_13d and np.isfinite(nmae_values).all() else np.nan)
    macro = {"comparison": data.comparison.iloc[0], "snr_db": data.snr_db.iloc[0],
             "evaluation_level": level, "aggregation": aggregation, "feature": "__macro_13d__",
             "n_total": len(data), "n_valid": int(np.isfinite(raw).sum()),
             "n_excluded": int((~np.isfinite(raw)).sum()), "clean_scale": np.nan,
             "clean_scale_method": "strict_13_feature_macro", "clean_p05": np.nan, "clean_p95": np.nan,
             "nmae": strict_nmae, "cosine_raw_13d": np.nanmean(raw) if np.isfinite(raw).any() else np.nan,
             "cosine_scaled_13d": np.nanmean(scaled) if np.isfinite(scaled).any() else np.nan}
    result = pd.DataFrame(rows)
    result["cosine_raw_13d"] = np.nan
    result["cosine_scaled_13d"] = np.nan
    return pd.concat([result, pd.DataFrame([macro])], ignore_index=True)


def bootstrap_sufficient_statistics(data, features=FEATURE_COLUMNS, scales=None):
    _validate_features(features)
    scales = scales or {feature: clean_scale(data[feature + "_clean"]) for feature in features}
    records, inverse = np.unique(data.RecordNumber.to_numpy(), return_inverse=True)
    nmae_sums, nmae_counts = np.zeros((len(records), len(features))), np.zeros((len(records), len(features)))
    for column, feature in enumerate(features):
        left, right = data[feature + "_clean"].to_numpy(float), data[feature + "_comparison"].to_numpy(float)
        valid = np.isfinite(left) & np.isfinite(right)
        nmae_sums[:, column] = np.bincount(inverse[valid], weights=np.abs(right[valid] - left[valid]) / scales[feature].value, minlength=len(records))
        nmae_counts[:, column] = np.bincount(inverse[valid], minlength=len(records))
    raw, scaled = _vector_cosines(data, features, scales)
    return records, nmae_sums, nmae_counts, np.bincount(inverse[np.isfinite(raw)], weights=raw[np.isfinite(raw)], minlength=len(records)), np.bincount(inverse[np.isfinite(raw)], minlength=len(records)), np.bincount(inverse[np.isfinite(scaled)], weights=scaled[np.isfinite(scaled)], minlength=len(records)), np.bincount(inverse[np.isfinite(scaled)], minlength=len(records))


def bootstrap_metrics(data, features=FEATURE_COLUMNS, scales=None, level="beat", aggregation="mean", iterations=1000, seed=0):
    if iterations < 1 or data.empty:
        return pd.DataFrame()
    records, nmae_sums, nmae_counts, raw_sums, raw_counts, scaled_sums, scaled_counts = bootstrap_sufficient_statistics(data, features, scales)
    rng, rows = np.random.default_rng(seed), []
    for iteration in range(iterations):
        weights = np.bincount(rng.integers(len(records), size=len(records)), minlength=len(records))
        per_feature = np.divide(weights @ nmae_sums, weights @ nmae_counts, out=np.full(len(features), np.nan), where=(weights @ nmae_counts) > 0)
        rows.append({"bootstrap_iteration": iteration, "comparison": data.comparison.iloc[0], "snr_db": data.snr_db.iloc[0], "evaluation_level": level, "aggregation": aggregation, "feature": "__macro_13d__", "nmae": np.mean(per_feature) if np.isfinite(per_feature).all() else np.nan, "cosine_raw_13d": (weights @ raw_sums) / (weights @ raw_counts) if weights @ raw_counts else np.nan, "cosine_scaled_13d": (weights @ scaled_sums) / (weights @ scaled_counts) if weights @ scaled_counts else np.nan})
    return pd.DataFrame(rows)


def confidence_intervals(point, samples, alpha=0.05):
    result = point.copy()
    for metric in ("nmae", "cosine_raw_13d", "cosine_scaled_13d"):
        result[metric + "_ci_low"] = np.nan
        result[metric + "_ci_high"] = np.nan
    if samples.empty:
        return result
    for index, row in result[result.feature.eq("__macro_13d__")].iterrows():
        selected = samples[(samples.comparison.eq(row.comparison)) & (samples.snr_db.eq(row.snr_db)) & (samples.evaluation_level.eq(row.evaluation_level)) & (samples.aggregation.eq(row.aggregation))]
        for metric in ("nmae", "cosine_raw_13d", "cosine_scaled_13d"):
            result.loc[index, metric + "_ci_low"] = selected[metric].quantile(alpha / 2)
            result.loc[index, metric + "_ci_high"] = selected[metric].quantile(1 - alpha / 2)
    return result
