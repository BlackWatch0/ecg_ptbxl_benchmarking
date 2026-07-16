"""Record-aligned Wavelet feature SNR robustness analysis.

This module deliberately does not load labels or train models.  Every metric
compares a noisy or denoised feature vector with the clean vector bearing the
same RecordName.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


LOGGER = logging.getLogger("wavelet_snr_robustness")
RECORD_COLUMN = "RecordName"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FEATURE_TYPES = ["cA4_Mean", "cA4_Std", "cA4_Energy", "cD2_cD3_Mean", "cD2_cD3_Std", "cD2_cD3_Energy"]
FEATURE_PATTERN = re.compile(
    r"^Lead_(?P<lead>I|II|III|aVR|aVL|aVF|V[1-6])_"
    r"(?P<band>cA4|cD2_cD3)_(?P<statistic>Mean|Std|Energy)$"
)


@dataclass
class LoadedFeatures:
    dataset_type: str
    snr: str
    path: Path
    frame: pd.DataFrame | None
    validation: dict[str, Any]
    error: str | None = None


def _snr_value(snr: str) -> float:
    try:
        return float(str(snr).replace("dB", "").strip())
    except ValueError:
        return -np.inf


def _metadata(feature: str) -> dict[str, str]:
    match = FEATURE_PATTERN.match(feature)
    if not match:
        return {"lead": "Unknown", "band": "Unknown", "statistic": "Unknown", "feature_type": feature}
    values = match.groupdict()
    values["feature_type"] = f"{values['band']}_{values['statistic']}"
    return values


def _empty_validation(dataset_type: str, snr: str, path: Path) -> dict[str, Any]:
    return {
        "dataset_type": dataset_type,
        "snr": snr,
        "file_path": str(path),
        "total_samples": 0,
        "matched_samples": 0,
        "missing_samples": 0,
        "extra_samples": 0,
        "duplicate_records": 0,
        "nan_count": 0,
        "inf_count": 0,
        "feature_count": 0,
        "non_numeric_columns": "",
        "constant_features": "",
        "column_match": False,
        "status": "not_loaded",
        "warnings": "",
    }


def _load_csv(dataset_type: str, snr: str, file_path: str | Path | None,
              record_id_mode: str = "strict", record_prefix_pattern: str = r"^(\d+_lr)") -> LoadedFeatures:
    path = Path(file_path).expanduser() if file_path else Path("")
    report = _empty_validation(dataset_type, snr, path)
    if not file_path:
        report.update(status="not_configured", warnings="No file path configured.")
        return LoadedFeatures(dataset_type, snr, path, None, report, report["warnings"])
    csv_paths = [path] if path.is_file() else sorted(path.rglob("*.csv")) if path.is_dir() else []
    if not csv_paths:
        report.update(status="missing_file", warnings="Configured file does not exist.")
        return LoadedFeatures(dataset_type, snr, path, None, report, report["warnings"])
    try:
        frame = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True)
    except Exception as error:  # noqa: BLE001 - show malformed CSV to notebook user.
        report.update(status="read_error", warnings=f"Unable to read CSV: {error}")
        return LoadedFeatures(dataset_type, snr, path, None, report, report["warnings"])
    source_id_column = RECORD_COLUMN if RECORD_COLUMN in frame.columns else "FileName" if "FileName" in frame.columns and record_id_mode == "prefix" else None
    if source_id_column is None:
        message = f"{path} has no {RECORD_COLUMN} column; reliable alignment is impossible."
        report.update(status="invalid", warnings=message)
        return LoadedFeatures(dataset_type, snr, path, None, report, message)
    if source_id_column != RECORD_COLUMN:
        frame = frame.rename(columns={source_id_column: RECORD_COLUMN})
    if record_id_mode == "prefix":
        frame[RECORD_COLUMN] = frame[RECORD_COLUMN].astype(str).str.extract(record_prefix_pattern, expand=False)
        if frame[RECORD_COLUMN].isna().any():
            message = f"{path} has record IDs that do not match prefix pattern {record_prefix_pattern!r}."
            report.update(status="invalid", warnings=message)
            return LoadedFeatures(dataset_type, snr, path, None, report, message)
    feature_columns = [column for column in frame.columns if column != RECORD_COLUMN]
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    duplicate_count = int(frame[RECORD_COLUMN].duplicated(keep=False).sum())
    invalid_ids = int(frame[RECORD_COLUMN].isna().sum() + (frame[RECORD_COLUMN].astype(str).str.strip() == "").sum())
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    constants = [column for column in feature_columns if numeric[column].dropna().nunique() <= 1]
    report.update(
        total_samples=len(frame), feature_count=len(feature_columns), duplicate_records=duplicate_count,
        nan_count=int(np.isnan(array).sum()), inf_count=int(np.isinf(array).sum()),
        non_numeric_columns=";".join(non_numeric), constant_features=";".join(constants),
    )
    warnings = []
    if duplicate_count or invalid_ids:
        warnings.append(f"{duplicate_count} duplicate and {invalid_ids} blank/missing RecordName values.")
    if non_numeric:
        warnings.append(f"Non-numeric feature columns: {', '.join(non_numeric)}.")
    if len(feature_columns) != 72:
        warnings.append(f"Expected 72 feature columns, found {len(feature_columns)}.")
    report.update(status="loaded" if not warnings else "loaded_with_warnings", warnings=" ".join(warnings))
    if duplicate_count or invalid_ids:
        return LoadedFeatures(dataset_type, snr, path, None, report, report["warnings"])
    frame = frame[[RECORD_COLUMN] + feature_columns].copy()
    frame[feature_columns] = numeric
    return LoadedFeatures(dataset_type, snr, path, frame, report)


def _align(clean: LoadedFeatures, candidate: LoadedFeatures) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if clean.frame is None or candidate.frame is None:
        return None
    clean_features = [column for column in clean.frame.columns if column != RECORD_COLUMN]
    candidate_features = [column for column in candidate.frame.columns if column != RECORD_COLUMN]
    missing_columns = sorted(set(clean_features) - set(candidate_features))
    extra_columns = sorted(set(candidate_features) - set(clean_features))
    clean_ids = set(clean.frame[RECORD_COLUMN])
    candidate_ids = set(candidate.frame[RECORD_COLUMN])
    matched_ids = sorted(clean_ids & candidate_ids)
    candidate.validation.update(
        matched_samples=len(matched_ids), missing_samples=len(clean_ids - candidate_ids),
        extra_samples=len(candidate_ids - clean_ids),
        column_match=not missing_columns and not extra_columns and clean_features == candidate_features,
    )
    warnings = [item for item in [candidate.validation.get("warnings", "")] if item]
    if missing_columns or extra_columns or clean_features != candidate_features:
        warnings.append(
            "Feature columns do not exactly match clean; comparison skipped. "
            f"Missing columns: {len(missing_columns)}; extra columns: {len(extra_columns)}."
        )
    missing_rate = 1 - len(matched_ids) / max(len(clean_ids), 1)
    if missing_rate >= 0.05:
        warnings.append(f"High clean-record missing rate: {missing_rate:.1%}.")
    candidate.validation["warnings"] = " ".join(warnings)
    if missing_columns or extra_columns or clean_features != candidate_features:
        candidate.validation["status"] = "incompatible_columns"
        return None
    if not matched_ids:
        candidate.validation["status"] = "no_matched_records"
        return None
    candidate.validation["status"] = "aligned" if not warnings else "aligned_with_warnings"
    clean_aligned = clean.frame.set_index(RECORD_COLUMN).loc[matched_ids].reset_index()
    candidate_aligned = candidate.frame.set_index(RECORD_COLUMN).loc[matched_ids].reset_index()
    return clean_aligned, candidate_aligned


def _clean_statistics(clean: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows = []
    for feature in clean.columns[1:]:
        values = clean[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        metadata = _metadata(feature)
        q1, q3 = np.nanpercentile(finite, [25, 75]) if finite.size else (np.nan, np.nan)
        std = np.nanstd(finite, ddof=1) if finite.size > 1 else np.nan
        rows.append({
            "feature": feature, **metadata, "mean": np.nanmean(finite) if finite.size else np.nan,
            "std": std, "median": np.nanmedian(finite) if finite.size else np.nan,
            "iqr": q3 - q1, "min": np.nanmin(finite) if finite.size else np.nan,
            "max": np.nanmax(finite) if finite.size else np.nan,
            "std_near_zero": bool(not np.isfinite(std) or abs(std) < epsilon),
        })
    return pd.DataFrame(rows)


def _safe_correlations(clean_values: np.ndarray, other_values: np.ndarray) -> tuple[float, float, str]:
    valid = np.isfinite(clean_values) & np.isfinite(other_values)
    if valid.sum() < 2:
        return np.nan, np.nan, "fewer than two finite paired values"
    x, y = clean_values[valid], other_values[valid]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, "constant clean or comparison feature"
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic), ""


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values.size else (np.nan, np.nan)


def _bootstrap(clean_z: np.ndarray, other_z: np.ndarray, nae: np.ndarray, cosine: np.ndarray,
               thresholds: list[float], iterations: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n_samples = nae.shape[0]
    values: dict[str, list[float]] = {"mean_nae": [], "median_nae": [], "mean_cosine_similarity": [], "mean_pearson": []}
    values.update({f"accuracy_{threshold}": [] for threshold in thresholds})
    for _ in range(iterations):
        index = rng.integers(0, n_samples, n_samples)
        boot_nae = nae[index]
        values["mean_nae"].append(float(np.nanmean(np.nanmean(boot_nae, axis=1))))
        values["median_nae"].append(float(np.nanmedian(np.nanmean(boot_nae, axis=1))))
        values["mean_cosine_similarity"].append(float(np.nanmean(cosine[index])))
        for threshold in thresholds:
            values[f"accuracy_{threshold}"].append(float(np.nanmean(boot_nae < threshold)))
        x, y = clean_z[index], other_z[index]
        valid = np.isfinite(x) & np.isfinite(y)
        count = valid.sum(axis=0)
        x_mean = np.divide(np.where(valid, x, 0).sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
        y_mean = np.divide(np.where(valid, y, 0).sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
        covariance = np.where(valid, (x - x_mean) * (y - y_mean), 0).sum(axis=0)
        x_ss = np.where(valid, (x - x_mean) ** 2, 0).sum(axis=0)
        y_ss = np.where(valid, (y - y_mean) ** 2, 0).sum(axis=0)
        corr = np.divide(covariance, np.sqrt(x_ss * y_ss), out=np.full(x.shape[1], np.nan), where=(x_ss > 0) & (y_ss > 0))
        values["mean_pearson"].append(float(np.nanmean(corr)))
    return {name: _confidence_interval(np.asarray(result)) for name, result in values.items()}


def _accuracy_columns(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    return {f"accuracy_{threshold}": float(np.nanmean(errors < threshold)) for threshold in thresholds}


def _analyse_pair(dataset_type: str, snr: str, clean: pd.DataFrame, other: pd.DataFrame,
                  clean_stats: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = list(clean.columns[1:])
    thresholds = [float(value) for value in config["preservation_thresholds"]]
    epsilon = float(config["epsilon"])
    clean_values, other_values = clean[features].to_numpy(float), other[features].to_numpy(float)
    means = clean_stats.set_index("feature").loc[features, "mean"].to_numpy(float)
    stds = clean_stats.set_index("feature").loc[features, "std"].to_numpy(float)
    scale = np.where(np.isfinite(stds) & (np.abs(stds) >= epsilon), stds, epsilon)
    delta = other_values - clean_values
    absolute = np.abs(delta)
    nae = absolute / scale
    relative = absolute / (np.abs(clean_values) + 0.1 * np.abs(scale) + epsilon)
    squared_z = (delta / scale) ** 2
    clean_z, other_z = (clean_values - means) / scale, (other_values - means) / scale
    valid_vector = np.isfinite(clean_z) & np.isfinite(other_z)
    dot = np.where(valid_vector, clean_z * other_z, 0).sum(axis=1)
    clean_norm = np.sqrt(np.where(valid_vector, clean_z ** 2, 0).sum(axis=1))
    other_norm = np.sqrt(np.where(valid_vector, other_z ** 2, 0).sum(axis=1))
    cosine = np.divide(dot, clean_norm * other_norm, out=np.full(len(clean), np.nan), where=(clean_norm > 0) & (other_norm > 0))
    euclidean = np.sqrt(np.where(valid_vector, (clean_z - other_z) ** 2, 0).sum(axis=1))
    correlations = [_safe_correlations(clean_values[:, index], other_values[:, index]) for index in range(len(features))]
    metadata = pd.DataFrame([_metadata(feature) for feature in features])
    feature_metrics = pd.DataFrame({
        "dataset_type": dataset_type, "snr": snr, "feature": features,
        "mean_absolute_error": np.nanmean(absolute, axis=0), "mean_nae": np.nanmean(nae, axis=0),
        "median_nae": np.nanmedian(nae, axis=0), "mean_relative_error": np.nanmean(relative, axis=0),
        "rmse_z": np.sqrt(np.nanmean(squared_z, axis=0)), "pearson": [item[0] for item in correlations],
        "spearman": [item[1] for item in correlations], "correlation_note": [item[2] for item in correlations],
    })
    feature_metrics = pd.concat([feature_metrics, metadata], axis=1)
    for threshold, accuracy in _accuracy_columns(nae, thresholds).items():
        feature_metrics[threshold] = np.nanmean(nae < float(threshold.replace("accuracy_", "")), axis=0)
    sample_metrics = pd.DataFrame({
        "dataset_type": dataset_type, "snr": snr, RECORD_COLUMN: clean[RECORD_COLUMN],
        "mean_nae": np.nanmean(nae, axis=1), "median_nae": np.nanmedian(nae, axis=1),
        "relative_error": np.nanmean(relative, axis=1), "cosine_similarity": cosine,
        "euclidean_distance": euclidean,
    })
    group_columns = {"lead": "lead", "feature_type": "feature_type"}
    grouped: dict[str, pd.DataFrame] = {}
    for output_name, column in group_columns.items():
        rows = []
        for group, group_frame in feature_metrics.groupby(column, sort=False):
            indices = group_frame.index.to_numpy()
            row = {"dataset_type": dataset_type, "snr": snr, output_name: group,
                   "mean_nae": float(np.nanmean(nae[:, indices])), "median_nae": float(np.nanmedian(nae[:, indices])),
                   "mean_absolute_error": float(np.nanmean(absolute[:, indices])),
                   "mean_relative_error": float(np.nanmean(relative[:, indices])),
                   "rmse_z": float(np.sqrt(np.nanmean(squared_z[:, indices]))),
                   "mean_pearson": float(np.nanmean(group_frame["pearson"])),
                   "mean_spearman": float(np.nanmean(group_frame["spearman"])),
                   "mean_cosine_similarity": float(np.nanmean(cosine))}
            row.update({key: float(np.nanmean(nae[:, indices] < float(key.replace("accuracy_", "")))) for key in _accuracy_columns(nae, thresholds)})
            rows.append(row)
        grouped[output_name] = pd.DataFrame(rows)
    overall = {
        "dataset_type": dataset_type, "snr": snr, "matched_samples": len(clean),
        "mean_nae": float(np.nanmean(sample_metrics["mean_nae"])), "median_nae": float(np.nanmedian(sample_metrics["mean_nae"])),
        "mean_relative_error": float(np.nanmean(relative)), "rmse_z": float(np.sqrt(np.nanmean(squared_z))),
        "mean_pearson": float(np.nanmean(feature_metrics["pearson"])), "mean_spearman": float(np.nanmean(feature_metrics["spearman"])),
        "mean_cosine_similarity": float(np.nanmean(cosine)), "median_cosine_similarity": float(np.nanmedian(cosine)),
        "cosine_std": float(np.nanstd(cosine, ddof=1)), "cosine_p25": float(np.nanpercentile(cosine, 25)),
        "cosine_p75": float(np.nanpercentile(cosine, 75)), "mean_euclidean_distance": float(np.nanmean(euclidean)),
        "euclidean_median": float(np.nanmedian(euclidean)), "euclidean_std": float(np.nanstd(euclidean, ddof=1)),
        "euclidean_p25": float(np.nanpercentile(euclidean, 25)), "euclidean_p75": float(np.nanpercentile(euclidean, 75)),
    }
    overall.update(_accuracy_columns(nae, thresholds))
    if config["enable_bootstrap_ci"]:
        confidence = _bootstrap(clean_z, other_z, nae, cosine, thresholds, int(config["bootstrap_iterations"]), int(config["random_seed"]))
        for name, (low, high) in confidence.items():
            overall[f"{name}_ci_low"], overall[f"{name}_ci_high"] = low, high
    return overall, grouped["lead"], grouped["feature_type"], feature_metrics, sample_metrics, {
        "nae": nae, "cosine": cosine, "feature_metrics": feature_metrics, "sample_metrics": sample_metrics,
    }


def _denoising_improvements(results: dict[tuple[str, str], dict[str, Any]], thresholds: list[float], epsilon: float) -> pd.DataFrame:
    rows = []
    shared_snrs = sorted({snr for dataset_type, snr in results if dataset_type == "noisy"} & {snr for dataset_type, snr in results if dataset_type == "denoised"}, key=_snr_value, reverse=True)
    for snr in shared_snrs:
        noisy, denoised = results[("noisy", snr)], results[("denoised", snr)]
        noisy_samples = noisy["sample_metrics"].set_index(RECORD_COLUMN)
        denoised_samples = denoised["sample_metrics"].set_index(RECORD_COLUMN)
        common = noisy_samples.index.intersection(denoised_samples.index)
        if not len(common):
            continue
        noisy_features = noisy["feature_metrics"].set_index("feature")
        denoised_features = denoised["feature_metrics"].set_index("feature")
        common_features = noisy_features.index.intersection(denoised_features.index)
        def append(group_level: str, group_name: str, n_nae: float, d_nae: float, n_cos: float, d_cos: float,
                   n_acc: dict[str, float], d_acc: dict[str, float]) -> None:
            change = n_nae - d_nae
            rows.append({"snr": snr, "group_level": group_level, "group_name": group_name,
                         "mean_nae_noisy": n_nae, "mean_nae_denoised": d_nae, "nae_improvement": change,
                         "relative_error_reduction_percent": change / (n_nae + epsilon) * 100,
                         "cosine_similarity_improvement": d_cos - n_cos,
                         "status": "improved" if change > epsilon else "degraded" if change < -epsilon else "unchanged",
                         **{f"feature_accuracy_improvement_{threshold}": d_acc[key] - n_acc[key] for threshold, key in [(t, f"accuracy_{t}") for t in thresholds]}})
        n_overall, d_overall = noisy["overall"], denoised["overall"]
        append("overall", "all_features", n_overall["mean_nae"], d_overall["mean_nae"], n_overall["mean_cosine_similarity"], d_overall["mean_cosine_similarity"], n_overall, d_overall)
        combined = pd.concat([noisy_features.add_prefix("noisy_"), denoised_features.add_prefix("denoised_")], axis=1, join="inner")
        for feature, row in combined.iterrows():
            append("feature", feature, row["noisy_mean_nae"], row["denoised_mean_nae"], n_overall["mean_cosine_similarity"], d_overall["mean_cosine_similarity"], row.filter(like="noisy_accuracy_").rename(lambda key: key.replace("noisy_", "")).to_dict(), row.filter(like="denoised_accuracy_").rename(lambda key: key.replace("denoised_", "")).to_dict())
        for group_level, field in [("lead", "lead"), ("feature_type", "feature_type")]:
            for group_name, group in combined.groupby(f"noisy_{field}"):
                n_acc = {f"accuracy_{t}": float(group[f"noisy_accuracy_{t}"].mean()) for t in thresholds}
                d_acc = {f"accuracy_{t}": float(group[f"denoised_accuracy_{t}"].mean()) for t in thresholds}
                append(group_level, group_name, float(group["noisy_mean_nae"].mean()), float(group["denoised_mean_nae"].mean()), n_overall["mean_cosine_similarity"], d_overall["mean_cosine_similarity"], n_acc, d_acc)
    return pd.DataFrame(rows)


def _save_line_plot(overall: pd.DataFrame, output: Path, metric: str, title: str, ylabel: str, ci: bool = False) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for dataset_type, group in overall.groupby("dataset_type"):
        group = group.sort_values("snr_value", ascending=False)
        x_labels = ["Clean"] + [f"{snr} dB" for snr in group["snr"]]
        values = [1.0 if metric.startswith("accuracy_") else 0.0] + group[metric].tolist()
        style = "-" if dataset_type == "noisy" else "--"
        axis.plot(range(len(values)), values, marker="o", linestyle=style, label=dataset_type)
        if ci and f"{metric}_ci_low" in group:
            low, high = group[f"{metric}_ci_low"].to_numpy(), group[f"{metric}_ci_high"].to_numpy()
            axis.errorbar(range(1, len(values)), group[metric], yerr=[group[metric] - low, high - group[metric]], fmt="none", capsize=3)
    axis.set_xticks(range(len(x_labels)), x_labels)
    axis.set_title(title); axis.set_ylabel(ylabel); axis.set_xlabel("SNR"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(output, dpi=300); plt.close(fig)


def _preservation_plot(overall: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for dataset_type, group in overall.groupby("dataset_type"):
        group = group.sort_values("snr_value", ascending=False)
        x_labels = ["Clean"] + [f"{snr} dB" for snr in group["snr"]]
        for threshold, color in [(0.1, "C0"), (0.25, "C1"), (0.5, "C2")]:
            metric = f"accuracy_{threshold}"
            style = "-" if dataset_type == "noisy" else "--"
            axis.plot(range(len(x_labels)), [1.0] + group[metric].tolist(), marker="o", color=color,
                      linestyle=style, label=f"{dataset_type} Accuracy@{threshold}")
    axis.set_xticks(range(len(x_labels)), x_labels)
    axis.set_title("Feature Preservation Accuracy vs SNR")
    axis.set_xlabel("SNR"); axis.set_ylabel("Feature Preservation Accuracy")
    axis.grid(alpha=.25); axis.legend(ncol=2); fig.tight_layout()
    fig.savefig(output, dpi=300); plt.close(fig)


def _heatmap(frame: pd.DataFrame, group_column: str, output: Path, title: str, columns: list[str] | None = None, separators: bool = False) -> None:
    pivot = frame.pivot(index="snr", columns=group_column, values="mean_nae")
    pivot = pivot.reindex(sorted(pivot.index, key=_snr_value, reverse=True))
    if columns:
        pivot = pivot.reindex(columns=columns)
    if pivot.empty:
        return
    fig, axis = plt.subplots(figsize=(max(9, pivot.shape[1] * .45), max(4, pivot.shape[0] * .6)))
    image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="magma")
    axis.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=90 if pivot.shape[1] > 12 else 0)
    axis.set_yticks(range(pivot.shape[0]), [f"{value} dB" for value in pivot.index])
    if separators:
        for boundary in range(6, pivot.shape[1], 6): axis.axvline(boundary - .5, color="white", linewidth=.8)
    axis.set_title(title); fig.colorbar(image, ax=axis, label="Mean Standardised Absolute Error"); fig.tight_layout()
    fig.savefig(output, dpi=300); plt.close(fig)


def _plots(overall: pd.DataFrame, lead: pd.DataFrame, feature_type: pd.DataFrame, feature: pd.DataFrame,
           sample: pd.DataFrame, improvement: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"; plot_dir.mkdir(exist_ok=True)
    _preservation_plot(overall, plot_dir / "feature_preservation_accuracy_vs_snr.png")
    _save_line_plot(overall, plot_dir / "normalised_feature_error_vs_snr.png", "mean_nae", "Normalised Feature Error vs SNR", "Mean Standardised Absolute Error", ci=True)
    for metric, label in [("mean_pearson", "Pearson correlation"), ("mean_spearman", "Spearman correlation"), ("mean_cosine_similarity", "Cosine similarity")]:
        _save_line_plot(overall, plot_dir / f"{metric}_vs_snr.png", metric, f"Feature Similarity: {label} vs SNR", label)
    for dataset_type in lead["dataset_type"].unique():
        _heatmap(lead[lead.dataset_type == dataset_type], "lead", plot_dir / f"lead_snr_heatmap_{dataset_type}.png", f"Lead x SNR Mean NAE ({dataset_type})", LEADS)
        _heatmap(feature_type[feature_type.dataset_type == dataset_type], "feature_type", plot_dir / f"feature_type_snr_heatmap_{dataset_type}.png", f"Feature Type x SNR Mean NAE ({dataset_type})", FEATURE_TYPES)
        ordered_features = feature[feature.dataset_type == dataset_type].sort_values(["lead", "feature_type"])["feature"].unique().tolist()
        _heatmap(feature[feature.dataset_type == dataset_type], "feature", plot_dir / f"full_72_feature_heatmap_{dataset_type}.png", f"Full Feature x SNR Mean NAE ({dataset_type})", ordered_features, separators=True)
    fig, axis = plt.subplots(figsize=(10, 5)); groups, labels = [], []
    for (dataset_type, snr), group in sample.sort_values("snr", key=lambda value: value.map(_snr_value), ascending=False).groupby(["dataset_type", "snr"]):
        groups.append(group["mean_nae"].dropna()); labels.append(f"{dataset_type}\n{snr} dB")
    if groups:
        axis.boxplot(groups, tick_labels=labels, showfliers=False); axis.set_ylabel("Sample Mean Standardised Absolute Error"); axis.set_title("Sample-level NAE"); fig.tight_layout(); fig.savefig(plot_dir / "sample_nae_boxplot.png", dpi=300)
    plt.close(fig)
    lowest_snr = feature.groupby("dataset_type")["snr"].transform(lambda values: values.map(_snr_value).min())
    lowest = feature[feature["snr"].map(_snr_value) == lowest_snr].nlargest(15, "mean_nae")
    if not lowest.empty:
        fig, axis = plt.subplots(figsize=(10, 6)); axis.barh(lowest["feature"], lowest["mean_nae"]); axis.invert_yaxis(); axis.set_title("Top 15 Most Noise-sensitive Features"); axis.set_xlabel("Mean Standardised Absolute Error"); fig.tight_layout(); fig.savefig(plot_dir / "top_15_noise_sensitive_features.png", dpi=300); plt.close(fig)
    if not improvement.empty:
        recovery = improvement[improvement.group_level == "overall"]
        fig, axis = plt.subplots(figsize=(8, 5)); axis.plot(recovery["snr"], recovery["relative_error_reduction_percent"], marker="o"); axis.set_title("Denoising Recovery Rate vs SNR"); axis.set_ylabel("NAE reduction (%)"); axis.set_xlabel("SNR (dB)"); axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(plot_dir / "denoising_recovery_rate.png", dpi=300); plt.close(fig)
        recovered = improvement[improvement.group_level == "feature"].nlargest(15, "nae_improvement")
        if not recovered.empty:
            fig, axis = plt.subplots(figsize=(10, 6)); axis.barh(recovered["group_name"], recovered["nae_improvement"]); axis.invert_yaxis(); axis.set_title("Top 15 Best Recovered Features After Denoising"); axis.set_xlabel("NAE improvement"); fig.tight_layout(); fig.savefig(plot_dir / "top_15_recovered_features.png", dpi=300); plt.close(fig)


def _summary(output: Path, validation: pd.DataFrame, overall: pd.DataFrame, lead: pd.DataFrame,
             feature_type: pd.DataFrame, feature: pd.DataFrame, improvement: pd.DataFrame) -> None:
    lines = ["# Wavelet Feature SNR Robustness Summary", "", "## Files and Alignment", ""]
    lines.extend(f"- {row.dataset_type} {row.snr} dB: `{row.file_path}`; {row.matched_samples}/{row.total_samples} matched samples; status={row.status}." for row in validation.itertuples())
    lines += ["", "## Per-SNR Overall Metrics", "", "```csv", overall.to_csv(index=False), "```", ""]
    if not overall.empty:
        lowest_snr = overall.loc[overall["snr"].map(_snr_value).idxmin(), "snr"]
        subset = lead[lead.snr == lowest_snr]
        if not subset.empty:
            lines += ["## Lowest-SNR Lead Stability", "", f"- Most sensitive lead: `{subset.loc[subset.mean_nae.idxmax(), 'lead']}`.", f"- Most stable lead: `{subset.loc[subset.mean_nae.idxmin(), 'lead']}`.", ""]
        subset = feature_type[feature_type.snr == lowest_snr]
        if not subset.empty:
            lines += ["## Lowest-SNR Feature-Type Stability", "", f"- Most sensitive feature type: `{subset.loc[subset.mean_nae.idxmax(), 'feature_type']}`.", f"- Most stable feature type: `{subset.loc[subset.mean_nae.idxmin(), 'feature_type']}`.", ""]
        top = feature[feature.snr == lowest_snr].nlargest(10, "mean_nae")
        lines += ["## Top 10 Sensitive Features", ""] + [f"- `{row.feature}`: mean NAE {row.mean_nae:.4g}." for row in top.itertuples()] + [""]
    if not improvement.empty:
        lines += ["## Denoising Recovery", ""]
        for row in improvement[improvement.group_level == "overall"].itertuples(): lines.append(f"- {row.snr} dB: mean NAE reduction {row.relative_error_reduction_percent:.2f}%.")
        degraded = improvement[(improvement.group_level == "feature") & (improvement.status == "degraded")]
        lines += ["", "Features degraded after denoising:", ""] + [f"- `{row.group_name}` at {row.snr} dB." for row in degraded.itertuples()] + [""]
    lines += ["## Interpretation Limits", "", "- High correlation does not imply small absolute or standardised error.", "- These comparisons are feature-preservation measurements, not classification accuracy or clinical evidence.", "- No medical causal inference should be drawn from this analysis.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run the complete no-label, no-model Wavelet feature robustness analysis.

    Required config keys: clean_feature_csv, noisy_feature_csvs, denoised_feature_csvs,
    output_dir. Each path may be one CSV or a directory of CSV chunks. SNR mapping
    values may be None; missing files are reported and skipped. Set
    record_id_mode='prefix' only when the configured prefix reliably identifies
    the same ECG record across clean and comparison files.
    """
    defaults = {"noisy_feature_csvs": {}, "denoised_feature_csvs": {}, "enable_bootstrap_ci": True,
                "bootstrap_iterations": 1000, "preservation_thresholds": [0.1, .25, .5, 1.0],
                "epsilon": 1e-8, "random_seed": 42, "record_id_mode": "strict",
                "record_prefix_pattern": r"^(\d+_lr)"}
    config = {**defaults, **config}
    output_dir = Path(config["output_dir"]).expanduser(); output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_loaded = _load_csv("clean", "clean", config.get("clean_feature_csv"), config["record_id_mode"], config["record_prefix_pattern"])
    if clean_loaded.frame is None:
        pd.DataFrame([clean_loaded.validation]).to_csv(output_dir / "data_validation_report.csv", index=False)
        raise ValueError(f"Clean CSV is required and invalid: {clean_loaded.error}")
    clean_loaded.validation.update(matched_samples=len(clean_loaded.frame), column_match=True, status="clean_reference")
    clean_stats = _clean_statistics(clean_loaded.frame, float(config["epsilon"]))
    clean_stats.to_csv(output_dir / "clean_feature_statistics.csv", index=False)
    loaded = [clean_loaded]
    for dataset_type, mapping in [("noisy", config["noisy_feature_csvs"]), ("denoised", config["denoised_feature_csvs"])]:
        for snr, path in mapping.items():
            loaded.append(_load_csv(dataset_type, str(snr), path, config["record_id_mode"], config["record_prefix_pattern"]))
    results: dict[tuple[str, str], dict[str, Any]] = {}
    all_overall: list[dict[str, Any]] = []; all_lead: list[pd.DataFrame] = []; all_type: list[pd.DataFrame] = []; all_feature: list[pd.DataFrame] = []; all_sample: list[pd.DataFrame] = []
    for candidate in loaded[1:]:
        aligned = _align(clean_loaded, candidate)
        if aligned is None:
            LOGGER.warning("Skipping %s %s dB: %s", candidate.dataset_type, candidate.snr, candidate.validation["warnings"])
            continue
        overall, lead, feature_type, feature, sample, internal = _analyse_pair(candidate.dataset_type, candidate.snr, *aligned, clean_stats, config)
        internal["overall"] = overall; results[(candidate.dataset_type, candidate.snr)] = internal
        all_overall.append(overall); all_lead.append(lead); all_type.append(feature_type); all_feature.append(feature); all_sample.append(sample)
    validation = pd.DataFrame([item.validation for item in loaded])
    overall = pd.DataFrame(all_overall); lead = pd.concat(all_lead, ignore_index=True) if all_lead else pd.DataFrame(); feature_type = pd.concat(all_type, ignore_index=True) if all_type else pd.DataFrame(); feature = pd.concat(all_feature, ignore_index=True) if all_feature else pd.DataFrame(); sample = pd.concat(all_sample, ignore_index=True) if all_sample else pd.DataFrame()
    if not overall.empty: overall["snr_value"] = overall["snr"].map(_snr_value)
    improvement = _denoising_improvements(results, [float(value) for value in config["preservation_thresholds"]], float(config["epsilon"])) if results else pd.DataFrame()
    validation.to_csv(output_dir / "data_validation_report.csv", index=False); overall.drop(columns="snr_value", errors="ignore").to_csv(output_dir / "overall_metrics.csv", index=False); lead.to_csv(output_dir / "per_lead_metrics.csv", index=False); feature_type.to_csv(output_dir / "per_feature_type_metrics.csv", index=False); feature.to_csv(output_dir / "per_feature_metrics.csv", index=False); sample.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    if not improvement.empty: improvement.to_csv(output_dir / "denoising_improvement.csv", index=False)
    if not overall.empty: _plots(overall, lead, feature_type, feature, sample, improvement, output_dir)
    _summary(output_dir, validation, overall.drop(columns="snr_value", errors="ignore"), lead, feature_type, feature, improvement)
    LOGGER.info("Analysis complete. Results written to %s", output_dir)
    return {"validation": validation, "clean_statistics": clean_stats, "overall": overall.drop(columns="snr_value", errors="ignore"), "per_lead": lead, "per_feature_type": feature_type, "per_feature": feature, "per_sample": sample, "improvement": improvement}
