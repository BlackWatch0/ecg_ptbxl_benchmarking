"""Composite matching, robust feature metrics, and clustered bootstrap inference."""

from dataclasses import dataclass
import logging
from time import perf_counter

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, KEY_COLUMNS


METRIC_COLUMNS = ("mae", "rmse", "nae", "signed_mean", "signed_median", "absolute_mean", "absolute_median", "pearson_r", "spearman_r", "cosine_raw", "cosine_scaled")
BOOTSTRAP_METRICS = ("nae", "cosine_raw", "cosine_scaled")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapInputs:
    """Record-level sufficient statistics for clustered bootstrap metrics."""

    record_ids: np.ndarray
    nae_sums: np.ndarray
    nae_counts: np.ndarray
    raw_dot_sums: np.ndarray
    raw_left_square_sums: np.ndarray
    raw_right_square_sums: np.ndarray
    raw_counts: np.ndarray
    scaled_left_sums: np.ndarray
    scaled_right_sums: np.ndarray
    scaled_left_square_sums: np.ndarray
    scaled_right_square_sums: np.ndarray
    scaled_cross_sums: np.ndarray
    scaled_counts: np.ndarray


def snr_sort_key(value):
    return (value is None or pd.isna(value), float(value) if value is not None and not pd.isna(value) else 0.0)


def matching_report(data, comparison, snr):
    keys = list(KEY_COLUMNS)
    clean = data[data.Condition == "clean"][keys].copy()
    other = data[(data.Condition == comparison) & (data.SNR.isna() if pd.isna(snr) else data.SNR.eq(snr))][keys].copy()
    report = clean.merge(other, on=keys, how="outer", indicator=True)
    report["comparison"], report["SNR"] = comparison, snr
    report["match_status"] = report.pop("_merge").map({"both": "matched", "left_only": "clean_only", "right_only": "comparison_only"})
    return report[["comparison", "SNR"] + keys + ["match_status"]]


def pair_condition(data, comparison, snr, features=FEATURE_COLUMNS):
    keys, columns = list(KEY_COLUMNS), list(features)
    clean = data[data.Condition == "clean"][keys + columns]
    other = data[data.Condition.eq(comparison) & (data.SNR.isna() if pd.isna(snr) else data.SNR.eq(snr))][keys + columns]
    paired = clean.merge(other, on=keys, how="inner", validate="one_to_one", suffixes=("_clean", "_comparison"))
    paired["comparison"], paired["SNR"] = comparison, snr
    return paired


def aggregate_pairs(pairs, level, aggregation, features=FEATURE_COLUMNS):
    if level not in {"beat", "record"} or aggregation not in {"mean", "median"}:
        raise ValueError("level must be beat/record and aggregation must be mean/median")
    if level == "beat":
        return pairs.copy()
    columns = ["{}_{}".format(feature, side) for feature in features for side in ("clean", "comparison")]
    return pairs.groupby(["RecordNumber", "comparison", "SNR"], dropna=False, as_index=False)[columns].agg(aggregation)


def _rank(values):
    return pd.Series(values).rank(method="average").to_numpy()


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else np.nan


def _metrics(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    excluded = len(finite) - len(left)
    empty = {name: np.nan for name in METRIC_COLUMNS}
    if not len(left):
        return dict(empty, n_total=len(finite), n_valid=0, n_excluded=excluded, epsilon=np.nan)
    error, scale = right - left, np.median(np.abs(left))
    epsilon = max(np.finfo(float).eps, scale * 1e-8)
    correlation = np.corrcoef(left, right)[0, 1] if len(left) > 1 and np.std(left) and np.std(right) else np.nan
    spearman = np.corrcoef(_rank(left), _rank(right))[0, 1] if len(left) > 1 and np.std(_rank(left)) and np.std(_rank(right)) else np.nan
    scaled_left = (left - left.mean()) / left.std() if left.std() else np.zeros_like(left)
    scaled_right = (right - right.mean()) / right.std() if right.std() else np.zeros_like(right)
    return {"n_total": len(finite), "n_valid": len(left), "n_excluded": excluded, "epsilon": epsilon, "mae": np.mean(np.abs(error)), "rmse": np.sqrt(np.mean(error ** 2)), "nae": np.mean(np.abs(error) / (np.abs(left) + epsilon)), "signed_mean": np.mean(error), "signed_median": np.median(error), "absolute_mean": np.mean(np.abs(error)), "absolute_median": np.median(np.abs(error)), "pearson_r": correlation, "spearman_r": spearman, "cosine_raw": _cosine(left, right), "cosine_scaled": _cosine(scaled_left, scaled_right)}


def compute_metrics(data, features=FEATURE_COLUMNS, level="beat", aggregation="mean"):
    rows = []
    for feature in features:
        row = _metrics(data[feature + "_clean"].to_numpy(), data[feature + "_comparison"].to_numpy())
        row.update({"comparison": data.comparison.iloc[0], "SNR": data.SNR.iloc[0], "evaluation_level": level, "aggregation": aggregation, "feature": feature})
        rows.append(row)
    result = pd.DataFrame(rows)
    macro = {name: result[name].mean() for name in METRIC_COLUMNS}
    macro.update({"n_total": result.n_total.sum(), "n_valid": result.n_valid.sum(), "n_excluded": result.n_excluded.sum(), "epsilon": np.nan, "comparison": data.comparison.iloc[0], "SNR": data.SNR.iloc[0], "evaluation_level": level, "aggregation": aggregation, "feature": "__macro__"})
    return pd.concat([result, pd.DataFrame([macro])], ignore_index=True)


def bootstrap_inputs(data, features=FEATURE_COLUMNS):
    """Build per-record NAE and cosine sufficient statistics without DataFrame grouping."""
    record_ids, inverse = np.unique(data.RecordNumber.to_numpy(), return_inverse=True)
    shape = (len(record_ids), len(features))
    statistics = [np.zeros(shape, dtype=float) for _ in range(12)]
    for column, feature in enumerate(features):
        left = data[feature + "_clean"].to_numpy(dtype=float)
        right = data[feature + "_comparison"].to_numpy(dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        if not finite.any():
            continue
        valid_inverse, valid_left, valid_right = inverse[finite], left[finite], right[finite]
        epsilon = max(np.finfo(float).eps, np.median(np.abs(valid_left)) * 1e-8)
        values = (
            np.abs(valid_right - valid_left) / (np.abs(valid_left) + epsilon),
            np.ones(len(valid_left)),
            valid_left * valid_right,
            valid_left ** 2,
            valid_right ** 2,
            np.ones(len(valid_left)),
            valid_left,
            valid_right,
            valid_left ** 2,
            valid_right ** 2,
            valid_left * valid_right,
            np.ones(len(valid_left)),
        )
        for destination, values_for_statistic in enumerate(values):
            statistics[destination][:, column] = np.bincount(valid_inverse, weights=values_for_statistic, minlength=len(record_ids))
    return BootstrapInputs(record_ids, *statistics)


def _bootstrap_weights(indices, records):
    """Count cluster appearances per draw, keeping duplicate records as weights."""
    offsets = np.arange(len(indices))[:, None] * records
    return np.bincount((indices + offsets).ravel(), minlength=len(indices) * records).reshape(len(indices), records)


def _safe_ratio(numerator, denominator):
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _macro_mean(values):
    return np.nanmean(values) if np.isfinite(values).any() else np.nan


def _bootstrap_draw_values(inputs, weights):
    """Calculate each draw's feature values from cluster weights and sufficient statistics."""
    weighted = lambda statistic: weights @ statistic
    nae = _safe_ratio(weighted(inputs.nae_sums), weighted(inputs.nae_counts))
    raw_dot, raw_left, raw_right = (weighted(statistic) for statistic in (inputs.raw_dot_sums, inputs.raw_left_square_sums, inputs.raw_right_square_sums))
    raw = _safe_ratio(raw_dot, np.sqrt(raw_left * raw_right))
    count = weighted(inputs.scaled_counts)
    left_sum, right_sum, left_square, right_square, cross = (weighted(statistic) for statistic in (inputs.scaled_left_sums, inputs.scaled_right_sums, inputs.scaled_left_square_sums, inputs.scaled_right_square_sums, inputs.scaled_cross_sums))
    covariance = cross - left_sum * right_sum / np.where(count, count, 1)
    left_variance = left_square - left_sum ** 2 / np.where(count, count, 1)
    right_variance = right_square - right_sum ** 2 / np.where(count, count, 1)
    scaled = _safe_ratio(covariance, np.sqrt(left_variance * right_variance))
    return nae, raw, scaled


def bootstrap_metrics(data, features=FEATURE_COLUMNS, level="beat", aggregation="mean", iterations=1000, seed=0, batch_size=100, per_feature=False):
    """Cluster-bootstrap macro NAE/cosines using batched record-count weights."""
    if iterations < 1 or data.empty:
        return pd.DataFrame()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    started = perf_counter()
    inputs = bootstrap_inputs(data, features)
    preparation_seconds = perf_counter() - started
    rng, rows, records = np.random.default_rng(seed), [], len(inputs.record_ids)
    for first in range(0, iterations, batch_size):
        batch_iterations = min(batch_size, iterations - first)
        indices = rng.integers(records, size=(batch_iterations, records))
        nae, raw, scaled = _bootstrap_draw_values(inputs, _bootstrap_weights(indices, records))
        for offset in range(batch_iterations):
            values = (nae[offset], raw[offset], scaled[offset])
            if per_feature:
                for feature, feature_values in zip(features, zip(*values)):
                    rows.append((first + offset, feature, *feature_values))
            rows.append((first + offset, "__macro__", *(_macro_mean(value) for value in values)))
    LOGGER.info("Bootstrap prepared %d records in %.3fs; generated %d draws in %.3fs", records, preparation_seconds, iterations, perf_counter() - started - preparation_seconds)
    result = pd.DataFrame(rows, columns=["bootstrap_iteration", "feature", *BOOTSTRAP_METRICS])
    result["comparison"] = data.comparison.iloc[0]
    result["SNR"] = data.SNR.iloc[0]
    result["evaluation_level"] = level
    result["aggregation"] = aggregation
    return result[["bootstrap_iteration", "comparison", "SNR", "evaluation_level", "aggregation", "feature", *BOOTSTRAP_METRICS]]


def confidence_intervals(point, samples, alpha=0.05):
    if samples.empty:
        return point
    keys, rows = ["comparison", "SNR", "evaluation_level", "aggregation", "feature"], []
    for _, row in point.iterrows():
        selected = samples
        for key in keys:
            selected = selected[selected[key].isna()] if pd.isna(row[key]) else selected[selected[key].eq(row[key])]
        output = row.to_dict()
        for metric in METRIC_COLUMNS:
            if metric in BOOTSTRAP_METRICS and metric in selected:
                output[metric + "_ci_low"], output[metric + "_ci_high"] = selected[metric].quantile(alpha / 2), selected[metric].quantile(1 - alpha / 2)
            else:
                output[metric + "_ci_low"], output[metric + "_ci_high"] = np.nan, np.nan
        rows.append(output)
    return pd.DataFrame(rows)


def sample_errors(data, features=FEATURE_COLUMNS, limit=100):
    rows = []
    keys = [key for key in KEY_COLUMNS if key in data.columns]
    for feature in features:
        table = data[keys + ["comparison", "SNR", feature + "_clean", feature + "_comparison"]].copy()
        table = table.rename(columns={feature + "_clean": "clean_value", feature + "_comparison": "comparison_value"})
        table["feature"], table["signed_error"] = feature, table.comparison_value - table.clean_value
        table["absolute_error"] = table.signed_error.abs()
        rows.append(table[np.isfinite(table.clean_value) & np.isfinite(table.comparison_value)])
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return result.nlargest(limit, "absolute_error") if limit is not None else result
