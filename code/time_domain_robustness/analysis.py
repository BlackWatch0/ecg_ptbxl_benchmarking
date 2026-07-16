"""Composite matching, robust feature metrics, and clustered bootstrap inference."""

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, KEY_COLUMNS


METRIC_COLUMNS = ("mae", "rmse", "nae", "signed_mean", "signed_median", "absolute_mean", "absolute_median", "pearson_r", "spearman_r", "cosine_raw", "cosine_scaled")


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


def bootstrap_metrics(data, features=FEATURE_COLUMNS, level="beat", aggregation="mean", iterations=1000, seed=0):
    if iterations < 1 or data.empty:
        return pd.DataFrame()
    groups, rng, samples = [group for _, group in data.groupby("RecordNumber", sort=False)], np.random.RandomState(seed), []
    for iteration in range(iterations):
        sample = pd.concat([groups[index] for index in rng.randint(len(groups), size=len(groups))], ignore_index=True)
        point = compute_metrics(sample, features, level, aggregation)
        point.insert(0, "bootstrap_iteration", iteration)
        samples.append(point)
    return pd.concat(samples, ignore_index=True)


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
            output[metric + "_ci_low"], output[metric + "_ci_high"] = selected[metric].quantile(alpha / 2), selected[metric].quantile(1 - alpha / 2)
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
