"""Domain robustness summaries and denoising recovery metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ROBUSTNESS_VALUE_COLUMNS: Tuple[str, ...] = (
    "clean_value",
    "mean_value",
    "worst_value",
    "mean_absolute_drop",
    "worst_absolute_drop",
    "mean_relative_drop",
    "worst_relative_drop",
    "auc_over_snr",
    "normalized_auc_over_snr",
    "mean_performance_retention",
    "clean_to_min_snr_absolute_drop",
    "clean_to_min_snr_relative_drop",
    "max_adjacent_drop",
    "max_drop_from_snr",
    "max_drop_to_snr",
    "n_conditions",
)

DENOISING_RECOVERY_COLUMNS: Tuple[str, ...] = (
    "clean_value",
    "noisy_value",
    "denoised_value",
    "degradation",
    "denoising_improvement",
    "recovery_fraction",
    "recovery_percent",
    "remaining_gap",
)


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional sequence".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def compute_robustness_summary(
    clean_value: float,
    condition_values: Sequence[float],
    snr_db: Optional[Sequence[float]] = None,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Summarize performance across perturbed conditions relative to clean."""
    clean = float(clean_value)
    if not np.isfinite(clean):
        raise ValueError("clean_value must be finite")
    values = _finite_vector(condition_values, "condition_values")
    direction = 1.0 if higher_is_better else -1.0
    drops = direction * (clean - values)
    denominator = abs(clean)
    relative = drops / denominator if denominator > 0.0 else np.full(values.shape, np.nan)
    oriented = direction * values
    worst_index = int(np.argmin(oriented))
    auc = float("nan")
    normalized_auc = float("nan")
    min_snr_drop = min_snr_relative = max_adjacent = float("nan")
    max_from = max_to = float("nan")
    if snr_db is not None:
        snr = _finite_vector(snr_db, "snr_db")
        if snr.shape != values.shape:
            raise ValueError("snr_db and condition_values must have equal lengths")
        if len(snr) >= 2:
            order = np.argsort(snr)
            span = float(snr[order][-1] - snr[order][0])
            integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            auc = float(integrate(values[order], snr[order]))
            normalized_auc = auc / span if span > 0.0 else float("nan")
            descending = np.argsort(snr)[::-1]
            adjacent = direction * (values[descending][:-1] - values[descending][1:])
            if len(adjacent):
                location = int(np.argmax(adjacent))
                max_adjacent = float(adjacent[location])
                max_from = float(snr[descending][location])
                max_to = float(snr[descending][location + 1])
            minimum = int(np.argmin(snr))
            min_snr_drop = float(direction * (clean - values[minimum]))
            min_snr_relative = (min_snr_drop / abs(clean) if clean != 0 else float("nan"))
    result = {
        "clean_value": clean,
        "mean_value": float(np.mean(values)),
        "worst_value": float(values[worst_index]),
        "mean_absolute_drop": float(np.mean(drops)),
        "worst_absolute_drop": float(np.max(drops)),
        "mean_relative_drop": float(np.mean(relative)) if np.isfinite(relative).any() else float("nan"),
        "worst_relative_drop": float(np.max(relative)) if np.isfinite(relative).any() else float("nan"),
        "auc_over_snr": auc,
        "normalized_auc_over_snr": normalized_auc,
        "mean_performance_retention": (float(np.mean(values / clean))
                                       if clean != 0 else float("nan")),
        "clean_to_min_snr_absolute_drop": min_snr_drop,
        "clean_to_min_snr_relative_drop": min_snr_relative,
        "max_adjacent_drop": max_adjacent,
        "max_drop_from_snr": max_from,
        "max_drop_to_snr": max_to,
        "n_conditions": int(values.size),
    }
    return {column: result[column] for column in ROBUSTNESS_VALUE_COLUMNS}


def compute_denoising_recovery(
    clean_value: float,
    noisy_value: float,
    denoised_value: float,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Quantify the fraction of noisy-to-clean degradation recovered.

    Values above one indicate improvement beyond clean performance; negative
    values indicate that denoising made the metric worse.
    """
    clean, noisy, denoised = float(clean_value), float(noisy_value), float(denoised_value)
    if not np.isfinite([clean, noisy, denoised]).all():
        raise ValueError("clean_value, noisy_value, and denoised_value must be finite")
    direction = 1.0 if higher_is_better else -1.0
    degradation = direction * (clean - noisy)
    improvement = direction * (denoised - noisy)
    recovery = improvement / degradation if degradation != 0.0 else float("nan")
    result = {
        "clean_value": clean,
        "noisy_value": noisy,
        "denoised_value": denoised,
        "degradation": float(degradation),
        "denoising_improvement": float(improvement),
        "recovery_fraction": float(recovery),
        "recovery_percent": float(100.0 * recovery),
        "remaining_gap": float(direction * (clean - denoised)),
    }
    return {column: result[column] for column in DENOISING_RECOVERY_COLUMNS}


def _metric_direction(metric: str, higher_is_better: Union[bool, Mapping[str, bool]]) -> bool:
    return bool(higher_is_better[metric]) if isinstance(higher_is_better, Mapping) else bool(higher_is_better)


def summarize_robustness(
    results: pd.DataFrame,
    metric_columns: Iterable[str],
    group_columns: Sequence[str] = ("model",),
    domain_column: str = "domain",
    snr_column: str = "snr_db",
    clean_domain: str = "clean",
    higher_is_better: Union[bool, Mapping[str, bool]] = True,
) -> pd.DataFrame:
    """Create per-group, per-domain robustness rows from a wide result table."""
    metrics = tuple(metric_columns)
    required = set(group_columns).union({domain_column}).union(metrics)
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("results is missing columns: {}".format(", ".join(missing)))
    rows = []
    grouper = list(group_columns) if len(group_columns) > 1 else group_columns[0]
    for group_key, group in results.groupby(grouper, sort=False, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_columns, keys))
        clean_rows = group[group[domain_column] == clean_domain]
        if clean_rows.empty:
            raise ValueError("missing clean row for group {}".format(identity))
        for domain, domain_rows in group[group[domain_column] != clean_domain].groupby(domain_column, sort=False):
            for metric in metrics:
                values = domain_rows[metric].to_numpy(dtype=float)
                snr = domain_rows[snr_column].to_numpy(dtype=float) if snr_column in domain_rows else None
                summary = compute_robustness_summary(
                    float(clean_rows[metric].mean()), values, snr, _metric_direction(metric, higher_is_better)
                )
                rows.append(dict(identity, **{domain_column: domain, "metric": metric}, **summary))
    columns = tuple(group_columns) + (domain_column, "metric") + ROBUSTNESS_VALUE_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def denoising_recovery(
    results: pd.DataFrame,
    metric_columns: Iterable[str],
    group_columns: Sequence[str] = ("model",),
    domain_column: str = "domain",
    snr_column: str = "snr_db",
    clean_domain: str = "clean",
    noisy_domain: str = "noisy",
    denoised_domain: str = "denoised",
    higher_is_better: Union[bool, Mapping[str, bool]] = True,
) -> pd.DataFrame:
    """Compute denoising recovery after matching group, SNR, and metric."""
    metrics = tuple(metric_columns)
    required = set(group_columns).union({domain_column, snr_column}).union(metrics)
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError("results is missing columns: {}".format(", ".join(missing)))
    rows = []
    grouper = list(group_columns) if len(group_columns) > 1 else group_columns[0]
    for group_key, group in results.groupby(grouper, sort=False, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_columns, keys))
        clean_rows = group[group[domain_column] == clean_domain]
        noisy = group[group[domain_column] == noisy_domain]
        denoised = group[group[domain_column] == denoised_domain]
        if clean_rows.empty:
            raise ValueError("missing clean row for group {}".format(identity))
        merged = noisy.merge(denoised, on=list(group_columns) + [snr_column], suffixes=("_noisy", "_denoised"))
        for _, pair in merged.iterrows():
            for metric in metrics:
                recovery = compute_denoising_recovery(
                    float(clean_rows[metric].mean()),
                    float(pair[metric + "_noisy"]),
                    float(pair[metric + "_denoised"]),
                    _metric_direction(metric, higher_is_better),
                )
                rows.append(dict(identity, **{snr_column: pair[snr_column], "metric": metric}, **recovery))
    columns = tuple(group_columns) + (snr_column, "metric") + DENOISING_RECOVERY_COLUMNS
    return pd.DataFrame(rows, columns=columns)


robustness_summary = compute_robustness_summary


def compute_robustness(records: Sequence[object], class_names: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """Evaluate record conditions and return robustness and recovery tables.

    Records must expose ``probabilities`` and ``labels`` arrays. Optional
    ``condition``, ``snr``, ``model_name``, and ``scenario_name`` attributes are
    used to partition rows and identify clean/noisy/denoised domains.
    """
    del class_names  # Aggregate metrics do not depend on display labels.
    from .metrics import OVERALL_METRIC_COLUMNS, compute_multilabel_metrics

    rows = []
    for record in records:
        truth = np.asarray(getattr(record, "labels"))
        probability = np.asarray(getattr(record, "probabilities"))
        if truth.shape != probability.shape:
            raise ValueError("record labels and probabilities must have identical shapes")
        n_samples = truth.shape[0]
        conditions = np.asarray(getattr(record, "condition", np.repeat("unknown", n_samples))).astype(str)
        snr = np.asarray(getattr(record, "snr", np.repeat(np.nan, n_samples)), dtype=float)
        if conditions.ndim == 0:
            conditions = np.repeat(conditions, n_samples)
        if snr.ndim == 0:
            snr = np.repeat(snr, n_samples)
        if len(conditions) != n_samples or len(snr) != n_samples:
            raise ValueError("record condition and snr arrays must align with samples")
        model = str(getattr(record, "model_name", "model"))
        scenario = str(getattr(record, "scenario_name", "scenario"))
        partition = pd.DataFrame({"condition": conditions, "snr_db": snr}).groupby(
            ["condition", "snr_db"], sort=False, dropna=False
        ).indices
        for (condition, snr_value), indices in partition.items():
            metrics = compute_multilabel_metrics(truth[indices], probability[indices])
            rows.append(
                dict(
                    {"model": model, "scenario": scenario, "domain": condition, "snr_db": snr_value},
                    **metrics
                )
            )
    condition_columns = ("model", "scenario", "domain", "snr_db") + OVERALL_METRIC_COLUMNS
    condition_metrics = pd.DataFrame(rows, columns=condition_columns)
    summary_columns = ("model", "domain", "metric") + ROBUSTNESS_VALUE_COLUMNS
    recovery_columns = ("model", "snr_db", "metric") + DENOISING_RECOVERY_COLUMNS
    summary = pd.DataFrame(columns=summary_columns)
    recovery = pd.DataFrame(columns=recovery_columns)
    domains = set(condition_metrics["domain"]) if len(condition_metrics) else set()
    metric_names = ("macro_roc_auc", "macro_pr_auc", "macro_f1", "micro_f1")
    if "clean" in domains and domains.difference({"clean"}):
        summary = summarize_robustness(condition_metrics, metric_names)
    if {"clean", "noisy", "denoised"}.issubset(domains):
        recovery = denoising_recovery(condition_metrics, metric_names)
    return {"condition_metrics": condition_metrics, "summary": summary, "denoising_recovery": recovery}
