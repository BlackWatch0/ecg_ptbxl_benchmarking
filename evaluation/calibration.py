"""Calibration summaries and reliability tables for multilabel predictions."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import ArrayLike, _class_names, _validated_arrays


CALIBRATION_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "scope",
    "class_name",
    "n_samples",
    "n_positive",
    "prevalence",
    "brier_score",
    "log_loss",
    "ece",
    "mce",
    "mean_confidence",
    "positive_mean_probability",
    "negative_mean_probability",
)

CALIBRATION_BIN_COLUMNS: Tuple[str, ...] = (
    "scope",
    "class_name",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "fraction",
    "mean_probability",
    "observed_frequency",
    "calibration_error",
)


def flatten_multilabel(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten sample-class pairs after validating aligned multilabel arrays."""
    truth, probability = _validated_arrays(y_true, y_prob)
    return truth.ravel(), probability.ravel()


def _bin_edges(probability: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.maximum.accumulate(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def _binary_calibration(
    truth: np.ndarray,
    probability: np.ndarray,
    scope: str,
    class_name: Optional[str],
    n_bins: int,
    strategy: str,
) -> Tuple[dict, list]:
    edges = _bin_edges(probability, n_bins, strategy)
    # searchsorted assigns 1.0 to the final bin and handles duplicate quantiles.
    assignments = np.searchsorted(edges[1:-1], probability, side="right")
    bin_rows = []
    weighted_error = 0.0
    maximum_error = 0.0
    total = len(truth)
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability = float(np.mean(probability[mask])) if count else float("nan")
        observed_frequency = float(np.mean(truth[mask])) if count else float("nan")
        error = abs(mean_probability - observed_frequency) if count else float("nan")
        if count:
            weighted_error += count / float(total) * error
            maximum_error = max(maximum_error, error)
        bin_rows.append(
            {
                "scope": scope,
                "class_name": class_name,
                "bin_index": index,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "fraction": count / float(total),
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "calibration_error": error,
            }
        )
    epsilon = np.finfo(float).eps
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    summary = {
        "scope": scope,
        "class_name": class_name,
        "n_samples": total,
        "n_positive": int(np.sum(truth)),
        "prevalence": float(np.mean(truth)),
        "brier_score": float(np.mean((probability - truth) ** 2)),
        "log_loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1.0 - clipped))),
        "ece": float(weighted_error),
        "mce": float(maximum_error),
        "mean_confidence": float(np.mean(probability)),
        "positive_mean_probability": (float(np.mean(probability[truth == 1]))
                                      if np.any(truth == 1) else float("nan")),
        "negative_mean_probability": (float(np.mean(probability[truth == 0]))
                                      if np.any(truth == 0) else float("nan")),
    }
    return summary, bin_rows


def calibration_summary(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    include_flattened: bool = True,
    include_per_class: bool = True,
) -> pd.DataFrame:
    """Return flattened and/or per-class calibration metrics."""
    truth, probability = _validated_arrays(y_true, y_prob)
    names = _class_names(class_names, truth.shape[1])
    rows = []
    if include_flattened:
        summary, _ = _binary_calibration(truth.ravel(), probability.ravel(), "flattened", None, n_bins, strategy)
        rows.append(summary)
    if include_per_class:
        class_rows = []
        for index, name in enumerate(names):
            summary, _ = _binary_calibration(
                truth[:, index], probability[:, index], "per_class", name, n_bins, strategy
            )
            rows.append(summary)
            class_rows.append(summary)
        macro = {column: float(np.nanmean([row[column] for row in class_rows]))
                 for column in ("prevalence", "brier_score", "log_loss", "ece", "mce",
                                "mean_confidence", "positive_mean_probability",
                                "negative_mean_probability")}
        rows.append(dict(scope="macro", class_name=None, n_samples=len(truth),
                         n_positive=int(truth.sum()), **macro))
    if not rows:
        raise ValueError("at least one of include_flattened or include_per_class must be true")
    return pd.DataFrame(rows, columns=CALIBRATION_SUMMARY_COLUMNS)


def calibration_table(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    include_flattened: bool = True,
    include_per_class: bool = True,
) -> pd.DataFrame:
    """Return stable reliability-bin rows, including empty bins."""
    truth, probability = _validated_arrays(y_true, y_prob)
    names = _class_names(class_names, truth.shape[1])
    rows = []
    if include_flattened:
        _, bins = _binary_calibration(truth.ravel(), probability.ravel(), "flattened", None, n_bins, strategy)
        rows.extend(bins)
    if include_per_class:
        for index, name in enumerate(names):
            _, bins = _binary_calibration(
                truth[:, index], probability[:, index], "per_class", name, n_bins, strategy
            )
            rows.extend(bins)
    if not rows:
        raise ValueError("at least one of include_flattened or include_per_class must be true")
    return pd.DataFrame(rows, columns=CALIBRATION_BIN_COLUMNS)


def flattened_calibration(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """Return a flattened reliability table for all sample-class pairs."""
    return calibration_table(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, include_flattened=True, include_per_class=False
    )


def per_class_calibration(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Return reliability bins independently for every class."""
    return calibration_table(
        y_true,
        y_prob,
        class_names=class_names,
        n_bins=n_bins,
        strategy=strategy,
        include_flattened=False,
        include_per_class=True,
    )


compute_calibration_metrics = calibration_summary


def compute_calibration(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Return summary metrics and reliability bins through one stable API."""
    return {
        "summary": calibration_summary(y_true, y_prob, class_names, n_bins, strategy),
        "bins": calibration_table(y_true, y_prob, class_names, n_bins, strategy),
    }
