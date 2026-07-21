"""Reusable metrics and threshold handling for multilabel evaluation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (average_precision_score, coverage_error, f1_score,
                             hamming_loss, label_ranking_average_precision_score,
                             label_ranking_loss, precision_score, recall_score,
                             roc_auc_score)


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

OVERALL_METRIC_COLUMNS: Tuple[str, ...] = (
    "macro_roc_auc",
    "micro_roc_auc",
    "weighted_roc_auc",
    "macro_pr_auc",
    "micro_pr_auc",
    "weighted_pr_auc",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "macro_precision",
    "micro_precision",
    "weighted_precision",
    "macro_recall",
    "micro_recall",
    "weighted_recall",
    "samples_f1",
    "samples_precision",
    "samples_recall",
    "label_accuracy",
    "exact_match_accuracy",
    "subset_accuracy",
    "hamming_loss",
    "predicted_positive_rate",
    "true_positive_rate",
    "mean_predicted_labels",
    "mean_true_labels",
    "all_zero_prediction_rate",
    "label_ranking_average_precision",
    "label_ranking_loss",
    "coverage_error",
    "average_test_loss",
    "batch_loss_mean",
    "batch_loss_std",
    "batch_loss_min",
    "batch_loss_max",
    "valid_roc_auc_class_count",
    "valid_pr_auc_class_count",
    "skipped_roc_auc_class_count",
    "skipped_pr_auc_class_count",
)

PER_CLASS_METRIC_COLUMNS: Tuple[str, ...] = (
    "class_index",
    "class_name",
    "support",
    "positive_count",
    "negative_count",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "predicted_positive_count",
    "predicted_positive_rate",
    "threshold",
    "valid_roc_auc",
    "valid_pr_auc",
    "warning",
)

BOOTSTRAP_CI_COLUMNS: Tuple[str, ...] = (
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "bootstrap_iterations",
    "seed",
    "n_valid",
)


def _validated_arrays(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated two-dimensional labels and probabilities."""
    truth = np.asarray(y_true)
    probability = np.asarray(y_prob, dtype=float)
    if truth.ndim != 2 or probability.ndim != 2:
        raise ValueError("y_true and y_prob must both have shape (n_samples, n_classes)")
    if truth.shape != probability.shape:
        raise ValueError("y_true and y_prob must have identical shapes")
    if truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("y_true and y_prob must not be empty")
    if not np.isfinite(probability).all():
        raise ValueError("y_prob must contain only finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("y_prob values must be probabilities in [0, 1]")
    if not np.isin(truth, (0, 1, False, True)).all():
        raise ValueError("y_true must contain only binary values")
    return truth.astype(np.uint8, copy=False), probability


def _class_names(class_names: Optional[Sequence[str]], n_classes: int) -> Tuple[str, ...]:
    names = tuple(str(name) for name in class_names) if class_names is not None else tuple(
        "class_{}".format(index) for index in range(n_classes)
    )
    if len(names) != n_classes:
        raise ValueError("class_names must contain one unique name per class")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def normalize_thresholds(thresholds: Union[float, Sequence[float], np.ndarray], n_classes: int) -> np.ndarray:
    """Normalize a scalar or per-class threshold specification."""
    values = np.asarray(thresholds, dtype=float)
    if values.ndim == 0:
        values = np.full(n_classes, float(values), dtype=float)
    elif values.ndim == 1 and len(values) == 1:
        values = np.full(n_classes, float(values[0]), dtype=float)
    elif values.ndim == 1 and len(values) == n_classes:
        values = values.copy()
    else:
        raise ValueError("thresholds must be a scalar or have length n_classes")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("thresholds must be finite values in [0, 1]")
    return values


def apply_thresholds(y_prob: ArrayLike, thresholds: Union[float, Sequence[float], np.ndarray] = 0.5) -> np.ndarray:
    """Convert a probability matrix to binary predictions."""
    probability = np.asarray(y_prob, dtype=float)
    if probability.ndim != 2:
        raise ValueError("y_prob must have shape (n_samples, n_classes)")
    if not np.isfinite(probability).all():
        raise ValueError("y_prob must contain only finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("y_prob values must be probabilities in [0, 1]")
    values = normalize_thresholds(thresholds, probability.shape[1])
    return (probability >= values[None, :]).astype(np.uint8)


def _undefined_auc(metric: str, class_name: Optional[str] = None) -> float:
    context = " for class {!r}".format(class_name) if class_name is not None else ""
    warnings.warn(
        "{} is undefined{} because both positive and negative labels are required; returning NaN".format(
            metric, context
        ),
        UndefinedMetricWarning,
        stacklevel=3,
    )
    return float("nan")


def _binary_auc(y_true: np.ndarray, y_prob: np.ndarray, metric: str, class_name: Optional[str] = None) -> float:
    if np.unique(y_true).size < 2:
        return _undefined_auc(metric, class_name)
    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    return float(scorer(y_true, y_prob))


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def compute_per_class_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    thresholds: Union[float, Sequence[float], np.ndarray] = 0.5,
    class_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute stable, one-row-per-class multilabel metrics.

    ROC-AUC and PR-AUC are reported as NaN with ``UndefinedMetricWarning`` when
    a class has only one observed label value.
    """
    truth, probability = _validated_arrays(y_true, y_prob)
    names = _class_names(class_names, truth.shape[1])
    threshold_values = normalize_thresholds(thresholds, truth.shape[1])
    prediction = apply_thresholds(probability, threshold_values)
    rows = []
    for index, name in enumerate(names):
        actual = truth[:, index]
        predicted = prediction[:, index]
        tp = int(np.sum((actual == 1) & (predicted == 1)))
        tn = int(np.sum((actual == 0) & (predicted == 0)))
        fp = int(np.sum((actual == 0) & (predicted == 1)))
        fn = int(np.sum((actual == 1) & (predicted == 0)))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        specificity = tn / float(tn + fp) if tn + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        valid_roc = bool((tp + fn) > 0 and (tn + fp) > 0)
        valid_pr = bool((tp + fn) > 0)
        warning_parts = []
        if not valid_roc:
            warning_parts.append("ROC-AUC requires positive and negative samples")
        if not valid_pr:
            warning_parts.append("PR-AUC requires at least one positive sample")
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "support": len(actual),
                "positive_count": tp + fn,
                "negative_count": tn + fp,
                "prevalence": float(np.mean(actual)),
                "roc_auc": _binary_auc(actual, probability[:, index], "roc_auc", name),
                "pr_auc": (float(average_precision_score(actual, probability[:, index]))
                           if valid_pr else _undefined_auc("pr_auc", name)),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "sensitivity": recall,
                "balanced_accuracy": (recall + specificity) / 2.0,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "predicted_positive_count": tp + fp,
                "predicted_positive_rate": float(np.mean(predicted)),
                "threshold": float(threshold_values[index]),
                "valid_roc_auc": valid_roc,
                "valid_pr_auc": valid_pr,
                "warning": "; ".join(warning_parts),
            }
        )
    return pd.DataFrame(rows, columns=PER_CLASS_METRIC_COLUMNS)


def compute_multilabel_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    thresholds: Union[float, Sequence[float], np.ndarray] = 0.5,
    batch_losses: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute aggregate discrimination and thresholded multilabel metrics."""
    truth, probability = _validated_arrays(y_true, y_prob)
    threshold_values = normalize_thresholds(thresholds, truth.shape[1])
    prediction = apply_thresholds(probability, threshold_values)
    roc_per_class = [_binary_auc(truth[:, i], probability[:, i], "roc_auc") for i in range(truth.shape[1])]
    pr_per_class = [float(average_precision_score(truth[:, i], probability[:, i]))
                    if truth[:, i].any() else _undefined_auc("pr_auc")
                    for i in range(truth.shape[1])]
    supports = truth.sum(axis=0).astype(float)
    def weighted(values: Sequence[float]) -> float:
        values_array = np.asarray(values, dtype=float)
        valid = np.isfinite(values_array) & (supports > 0)
        return float(np.average(values_array[valid], weights=supports[valid])) if valid.any() else float("nan")
    flattened_truth = truth.ravel()
    flattened_probability = probability.ravel()
    epsilon = np.finfo(float).eps
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    sample_losses = -np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1.0 - clipped), axis=1)
    reported_losses = np.asarray(batch_losses if batch_losses is not None else sample_losses, dtype=float)
    if reported_losses.ndim != 1 or not len(reported_losses) or not np.isfinite(reported_losses).all():
        raise ValueError("batch_losses must be a non-empty finite one-dimensional sequence")
    metrics = {
        "macro_roc_auc": _nanmean(roc_per_class),
        "micro_roc_auc": _binary_auc(flattened_truth, flattened_probability, "micro_roc_auc"),
        "weighted_roc_auc": weighted(roc_per_class),
        "macro_pr_auc": _nanmean(pr_per_class),
        "micro_pr_auc": _binary_auc(flattened_truth, flattened_probability, "micro_pr_auc"),
        "weighted_pr_auc": weighted(pr_per_class),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(truth, prediction, average="micro", zero_division=0)),
        "weighted_precision": float(precision_score(truth, prediction, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(truth, prediction, average="micro", zero_division=0)),
        "weighted_recall": float(recall_score(truth, prediction, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(truth, prediction, average="samples", zero_division=0)),
        "samples_precision": float(precision_score(truth, prediction, average="samples", zero_division=0)),
        "samples_recall": float(recall_score(truth, prediction, average="samples", zero_division=0)),
        "label_accuracy": float(np.mean(truth == prediction)),
        "exact_match_accuracy": float(np.mean(np.all(truth == prediction, axis=1))),
        "subset_accuracy": float(np.mean(np.all(truth == prediction, axis=1))),
        "hamming_loss": float(hamming_loss(truth, prediction)),
        "predicted_positive_rate": float(np.mean(prediction)),
        "true_positive_rate": float(np.mean(truth)),
        "mean_predicted_labels": float(np.mean(np.sum(prediction, axis=1))),
        "mean_true_labels": float(np.mean(np.sum(truth, axis=1))),
        "all_zero_prediction_rate": float(np.mean(np.sum(prediction, axis=1) == 0)),
        "label_ranking_average_precision": float(label_ranking_average_precision_score(truth, probability)),
        "label_ranking_loss": float(label_ranking_loss(truth, probability)),
        "coverage_error": float(coverage_error(truth, probability)),
        "average_test_loss": float(np.mean(sample_losses)),
        "batch_loss_mean": float(np.mean(reported_losses)),
        "batch_loss_std": float(np.std(reported_losses)),
        "batch_loss_min": float(np.min(reported_losses)),
        "batch_loss_max": float(np.max(reported_losses)),
        "valid_roc_auc_class_count": int(np.isfinite(roc_per_class).sum()),
        "valid_pr_auc_class_count": int(np.isfinite(pr_per_class).sum()),
        "skipped_roc_auc_class_count": int(np.isnan(roc_per_class).sum()),
        "skipped_pr_auc_class_count": int(np.isnan(pr_per_class).sum()),
    }
    return {column: metrics[column] for column in OVERALL_METRIC_COLUMNS}


def bootstrap_confidence_intervals(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    thresholds: Union[float, Sequence[float], np.ndarray] = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 0,
    metric_names: Optional[Iterable[str]] = None,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Mapping[str, float]]] = None,
) -> pd.DataFrame:
    """Estimate percentile confidence intervals by resampling complete samples.

    Resampling rows keeps all labels for a patient/sample together. Undefined
    bootstrap draws are omitted metric-by-metric and reflected in ``n_valid``.
    """
    truth, probability = _validated_arrays(y_true, y_prob)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    threshold_values = normalize_thresholds(thresholds, truth.shape[1])

    def default_metric_fn(sample_truth: np.ndarray, sample_probability: np.ndarray, values: np.ndarray) -> Mapping[str, float]:
        return compute_multilabel_metrics(sample_truth, sample_probability, values)

    evaluate = metric_fn or default_metric_fn
    estimate = dict(evaluate(truth, probability, threshold_values))
    selected = tuple(metric_names) if metric_names is not None else tuple(estimate.keys())
    unknown = set(selected).difference(estimate)
    if unknown:
        raise ValueError("unknown metric_names: {}".format(", ".join(sorted(unknown))))
    draws = {name: [] for name in selected}
    rng = np.random.RandomState(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for _ in range(n_bootstrap):
            indices = rng.randint(0, truth.shape[0], size=truth.shape[0])
            values = evaluate(truth[indices], probability[indices], threshold_values)
            for name in selected:
                value = float(values[name])
                if np.isfinite(value):
                    draws[name].append(value)
    tail = (1.0 - confidence_level) / 2.0
    rows = []
    for name in selected:
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": float(estimate[name]),
                "ci_low": float(np.quantile(values, tail)) if values.size else float("nan"),
                "ci_high": float(np.quantile(values, 1.0 - tail)) if values.size else float("nan"),
                "bootstrap_iterations": int(n_bootstrap),
                "seed": random_state,
                "n_valid": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_CI_COLUMNS)


@dataclass(frozen=True)
class ThresholdManager:
    """Resolve explicit thresholds without fitting or searching evaluation data.

    ``source_split`` documents provenance. A ``test`` source is rejected so a
    saved threshold artifact cannot claim to have been selected on test data.
    """

    strategy: str = "fixed_global"
    global_threshold: float = 0.5
    per_class_thresholds: Optional[Union[Sequence[float], Mapping[str, float], np.ndarray]] = None
    path: Optional[Union[str, Path]] = None
    source_split: str = "validation"

    def __post_init__(self) -> None:
        allowed = {"fixed_global", "fixed_per_class", "load_from_file"}
        if self.strategy not in allowed:
            raise ValueError("strategy must be one of {}".format(sorted(allowed)))
        if self.source_split.lower() == "test":
            raise ValueError("thresholds must not be selected or searched on the test split")
        if self.strategy == "fixed_per_class" and self.per_class_thresholds is None:
            raise ValueError("fixed_per_class requires per_class_thresholds")
        if self.strategy == "load_from_file" and self.path is None:
            raise ValueError("load_from_file requires path")
        if self.strategy == "fixed_global":
            normalize_thresholds(self.global_threshold, 1)

    def resolve(self, n_classes: int, class_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return one threshold per class in the requested class order."""
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        names = _class_names(class_names, n_classes)
        if self.strategy == "fixed_global":
            return normalize_thresholds(self.global_threshold, n_classes)
        specification = self.per_class_thresholds if self.strategy == "fixed_per_class" else self._load()
        if isinstance(specification, Mapping):
            missing = [name for name in names if name not in specification]
            extra = [name for name in specification if name not in names]
            if missing or extra:
                raise ValueError("threshold class names do not match; missing={}, extra={}".format(missing, extra))
            specification = [specification[name] for name in names]
        return normalize_thresholds(specification, n_classes)  # type: ignore[arg-type]

    def apply(self, y_prob: ArrayLike, class_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Apply the configured thresholds to probabilities."""
        probability = np.asarray(y_prob)
        if probability.ndim != 2:
            raise ValueError("y_prob must have shape (n_samples, n_classes)")
        return apply_thresholds(probability, self.resolve(probability.shape[1], class_names))

    def _load(self) -> Union[Sequence[float], Mapping[str, float], np.ndarray]:
        path = Path(self.path)  # type: ignore[arg-type]
        if not path.is_file():
            raise FileNotFoundError("threshold file does not exist: {}".format(path))
        suffix = path.suffix.lower()
        if suffix == ".npy":
            raise ValueError("NPY threshold files lack required source_split provenance; use JSON or CSV")
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or not payload.get("source_split"):
                raise ValueError("JSON threshold file requires source_split provenance")
            if str(payload.get("source_split", "")).lower() == "test":
                raise ValueError("threshold files selected or searched on the test split are not allowed")
            if str(payload["source_split"]).lower() != self.source_split.lower():
                raise ValueError("threshold file source_split does not match configured provenance")
            if isinstance(payload, dict) and "thresholds" in payload:
                payload = payload["thresholds"]
            if not isinstance(payload, (dict, list, tuple, float, int)):
                raise ValueError("JSON threshold file must contain a number, list, mapping, or 'thresholds' field")
            return payload
        if suffix == ".csv":
            frame = pd.read_csv(path)
            if "source_split" not in frame or frame["source_split"].isna().any():
                raise ValueError("CSV threshold file requires source_split provenance")
            if "source_split" in frame and frame["source_split"].astype(str).str.lower().eq("test").any():
                raise ValueError("threshold files selected or searched on the test split are not allowed")
            sources = set(frame["source_split"].astype(str).str.lower())
            if sources != {self.source_split.lower()}:
                raise ValueError("threshold file source_split does not match configured provenance")
            if "threshold" not in frame.columns:
                raise ValueError("CSV threshold file requires a 'threshold' column")
            if "class_name" in frame.columns:
                return dict(zip(frame["class_name"].astype(str), frame["threshold"].astype(float)))
            return frame["threshold"].to_numpy(dtype=float)
        raise ValueError("threshold files must use .json, .csv, or .npy")


# Concise aliases for callers that use evaluation-oriented naming.
multilabel_metrics = compute_multilabel_metrics
per_class_metrics = compute_per_class_metrics
bootstrap_multilabel_metrics = bootstrap_confidence_intervals


def paired_bootstrap_difference(
    y_true: ArrayLike,
    first_prob: ArrayLike,
    second_prob: ArrayLike,
    metric_names: Sequence[str] = ("macro_roc_auc", "macro_f1"),
    thresholds: Union[float, Sequence[float], np.ndarray] = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Paired sample bootstrap for two ID-aligned model prediction matrices."""
    truth, first = _validated_arrays(y_true, first_prob)
    _, second = _validated_arrays(y_true, second_prob)
    values = normalize_thresholds(thresholds, truth.shape[1])
    first_estimate = compute_multilabel_metrics(truth, first, values)
    second_estimate = compute_multilabel_metrics(truth, second, values)
    rng = np.random.RandomState(random_state)
    draws = {name: [] for name in metric_names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for _ in range(n_bootstrap):
            indices = rng.randint(0, len(truth), len(truth))
            left = compute_multilabel_metrics(truth[indices], first[indices], values)
            right = compute_multilabel_metrics(truth[indices], second[indices], values)
            for name in metric_names:
                difference = float(left[name] - right[name])
                if np.isfinite(difference):
                    draws[name].append(difference)
    rows = []
    for name in metric_names:
        samples = np.asarray(draws[name], dtype=float)
        rows.append({"metric": name,
                     "estimate": float(first_estimate[name] - second_estimate[name]),
                     "ci_low": float(np.quantile(samples, .025)) if len(samples) else np.nan,
                     "ci_high": float(np.quantile(samples, .975)) if len(samples) else np.nan,
                     "bootstrap_iterations": n_bootstrap, "seed": random_state,
                     "n_valid": len(samples)})
    return pd.DataFrame(rows)


def compute_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    thresholds: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    batch_losses: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return aggregate and per-class tables through one evaluator-facing API."""
    selected = 0.5 if thresholds is None else thresholds
    truth, probability = _validated_arrays(y_true, y_prob)
    values = normalize_thresholds(selected, truth.shape[1])
    return {
        "overall": compute_multilabel_metrics(truth, probability, values, batch_losses),
        "per_class": compute_per_class_metrics(truth, probability, values, class_names),
        "thresholds": values,
    }
